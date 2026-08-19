"""Five isolated backends for the formal H20 RawData1 benchmark.

Each performance invocation imports exactly one source tree.  The Slurm
driver launches variants in separate processes, so Python module state never
mixes the official extension and the ProbeEP fork.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


class BackendUnavailable(RuntimeError):
    """The selected build does not expose the requested test entry point."""


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


@dataclass(frozen=True)
class OfficialLayout:
    num_tokens_per_rank: torch.Tensor
    num_tokens_per_rdma_rank: torch.Tensor
    num_tokens_per_expert: torch.Tensor
    is_token_in_rank: torch.Tensor
    event: Any | None = None


@dataclass(frozen=True)
class DispatchResult:
    exec_x: torch.Tensor
    exec_scales: torch.Tensor
    handle: Any
    recv_topk_idx: torch.Tensor | None
    recv_topk_weights: torch.Tensor | None
    exec_counts: torch.Tensor | None
    event: Any | None = None
    # Post-timing wire telemetry.  ``wire_units`` is kept on device and is
    # consumed only after the measured CUDA-event interval; retaining it here
    # avoids a second routing/counting pass in the benchmark hot path.
    wire_units: torch.Tensor | None = None
    wire_unit_scope: str = ""
    wire_bytes_per_unit: int = 0
    wire_traffic_source: str = ""
    wire_routing_map: torch.Tensor | None = None


@dataclass(frozen=True)
class TorchEventHandle:
    event: torch.cuda.Event

    def current_stream_wait(self) -> None:
        torch.cuda.current_stream().wait_event(self.event)


@dataclass(frozen=True)
class NCCLLayout:
    routing_ref: torch.Tensor
    order: torch.Tensor
    source_token: torch.Tensor
    local_expert: torch.Tensor
    send_counts: tuple[int, ...]
    recv_counts: tuple[int, ...]
    send_counts_device: torch.Tensor
    event: TorchEventHandle | None = None


@dataclass(frozen=True)
class NCCLHandle:
    source_token: torch.Tensor
    num_source_tokens: int
    send_counts: tuple[int, ...]
    recv_counts: tuple[int, ...]


@dataclass
class NCCLBackend:
    """Route-occurrence NCCL all-to-all baseline without expert balancing."""

    variant: str
    group: Any
    num_experts: int
    local_experts: int
    comm_stream: torch.cuda.Stream = field(init=False)
    _layout_cache: dict[tuple[int, tuple[int, ...]], NCCLLayout] = field(
        default_factory=dict, init=False
    )

    def __post_init__(self) -> None:
        self.comm_stream = torch.cuda.Stream()

    @classmethod
    def load(
        cls,
        group: Any,
        *,
        num_experts: int,
        local_experts: int | None,
    ) -> "NCCLBackend":
        if local_experts is None or local_experts <= 0:
            raise BackendUnavailable("NCCL baseline requires local_experts")
        if num_experts != int(group.size()) * int(local_experts):
            raise BackendUnavailable(
                "NCCL baseline requires canonical contiguous expert ownership"
            )
        return cls("nccl", group, num_experts, int(local_experts))

    @property
    def balanced(self) -> bool:
        return False

    @property
    def buffer(self) -> "NCCLBackend":
        return self

    def get_comm_stream(self) -> torch.cuda.Stream:
        return self.comm_stream

    def capture_event(self) -> TorchEventHandle:
        event = torch.cuda.Event()
        event.record()
        return TorchEventHandle(event)

    @staticmethod
    def _wait(previous_event: Any | None) -> None:
        if previous_event is not None:
            previous_event.current_stream_wait()

    def plan(
        self,
        topk_idx: torch.Tensor,
        *,
        previous_event: Any | None = None,
        async_finish: bool = False,
    ) -> NCCLLayout:
        key = (int(topk_idx.data_ptr()), tuple(topk_idx.shape))
        cached = self._layout_cache.get(key)
        if cached is not None:
            return cached
        with torch.cuda.stream(self.comm_stream):
            self._wait(previous_event)
            tokens, topk = topk_idx.shape
            flat_expert = topk_idx.reshape(-1)
            source_token = torch.arange(
                tokens, device=topk_idx.device, dtype=torch.int64
            ).repeat_interleave(topk)
            destination = torch.div(
                flat_expert, self.local_experts, rounding_mode="floor"
            )
            order = torch.argsort(
                destination * self.num_experts + flat_expert, stable=True
            )
            send_counts_tensor = torch.bincount(
                destination.index_select(0, order),
                minlength=int(self.group.size()),
            ).to(torch.int64)
            recv_counts_tensor = torch.empty_like(send_counts_tensor)
            dist.all_to_all_single(
                recv_counts_tensor, send_counts_tensor, group=self.group
            )
            event = torch.cuda.Event()
            event.record(self.comm_stream)
        # PyTorch requires host split sizes. This one-time conversion is
        # completed during untimed warmup and cached for the immutable layer.
        event.synchronize()
        layout = NCCLLayout(
            routing_ref=topk_idx,
            order=order,
            source_token=source_token.index_select(0, order),
            local_expert=(flat_expert.index_select(0, order) % self.local_experts),
            send_counts=tuple(int(value) for value in send_counts_tensor.cpu().tolist()),
            recv_counts=tuple(int(value) for value in recv_counts_tensor.cpu().tolist()),
            send_counts_device=send_counts_tensor,
            event=TorchEventHandle(event) if async_finish else None,
        )
        self._layout_cache[key] = layout
        return layout

    def dispatch(
        self,
        x_fp8: torch.Tensor,
        x_scales: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        layout: NCCLLayout | None,
        compute_kind: int = 1,
        previous_event: Any | None = None,
        async_finish: bool = False,
    ) -> DispatchResult:
        del topk_idx, compute_kind
        if layout is None:
            raise ValueError("NCCL dispatch requires a cached occurrence layout")
        total_recv = sum(layout.recv_counts)
        with torch.cuda.stream(self.comm_stream):
            self._wait(previous_event or layout.event)
            send_x = x_fp8.index_select(0, layout.source_token)
            recv_x_bytes = torch.empty(
                (total_recv, x_fp8.size(1)), dtype=torch.uint8, device=x_fp8.device
            )
            dist.all_to_all_single(
                recv_x_bytes,
                send_x.view(torch.uint8),
                output_split_sizes=list(layout.recv_counts),
                input_split_sizes=list(layout.send_counts),
                group=self.group,
            )
            send_scales = x_scales.index_select(0, layout.source_token)
            recv_scales = torch.empty(
                (total_recv, x_scales.size(1)),
                dtype=x_scales.dtype,
                device=x_scales.device,
            )
            dist.all_to_all_single(
                recv_scales,
                send_scales,
                output_split_sizes=list(layout.recv_counts),
                input_split_sizes=list(layout.send_counts),
                group=self.group,
            )
            recv_expert = torch.empty(
                total_recv, dtype=torch.int64, device=x_fp8.device
            )
            dist.all_to_all_single(
                recv_expert,
                layout.local_expert,
                output_split_sizes=list(layout.recv_counts),
                input_split_sizes=list(layout.send_counts),
                group=self.group,
            )
            send_weights = topk_weights.reshape(-1).index_select(0, layout.order)
            recv_weights = torch.empty(
                total_recv, dtype=topk_weights.dtype, device=topk_weights.device
            )
            dist.all_to_all_single(
                recv_weights,
                send_weights,
                output_split_sizes=list(layout.recv_counts),
                input_split_sizes=list(layout.send_counts),
                group=self.group,
            )
            event = torch.cuda.Event()
            event.record(self.comm_stream)
        return DispatchResult(
            exec_x=recv_x_bytes.view(torch.float8_e4m3fn),
            exec_scales=recv_scales,
            handle=NCCLHandle(
                layout.source_token,
                x_fp8.size(0),
                layout.send_counts,
                layout.recv_counts,
            ),
            recv_topk_idx=recv_expert.view(-1, 1),
            recv_topk_weights=recv_weights.view(-1, 1),
            exec_counts=None,
            event=TorchEventHandle(event) if async_finish else None,
            wire_units=layout.send_counts_device,
            wire_unit_scope="rank",
            wire_bytes_per_unit=(
                x_fp8.size(1) * x_fp8.element_size()
                + x_scales.size(1) * x_scales.element_size()
                + layout.local_expert.element_size()
                + topk_weights.element_size()
            ),
            wire_traffic_source="runtime_nccl_route_occurrences",
        )

    def update_probe_feedback(
        self, *args: Any, previous_event: Any | None = None, **kwargs: Any
    ) -> Any | None:
        del args, kwargs
        return previous_event

    def prefetch(
        self,
        dispatch: DispatchResult,
        *,
        previous_event: Any | None = None,
        async_finish: bool = False,
    ) -> Any | None:
        del async_finish
        return previous_event or dispatch.event

    def combine_async(
        self,
        dispatch: DispatchResult,
        exec_y: torch.Tensor,
        *,
        previous_event: Any | None = None,
        async_finish: bool = False,
    ) -> tuple[torch.Tensor, TorchEventHandle | None]:
        handle = dispatch.handle
        if not isinstance(handle, NCCLHandle):
            raise TypeError("invalid NCCL combine handle")
        with torch.cuda.stream(self.comm_stream):
            self._wait(previous_event)
            returned = torch.empty(
                (sum(handle.send_counts), exec_y.size(1)),
                dtype=exec_y.dtype,
                device=exec_y.device,
            )
            dist.all_to_all_single(
                returned,
                exec_y,
                output_split_sizes=list(handle.send_counts),
                input_split_sizes=list(handle.recv_counts),
                group=self.group,
            )
            output = torch.zeros(
                (handle.num_source_tokens, exec_y.size(1)),
                dtype=exec_y.dtype,
                device=exec_y.device,
            )
            output.index_add_(0, handle.source_token, returned)
            event = torch.cuda.Event()
            event.record(self.comm_stream)
        if not async_finish:
            event.synchronize()
        return output, TorchEventHandle(event) if async_finish else None

    def combine(self, dispatch: DispatchResult, exec_y: torch.Tensor) -> torch.Tensor:
        output, _ = self.combine_async(dispatch, exec_y)
        return output

    def register_expert_pools(
        self, views: tuple[torch.Tensor, ...], local_experts: int
    ) -> None:
        del views, local_experts
        raise ValueError("NCCL baseline does not migrate expert weights")

    def destroy(self) -> None:
        self._layout_cache.clear()


@dataclass
class UltraEPLayout:
    layer_id: int
    routing_map: torch.Tensor
    probs: torch.Tensor
    hybrid_handle: tuple[Any, ...] | None = None
    padded_rows: int = 0
    padded_counts: torch.Tensor | None = None
    event: TorchEventHandle | None = None
    weight_event: Any | None = None


@dataclass(frozen=True)
class UltraEPHandle:
    hybrid_handle: tuple[Any, ...]
    dispatched_probs: torch.Tensor
    local_padded_tokens_per_expert: torch.Tensor
    local_tokens_per_expert: torch.Tensor
    layer_id: int


@dataclass
class UltraEPHybridBackend:
    """Official UltraEP placement/weight runtime plus locked HybridEP A2A."""

    variant: str
    root: Path
    manager: Any
    hybrid_buffer: Any
    group: Any
    num_experts: int
    local_experts: int
    replica_slots: int
    hidden: int
    intermediate: int
    comm_stream: torch.cuda.Stream = field(init=False)
    _layouts: dict[tuple[int, int, int], UltraEPLayout] = field(
        default_factory=dict, init=False
    )
    _scales: dict[tuple[int, int, int], torch.Tensor] = field(default_factory=dict, init=False)
    _master_fc1: torch.Tensor | None = field(default=None, init=False)
    _master_fc2: torch.Tensor | None = field(default=None, init=False)
    _execution_gate: torch.Tensor | None = field(default=None, init=False)
    _execution_up: torch.Tensor | None = field(default=None, init=False)
    _execution_down: torch.Tensor | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.comm_stream = self.manager.get_comm_stream()

    @classmethod
    def load(
        cls,
        root: str | Path,
        group: Any,
        *,
        num_experts: int,
        num_sms: int,
        hidden: int | None,
        max_num_tokens_per_rank: int | None,
        local_experts: int | None,
        replica_slots: int | None,
        intermediate: int | None,
    ) -> "UltraEPHybridBackend":
        required = {
            "hidden": hidden,
            "max_num_tokens_per_rank": max_num_tokens_per_rank,
            "local_experts": local_experts,
            "replica_slots": replica_slots,
            "intermediate": intermediate,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise BackendUnavailable(
                "UltraEP+HybridEP load parameters are missing: " + ", ".join(missing)
            )
        source_root = Path(root).resolve()
        hybrid_root = source_root / "HybridEP"
        if not hybrid_root.is_dir():
            raise BackendUnavailable(f"locked HybridEP tree is missing: {hybrid_root}")
        for name in tuple(sys.modules):
            if name == "deep_ep" or name.startswith("deep_ep.") or name in {"deep_ep_cpp", "hybrid_ep_cpp"}:
                sys.modules.pop(name, None)
        sys.path[:0] = [str(hybrid_root), str(source_root)]
        try:
            ultra_module = importlib.import_module("ultra_ep")
            hybrid_extension = importlib.import_module("hybrid_ep_cpp")
            adapter_path = hybrid_root / "deep_ep/hybrid_ep_buffer.py"
            adapter_spec = importlib.util.spec_from_file_location(
                "_probeep_hybrid_ep_buffer", adapter_path
            )
            if adapter_spec is None or adapter_spec.loader is None:
                raise ImportError(f"cannot load HybridEP adapter: {adapter_path}")
            hybrid_module = importlib.util.module_from_spec(adapter_spec)
            adapter_spec.loader.exec_module(hybrid_module)
        except (ImportError, ModuleNotFoundError) as error:
            raise BackendUnavailable(
                f"UltraEP/HybridEP is not built or importable from {source_root}"
            ) from error
        if not Path(ultra_module.__file__).resolve().is_relative_to(source_root):
            raise BackendUnavailable("ultra_ep import provenance mismatch")
        if not Path(hybrid_extension.__file__).resolve().is_relative_to(hybrid_root):
            raise BackendUnavailable("HybridEP import provenance mismatch")

        hidden_value = int(hidden)
        intermediate_value = int(intermediate)
        local_value = int(local_experts)
        replica_value = int(os.getenv("ULTRAEP_REPLICA_SLOTS", "2"))
        local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", str(dist.get_world_size(group))))
        is_multinode = dist.get_world_size(group) > local_world_size
        hybrid_num_sms = int(
            os.getenv("HYBRIDEP_NUM_SMS", "8" if is_multinode else str(num_sms))
        )
        preprocessing_sms = int(
            os.getenv("HYBRIDEP_PREPROCESSING_SMS", str(num_sms))
        )
        permute_blocks = (
            int(os.getenv("HYBRIDEP_PERMUTE_BLOCKS", "24")) if is_multinode else None
        )
        manager = ultra_module.Manager(
            group=group,
            num_layers=1,
            num_local_master_experts=local_value,
            num_local_redundant_experts=replica_value,
            expert_fc1_numel=2 * hidden_value * intermediate_value,
            expert_fc2_numel=hidden_value * intermediate_value,
            is_train=False,
            explicitly_destroy=True,
            max_microbatches=2,
            weight_data_dtype=torch.bfloat16,
        )
        hybrid_buffer = hybrid_module.HybridEPBuffer(
            group=group,
            hidden_dim=hidden_value,
            max_num_of_tokens_per_rank=int(max_num_tokens_per_rank),
            num_local_experts=local_value + replica_value,
            use_fp8=True,
            num_sms_dispatch_api=hybrid_num_sms,
            num_sms_combine_api=hybrid_num_sms,
            num_sms_preprocessing_api=preprocessing_sms,
            num_blocks_permute=permute_blocks,
            num_blocks_unpermute=permute_blocks,
            load_cached_kernels=_env_bool("ULTRAEP_HYBRIDEP_LOAD_CACHED_KERNELS", False),
            use_shared_buffer=_env_bool("ULTRAEP_HYBRIDEP_USE_SHARED_BUFFER", True),
            enable_custom_allgather=_env_bool(
                "ULTRAEP_HYBRIDEP_ENABLE_CUSTOM_ALLGATHER", True
            ),
        )
        return cls(
            "ultraep_hybridep",
            source_root,
            manager,
            hybrid_buffer,
            group,
            num_experts,
            local_value,
            replica_value,
            hidden_value,
            intermediate_value,
        )

    @property
    def balanced(self) -> bool:
        return True

    @property
    def probeep_hybrid(self) -> bool:
        # The benchmark's generic physical-expert path is also the correct
        # shape contract for UltraEP+HybridEP.
        return True

    @property
    def dynamic_expert_weights_ready(self) -> bool:
        return True

    @property
    def buffer(self) -> "UltraEPHybridBackend":
        return self

    def get_comm_stream(self) -> torch.cuda.Stream:
        return self.comm_stream

    def capture_event(self) -> TorchEventHandle:
        event = torch.cuda.Event()
        event.record()
        return TorchEventHandle(event)

    @staticmethod
    def _wait(previous_event: Any | None) -> None:
        if previous_event is not None:
            previous_event.current_stream_wait()

    def plan(
        self,
        topk_idx: torch.Tensor,
        *,
        previous_event: Any | None = None,
        async_finish: bool = False,
    ) -> UltraEPLayout:
        route_key = (int(topk_idx.data_ptr()), topk_idx.size(0), topk_idx.size(1))
        layout = self._layouts.get(route_key)
        if layout is None:
            routing = torch.zeros(
                (topk_idx.size(0), self.num_experts),
                dtype=torch.bool,
                device=topk_idx.device,
            )
            routing.scatter_(1, topk_idx.to(torch.int64), True)
            layout = UltraEPLayout(
                layer_id=self.manager.allocate_microbatch_slot(0),
                routing_map=routing,
                probs=torch.zeros_like(routing, dtype=torch.float32),
            )
            self._layouts[route_key] = layout
        self._wait(previous_event)
        self.manager.update_placement(layout.layer_id, layout.routing_map)
        layout.weight_event = self.manager.weight_sync(
            layout.layer_id, async_finish=True
        )
        event = torch.cuda.Event()
        event.record()
        layout.event = TorchEventHandle(event)
        if not async_finish:
            layout.event.current_stream_wait()
        return layout

    def dispatch(
        self,
        x_fp8: torch.Tensor,
        x_scales: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        layout: UltraEPLayout | None,
        compute_kind: int = 1,
        previous_event: Any | None = None,
        async_finish: bool = False,
    ) -> DispatchResult:
        del compute_kind
        if layout is None:
            raise ValueError("UltraEP+HybridEP dispatch requires placement layout")
        with torch.cuda.stream(self.comm_stream):
            self._wait(previous_event or layout.event)
            scale_key = (x_scales.data_ptr(), x_scales.size(0), x_scales.size(1))
            contiguous_scales = self._scales.get(scale_key)
            if contiguous_scales is None:
                contiguous_scales = x_scales.contiguous()
                self._scales[scale_key] = contiguous_scales
            layout.probs.zero_()
            layout.probs.scatter_(
                1, topk_idx.to(torch.int64), topk_weights.to(torch.float32)
            )
            physical_probs, physical_routing = self.manager.reroute(
                layout.layer_id, layout.probs, layout.routing_map
            )
            layout.weight_event.current_stream_wait()
            if layout.hybrid_handle is None:
                dispatched, dispatched_probs, dispatched_scales, counts, hybrid_handle = (
                    self.hybrid_buffer.dispatch_with_permute(
                        hidden=x_fp8.view(torch.uint8),
                        routing_map=physical_routing,
                        probs=physical_probs,
                        scaling_factor=contiguous_scales,
                        num_of_experts_per_rank=self.local_experts + self.replica_slots,
                        use_fp8=True,
                        pad_multiple=8,
                        fuse_permute_dispatch=True,
                    )
                )
                layout.hybrid_handle = hybrid_handle
                layout.padded_counts = counts
                layout.padded_rows = int(counts.sum().item())
            else:
                dispatched, dispatched_probs, dispatched_scales, _, _ = (
                    self.hybrid_buffer.dispatch_with_permute(
                        hidden=x_fp8.view(torch.uint8),
                        probs=physical_probs,
                        scaling_factor=contiguous_scales,
                        handle=layout.hybrid_handle,
                        num_permuted_tokens=layout.padded_rows,
                        use_fp8=True,
                        pad_multiple=8,
                        non_blocking=True,
                        fuse_permute_dispatch=True,
                    )
                )
            event = torch.cuda.Event()
            event.record(self.comm_stream)
        event_handle = TorchEventHandle(event)
        if not async_finish:
            event_handle.current_stream_wait()
        if dispatched_scales is None:
            raise RuntimeError("HybridEP FP8 dispatch did not return scaling factors")
        if layout.hybrid_handle is None or layout.padded_counts is None:
            raise RuntimeError("HybridEP dispatch cache was not initialized")
        wrapper = UltraEPHandle(
            hybrid_handle=layout.hybrid_handle,
            dispatched_probs=dispatched_probs,
            local_padded_tokens_per_expert=layout.padded_counts.to(torch.int32),
            local_tokens_per_expert=layout.hybrid_handle[7].to(torch.int32),
            layer_id=layout.layer_id,
        )
        return DispatchResult(
            exec_x=dispatched.view(torch.float8_e4m3fn),
            exec_scales=dispatched_scales,
            handle=wrapper,
            recv_topk_idx=None,
            recv_topk_weights=None,
            exec_counts=wrapper.local_tokens_per_expert,
            event=event_handle if async_finish else None,
        )

    def register_expert_pools(
        self, views: tuple[torch.Tensor, ...], local_experts: int
    ) -> None:
        if len(views) < 3 or local_experts != self.local_experts:
            raise ValueError("UltraEP requires gate/up/down master weights")
        gate, up, down = (tensor[:local_experts].contiguous() for tensor in views[:3])
        self._master_fc1 = torch.cat(
            (gate.reshape(local_experts, -1), up.reshape(local_experts, -1)), dim=1
        ).contiguous()
        self._master_fc2 = down.reshape(local_experts, -1).contiguous()
        self._execution_gate = torch.empty(
            (local_experts + self.replica_slots, self.hidden, self.intermediate),
            dtype=gate.dtype,
            device=gate.device,
        )
        self._execution_up = torch.empty_like(self._execution_gate)
        self._execution_down = torch.empty(
            (local_experts + self.replica_slots, self.intermediate, self.hidden),
            dtype=down.dtype,
            device=down.device,
        )
        self._execution_gate[:local_experts].copy_(gate)
        self._execution_up[:local_experts].copy_(up)
        self._execution_down[:local_experts].copy_(down)
        self.manager.construct_local_master_ptr_pool(
            0,
            [self._master_fc1[index] for index in range(local_experts)],
            [self._master_fc2[index] for index in range(local_experts)],
        )

    def grouped_weights_for(
        self, dispatch: DispatchResult
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del dispatch
        if any(
            tensor is None
            for tensor in (
                self._execution_gate,
                self._execution_up,
                self._execution_down,
            )
        ):
            raise RuntimeError("UltraEP grouped weights were not registered")
        return self._execution_gate, self._execution_up, self._execution_down

    def execution_experts(self, dispatch: DispatchResult) -> torch.Tensor:
        handle = dispatch.handle
        if not isinstance(handle, UltraEPHandle):
            raise TypeError("invalid UltraEP dispatch handle")
        rank = int(self.group.rank())
        slots = self.local_experts + self.replica_slots
        begin = rank * slots
        return self.manager.physical_to_logical_map[
            handle.layer_id, begin : begin + slots
        ]

    @staticmethod
    def execution_raw_counts(dispatch: DispatchResult) -> torch.Tensor:
        handle = dispatch.handle
        if not isinstance(handle, UltraEPHandle):
            raise TypeError("invalid UltraEP dispatch handle")
        return handle.local_tokens_per_expert

    def update_probe_feedback(
        self, *args: Any, previous_event: Any | None = None, **kwargs: Any
    ) -> Any | None:
        del args, kwargs
        return previous_event

    def prefetch(
        self,
        dispatch: DispatchResult,
        *,
        previous_event: Any | None = None,
        async_finish: bool = False,
    ) -> Any | None:
        if self._master_fc1 is None:
            return previous_event or dispatch.event
        with torch.cuda.stream(self.comm_stream):
            self._wait(previous_event or dispatch.event)
            replica_fc1 = self.manager.local_replica_fc1_weight_buffer.view(
                self.replica_slots, 2, self.hidden, self.intermediate
            )
            self._execution_gate[self.local_experts :].copy_(replica_fc1[:, 0])
            self._execution_up[self.local_experts :].copy_(replica_fc1[:, 1])
            self._execution_down[self.local_experts :].copy_(
                self.manager.local_replica_fc2_weight_buffer.view(
                    self.replica_slots, self.intermediate, self.hidden
                )
            )
            event = torch.cuda.Event()
            event.record(self.comm_stream)
        event_handle = TorchEventHandle(event)
        if not async_finish:
            event_handle.current_stream_wait()
        return event_handle

    def combine_async(
        self,
        dispatch: DispatchResult,
        exec_y: torch.Tensor,
        *,
        previous_event: Any | None = None,
        async_finish: bool = False,
    ) -> tuple[torch.Tensor, TorchEventHandle | None]:
        handle = dispatch.handle
        if not isinstance(handle, UltraEPHandle):
            raise TypeError("invalid UltraEP combine handle")
        with torch.cuda.stream(self.comm_stream):
            self._wait(previous_event)
            exec_y.mul_(
                handle.dispatched_probs.to(torch.bfloat16).unsqueeze(1)
            )
            output, _ = self.hybrid_buffer.combine_with_unpermute(
                hidden=exec_y,
                handle=handle.hybrid_handle,
                pad_multiple=8,
                fuse_unpermute_combine=True,
            )
            event = torch.cuda.Event()
            event.record(self.comm_stream)
        event_handle = TorchEventHandle(event)
        if not async_finish:
            event_handle.current_stream_wait()
        return output, event_handle if async_finish else None

    def combine(self, dispatch: DispatchResult, exec_y: torch.Tensor) -> torch.Tensor:
        output, _ = self.combine_async(dispatch, exec_y)
        return output

    def destroy(self) -> None:
        self._layouts.clear()
        runtime = getattr(self.hybrid_buffer, "runtime", None)
        if runtime is not None and hasattr(runtime, "destroy"):
            runtime.destroy()
        self.manager.destroy()


@dataclass
class RuntimeBackend:
    """One initialized DeepEP buffer and a uniform forward-only API."""

    variant: str
    root: Path
    module: Any
    extension: Any
    buffer: Any
    dispatch_config: Any
    combine_config: Any
    num_experts: int
    expert_pools_registered: bool = False
    _pending_probe_feedback: dict[int, tuple[Any, ...]] = field(default_factory=dict)

    @property
    def balanced(self) -> bool:
        return self.variant in {"deepep_moonep_on", "probeep"}

    def capture_event(self) -> Any:
        """Capture a CUDA event on the current compute stream."""

        return self.buffer.capture()

    def reset_probe_controller(
        self, fallback_budget_bytes: int = 32 * 1024 * 1024
    ) -> None:
        """Start one independent layer sequence from explicit A/M bootstrap."""

        if self.variant != "probeep":
            return
        self._pending_probe_feedback.clear()
        self.buffer.reset_balanced_probe_controller(fallback_budget_bytes)

    @classmethod
    def load(
        cls,
        variant: str,
        root: str | Path,
        group: Any,
        *,
        num_experts: int,
        num_sms: int,
        num_nvl_bytes: int,
        num_rdma_bytes: int,
        nvl_chunk_size: int,
        nvl_buffer_size: int,
        rdma_chunk_size: int,
        rdma_buffer_size: int,
        hidden: int | None = None,
        max_num_tokens_per_rank: int | None = None,
        topk: int | None = None,
        local_experts: int | None = None,
        replica_slots: int | None = None,
        token_padding: int | None = None,
        ranks_per_server: int | None = None,
        intermediate: int | None = None,
    ) -> "RuntimeBackend":
        if variant == "nccl":
            return NCCLBackend.load(
                group,
                num_experts=num_experts,
                local_experts=local_experts,
            )
        if variant == "ultraep_hybridep":
            from ultraep_backend import UltraEPRuntimeBackend

            return UltraEPRuntimeBackend.load(
                root,
                group,
                num_experts=num_experts,
                num_sms=num_sms,
                hidden=hidden,
                max_num_tokens_per_rank=max_num_tokens_per_rank,
                local_experts=local_experts,
                replica_slots=replica_slots,
                intermediate=intermediate,
            )
        source_root = Path(root).resolve()
        sys.path.insert(0, str(source_root))
        try:
            module = importlib.import_module("deep_ep")
            extension = importlib.import_module("deep_ep_cpp")
        except (ImportError, ModuleNotFoundError) as error:
            raise BackendUnavailable(
                f"deep_ep is not built or importable from {source_root}"
            ) from error

        package_origin = Path(module.__file__).resolve()
        extension_origin = Path(extension.__file__).resolve()
        if not package_origin.is_relative_to(source_root):
            raise BackendUnavailable(
                f"deep_ep resolved to {package_origin}, not {source_root}"
            )
        if not extension_origin.is_relative_to(source_root):
            raise BackendUnavailable(
                f"deep_ep_cpp resolved to {extension_origin}, not {source_root}"
            )
        module.Buffer.set_num_sms(num_sms)
        buffer_kwargs = {
            "num_qps_per_rank": num_sms,
            "explicitly_destroy": True,
        }
        if variant in {"deepep_moonep_on", "probeep"}:
            buffer_kwargs["balanced_mode"] = True
        buffer = module.Buffer(
            group, num_nvl_bytes, num_rdma_bytes, **buffer_kwargs
        )
        if variant in {"deepep_moonep_on", "probeep"}:
            buffer.configure_balanced()
            dispatch_config = buffer.get_balanced_dispatch_config()
            combine_config = buffer.get_balanced_combine_config()
        else:
            dispatch_config = module.Buffer.get_dispatch_config(group.size())
            combine_config = module.Buffer.get_combine_config(group.size())
        return cls(
            variant=variant,
            root=source_root,
            module=module,
            extension=extension,
            buffer=buffer,
            dispatch_config=dispatch_config,
            combine_config=combine_config,
            num_experts=num_experts,
        )

    def plan(
        self,
        topk_idx: torch.Tensor,
        *,
        previous_event: Any | None = None,
        async_finish: bool = False,
    ) -> OfficialLayout | None:
        """Run the official routing layout phase.

        ProbeEP's count exchange, plan materialization and dispatch share one
        fused binding.  For that variant this method is intentionally empty;
        the combined phase is timed by :meth:`dispatch` exactly once.
        """

        if self.balanced:
            return None
        rank_counts, rdma_counts, expert_counts, membership, event = (
            self.buffer.get_dispatch_layout(
                topk_idx,
                self.num_experts,
                previous_event=previous_event,
                async_finish=async_finish,
                allocate_on_comm_stream=(
                    previous_event is not None and async_finish
                ),
            )
        )
        if rdma_counts is None:
            raise RuntimeError("the EP16 benchmark requires internode DeepEP")
        return OfficialLayout(
            num_tokens_per_rank=rank_counts,
            num_tokens_per_rdma_rank=rdma_counts,
            num_tokens_per_expert=expert_counts,
            is_token_in_rank=membership,
            event=event,
        )

    def dispatch(
        self,
        x_fp8: torch.Tensor,
        x_scales: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        layout: OfficialLayout | None,
        compute_kind: int = 1,
        previous_event: Any | None = None,
        async_finish: bool = False,
    ) -> DispatchResult:
        if self.balanced:
            dispatch_kwargs = {
                "config": self.dispatch_config,
                "previous_event": previous_event,
                "async_finish": async_finish,
            }
            if self.variant == "probeep":
                dispatch_kwargs["compute_kind"] = compute_kind
                dispatch_kwargs["expert_weight_version"] = int(
                    os.getenv("PROBEEP_WEIGHT_VERSION", "1")
                )
                completed = self._pending_probe_feedback.pop(compute_kind, None)
                if completed is not None:
                    dispatch_kwargs.update(
                        completed_observation=completed[:6],
                        feedback_valid=True,
                        rdma_path_bandwidth_gbps=completed[6],
                        controller_alpha=completed[7],
                    )
            (exec_x, exec_scales), _, exec_counts, handle, event = (
                self.buffer.balanced_dispatch(
                    (x_fp8, x_scales),
                    topk_idx,
                    topk_weights,
                    **dispatch_kwargs,
                )
            )
            return DispatchResult(
                exec_x=exec_x,
                exec_scales=exec_scales,
                handle=handle,
                recv_topk_idx=None,
                recv_topk_weights=None,
                exec_counts=exec_counts,
                event=event,
                wire_units=handle.num_tokens_per_rdma_rank,
                wire_unit_scope="server",
                wire_bytes_per_unit=(
                    (
                        x_fp8.size(1) * x_fp8.element_size()
                        + x_scales.size(1) * x_scales.element_size()
                        + 8
                        + topk_idx.size(1) * 8
                        + 15
                    )
                    // 16
                    * 16
                ),
                wire_traffic_source="runtime_deepep_n2n_tokens",
            )

        if layout is None:
            raise ValueError("official DeepEP dispatch requires its layout")
        dispatch_previous_event = previous_event or layout.event
        recv_x, recv_idx, recv_weights, _, handle, event = self.buffer.dispatch(
            (x_fp8, x_scales),
            num_tokens_per_rank=layout.num_tokens_per_rank,
            num_tokens_per_rdma_rank=layout.num_tokens_per_rdma_rank,
            is_token_in_rank=layout.is_token_in_rank,
            num_tokens_per_expert=layout.num_tokens_per_expert,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            config=self.dispatch_config,
            previous_event=dispatch_previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=(
                dispatch_previous_event is not None and async_finish
            ),
        )
        exec_x, exec_scales = recv_x
        return DispatchResult(
            exec_x=exec_x,
            exec_scales=exec_scales,
            handle=handle,
            recv_topk_idx=recv_idx,
            recv_topk_weights=recv_weights,
            exec_counts=None,
            event=event,
            wire_units=layout.num_tokens_per_rdma_rank,
            wire_unit_scope="server",
            wire_bytes_per_unit=(
                (
                    x_fp8.size(1) * x_fp8.element_size()
                    + x_scales.size(1) * x_scales.element_size()
                    + 8
                    + topk_idx.size(1) * 8
                    + 15
                )
                // 16
                * 16
            ),
            wire_traffic_source="runtime_deepep_n2n_tokens",
        )

    def update_probe_feedback(
        self,
        compute_ns: torch.Tensor,
        network_ns: torch.Tensor,
        dispatch_tx_bytes: torch.Tensor,
        dispatch_rx_bytes: torch.Tensor,
        migration_tx_bytes: torch.Tensor,
        migration_rx_bytes: torch.Tensor,
        *,
        compute_kind: int,
        rdma_path_bandwidth_gbps: float,
        alpha: float,
        previous_event: Any | None = None,
        async_finish: bool = False,
    ) -> Any | None:
        """Launch the CUDA feedback controller for the ProbeEP variant."""

        if self.variant != "probeep":
            return previous_event
        del async_finish
        self._pending_probe_feedback[compute_kind] = (
            compute_ns,
            network_ns,
            dispatch_tx_bytes,
            dispatch_rx_bytes,
            migration_tx_bytes,
            migration_rx_bytes,
            rdma_path_bandwidth_gbps,
            alpha,
        )
        return previous_event

    def combine(
        self, dispatch: DispatchResult, exec_y: torch.Tensor
    ) -> torch.Tensor:
        combined, _ = self.combine_async(dispatch, exec_y, async_finish=False)
        return combined

    def combine_async(
        self,
        dispatch: DispatchResult,
        exec_y: torch.Tensor,
        *,
        previous_event: Any | None = None,
        async_finish: bool = False,
    ) -> tuple[torch.Tensor, Any | None]:
        if self.balanced:
            combined, event = self.buffer.balanced_combine(
                exec_y,
                dispatch.handle,
                config=self.combine_config,
                previous_event=previous_event,
                async_finish=async_finish,
                release_after_combine=True,
            )
            return combined, event
        combined, _, event = self.buffer.combine(
            exec_y,
            dispatch.handle,
            config=self.combine_config,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=previous_event is not None and async_finish,
        )
        return combined, event

    def register_expert_pools(
        self, views: tuple[torch.Tensor, ...], local_experts: int
    ) -> None:
        if not self.balanced:
            raise ValueError("expert IPC pools belong to the balanced runtime")
        weights = views[:3]
        grads = views[3:]
        replica_begin = 16 if self.variant == "probeep" else local_experts
        self.buffer.register_balanced_expert_pools(
            [tensor[:local_experts] for tensor in weights],
            [tensor[replica_begin:] for tensor in weights],
            [tensor[:local_experts] for tensor in grads],
            [tensor[replica_begin:] for tensor in grads],
        )
        self.expert_pools_registered = True

    def prefetch(
        self,
        dispatch: DispatchResult,
        *,
        previous_event: Any | None = None,
        async_finish: bool = False,
    ) -> Any | None:
        # ProbeEP lowers registered expert Weight transfer before Dispatch in
        # balanced_dispatch itself.  Reissuing the public idempotent hook would
        # only measure Python/C++ wrapper overhead as a fake prefetch phase.
        if self.variant == "probeep":
            return previous_event or dispatch.event
        if self.balanced and self.expert_pools_registered:
            return self.buffer.balanced_weight_sync(
                dispatch.handle,
                previous_event=previous_event or dispatch.event,
                async_finish=async_finish,
            )
        return previous_event or dispatch.event

    def destroy(self) -> None:
        self.buffer.destroy()


def root_for_variant(
    variant: str,
    deepep_root: str,
    fork_root: str,
    probeep_root: str,
    ultraep_hybridep_root: str | None = None,
) -> Path:
    if variant == "nccl":
        return Path(deepep_root)
    if variant == "deepep":
        return Path(deepep_root)
    if variant == "deepep_moonep_on":
        return Path(fork_root)
    if variant == "ultraep_hybridep":
        if ultraep_hybridep_root is None:
            raise ValueError("ultraep_hybridep_root is required")
        return Path(ultraep_hybridep_root)
    if variant == "probeep":
        return Path(probeep_root)
    raise ValueError(f"unknown benchmark variant: {variant}")
