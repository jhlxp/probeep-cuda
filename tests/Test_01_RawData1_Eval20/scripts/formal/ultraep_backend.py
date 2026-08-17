"""Proven UltraEP placement/weight-sync path with the pinned HybridEP data plane."""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from backend import BackendUnavailable, DispatchResult


class _TorchEvent:
    def __init__(self, event: torch.cuda.Event):
        self.event = event

    def current_stream_wait(self) -> None:
        torch.cuda.current_stream().wait_event(self.event)


@dataclass
class _HybridCache:
    handle: tuple[Any, ...] | None = None
    padded_rows: int = 0
    padded_counts: torch.Tensor | None = None
    exec_y: torch.Tensor | None = None


@dataclass
class UltraLayout:
    topk_idx_ref: torch.Tensor
    routing_map: torch.Tensor
    probs: torch.Tensor
    cache: _HybridCache
    virtual_layer: int = 0
    weight_event: Any | None = None
    event: Any | None = None


@dataclass
class UltraHandle:
    hybrid_handle: tuple[Any, ...]
    exec_counts: torch.Tensor
    route_weights: torch.Tensor
    exec_y: torch.Tensor
    weight_event: Any
    virtual_layer: int


@dataclass
class UltraEPRuntimeBackend:
    variant: str
    root: Path
    module: Any
    extension: Any
    buffer: Any
    dispatch_config: Any
    combine_config: Any
    num_experts: int
    manager: Any
    hybrid_buffer: Any
    local_experts: int
    replica_slots: int
    hidden: int
    intermediate: int
    expert_pools_registered: bool = False

    def __post_init__(self) -> None:
        self._layouts: dict[tuple[int, int, int], UltraLayout] = {}
        self._scales: dict[tuple[int, int, int], torch.Tensor] = {}
        self._physical_gate: torch.Tensor | None = None
        self._physical_up: torch.Tensor | None = None
        self._physical_down: torch.Tensor | None = None
        self._master_fc1: torch.Tensor | None = None
        self._master_down: torch.Tensor | None = None
        self._replica_fc1: torch.Tensor | None = None
        self._replica_down: torch.Tensor | None = None

    @property
    def balanced(self) -> bool:
        return True

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
    ) -> "UltraEPRuntimeBackend":
        ultra_root = Path(root).resolve()
        hybrid_root = ultra_root / "HybridEP"
        for name in tuple(sys.modules):
            if name == "ultra_ep" or name.startswith("ultra_ep.") or name in {
                "deep_ep",
                "deep_ep_cpp",
                "hybrid_ep_cpp",
            }:
                sys.modules.pop(name, None)
        sys.path[:0] = [str(hybrid_root), str(ultra_root)]
        try:
            ultra_ep = importlib.import_module("ultra_ep")
            ultra_extension = importlib.import_module("ultra_ep._C")
            hybrid_extension = importlib.import_module("hybrid_ep_cpp")
        except (ImportError, ModuleNotFoundError) as error:
            raise BackendUnavailable(
                "UltraEP or the pinned multi-node HybridEP extension is not importable"
            ) from error

        buffer_spec = importlib.util.spec_from_file_location(
            "_probeep_hybrid_ep_buffer",
            hybrid_root / "deep_ep" / "hybrid_ep_buffer.py",
        )
        if buffer_spec is None or buffer_spec.loader is None:
            raise BackendUnavailable("the pinned HybridEP Python adapter is missing")
        buffer_module = importlib.util.module_from_spec(buffer_spec)
        buffer_spec.loader.exec_module(buffer_module)

        world_size = dist.get_world_size(group)
        local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", str(world_size)))
        is_multinode = world_size > local_world_size
        hidden_value = int(hidden)
        intermediate_value = int(intermediate)
        local_value = int(local_experts)
        replica_value = int(os.getenv("ULTRAEP_REPLICA_SLOTS", "2"))
        hybrid_num_sms = int(
            os.getenv("HYBRIDEP_NUM_SMS", "8" if is_multinode else str(num_sms))
        )
        hybrid_preprocessing_sms = int(
            os.getenv("HYBRIDEP_PREPROCESSING_SMS", str(num_sms))
        )
        hybrid_permute_blocks = (
            int(os.getenv("HYBRIDEP_PERMUTE_BLOCKS", "24"))
            if is_multinode
            else None
        )
        manager = ultra_ep.Manager(
            group=group,
            num_layers=1,
            num_local_master_experts=local_value,
            num_local_redundant_experts=replica_value,
            expert_fc1_numel=2 * hidden_value * intermediate_value,
            expert_fc2_numel=hidden_value * intermediate_value,
            weight_data_dtype=torch.bfloat16,
            is_train=False,
            max_microbatches=2,
            explicitly_destroy=True,
        )
        hybrid_buffer = buffer_module.HybridEPBuffer(
            group=group,
            hidden_dim=hidden_value,
            max_num_of_tokens_per_rank=int(max_num_tokens_per_rank),
            num_local_experts=local_value + replica_value,
            use_fp8=True,
            num_sms_dispatch_api=hybrid_num_sms,
            num_sms_combine_api=hybrid_num_sms,
            num_sms_preprocessing_api=hybrid_preprocessing_sms,
            num_blocks_permute=hybrid_permute_blocks,
            num_blocks_unpermute=hybrid_permute_blocks,
            load_cached_kernels=True,
            use_shared_buffer=True,
            enable_custom_allgather=True,
        )
        backend = cls(
            variant="ultraep_hybridep",
            root=ultra_root,
            module=ultra_ep,
            extension=(ultra_extension, hybrid_extension),
            buffer=manager,
            dispatch_config=None,
            combine_config=None,
            num_experts=num_experts,
            manager=manager,
            hybrid_buffer=hybrid_buffer,
            local_experts=local_value,
            replica_slots=replica_value,
            hidden=hidden_value,
            intermediate=intermediate_value,
        )
        backend._allocate_master_weights()
        return backend

    def _allocate_master_weights(self) -> None:
        self._master_fc1 = torch.empty(
            (self.local_experts, 2 * self.hidden, self.intermediate),
            dtype=torch.bfloat16,
            device="cuda",
        )
        self._master_down = torch.empty(
            (self.local_experts, self.intermediate, self.hidden),
            dtype=torch.bfloat16,
            device="cuda",
        )
        self.manager.construct_local_master_ptr_pool(
            0,
            list(self._master_fc1.unbind(0)),
            list(self._master_down.unbind(0)),
        )

    @staticmethod
    def _record_event(stream: torch.cuda.Stream) -> _TorchEvent:
        event = torch.cuda.Event()
        event.record(stream)
        return _TorchEvent(event)

    @staticmethod
    def _wait(event: Any | None, stream: torch.cuda.Stream) -> None:
        if isinstance(event, _TorchEvent):
            stream.wait_event(event.event)
        elif event is not None and hasattr(event, "current_stream_wait"):
            with torch.cuda.stream(stream):
                event.current_stream_wait()

    def capture_event(self) -> _TorchEvent:
        return self._record_event(torch.cuda.current_stream())

    def configure_grouped_weights(
        self,
        gate: torch.Tensor,
        up: torch.Tensor,
        down: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        physical_experts = self.local_experts + self.replica_slots
        physical_gate = gate.new_zeros(
            (physical_experts, self.hidden, self.intermediate)
        )
        physical_up = up.new_zeros(
            (physical_experts, self.hidden, self.intermediate)
        )
        physical_down = down.new_zeros(
            (physical_experts, self.intermediate, self.hidden)
        )
        physical_gate[: self.local_experts].copy_(gate)
        physical_up[: self.local_experts].copy_(up)
        physical_down[: self.local_experts].copy_(down)

        self._master_fc1.copy_(
            torch.cat(
                (
                    physical_gate[: self.local_experts],
                    physical_up[: self.local_experts],
                ),
                dim=1,
            ).contiguous()
        )
        self._master_down.copy_(physical_down[: self.local_experts])
        self._physical_gate = physical_gate
        self._physical_up = physical_up
        self._physical_down = physical_down
        self._replica_fc1 = self.manager.local_replica_fc1_weight_buffer.view(
            self.replica_slots, 2 * self.hidden, self.intermediate
        )
        self._replica_down = self.manager.local_replica_fc2_weight_buffer.view(
            self.replica_slots, self.intermediate, self.hidden
        )
        self.expert_pools_registered = True
        return physical_gate, physical_up, physical_down

    def plan(
        self,
        topk_idx: torch.Tensor,
        *,
        previous_event: Any | None = None,
        async_finish: bool = False,
    ) -> UltraLayout:
        key = (topk_idx.data_ptr(), topk_idx.size(0), topk_idx.size(1))
        layout = self._layouts.get(key)
        if layout is None:
            routing = torch.zeros(
                (topk_idx.size(0), self.num_experts),
                dtype=torch.bool,
                device="cuda",
            )
            routing.scatter_(1, topk_idx.to(torch.int64), True)
            layout = UltraLayout(
                topk_idx_ref=topk_idx,
                routing_map=routing,
                probs=torch.zeros_like(routing, dtype=torch.float32),
                cache=_HybridCache(),
            )
            self._layouts[key] = layout

        stream = torch.cuda.current_stream()
        self._wait(previous_event, stream)
        virtual_layer = self.manager.allocate_microbatch_slot(0)
        self.manager.update_placement(virtual_layer, layout.routing_map)
        layout.weight_event = self.manager.weight_sync(
            virtual_layer, async_finish=True
        )
        layout.virtual_layer = virtual_layer
        layout.event = self._record_event(stream)
        if not async_finish:
            layout.event.current_stream_wait()
        return layout

    def dispatch(
        self,
        x_fp8: torch.Tensor,
        x_scales: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        layout: UltraLayout | None,
        compute_kind: int = 1,
        previous_event: Any | None = None,
        async_finish: bool = False,
    ) -> DispatchResult:
        del compute_kind
        if layout is None:
            raise ValueError("UltraEP dispatch requires its placement result")
        stream = self.manager.get_comm_stream()
        self._wait(previous_event or layout.event, stream)
        with torch.cuda.stream(stream):
            scale_key = (
                x_scales.data_ptr(),
                x_scales.size(0),
                x_scales.size(1),
            )
            contiguous_scales = self._scales.get(scale_key)
            if contiguous_scales is None:
                contiguous_scales = x_scales.contiguous()
                self._scales[scale_key] = contiguous_scales
            layout.probs.zero_()
            layout.probs.scatter_(
                1, topk_idx.to(torch.int64), topk_weights.to(torch.float32)
            )
            expanded_probs, expanded_routing = self.manager.reroute(
                layout.virtual_layer, layout.probs, layout.routing_map
            )
            layout.weight_event.current_stream_wait()
            cache = layout.cache
            if cache.handle is None:
                exec_x, route_weights, exec_scales, padded_counts, handle = (
                    self.hybrid_buffer.dispatch_with_permute(
                        hidden=x_fp8.view(torch.uint8),
                        scaling_factor=contiguous_scales,
                        routing_map=expanded_routing,
                        probs=expanded_probs,
                        num_of_experts_per_rank=(
                            self.local_experts + self.replica_slots
                        ),
                        use_fp8=True,
                        pad_multiple=int(os.getenv("TOKEN_PADDING", "8")),
                        fuse_permute_dispatch=True,
                    )
                )
                cache.handle = handle
                cache.padded_counts = padded_counts
                cache.padded_rows = int(padded_counts.sum().item())
                cache.exec_y = torch.empty(
                    (cache.padded_rows, x_fp8.size(1)),
                    dtype=torch.bfloat16,
                    device="cuda",
                )
            else:
                exec_x, route_weights, exec_scales, _, _ = (
                    self.hybrid_buffer.dispatch_with_permute(
                        hidden=x_fp8.view(torch.uint8),
                        scaling_factor=contiguous_scales,
                        probs=expanded_probs,
                        handle=cache.handle,
                        num_permuted_tokens=cache.padded_rows,
                        use_fp8=True,
                        pad_multiple=int(os.getenv("TOKEN_PADDING", "8")),
                        non_blocking=True,
                        fuse_permute_dispatch=True,
                    )
                )
            event = self._record_event(stream)
        if not async_finish:
            event.current_stream_wait()
        return DispatchResult(
            exec_x=exec_x.view(torch.float8_e4m3fn),
            exec_scales=exec_scales,
            handle=UltraHandle(
                hybrid_handle=cache.handle,
                exec_counts=cache.padded_counts,
                route_weights=route_weights,
                exec_y=cache.exec_y,
                weight_event=layout.weight_event,
                virtual_layer=layout.virtual_layer,
            ),
            recv_topk_idx=None,
            recv_topk_weights=None,
            exec_counts=cache.padded_counts,
            event=event,
            wire_unit_scope="server",
            wire_bytes_per_unit=(
                x_fp8.size(1) * x_fp8.element_size()
                + x_scales.size(1) * x_scales.element_size()
                + (
                    (self.local_experts + self.replica_slots)
                    * int(os.getenv("LOCAL_WORLD_SIZE", "8"))
                    * expanded_probs.element_size()
                )
            ),
            wire_traffic_source="runtime_hybridep_destination_membership",
            wire_routing_map=expanded_routing,
        )

    def prefetch(
        self,
        dispatch: DispatchResult,
        *,
        previous_event: Any | None = None,
        async_finish: bool = False,
    ) -> Any | None:
        if not self.expert_pools_registered:
            return previous_event or dispatch.event
        stream = self.manager.get_comm_stream()
        self._wait(previous_event or dispatch.event, stream)
        with torch.cuda.stream(stream):
            self._physical_gate[self.local_experts :].copy_(
                self._replica_fc1[:, : self.hidden]
            )
            self._physical_up[self.local_experts :].copy_(
                self._replica_fc1[:, self.hidden :]
            )
            self._physical_down[self.local_experts :].copy_(self._replica_down)
            event = self._record_event(stream)
        if not async_finish:
            event.current_stream_wait()
        return event

    def combine(
        self, dispatch: DispatchResult, exec_y: torch.Tensor
    ) -> torch.Tensor:
        output, _ = self.combine_async(
            dispatch,
            exec_y,
            previous_event=self.capture_event(),
            async_finish=False,
        )
        return output

    def combine_async(
        self,
        dispatch: DispatchResult,
        exec_y: torch.Tensor,
        *,
        previous_event: Any | None = None,
        async_finish: bool = False,
    ) -> tuple[torch.Tensor, Any | None]:
        stream = self.manager.get_comm_stream()
        self._wait(previous_event, stream)
        with torch.cuda.stream(stream):
            output, _ = self.hybrid_buffer.combine_with_unpermute(
                hidden=exec_y,
                handle=dispatch.handle.hybrid_handle,
                pad_multiple=int(os.getenv("TOKEN_PADDING", "8")),
                fuse_unpermute_combine=True,
            )
            event = self._record_event(stream)
        if not async_finish:
            event.current_stream_wait()
        return output, event

    def execution_experts(self, dispatch: DispatchResult) -> torch.Tensor:
        rank = dist.get_rank()
        begin = rank * (self.local_experts + self.replica_slots)
        end = begin + self.local_experts + self.replica_slots
        return self.manager.physical_to_logical_map[
            dispatch.handle.virtual_layer, begin:end
        ]

    @staticmethod
    def execution_raw_counts(dispatch: DispatchResult) -> torch.Tensor:
        return dispatch.handle.hybrid_handle[7]

    def update_probe_feedback(self, *args: Any, **kwargs: Any) -> Any | None:
        return kwargs.get("previous_event")

    def register_expert_pools(self, *args: Any, **kwargs: Any) -> None:
        raise ValueError("UltraEP registers its master weights through Manager")

    def destroy(self) -> None:
        self.manager.destroy()
        self._layouts.clear()
        self._scales.clear()
