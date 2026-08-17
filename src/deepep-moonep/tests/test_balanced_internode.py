"""Two-node EP16 correctness test for the balanced ProbeEP data path.

Run this file the same way as ``test_internode.py``: launch one process on
each node and let each process spawn the eight local CUDA workers.  ``WORLD_SIZE``
and ``RANK`` therefore describe nodes, while the NCCL group contains 16 ranks.

The adapter below is deliberately the only place that knows the Python API
shape.  If the C++ binding changes while it is being integrated, the rest of
the workload and oracle stay untouched.
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist

import deep_ep
from utils import calc_diff, init_dist, per_token_cast_back, per_token_cast_to_fp8


WORLD_SIZE = 16
RANKS_PER_SERVER = 8
NUM_SERVERS = 2
NUM_EXPERTS = 256
EXPERTS_PER_SERVER = NUM_EXPERTS // NUM_SERVERS
TOPK = 8
ROUTES_PER_SERVER = TOPK // NUM_SERVERS
LOCAL_EXPERTS = 16
REPLICA_SLOTS = 16
EXECUTION_SLOTS = LOCAL_EXPERTS + REPLICA_SLOTS
TOKEN_PADDING = 8
HIDDEN = 7168
FP8_BLOCK = 128


class BalancedAdapter:
    """Small, exact adapter around the new ``deep_ep.Buffer`` API."""

    def __init__(self, buffer: deep_ep.Buffer, config: deep_ep.Config):
        self.buffer = buffer
        self.config = config

    def dispatch(
        self,
        x_fp8: torch.Tensor,
        x_scales: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        async_finish: bool,
    ):
        result = self.buffer.balanced_dispatch(
            (x_fp8, x_scales),
            topk_idx,
            topk_weights,
            config=self.config,
            previous_event=self.buffer.capture(),
            async_finish=async_finish,
        )
        (exec_x, exec_scales), exec_weights, exec_counts, handle, event = result
        if async_finish:
            event.current_stream_wait()
        return exec_x, exec_scales, exec_weights, exec_counts, handle

    def identity_expert(
        self,
        exec_x: torch.Tensor,
        exec_scales: torch.Tensor,
        handle,
        num_tokens: int,
    ) -> torch.Tensor:
        # route_dst uses this per-rank stride.  Only this prefix can be
        # referenced, so the fixed-capacity tail need not be materialized.
        nvs = NUM_SERVERS * num_tokens * TOPK + (TOKEN_PADDING - 1) * EXECUTION_SLOTS
        handle.exec_y[:nvs].copy_(
            per_token_cast_back(exec_x[:nvs], exec_scales[:nvs])
        )
        return handle.exec_y

    def combine(
        self,
        exec_y: torch.Tensor,
        handle,
        *,
        async_finish: bool,
        release_after_combine: bool,
    ) -> torch.Tensor:
        combined, event = self.buffer.balanced_combine(
            exec_y,
            handle,
            config=self.config,
            previous_event=self.buffer.capture(),
            async_finish=async_finish,
            release_after_combine=release_after_combine,
        )
        if async_finish:
            event.current_stream_wait()
        return combined[: handle.num_tokens]


def make_server_preserving_routes(
    mode: str, rank: int, num_tokens: int
) -> torch.Tensor:
    """Build four unique routes on each server for every source token."""

    token = torch.arange(num_tokens, device="cuda", dtype=torch.int64)
    global_token = token + rank * num_tokens
    route_lane = torch.arange(
        ROUTES_PER_SERVER, device="cuda", dtype=torch.int64
    )

    if mode == "balanced":
        within_server = (
            global_token[:, None] * ROUTES_PER_SERVER + route_lane[None, :]
        ) % EXPERTS_PER_SERVER
    elif mode == "server_preserving_skew":
        # All traffic is concentrated on four experts per server.  This is an
        # intentionally hard planner case while retaining the same 4+4 server
        # split as the balanced workload.
        within_server = route_lane[None, :].expand(num_tokens, -1)
    else:
        raise ValueError(f"unknown routing mode: {mode}")

    server_zero = within_server
    server_one = within_server + EXPERTS_PER_SERVER
    return torch.cat((server_zero, server_one), dim=1).contiguous()


def make_route_weights(rank: int, num_tokens: int) -> torch.Tensor:
    """Deterministic, positive and deliberately non-normalized gate weights."""

    token = torch.arange(num_tokens, device="cuda", dtype=torch.int64)[:, None]
    lane = torch.arange(TOPK, device="cuda", dtype=torch.int64)[None, :]
    numerator = (token * 11 + lane * 7 + rank * 13) % 29 + 1
    return (numerator.to(torch.float32) / 64.0).contiguous()


def check_result(
    combined: torch.Tensor,
    source_bf16: torch.Tensor,
    source_after_fp8: torch.Tensor,
    topk_weights: torch.Tensor,
    mode: str,
    async_finish: bool,
    round_index: int,
) -> None:
    weight_sum = topk_weights.sum(dim=1, keepdim=True)
    exact_fp8_reference = source_after_fp8.float() * weight_sum
    original_token_reference = source_bf16.float() * weight_sum

    fp8_path_diff = calc_diff(combined.float(), exact_fp8_reference)
    original_diff = calc_diff(combined.float(), original_token_reference)
    assert fp8_path_diff < 2e-5, (
        f"{mode=} {async_finish=} {round_index=}: "
        f"identity transport diff against dequantized FP8 input is {fp8_path_diff}"
    )
    assert original_diff < 5e-4, (
        f"{mode=} {async_finish=} {round_index=}: "
        f"weighted original-token diff is {original_diff}"
    )


def run_case(
    adapter: BalancedAdapter,
    group: dist.ProcessGroup,
    rank: int,
    num_tokens: int,
    mode: str,
    async_finish: bool,
    seed: int,
    rounds: int,
) -> list[tuple[int, int]]:
    torch.manual_seed(seed + rank)
    source_bf16 = torch.randn(
        (num_tokens, HIDDEN), device="cuda", dtype=torch.bfloat16
    )
    x_fp8, x_scales_contiguous = per_token_cast_to_fp8(source_bf16)
    source_after_fp8 = per_token_cast_back(x_fp8, x_scales_contiguous)
    # DeepEP accepts the token dimension as the contiguous scale dimension;
    # this is the same layout exercised by the upstream inter-node test.
    x_scales = x_scales_contiguous.T.contiguous().T

    topk_idx = make_server_preserving_routes(mode, rank, num_tokens)
    topk_weights = make_route_weights(rank, num_tokens)
    assert topk_idx.dtype == torch.int64 and topk_idx.shape == (num_tokens, TOPK)
    assert torch.all(topk_idx[:, :ROUTES_PER_SERVER] < EXPERTS_PER_SERVER)
    assert torch.all(topk_idx[:, ROUTES_PER_SERVER:] >= EXPERTS_PER_SERVER)

    handle_history = []
    # Every round is a complete acquire/dispatch/expert/combine/release
    # lifecycle.  Long runs repeatedly wrap the same three persistent slots.
    for round_index in range(rounds):
        exec_x, exec_scales, exec_weights, exec_counts, handle = adapter.dispatch(
            x_fp8,
            x_scales,
            topk_idx,
            topk_weights,
            async_finish=async_finish,
        )

        assert handle.num_tokens == num_tokens
        assert handle.route_dst.shape[1] == TOPK
        assert handle.slot_count.shape == (WORLD_SIZE, EXECUTION_SLOTS)
        assert exec_weights.shape[0] == exec_x.shape[0]
        assert exec_counts.dtype == torch.int32
        assert exec_counts.shape == (EXECUTION_SLOTS,)
        torch.testing.assert_close(
            exec_counts, handle.slot_count[rank], rtol=0, atol=0
        )

        # Every assignment must appear in exactly one grouped execution slot.
        local_assignment_count = exec_counts.sum().to(torch.int64)
        global_assignment_count = local_assignment_count.clone()
        dist.all_reduce(global_assignment_count, group=group)
        assert global_assignment_count.item() == WORLD_SIZE * num_tokens * TOPK

        exec_y = adapter.identity_expert(exec_x, exec_scales, handle, num_tokens)
        handle_history.append((handle.slot, handle.generation))
        combined = adapter.combine(
            exec_y,
            handle,
            async_finish=async_finish,
            release_after_combine=True,
        )
        check_result(
            combined,
            source_bf16,
            source_after_fp8,
            topk_weights,
            mode,
            async_finish,
            round_index,
        )
    return handle_history


def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    assert num_local_ranks == RANKS_PER_SERVER
    assert num_ranks == WORLD_SIZE
    assert int(os.environ["WORLD_SIZE"]) == NUM_SERVERS
    assert HIDDEN % FP8_BLOCK == 0

    buffer = deep_ep.Buffer(
        group,
        int(2e9),
        int(1e9),
        num_qps_per_rank=24,
        explicitly_destroy=True,
        balanced_mode=True,
    )
    buffer.configure_balanced()
    adapter = BalancedAdapter(buffer, deep_ep.Config(24, 8, 512, 16, 128))
    handle_history = []

    for mode in ("balanced", "server_preserving_skew"):
        for async_finish in (False, True):
            if local_rank == 0:
                print(
                    f"[balanced] {mode=}, {async_finish=}, "
                    f"tokens={args.num_tokens}, rounds={args.rounds}",
                    flush=True,
                )
            handle_history.extend(
                run_case(
                    adapter,
                    group,
                    rank,
                    args.num_tokens,
                    mode,
                    async_finish,
                    args.seed,
                    args.rounds,
                )
            )
            group.barrier()

    assert len(handle_history) == 4 * args.rounds
    for lifecycle_index, (slot, generation) in enumerate(handle_history):
        assert slot == lifecycle_index % 3
        if lifecycle_index >= 3:
            previous_slot, previous_generation = handle_history[
                lifecycle_index - 3
            ]
            assert slot == previous_slot
            assert generation > previous_generation

    if local_rank == 0:
        print(
            "[balanced] all EP16 identity-path checks passed, "
            f"lifecycles={len(handle_history)}",
            flush=True,
        )
    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test ProbeEP balanced dispatch/combine on two 8-GPU nodes"
    )
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--num-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--rounds", type=int, default=2)
    arguments = parser.parse_args()
    torch.multiprocessing.spawn(
        test_loop,
        args=(arguments.num_processes, arguments),
        nprocs=arguments.num_processes,
    )
