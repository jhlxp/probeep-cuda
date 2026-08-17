"""EP16 identity-backward test for the balanced ProbeEP data path.

Launch one process per node and let each process spawn the eight local CUDA
workers.  ``WORLD_SIZE`` and ``RANK`` describe nodes; the NCCL group therefore
contains sixteen ranks.
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
EXECUTION_SLOTS = 48
TOKEN_PADDING = 8
HIDDEN = 7168
FP8_BLOCK = 128


def make_server_preserving_routes(
    mode: str, rank: int, num_tokens: int
) -> torch.Tensor:
    """Route four unique experts on each server for every source token."""

    token = torch.arange(num_tokens, device="cuda", dtype=torch.int64)
    global_token = token + rank * num_tokens
    lane = torch.arange(ROUTES_PER_SERVER, device="cuda", dtype=torch.int64)

    if mode == "balanced":
        within_server = (
            global_token[:, None] * ROUTES_PER_SERVER + lane[None, :]
        ) % EXPERTS_PER_SERVER
    elif mode == "server_preserving_skew":
        within_server = lane[None, :].expand(num_tokens, -1)
    elif mode == "server_imbalanced":
        lane0 = torch.arange(6, device="cuda", dtype=torch.int64)
        lane1 = torch.arange(2, device="cuda", dtype=torch.int64)
        server_zero = (global_token[:, None] * 6 + lane0[None, :]) % EXPERTS_PER_SERVER
        server_one = (
            global_token[:, None] * 2 + lane1[None, :]
        ) % EXPERTS_PER_SERVER + EXPERTS_PER_SERVER
        return torch.cat((server_zero, server_one), dim=1).contiguous()
    else:
        raise ValueError(f"unknown routing mode: {mode}")

    return torch.cat(
        (within_server, within_server + EXPERTS_PER_SERVER), dim=1
    ).contiguous()


def make_route_weights(rank: int, num_tokens: int) -> torch.Tensor:
    """Create deterministic positive, non-normalized gate weights."""

    token = torch.arange(num_tokens, device="cuda", dtype=torch.int64)[:, None]
    lane = torch.arange(TOPK, device="cuda", dtype=torch.int64)[None, :]
    numerator = (token * 11 + lane * 7 + rank * 13) % 29 + 1
    return (numerator.to(torch.float32) / 64.0).contiguous()


def wait_if_async(event, async_finish: bool) -> None:
    if async_finish:
        event.current_stream_wait()


def run_case(
    buffer: deep_ep.Buffer,
    config: deep_ep.Config,
    rank: int,
    num_tokens: int,
    mode: str,
    async_finish: bool,
    seed: int,
) -> tuple[int, int]:
    torch.manual_seed(seed + rank)
    source_bf16 = torch.randn(
        (num_tokens, HIDDEN), device="cuda", dtype=torch.bfloat16
    )
    x_fp8, x_scales_contiguous = per_token_cast_to_fp8(source_bf16)
    source_after_fp8 = per_token_cast_back(x_fp8, x_scales_contiguous)
    x_scales = x_scales_contiguous.T.contiguous().T

    topk_idx = make_server_preserving_routes(mode, rank, num_tokens)
    topk_weights = make_route_weights(rank, num_tokens)
    weight_sum = topk_weights.sum(dim=1, keepdim=True)

    (exec_x, exec_scales), exec_weights, exec_counts, handle, event = (
        buffer.balanced_dispatch(
            (x_fp8, x_scales),
            topk_idx,
            topk_weights,
            config=config,
            previous_event=buffer.capture(),
            async_finish=async_finish,
        )
    )
    wait_if_async(event, async_finish)

    nvs = (
        NUM_SERVERS * num_tokens * TOPK
        + (TOKEN_PADDING - 1) * EXECUTION_SLOTS
    )
    assert exec_x.shape == (nvs, HIDDEN)
    assert exec_scales.shape == (nvs, HIDDEN // FP8_BLOCK)
    assert exec_weights.shape == (nvs,)
    assert exec_counts.shape == (EXECUTION_SLOTS,)
    if mode == "server_imbalanced":
        assert int(handle.probe_server_load_after.max().cpu()) < int(
            handle.probe_server_load_before.max().cpu()
        )
        assert int(handle.probe_plan_counts[0].cpu()) >= 2

    handle.exec_y[:nvs].copy_(per_token_cast_back(exec_x, exec_scales))
    forward, event = buffer.balanced_combine(
        handle.exec_y,
        handle,
        config=config,
        previous_event=buffer.capture(),
        async_finish=async_finish,
        release_after_combine=False,
    )
    wait_if_async(event, async_finish)

    forward_reference = source_after_fp8.float() * weight_sum
    forward_diff = calc_diff(forward.float(), forward_reference)
    assert forward_diff < 2e-5, (
        f"{mode=} {async_finish=}: identity forward diff is {forward_diff}"
    )

    grad_out = torch.randn(
        (num_tokens, HIDDEN), device="cuda", dtype=torch.bfloat16
    )
    exec_grad_out, event = buffer.balanced_dispatch_backward(
        grad_out,
        handle,
        config=config,
        previous_event=buffer.capture(),
        async_finish=async_finish,
    )
    wait_if_async(event, async_finish)
    assert exec_grad_out.shape == (nvs, HIDDEN)

    # The identity expert's derivative is one, so its grouped input gradient
    # is the weighted gradient produced by the cached backward dispatch.
    exec_grad_x = exec_grad_out.clone()
    grad_x, event = buffer.balanced_combine_backward(
        exec_grad_x,
        handle,
        config=config,
        previous_event=buffer.capture(),
        async_finish=async_finish,
    )
    wait_if_async(event, async_finish)

    backward_reference = grad_out.float() * weight_sum
    backward_diff = calc_diff(grad_x.float(), backward_reference)
    assert backward_diff < 5e-6, (
        f"{mode=} {async_finish=}: identity backward diff is {backward_diff}"
    )

    finish_event = buffer.balanced_finish_backward(
        handle,
        previous_event=buffer.capture(),
        async_finish=async_finish,
    )
    wait_if_async(finish_event, async_finish)
    return handle.slot, handle.generation


def run_worker(
    local_rank: int, num_local_ranks: int, args: argparse.Namespace
) -> None:
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    assert num_local_ranks == RANKS_PER_SERVER
    assert num_ranks == WORLD_SIZE
    assert int(os.environ["WORLD_SIZE"]) == NUM_SERVERS

    buffer = deep_ep.Buffer(
        group,
        int(2e9),
        int(1e9),
        num_qps_per_rank=24,
        explicitly_destroy=True,
        balanced_mode=True,
    )
    buffer.configure_balanced()
    config = deep_ep.Config(24, 8, 512, 16, 128)

    history = []
    for mode in ("balanced", "server_preserving_skew", "server_imbalanced"):
        for async_finish in (False, True):
            if local_rank == 0:
                print(
                    f"[balanced backward] {mode=}, {async_finish=}, "
                    f"tokens={args.num_tokens}",
                    flush=True,
                )
            history.append(
                run_case(
                    buffer,
                    config,
                    rank,
                    args.num_tokens,
                    mode,
                    async_finish,
                    args.seed,
                )
            )
            group.barrier()

    # Sequential work reuses the first released slot so a stable A/B
    # microbatch layout keeps its per-bank weight cache hot.  Generation is
    # still advanced on every acquisition, preserving stale-handle detection.
    assert [slot for slot, _ in history] == [0] * len(history)
    generations = [generation for _, generation in history]
    assert generations == sorted(generations)
    assert len(set(generations)) == len(generations)

    if local_rank == 0:
        print(
            "[balanced backward] all EP16 identity-gradient checks passed",
            flush=True,
        )
    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test ProbeEP balanced forward/backward on two NVL8 nodes"
    )
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--num-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260813)
    arguments = parser.parse_args()
    torch.multiprocessing.spawn(
        run_worker,
        args=(arguments.num_processes, arguments),
        nprocs=arguments.num_processes,
    )
