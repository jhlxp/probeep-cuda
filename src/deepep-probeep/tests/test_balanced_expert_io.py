"""EP16 correctness test for the fixed IPC expert pool.

Run this file with the same two-node launcher as ``test_internode.py``.  The
node processes each spawn eight CUDA workers, so the NCCL process group has 16
ranks while ``WORLD_SIZE`` in the launcher environment is two.

The tiny adapter exercises the public ``Buffer`` view binding, which returns
the six non-owning tensors produced by ``moonep::make_expert_pool_views``.
"""

from __future__ import annotations

import argparse
import os
import struct

import torch
import torch.distributed as dist

import deep_ep
from utils import init_dist, per_token_cast_to_fp8


WORLD_SIZE = 16
RANKS_PER_SERVER = 8
NUM_SERVERS = 2
NUM_EXPERTS = 256
LOCAL_EXPERTS = 16
REPLICA_SLOTS = 32
PLAN_SLOTS = 3
EXECUTION_SLOTS = LOCAL_EXPERTS + REPLICA_SLOTS
POOL_SLOTS = LOCAL_EXPERTS + PLAN_SLOTS * REPLICA_SLOTS
TOPK = 8
HIDDEN = 7168
INTERMEDIATE = 2048
TOKEN_PADDING = 8

WEIGHT_SHARD_BYTES = POOL_SLOTS * HIDDEN * INTERMEDIATE * 2
GRAD_SHARD_BYTES = POOL_SLOTS * HIDDEN * INTERMEDIATE * 4
POOL_BYTES = 3 * WEIGHT_SHARD_BYTES + 3 * GRAD_SHARD_BYTES

WEIGHT_SENTINEL = -320.0
GRAD_SENTINEL = -987.25


class ExpertPoolAdapter:
    """Keep the view-binding seam separate from the correctness oracle."""

    def __init__(self, buffer: deep_ep.Buffer):
        self.buffer = buffer

    def views(self) -> tuple[torch.Tensor, ...]:
        views = self.buffer.get_balanced_expert_pool_views()
        assert len(views) == 6
        return views

    def register(self, views: tuple[torch.Tensor, ...]) -> None:
        weights = views[:3]
        grads = views[3:]
        self.buffer.register_balanced_expert_pools(
            [tensor[:LOCAL_EXPERTS] for tensor in weights],
            [tensor[LOCAL_EXPERTS:] for tensor in weights],
            [tensor[:LOCAL_EXPERTS] for tensor in grads],
            [tensor[LOCAL_EXPERTS:] for tensor in grads],
        )


def _f32(value: float) -> float:
    return struct.unpack("=f", struct.pack("=f", value))[0]


def _f32_add(left: float, right: float) -> float:
    return _f32(_f32(left) + _f32(right))


def _weight_value(global_expert: int, shard: int) -> float:
    if shard == 0:
        return float(global_expert + 1)
    if shard == 1:
        return float(-(global_expert + 1))
    return float(2 * (global_expert + 1))


def _home_grad_value(global_expert: int, shard: int) -> float:
    return _f32((global_expert + 1) * 0.03125 + (shard + 1) * 0.00037)


def _replica_grad_value(global_rank: int, replica_slot: int, shard: int) -> float:
    # Decimal components make the expected result sensitive to the launcher's
    # documented peer-rank/slot FP32 addition order.  _f32_add mirrors
    # __fadd_rn after every contribution.
    return _f32(
        (global_rank + 1) * 0.101
        + (replica_slot + 1) * 0.0031
        + (shard + 1) * 0.000071
    )


def _assert_constant(tensor: torch.Tensor, expected: float, label: str) -> None:
    minimum, maximum = torch.aminmax(tensor)
    if tensor.dtype == torch.bfloat16:
        expected = torch.tensor(expected, dtype=torch.bfloat16).item()
    else:
        expected = _f32(expected)
    actual_minimum = minimum.item()
    actual_maximum = maximum.item()
    assert actual_minimum == expected and actual_maximum == expected, (
        f"{label}: expected every element to be {expected}, "
        f"got range [{actual_minimum}, {actual_maximum}]"
    )


def _check_layout(views: tuple[torch.Tensor, ...]) -> None:
    gate_weight, up_weight, down_weight, gate_grad, up_grad, down_grad = views
    assert gate_weight.shape == (POOL_SLOTS, HIDDEN, INTERMEDIATE)
    assert up_weight.shape == (POOL_SLOTS, HIDDEN, INTERMEDIATE)
    assert down_weight.shape == (POOL_SLOTS, INTERMEDIATE, HIDDEN)
    assert gate_grad.shape == (POOL_SLOTS, HIDDEN, INTERMEDIATE)
    assert up_grad.shape == (POOL_SLOTS, HIDDEN, INTERMEDIATE)
    assert down_grad.shape == (POOL_SLOTS, INTERMEDIATE, HIDDEN)
    assert all(tensor.is_cuda and tensor.is_contiguous() for tensor in views)
    assert all(tensor.dtype == torch.bfloat16 for tensor in views[:3])
    assert all(tensor.dtype == torch.float32 for tensor in views[3:])

    base = gate_weight.data_ptr()
    expected_offsets = (
        0,
        WEIGHT_SHARD_BYTES,
        2 * WEIGHT_SHARD_BYTES,
        3 * WEIGHT_SHARD_BYTES,
        3 * WEIGHT_SHARD_BYTES + GRAD_SHARD_BYTES,
        3 * WEIGHT_SHARD_BYTES + 2 * GRAD_SHARD_BYTES,
    )
    assert tuple(tensor.data_ptr() - base for tensor in views) == expected_offsets
    assert POOL_BYTES == 29_595_009_024

    for tensor in views:
        bytes_per_slot = tensor[0].numel() * tensor.element_size()
        assert (
            tensor[LOCAL_EXPERTS].data_ptr() - tensor.data_ptr()
            == LOCAL_EXPERTS * bytes_per_slot
        )


def _initialize_home_and_sentinels(
    views: tuple[torch.Tensor, ...], rank: int
) -> None:
    weights = views[:3]
    grads = views[3:]
    for shard, tensor in enumerate(weights):
        for local_expert in range(LOCAL_EXPERTS):
            global_expert = rank * LOCAL_EXPERTS + local_expert
            tensor[local_expert].fill_(_weight_value(global_expert, shard))
        tensor[LOCAL_EXPERTS:].fill_(WEIGHT_SENTINEL)

    for shard, tensor in enumerate(grads):
        for local_expert in range(LOCAL_EXPERTS):
            global_expert = rank * LOCAL_EXPERTS + local_expert
            tensor[local_expert].fill_(
                _home_grad_value(global_expert, shard)
            )
        tensor[LOCAL_EXPERTS:].fill_(GRAD_SENTINEL)


def _make_plan(
    buffer: deep_ep.Buffer,
    config: deep_ep.Config,
    rank: int,
    num_tokens: int,
    expert_offset: int = 0,
):
    torch.manual_seed(20260813 + rank)
    source = torch.randn(
        (num_tokens, HIDDEN), device="cuda", dtype=torch.bfloat16
    )
    x_fp8, x_scales_contiguous = per_token_cast_to_fp8(source)
    x_scales = x_scales_contiguous.T.contiguous().T

    # Expert zero is the only route on server zero.  It must execute on all
    # eight ranks there, so its owner pulls seven independent replica grads.
    # The other seven unique routes exercise the second server at the same
    # time without relying on duplicate top-k entries.
    route_row = torch.tensor(
        [
            expert_offset,
            128 + expert_offset,
            129 + expert_offset,
            130 + expert_offset,
            131 + expert_offset,
            132 + expert_offset,
            133 + expert_offset,
            134 + expert_offset,
        ],
        device="cuda",
        dtype=torch.int64,
    )
    topk_idx = route_row.expand(num_tokens, -1).contiguous()
    topk_weights = torch.full(
        (num_tokens, TOPK),
        1.0 / TOPK,
        device="cuda",
        dtype=torch.float32,
    )

    (_, _), _, _, handle, event = buffer.balanced_dispatch(
        (x_fp8, x_scales),
        topk_idx,
        topk_weights,
        config=config,
        previous_event=buffer.capture(),
        async_finish=True,
    )
    event.current_stream_wait()
    return handle


def _initialize_replica_grads(
    grads: tuple[torch.Tensor, ...],
    local_replica_map: torch.Tensor,
    rank: int,
    plan_slot: int,
) -> None:
    for shard, tensor in enumerate(grads):
        begin = LOCAL_EXPERTS + plan_slot * REPLICA_SLOTS
        replica = tensor[begin : begin + REPLICA_SLOTS]
        for slot in range(REPLICA_SLOTS):
            if int(local_replica_map[slot]) >= 0:
                replica[slot].fill_(_replica_grad_value(rank, slot, shard))


def _check_weight_sync(
    weights: tuple[torch.Tensor, ...],
    local_replica_map: torch.Tensor,
    rank: int,
    plan_slot: int,
) -> None:
    for shard, tensor in enumerate(weights):
        begin = LOCAL_EXPERTS + plan_slot * REPLICA_SLOTS
        replica = tensor[begin : begin + REPLICA_SLOTS]
        for slot in range(REPLICA_SLOTS):
            global_expert = int(local_replica_map[slot])
            expected = (
                WEIGHT_SENTINEL
                if global_expert < 0
                else _weight_value(global_expert, shard)
            )
            _assert_constant(
                replica[slot], expected,
                f"rank={rank} weight_shard={shard} replica_slot={slot}",
            )


def _expected_home_grad(
    global_replica_map: torch.Tensor,
    global_expert: int,
    shard: int,
) -> float:
    expected = _home_grad_value(global_expert, shard)
    owner_server = global_expert // (RANKS_PER_SERVER * LOCAL_EXPERTS)
    remote_sum = _f32(0.0)
    remote_found = False
    for peer_global_rank in range(WORLD_SIZE):
        for replica_slot in range(REPLICA_SLOTS):
            if int(global_replica_map[peer_global_rank, replica_slot]) != global_expert:
                continue
            value = _replica_grad_value(peer_global_rank, replica_slot, shard)
            if peer_global_rank // RANKS_PER_SERVER == owner_server:
                expected = _f32_add(expected, value)
            else:
                remote_sum = _f32_add(remote_sum, value)
                remote_found = True
    if remote_found:
        expected = _f32_add(expected, remote_sum)
    return expected


def _check_grad_reduce(
    grads: tuple[torch.Tensor, ...],
    global_replica_map: torch.Tensor,
    local_replica_map: torch.Tensor,
    rank: int,
    plan_slot: int,
) -> None:
    for shard, tensor in enumerate(grads):
        home = tensor[:LOCAL_EXPERTS]
        begin = LOCAL_EXPERTS + plan_slot * REPLICA_SLOTS
        replica = tensor[begin : begin + REPLICA_SLOTS]
        for local_expert in range(LOCAL_EXPERTS):
            global_expert = rank * LOCAL_EXPERTS + local_expert
            expected = _expected_home_grad(
                global_replica_map, global_expert, shard
            )
            _assert_constant(
                home[local_expert], expected,
                f"rank={rank} grad_shard={shard} home_expert={global_expert}",
            )
        for slot in range(REPLICA_SLOTS):
            expected = (
                0.0 if int(local_replica_map[slot]) >= 0 else GRAD_SENTINEL
            )
            _assert_constant(
                replica[slot], expected,
                f"rank={rank} grad_shard={shard} replica_slot={slot}",
            )


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
    adapter = ExpertPoolAdapter(buffer)
    views = adapter.views()
    _check_layout(views)
    _initialize_home_and_sentinels(views, rank)
    adapter.register(views)

    handle = _make_plan(buffer, config, rank, args.num_tokens)
    second_handle = _make_plan(
        buffer, config, rank, args.num_tokens, expert_offset=16
    )
    third_handle = _make_plan(
        buffer, config, rank, args.num_tokens, expert_offset=32
    )
    assert (handle.slot, second_handle.slot, third_handle.slot) == (0, 1, 2)
    server = rank // RANKS_PER_SERVER
    global_replica_map = handle.replica_expert.cpu()
    domain_replica_map = global_replica_map[
        server * RANKS_PER_SERVER : (server + 1) * RANKS_PER_SERVER
    ].cpu()
    local_replica_map = domain_replica_map[rank % RANKS_PER_SERVER]
    second_global_replica_map = second_handle.replica_expert.cpu()
    second_local_replica_map = second_global_replica_map[
        server * RANKS_PER_SERVER : (server + 1) * RANKS_PER_SERVER
    ][rank % RANKS_PER_SERVER]
    third_global_replica_map = third_handle.replica_expert.cpu()
    third_local_replica_map = third_global_replica_map[
        server * RANKS_PER_SERVER : (server + 1) * RANKS_PER_SERVER
    ][rank % RANKS_PER_SERVER]
    admitted = int(handle.probe_plan_counts[0].cpu())
    assert admitted >= 2
    remote_replicas = sum(
        expert >= 0
        and expert // (RANKS_PER_SERVER * LOCAL_EXPERTS) != rank_index // 8
        for rank_index, row in enumerate(global_replica_map.tolist())
        for expert in row
    )
    assert remote_replicas >= admitted
    _initialize_replica_grads(
        views[3:], local_replica_map, rank, handle.slot
    )
    dist.barrier(group=group)

    weight_event = buffer.balanced_weight_sync(
        handle,
        previous_event=buffer.capture(),
        async_finish=True,
    )
    weight_event.current_stream_wait()
    second_weight_event = buffer.balanced_weight_sync(
        second_handle,
        previous_event=buffer.capture(),
        async_finish=True,
    )
    second_weight_event.current_stream_wait()
    third_weight_event = buffer.balanced_weight_sync(
        third_handle,
        previous_event=buffer.capture(),
        async_finish=True,
    )
    third_weight_event.current_stream_wait()
    _check_weight_sync(
        views[:3], local_replica_map, rank, handle.slot
    )
    _check_weight_sync(
        views[:3], second_local_replica_map, rank, second_handle.slot
    )
    _check_weight_sync(
        views[:3], third_local_replica_map, rank, third_handle.slot
    )

    nvs = (
        NUM_SERVERS * args.num_tokens * TOPK
        + (TOKEN_PADDING - 1) * EXECUTION_SLOTS
    )
    second_handle.exec_y[:nvs].zero_()
    _, second_combine_event = buffer.balanced_combine(
        second_handle.exec_y,
        second_handle,
        config=config,
        previous_event=buffer.capture(),
        async_finish=True,
        release_after_combine=True,
    )
    second_combine_event.current_stream_wait()
    third_handle.exec_y[:nvs].zero_()
    _, third_combine_event = buffer.balanced_combine(
        third_handle.exec_y,
        third_handle,
        config=config,
        previous_event=buffer.capture(),
        async_finish=True,
        release_after_combine=True,
    )
    third_combine_event.current_stream_wait()
    dist.barrier(group=group)

    handle.exec_y[:nvs].zero_()
    _, combine_event = buffer.balanced_combine(
        handle.exec_y,
        handle,
        config=config,
        previous_event=buffer.capture(),
        async_finish=True,
        release_after_combine=False,
    )
    combine_event.current_stream_wait()

    grad_event = buffer.balanced_grad_reduce(
        handle,
        previous_event=buffer.capture(),
        async_finish=True,
    )
    grad_event.current_stream_wait()
    _check_grad_reduce(
        views[3:], global_replica_map, local_replica_map, rank, handle.slot
    )
    dist.barrier(group=group)

    if rank == 0:
        print(
            "[ProbeEP expert I/O] local+IBGDA chunked weights, local+remote "
            "FP32 grads, and replica clearing passed",
            flush=True,
        )
    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the ProbeEP fixed expert pool on two NVL8 nodes"
    )
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--num-tokens", type=int, default=64)
    arguments = parser.parse_args()
    torch.multiprocessing.spawn(
        run_worker,
        args=(arguments.num_processes, arguments),
        nprocs=arguments.num_processes,
    )
