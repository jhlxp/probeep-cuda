"""CPU/PyTorch reference for ProbeEP's server-local execution planner.

The production CUDA planner sees routes from all source ranks, but is only
allowed to move an expert assignment within the expert's home server.  This
module intentionally favors a literal, deterministic implementation over
speed.  It is a test oracle and must never be imported by the timed path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PlanningReference:
    """All integer products needed to compare a CUDA execution plan."""

    exec_rank: torch.Tensor
    exec_slot: torch.Tensor
    synthetic_expert: torch.Tensor
    tokens_per_exec_slot: torch.Tensor
    padded_tokens_per_exec_slot: torch.Tensor
    cu_seqlens: torch.Tensor
    replica_expert: torch.Tensor
    is_token_in_exec_rank: torch.Tensor
    alloc: torch.Tensor
    tokens_per_expert: torch.Tensor
    route_ordinal: torch.Tensor
    rank_capacity: torch.Tensor
    rank_load_before: torch.Tensor
    rank_load_after: torch.Tensor
    ranks_per_server: int
    local_experts: int
    replica_slots: int
    token_padding: int


def _smallest_argmax(values: list[int]) -> int:
    return max(range(len(values)), key=lambda index: (values[index], -index))


def _smallest_argmin(values: list[int]) -> int:
    return min(range(len(values)), key=lambda index: (values[index], index))


def _validate_topk(
    topk_experts: torch.Tensor,
    ranks_per_server: int,
    local_experts: int,
) -> tuple[int, int, int, int]:
    if topk_experts.device.type != "cpu":
        raise ValueError("planning reference requires a CPU topk tensor")
    if topk_experts.ndim != 3:
        raise ValueError("topk_experts must have shape [world_size, tokens, topk]")

    world_size, num_tokens, topk = topk_experts.shape
    num_experts = world_size * local_experts
    if world_size % ranks_per_server != 0:
        raise ValueError("world_size must be divisible by ranks_per_server")
    if bool(((topk_experts < 0) | (topk_experts >= num_experts)).any().item()):
        raise ValueError("topk expert ids must be in [0, num_experts)")

    sorted_topk = topk_experts.to(torch.int64).sort(dim=-1).values
    if topk > 1 and bool((sorted_topk[..., 1:] == sorted_topk[..., :-1]).any().item()):
        raise ValueError("duplicate global experts inside one token are unsupported")
    return world_size, num_tokens, topk, num_experts


def _build_allocation(
    expert_counts: list[int],
    world_size: int,
    ranks_per_server: int,
    local_experts: int,
) -> tuple[list[list[int]], list[int], list[int]]:
    """Return alloc[expert][destination], capacity and original rank load."""

    num_experts = world_size * local_experts
    num_servers = world_size // ranks_per_server
    experts_per_server = ranks_per_server * local_experts
    alloc = [[0 for _ in range(world_size)] for _ in range(num_experts)]
    rank_capacity = [0 for _ in range(world_size)]
    rank_load_before = [0 for _ in range(world_size)]

    for expert, count in enumerate(expert_counts):
        home_rank = expert // local_experts
        alloc[expert][home_rank] = count
        rank_load_before[home_rank] += count

    for server in range(num_servers):
        rank_begin = server * ranks_per_server
        expert_begin = server * experts_per_server
        server_total = sum(expert_counts[expert_begin : expert_begin + experts_per_server])
        base, remainder = divmod(server_total, ranks_per_server)
        capacity = [base + int(local_rank < remainder) for local_rank in range(ranks_per_server)]
        for local_rank, value in enumerate(capacity):
            rank_capacity[rank_begin + local_rank] = value

        home_load = [rank_load_before[rank_begin + i] for i in range(ranks_per_server)]
        balance = [home_load[i] - capacity[i] for i in range(ranks_per_server)]
        quotas = [[0 for _ in range(ranks_per_server)] for _ in range(ranks_per_server)]

        # Match MoonEP: fill the roomiest receiver in one shot.  The sender
        # may temporarily become a receiver; this is part of the algorithm.
        while True:
            owner = _smallest_argmax(balance)
            destination = _smallest_argmin(balance)
            if balance[owner] <= 0:
                break
            move = -balance[destination]
            quotas[owner][destination] += move
            balance[owner] -= move
            balance[destination] = 0

        for owner_local in range(ranks_per_server):
            owner_rank = rank_begin + owner_local
            owner_expert_begin = owner_rank * local_experts
            remaining = expert_counts[
                owner_expert_begin : owner_expert_begin + local_experts
            ].copy()
            owner_quotas = quotas[owner_local].copy()

            while True:
                destination_local = _smallest_argmax(owner_quotas)
                quota = owner_quotas[destination_local]
                if quota <= 0:
                    break
                expert_local = _smallest_argmax(remaining)
                available = remaining[expert_local]
                if available <= 0:
                    raise AssertionError("planner quota exceeds the owner's assignments")

                take = min(quota, available)
                expert = owner_expert_begin + expert_local
                destination_rank = rank_begin + destination_local
                alloc[expert][destination_rank] += take
                alloc[expert][owner_rank] -= take
                remaining[expert_local] -= take
                owner_quotas[destination_local] -= take

    return alloc, rank_capacity, rank_load_before


def plan_server_local(
    topk_experts: torch.Tensor,
    *,
    ranks_per_server: int = 8,
    local_experts: int = 16,
    replica_slots: int | None = None,
    token_padding: int = 8,
) -> PlanningReference:
    """Build the deterministic server-local plan for a complete EP batch.

    ``topk_experts`` is ordered as ``[source_rank, token, k]``.  That flattened
    order is also the global route order used to split an expert across its
    execution ranks.
    """

    if replica_slots is None:
        replica_slots = local_experts
    world_size, num_tokens, topk, num_experts = _validate_topk(
        topk_experts, ranks_per_server, local_experts
    )
    if replica_slots <= 0 or token_padding <= 0:
        raise ValueError("replica_slots and token_padding must be positive")

    flat_experts = topk_experts.to(torch.int64).contiguous().view(-1)
    tokens_per_expert = torch.bincount(flat_experts, minlength=num_experts).to(torch.int32)
    alloc_list, capacity_list, before_list = _build_allocation(
        tokens_per_expert.tolist(), world_size, ranks_per_server, local_experts
    )
    alloc64 = torch.tensor(alloc_list, dtype=torch.int64)

    if not torch.equal(alloc64.sum(dim=1), tokens_per_expert.to(torch.int64)):
        raise AssertionError("per-expert route conservation failed")

    rank_capacity = torch.tensor(capacity_list, dtype=torch.int32)
    rank_load_before = torch.tensor(before_list, dtype=torch.int32)
    rank_load_after = alloc64.sum(dim=0).to(torch.int32)
    if not torch.equal(rank_load_after, rank_capacity):
        raise AssertionError("server-local allocation did not reach its capacity vector")

    # Stable expert sort turns the flattened source-rank/token/k order into an
    # expert-local ordinal without a Python loop over all routes.
    sorted_positions = torch.argsort(flat_experts, stable=True)
    sorted_experts = flat_experts[sorted_positions]
    expert_starts = torch.cumsum(tokens_per_expert.to(torch.int64), dim=0) - tokens_per_expert
    sorted_ordinals = torch.arange(flat_experts.numel(), dtype=torch.int64)
    sorted_ordinals -= expert_starts[sorted_experts]
    route_ordinal_flat = torch.empty_like(sorted_ordinals)
    route_ordinal_flat[sorted_positions] = sorted_ordinals

    alloc_prefix = alloc64.cumsum(dim=1)
    sorted_destinations = torch.empty_like(sorted_ordinals)
    cursor = 0
    for expert, count in enumerate(tokens_per_expert.tolist()):
        if count:
            ordinals = torch.arange(count, dtype=torch.int64)
            sorted_destinations[cursor : cursor + count] = torch.searchsorted(
                alloc_prefix[expert], ordinals, right=True
            )
            cursor += count
    exec_rank_flat = torch.empty_like(sorted_destinations)
    exec_rank_flat[sorted_positions] = sorted_destinations

    replica_expert = torch.full(
        (world_size, replica_slots), -1, dtype=torch.int32
    )
    slot_lookup = torch.full((world_size, num_experts), -1, dtype=torch.int64)
    tokens_per_exec_slot = torch.zeros(
        (world_size, local_experts + replica_slots), dtype=torch.int32
    )

    for destination in range(world_size):
        local_begin = destination * local_experts
        local_end = local_begin + local_experts
        for expert in range(local_begin, local_end):
            count = int(alloc64[expert, destination].item())
            slot = expert - local_begin
            slot_lookup[destination, expert] = slot
            tokens_per_exec_slot[destination, slot] = count

        remote = [
            expert
            for expert in range(num_experts)
            if int(alloc64[expert, destination].item()) > 0
            and not (local_begin <= expert < local_end)
        ]
        remote.sort(
            key=lambda expert: (int(alloc64[expert, destination].item()), expert),
            reverse=True,
        )
        if len(remote) > replica_slots:
            raise AssertionError(
                f"rank {destination} needs {len(remote)} replica slots, has {replica_slots}"
            )
        for replica_slot, expert in enumerate(remote):
            execution_slot = local_experts + replica_slot
            replica_expert[destination, replica_slot] = expert
            slot_lookup[destination, expert] = execution_slot
            tokens_per_exec_slot[destination, execution_slot] = alloc64[
                expert, destination
            ].to(torch.int32)

    exec_slot_flat = slot_lookup[exec_rank_flat, flat_experts]
    if bool((exec_slot_flat < 0).any().item()):
        raise AssertionError("an allocated route has no execution slot")

    shape = (world_size, num_tokens, topk)
    exec_rank = exec_rank_flat.to(torch.int32).view(shape)
    exec_slot = exec_slot_flat.to(torch.int32).view(shape)
    synthetic_expert = exec_rank * (local_experts + replica_slots) + exec_slot
    route_ordinal = route_ordinal_flat.to(torch.int32).view(shape)

    is_token_in_exec_rank = torch.zeros(
        (world_size, num_tokens, world_size), dtype=torch.bool
    )
    is_token_in_exec_rank.scatter_(2, exec_rank.to(torch.int64), True)

    padded_tokens_per_exec_slot = torch.where(
        tokens_per_exec_slot > 0,
        ((tokens_per_exec_slot + token_padding - 1) // token_padding) * token_padding,
        0,
    )
    cu_seqlens = torch.zeros(
        (world_size, local_experts + replica_slots + 1), dtype=torch.int32
    )
    cu_seqlens[:, 1:] = padded_tokens_per_exec_slot.cumsum(dim=1)

    return PlanningReference(
        exec_rank=exec_rank,
        exec_slot=exec_slot,
        synthetic_expert=synthetic_expert,
        tokens_per_exec_slot=tokens_per_exec_slot,
        padded_tokens_per_exec_slot=padded_tokens_per_exec_slot,
        cu_seqlens=cu_seqlens,
        replica_expert=replica_expert,
        is_token_in_exec_rank=is_token_in_exec_rank,
        alloc=alloc64.to(torch.int32),
        tokens_per_expert=tokens_per_expert,
        route_ordinal=route_ordinal,
        rank_capacity=rank_capacity,
        rank_load_before=rank_load_before,
        rank_load_after=rank_load_after,
        ranks_per_server=ranks_per_server,
        local_experts=local_experts,
        replica_slots=replica_slots,
        token_padding=token_padding,
    )
