"""Deterministic CPU capacity preview for the bounded ProbeEP planner.

This module sizes fixed benchmark buffers before the CUDA runtime is created.
It is not the production plan authority and must not be emitted as measured
placement evidence. It is never imported by the timed path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class ProbeConfig:
    ranks_per_server: int = 8
    local_experts: int = 16
    replica_slots: int = 16
    token_padding: int = 8
    weight_chunk_bytes: int = 4 * 1024 * 1024
    expert_weight_bytes: int = 3 * 7168 * 2048 * 2
    rail_bandwidth_gbps: float = 200.0
    alpha: float = 0.90
    initial_migration_budget_bytes: int = 0
    max_migration_budget_bytes: int = 64 * 1024 * 1024
    max_cross_iterations: int = 0

    def validate(self, world_size: int) -> None:
        if world_size % self.ranks_per_server != 0:
            raise ValueError("world_size must be divisible by ranks_per_server")
        num_servers = world_size // self.ranks_per_server
        if not 2 <= num_servers <= 16:
            raise ValueError("the ProbeEP oracle supports 2..16 servers")
        if min(
            self.ranks_per_server,
            self.local_experts,
            self.replica_slots,
            self.token_padding,
            self.weight_chunk_bytes,
            self.expert_weight_bytes,
            self.max_migration_budget_bytes,
        ) <= 0:
            raise ValueError("all ProbeEP sizes must be positive")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if self.rail_bandwidth_gbps <= 0:
            raise ValueError("rail bandwidth must be positive")

    @property
    def rail_bytes_per_ns(self) -> float:
        return self.rail_bandwidth_gbps / 8.0


@dataclass(frozen=True)
class ProbeFeedback:
    compute_ns: tuple[int, ...]
    network_ns: tuple[int, ...]
    dispatch_tx_bytes: tuple[int, ...]
    dispatch_rx_bytes: tuple[int, ...]
    migration_tx_bytes: tuple[int, ...]
    migration_rx_bytes: tuple[int, ...]
    valid: bool = True

    def validate(self, world_size: int) -> None:
        fields = (
            self.compute_ns,
            self.network_ns,
            self.dispatch_tx_bytes,
            self.dispatch_rx_bytes,
            self.migration_tx_bytes,
            self.migration_rx_bytes,
        )
        if any(len(field) != world_size for field in fields):
            raise ValueError("every feedback vector must have world_size entries")
        if any(value < 0 for field in fields for value in field):
            raise ValueError("feedback values must be non-negative")


@dataclass(frozen=True)
class ControllerResult:
    migration_budget_bytes: tuple[int, ...]
    compute_max_ns: int
    network_max_ns: int
    sampled_total_bytes: int
    probe_total_bytes: int
    theory_total_bytes: int
    target_total_bytes: int
    source: str


@dataclass(frozen=True)
class WeightChunk:
    expert_id: int
    replica_id: int
    chunk_id: int
    source_server: int
    destination_server: int
    source_rank: int
    destination_rank: int
    byte_offset: int
    num_bytes: int
    rail: int
    rail_packed_offset: int


@dataclass(frozen=True)
class RemoteReplica:
    replica_id: int
    expert_id: int
    source_server: int
    destination_server: int
    seed_rank: int
    moved_routes: int
    chunks: tuple[WeightChunk, ...]


@dataclass(frozen=True)
class ProbePlan:
    alloc: torch.Tensor
    exec_rank: torch.Tensor
    exec_slot: torch.Tensor
    replica_expert: torch.Tensor
    slot_count: torch.Tensor
    slot_begin: torch.Tensor
    rank_load_before: torch.Tensor
    rank_load_after: torch.Tensor
    server_load_before: torch.Tensor
    server_load_after: torch.Tensor
    server_padded_load_before: torch.Tensor
    server_padded_load_after: torch.Tensor
    controller: ControllerResult
    replicas: tuple[RemoteReplica, ...]
    deferred_experts: tuple[int, ...]
    assigned_tx_bytes: tuple[int, ...]
    assigned_rx_bytes: tuple[int, ...]
    token_padding: int


def compute_controller_budget(
    feedback: ProbeFeedback | None,
    *,
    world_size: int,
    config: ProbeConfig,
) -> ControllerResult:
    """Apply the global alpha*Cmax/Nmax controller from the design spec."""

    config.validate(world_size)
    if feedback is None:
        fallback = (config.initial_migration_budget_bytes,) * world_size
        return ControllerResult(
            fallback, 0, 0, 0, 0, 0,
            config.initial_migration_budget_bytes,
            "initial_fallback",
        )

    feedback.validate(world_size)
    compute_max = max(feedback.compute_ns, default=0)
    network_max = max(feedback.network_ns, default=0)
    if not feedback.valid or compute_max <= 0 or network_max <= 0:
        fallback = (config.initial_migration_budget_bytes,) * world_size
        return ControllerResult(
            fallback, compute_max, network_max, 0, 0, 0,
            config.initial_migration_budget_bytes,
            "invalid_fallback",
        )

    bottleneck = [
        rank for rank, elapsed in enumerate(feedback.network_ns)
        if elapsed == network_max
    ]
    endpoint_bytes = tuple(
        max(
            feedback.dispatch_tx_bytes[rank] + feedback.migration_tx_bytes[rank],
            feedback.dispatch_rx_bytes[rank] + feedback.migration_rx_bytes[rank],
        )
        for rank in range(world_size)
    )
    sampled_total = max((endpoint_bytes[rank] for rank in bottleneck), default=0)
    probe_total = math.floor(
        config.alpha * compute_max * sampled_total / network_max
    )
    theory_total = math.floor(config.rail_bytes_per_ns * compute_max)
    target_total = min(probe_total, theory_total)
    budgets = tuple(
        min(
            config.max_migration_budget_bytes,
            max(
                0,
                target_total
                - max(
                    feedback.dispatch_tx_bytes[rank],
                    feedback.dispatch_rx_bytes[rank],
                ),
            ),
        )
        for rank in range(world_size)
    )
    return ControllerResult(
        budgets,
        compute_max,
        network_max,
        sampled_total,
        probe_total,
        theory_total,
        target_total,
        "sampled_global_ratio",
    )


def _try_schedule_replica(
    *,
    replica_id: int,
    expert_id: int,
    source_server: int,
    destination_server: int,
    seed_rank: int,
    moved_routes: int,
    assigned_tx: list[int],
    assigned_rx: list[int],
    rail_offsets: list[int],
    budgets: tuple[int, ...],
    config: ProbeConfig,
) -> tuple[RemoteReplica, list[int], list[int], list[int]] | None:
    next_tx = assigned_tx.copy()
    next_rx = assigned_rx.copy()
    next_offsets = rail_offsets.copy()
    chunks: list[WeightChunk] = []
    byte_offset = 0
    chunk_id = 0
    while byte_offset < config.expert_weight_bytes:
        num_bytes = min(
            config.weight_chunk_bytes,
            config.expert_weight_bytes - byte_offset,
        )
        candidates: list[tuple[int, int, int, int]] = []
        for rail in range(config.ranks_per_server):
            source_rank = source_server * config.ranks_per_server + rail
            destination_rank = destination_server * config.ranks_per_server + rail
            if next_tx[source_rank] + num_bytes > budgets[source_rank]:
                continue
            if next_rx[destination_rank] + num_bytes > budgets[destination_rank]:
                continue
            projected_tx = next_tx[source_rank] + num_bytes
            projected_rx = next_rx[destination_rank] + num_bytes
            candidates.append(
                (
                    next_offsets[rail] + num_bytes,
                    max(projected_tx, projected_rx),
                    projected_tx + projected_rx,
                    rail,
                    source_rank,
                )
            )
        if not candidates:
            return None
        _, _, _, rail, source_rank = min(candidates)
        destination_rank = destination_server * config.ranks_per_server + rail
        chunks.append(
            WeightChunk(
                expert_id=expert_id,
                replica_id=replica_id,
                chunk_id=chunk_id,
                source_server=source_server,
                destination_server=destination_server,
                source_rank=source_rank,
                destination_rank=destination_rank,
                byte_offset=byte_offset,
                num_bytes=num_bytes,
                rail=rail,
                rail_packed_offset=next_offsets[rail],
            )
        )
        next_tx[source_rank] += num_bytes
        next_rx[destination_rank] += num_bytes
        next_offsets[rail] += num_bytes
        byte_offset += num_bytes
        chunk_id += 1

    return (
        RemoteReplica(
            replica_id=replica_id,
            expert_id=expert_id,
            source_server=source_server,
            destination_server=destination_server,
            seed_rank=seed_rank,
            moved_routes=moved_routes,
            chunks=tuple(chunks),
        ),
        next_tx,
        next_rx,
        next_offsets,
    )


def _pack_server(
    alloc: list[list[int]],
    *,
    server: int,
    ranks_per_server: int,
    local_experts: int,
    replica_slots: int,
    token_padding: int,
) -> None:
    """Mirror the CUDA padding-block placement without sorting experts."""
    rank_begin = server * ranks_per_server
    ranks = list(range(rank_begin, rank_begin + ranks_per_server))

    def padded(rows: int) -> int:
        return 0 if rows <= 0 else math.ceil(rows / token_padding) * token_padding

    initial_padded = [
        sum(padded(rows[rank]) for rows in alloc)
        for rank in ranks
    ]
    if len(set(initial_padded)) == 1:
        return

    expert_rows: list[int] = []
    for rows in alloc:
        total = sum(rows[rank] for rank in ranks)
        expert_rows.append(total)
        for rank in ranks:
            rows[rank] = 0

    total_blocks = sum(padded(rows) for rows in expert_rows) // token_padding
    floor_blocks, remainder_blocks = divmod(total_blocks, ranks_per_server)
    targets = [
        (floor_blocks + int(local < remainder_blocks)) * token_padding
        for local in range(ranks_per_server)
    ]
    raw_load = [0] * ranks_per_server
    padded_load = [0] * ranks_per_server
    used_slots = [0] * ranks_per_server

    def owner_rank(expert: int) -> int:
        return expert // local_experts

    remote_remaining = sum(
        rows > 0 and owner_rank(expert) // ranks_per_server != server
        for expert, rows in enumerate(expert_rows)
    )
    free_remote_slots = ranks_per_server * replica_slots

    for remote_phase in (1, 0):
        for expert, total_rows in enumerate(expert_rows):
            is_remote = owner_rank(expert) // ranks_per_server != server
            if total_rows <= 0 or int(is_remote) != remote_phase:
                continue
            remaining = total_rows
            home = owner_rank(expert)
            future_remote = remote_remaining - 1 if is_remote else 0
            while remaining > 0:
                selected: int | None = None
                selected_slack = -(1 << 60)
                for local, rank in enumerate(ranks):
                    already_present = alloc[expert][rank] > 0
                    if rank != home and not already_present:
                        if used_slots[local] >= replica_slots:
                            continue
                        if is_remote and free_remote_slots <= future_remote:
                            continue
                    slack = targets[local] - padded_load[local]
                    if (
                        selected is None
                        or slack > selected_slack
                        or (slack == selected_slack and rank == home)
                        or (
                            slack == selected_slack
                            and ranks[selected] != home
                            and rank < ranks[selected]
                        )
                    ):
                        selected = local
                        selected_slack = slack
                if selected is None:
                    if not is_remote:
                        selected = home - rank_begin
                    else:
                        selected = next(
                            (
                                local
                                for local, rank in enumerate(ranks)
                                if alloc[expert][rank] > 0
                            ),
                            None,
                        )
                if selected is None:
                    break

                rank = ranks[selected]
                slack = targets[selected] - padded_load[selected]
                quota = remaining
                if remaining > token_padding and slack >= token_padding:
                    quota = min(quota, slack)
                    if quota < remaining:
                        quota = quota // token_padding * token_padding
                quota = max(1, min(quota, remaining))
                previous = alloc[expert][rank]
                alloc[expert][rank] += quota
                raw_load[selected] += quota
                padded_load[selected] += (
                    padded(previous + quota) - padded(previous)
                )
                if previous == 0 and rank != home:
                    used_slots[selected] += 1
                    if is_remote:
                        free_remote_slots -= 1
                remaining -= quota
            if is_remote:
                remote_remaining -= 1


def _materialize_routes(
    topk_experts: torch.Tensor,
    alloc: torch.Tensor,
    *,
    local_experts: int,
    replica_slots: int,
    token_padding: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    world_size, num_tokens, topk = topk_experts.shape
    num_experts = world_size * local_experts
    flat = topk_experts.to(torch.int64).contiguous().view(-1)
    counts = torch.bincount(flat, minlength=num_experts).to(torch.int64)
    sorted_positions = torch.argsort(flat, stable=True)
    sorted_experts = flat[sorted_positions]
    starts = counts.cumsum(0) - counts
    sorted_ordinals = torch.arange(flat.numel(), dtype=torch.int64) - starts[sorted_experts]
    ordinals = torch.empty_like(sorted_ordinals)
    ordinals[sorted_positions] = sorted_ordinals
    prefix = alloc.to(torch.int64).cumsum(1)
    exec_rank_flat = torch.searchsorted(
        prefix[flat], ordinals.unsqueeze(1), right=True
    ).squeeze(1)

    replica_expert = torch.full((world_size, replica_slots), -1, dtype=torch.int32)
    slot_lookup = torch.full((world_size, num_experts), -1, dtype=torch.int64)
    slot_count = torch.zeros(
        (world_size, local_experts + replica_slots), dtype=torch.int32
    )
    for rank in range(world_size):
        local_begin = rank * local_experts
        for expert in range(local_begin, local_begin + local_experts):
            slot = expert - local_begin
            slot_lookup[rank, expert] = slot
            slot_count[rank, slot] = alloc[expert, rank]
        remote = [
            expert for expert in range(num_experts)
            if int(alloc[expert, rank]) > 0
            and not local_begin <= expert < local_begin + local_experts
        ]
        remote.sort()
        if len(remote) > replica_slots:
            raise AssertionError(
                f"rank {rank} needs {len(remote)} replica slots, has {replica_slots}"
            )
        for offset, expert in enumerate(remote):
            slot = local_experts + offset
            replica_expert[rank, offset] = expert
            slot_lookup[rank, expert] = slot
            slot_count[rank, slot] = alloc[expert, rank]

    exec_slot_flat = slot_lookup[exec_rank_flat, flat]
    if bool((exec_slot_flat < 0).any()):
        raise AssertionError("route materialization found an unassigned slot")
    padded = torch.where(
        slot_count > 0,
        ((slot_count + token_padding - 1) // token_padding) * token_padding,
        0,
    )
    slot_begin = torch.zeros_like(slot_count)
    if slot_count.size(1) > 1:
        slot_begin[:, 1:] = padded.cumsum(1)[:, :-1]
    shape = (world_size, num_tokens, topk)
    return (
        exec_rank_flat.to(torch.int32).view(shape),
        exec_slot_flat.to(torch.int32).view(shape),
        replica_expert,
        slot_count,
        slot_begin,
    )


def plan_probeep(
    topk_experts: torch.Tensor,
    *,
    feedback: ProbeFeedback | None = None,
    config: ProbeConfig = ProbeConfig(),
) -> ProbePlan:
    if topk_experts.device.type != "cpu" or topk_experts.ndim != 3:
        raise ValueError("topk_experts must be a CPU tensor [world,tokens,topk]")
    world_size, _, _ = topk_experts.shape
    config.validate(world_size)
    num_experts = world_size * config.local_experts
    if bool(((topk_experts < 0) | (topk_experts >= num_experts)).any()):
        raise ValueError("expert id is outside the configured EP range")

    controller = compute_controller_budget(
        feedback, world_size=world_size, config=config
    )
    expert_counts = torch.bincount(
        topk_experts.to(torch.int64).contiguous().view(-1),
        minlength=num_experts,
    ).tolist()
    alloc = [[0 for _ in range(world_size)] for _ in range(num_experts)]
    rank_before = [0 for _ in range(world_size)]
    for expert, count in enumerate(expert_counts):
        owner = expert // config.local_experts
        alloc[expert][owner] = count
        rank_before[owner] += count

    num_servers = world_size // config.ranks_per_server
    experts_per_server = config.ranks_per_server * config.local_experts
    server_before = [
        sum(expert_counts[s * experts_per_server : (s + 1) * experts_per_server])
        for s in range(num_servers)
    ]
    def padded(rows: int) -> int:
        return 0 if rows <= 0 else math.ceil(rows / config.token_padding) * config.token_padding

    server_padded_before = [
        sum(
            padded(expert_counts[expert])
            for expert in range(
                server * experts_per_server,
                (server + 1) * experts_per_server,
            )
        )
        for server in range(num_servers)
    ]
    server_padded = server_padded_before.copy()
    target_floor, target_remainder = divmod(sum(server_before), num_servers)
    server_target = [target_floor] * num_servers
    for server in sorted(
        range(num_servers), key=lambda candidate: (-server_before[candidate], candidate)
    )[:target_remainder]:
        server_target[server] += 1
    server_surplus = [
        max(server_before[server] - server_target[server], 0)
        for server in range(num_servers)
    ]
    server_deficit = [
        max(server_target[server] - server_before[server], 0)
        for server in range(num_servers)
    ]
    assigned_tx = [0] * world_size
    assigned_rx = [0] * world_size
    pair_offsets = [
        [[0] * config.ranks_per_server for _ in range(num_servers)]
        for _ in range(num_servers)
    ]
    blocked_pair = [[False] * num_servers for _ in range(num_servers)]
    rejected_destination = [
        [False] * num_servers for _ in range(num_experts)
    ]
    remote_slots = [0] * world_size
    replicas: list[RemoteReplica] = []
    deferred: set[int] = set()
    max_expert_rows = max(
        (
            expert_counts[expert]
            for expert in range(num_experts)
            if server_surplus[
                (expert // config.local_experts) // config.ranks_per_server
            ]
            > 0
        ),
        default=0,
    )
    hot_experts = [
        expert
        for bucket in range(31, -1, -1)
        for expert in range(num_experts)
        if max_expert_rows > 0
        and server_surplus[
            (expert // config.local_experts) // config.ranks_per_server
        ]
        > 0
        and expert_counts[expert] * 32 // (max_expert_rows + 1) == bucket
    ]
    rank_raw = rank_before.copy()
    max_cross_iterations = (
        config.max_cross_iterations
        if config.max_cross_iterations > 0
        else num_experts * max(num_servers - 1, 1)
    )
    attempts = 0
    for expert in hot_experts:
        source_rank = expert // config.local_experts
        source_server = source_rank // config.ranks_per_server
        while (
            server_surplus[source_server] > 0
            and alloc[expert][source_rank] > 0
            and attempts < max_cross_iterations
        ):
            candidates = [
                server
                for server in range(num_servers)
                if server != source_server
                and server_deficit[server] > 0
                and not blocked_pair[source_server][server]
                and not rejected_destination[expert][server]
            ]
            if not candidates:
                break
            destination_server = min(
                candidates,
                key=lambda server: (-server_deficit[server], server),
            )
            destination_begin = destination_server * config.ranks_per_server
            destination_ranks = range(
                destination_begin, destination_begin + config.ranks_per_server
            )
            if any(alloc[expert][rank] > 0 for rank in destination_ranks):
                rejected_destination[expert][destination_server] = True
                continue
            seed_candidates = [
                rank
                for rank in destination_ranks
                if remote_slots[rank] < config.replica_slots
            ]
            if not seed_candidates:
                rejected_destination[expert][destination_server] = True
                continue
            seed_rank = min(seed_candidates, key=lambda rank: (rank_raw[rank], rank))
            source_rows = alloc[expert][source_rank]
            moved = min(
                source_rows,
                server_surplus[source_server],
                server_deficit[destination_server],
            )
            if moved <= 0:
                break
            attempts += 1

            candidate_padded = server_padded.copy()
            candidate_padded[source_server] += (
                padded(source_rows - moved) - padded(source_rows)
            )
            candidate_padded[destination_server] += padded(moved)
            current_objective = (
                max(server_padded), max(server_padded) - min(server_padded)
            )
            candidate_objective = (
                max(candidate_padded),
                max(candidate_padded) - min(candidate_padded),
            )
            if candidate_objective >= current_objective:
                rejected_destination[expert][destination_server] = True
                deferred.add(expert)
                continue

            scheduled = _try_schedule_replica(
                replica_id=len(replicas),
                expert_id=expert,
                source_server=source_server,
                destination_server=destination_server,
                seed_rank=seed_rank,
                moved_routes=moved,
                assigned_tx=assigned_tx,
                assigned_rx=assigned_rx,
                rail_offsets=pair_offsets[source_server][destination_server],
                budgets=controller.migration_budget_bytes,
                config=config,
            )
            if scheduled is None:
                blocked_pair[source_server][destination_server] = True
                deferred.add(expert)
                continue
            replica, assigned_tx, assigned_rx, next_offsets = scheduled
            pair_offsets[source_server][destination_server] = next_offsets
            alloc[expert][source_rank] -= moved
            alloc[expert][seed_rank] += moved
            rank_raw[source_rank] -= moved
            rank_raw[seed_rank] += moved
            server_padded = candidate_padded
            server_surplus[source_server] -= moved
            server_deficit[destination_server] -= moved
            remote_slots[seed_rank] += 1
            replicas.append(replica)

    for server in range(num_servers):
        _pack_server(
            alloc,
            server=server,
            ranks_per_server=config.ranks_per_server,
            local_experts=config.local_experts,
            replica_slots=config.replica_slots,
            token_padding=config.token_padding,
        )

    alloc_tensor = torch.tensor(alloc, dtype=torch.int32)
    if not torch.equal(
        alloc_tensor.sum(1).to(torch.int64),
        torch.tensor(expert_counts, dtype=torch.int64),
    ):
        raise AssertionError("per-expert conservation failed")
    exec_rank, exec_slot, replica_expert, slot_count, slot_begin = _materialize_routes(
        topk_experts,
        alloc_tensor,
        local_experts=config.local_experts,
        replica_slots=config.replica_slots,
        token_padding=config.token_padding,
    )
    rank_after = alloc_tensor.sum(0)
    server_after = torch.stack(
        tuple(
            rank_after[
                server * config.ranks_per_server :
                (server + 1) * config.ranks_per_server
            ].sum()
            for server in range(num_servers)
        )
    ).to(torch.int32)
    server_padded_after = torch.tensor(
        [
            sum(
                padded(int(alloc_tensor[expert, rank]))
                for expert in range(num_experts)
                for rank in range(
                    server * config.ranks_per_server,
                    (server + 1) * config.ranks_per_server,
                )
            )
            for server in range(num_servers)
        ],
        dtype=torch.int32,
    )
    return ProbePlan(
        alloc=alloc_tensor,
        exec_rank=exec_rank,
        exec_slot=exec_slot,
        replica_expert=replica_expert,
        slot_count=slot_count,
        slot_begin=slot_begin,
        rank_load_before=torch.tensor(rank_before, dtype=torch.int32),
        rank_load_after=rank_after.to(torch.int32),
        server_load_before=torch.tensor(server_before, dtype=torch.int32),
        server_load_after=server_after,
        server_padded_load_before=torch.tensor(
            server_padded_before, dtype=torch.int32
        ),
        server_padded_load_after=server_padded_after,
        controller=controller,
        replicas=tuple(replicas),
        deferred_experts=tuple(sorted(deferred)),
        assigned_tx_bytes=tuple(assigned_tx),
        assigned_rx_bytes=tuple(assigned_rx),
        token_padding=config.token_padding,
    )
