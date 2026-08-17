"""Semantic invariant checks for CPU and CUDA ProbeEP plans."""

from __future__ import annotations

import torch

from planning_reference import PlanningReference


EXACT_PLAN_FIELDS = (
    "exec_rank",
    "exec_slot",
    "synthetic_expert",
    "tokens_per_exec_slot",
    "replica_expert",
    "is_token_in_exec_rank",
    "rank_capacity",
    "rank_load_after",
)


def assert_plan_equal(
    actual: PlanningReference,
    expected: PlanningReference,
    field_names: tuple[str, ...] = EXACT_PLAN_FIELDS,
) -> None:
    for field_name in field_names:
        actual_value = getattr(actual, field_name)
        expected_value = getattr(expected, field_name)
        if not torch.equal(actual_value.cpu(), expected_value.cpu()):
            mismatch = int((actual_value.cpu() != expected_value.cpu()).sum().item())
            raise AssertionError(f"{field_name} differs at {mismatch} entries")


def assert_plan_invariants(plan: PlanningReference, topk_experts: torch.Tensor) -> None:
    world_size, num_tokens, topk = topk_experts.shape
    local_experts = plan.local_experts
    ranks_per_server = plan.ranks_per_server
    slots_per_rank = local_experts + plan.replica_slots

    if tuple(plan.exec_rank.shape) != (world_size, num_tokens, topk):
        raise AssertionError("exec_rank shape does not match topk")
    if tuple(plan.exec_slot.shape) != tuple(plan.exec_rank.shape):
        raise AssertionError("exec_slot shape does not match exec_rank")
    if bool(((plan.exec_rank < 0) | (plan.exec_rank >= world_size)).any().item()):
        raise AssertionError("execution rank is out of range")
    if bool(((plan.exec_slot < 0) | (plan.exec_slot >= slots_per_rank)).any().item()):
        raise AssertionError("execution slot is out of range")

    home_rank = topk_experts.to(torch.int64) // local_experts
    if not torch.equal(
        home_rank // ranks_per_server,
        plan.exec_rank.to(torch.int64) // ranks_per_server,
    ):
        raise AssertionError("an assignment crossed its expert's home server")

    synthetic = plan.exec_rank * slots_per_rank + plan.exec_slot
    if not torch.equal(synthetic, plan.synthetic_expert):
        raise AssertionError("synthetic expert encoding is inconsistent")

    if int(plan.tokens_per_exec_slot.sum().item()) != topk_experts.numel():
        raise AssertionError("effective route count is not conserved")
    if not torch.equal(plan.tokens_per_exec_slot.sum(dim=1), plan.rank_load_after):
        raise AssertionError("execution slot counts do not match rank loads")
    if not torch.equal(plan.rank_load_after, plan.rank_capacity):
        raise AssertionError("rank load does not match the server capacity vector")

    for server_begin in range(0, world_size, ranks_per_server):
        server_load = plan.rank_load_after[
            server_begin : server_begin + ranks_per_server
        ]
        if int((server_load.max() - server_load.min()).item()) > 1:
            raise AssertionError("server-local rank load differs by more than one")

    expected_membership = torch.zeros_like(plan.is_token_in_exec_rank)
    expected_membership.scatter_(2, plan.exec_rank.to(torch.int64), True)
    if not torch.equal(expected_membership, plan.is_token_in_exec_rank):
        raise AssertionError("is_token_in_exec_rank is inconsistent")

    for destination in range(world_size):
        replicas = plan.replica_expert[destination]
        valid = replicas[replicas >= 0].to(torch.int64)
        if valid.numel() != torch.unique(valid).numel():
            raise AssertionError("a destination has duplicate replica experts")
        if valid.numel() and bool(
            ((valid // local_experts) // ranks_per_server
             != destination // ranks_per_server).any().item()
        ):
            raise AssertionError("a replica crossed its expert's home server")

    padded = plan.padded_tokens_per_exec_slot
    if bool((padded < plan.tokens_per_exec_slot).any().item()):
        raise AssertionError("padded slot count is smaller than its valid count")
    nonempty = padded > 0
    if bool((padded[nonempty] % plan.token_padding != 0).any().item()):
        raise AssertionError("nonempty slots are not token-padding aligned")
    if not torch.equal(plan.cu_seqlens[:, 0], torch.zeros(world_size, dtype=torch.int32)):
        raise AssertionError("cu_seqlens must start at zero")
    if not torch.equal(plan.cu_seqlens[:, 1:].diff(dim=1, prepend=plan.cu_seqlens[:, :1]), padded):
        raise AssertionError("cu_seqlens does not describe padded slot sizes")


def max_violation(load: torch.Tensor) -> float:
    values = load.to(torch.float64)
    mean = float(values.mean().item())
    return 0.0 if mean == 0.0 else float(values.max().item() / mean - 1.0)


def plan_statistics(plan: PlanningReference) -> dict[str, float | int]:
    replica_count = int((plan.replica_expert >= 0).sum().item())
    destination = torch.arange(plan.alloc.shape[1]).view(1, -1)
    home = (
        torch.arange(plan.alloc.shape[0]) // plan.local_experts
    ).view(-1, 1)
    moved = int((plan.alloc * (destination != home)).sum().item())
    return {
        "assignments": int(plan.tokens_per_expert.sum().item()),
        "replica_count": replica_count,
        "moved_assignments": moved,
        "rank_maxvio_before": max_violation(plan.rank_load_before),
        "rank_maxvio_after": max_violation(plan.rank_load_after),
    }
