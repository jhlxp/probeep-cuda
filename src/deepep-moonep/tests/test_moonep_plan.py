from pathlib import Path
import sys

import torch

import deep_ep_cpp


PROBEEP_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROBEEP_ROOT))

from test.correctness.planning_reference import plan_server_local  # noqa: E402
from test.correctness.workloads import make_routing_workload  # noqa: E402


def assert_server_local_cuda_matches_cpu_oracle(topk: torch.Tensor) -> None:
    world_size = 16
    ranks_per_server = 8
    local_experts = 16
    replica_slots = 16
    token_padding = 8
    num_slots = local_experts + replica_slots

    expected = plan_server_local(
        topk,
        ranks_per_server=ranks_per_server,
        local_experts=local_experts,
        replica_slots=replica_slots,
        token_padding=token_padding,
    )
    actual = deep_ep_cpp.plan_server_local(
        topk.cuda(),
        ranks_per_server=ranks_per_server,
        local_experts=local_experts,
        replica_slots=replica_slots,
        token_padding=token_padding,
    )

    slot_expert = torch.full((world_size, num_slots), -1, dtype=torch.int32)
    for rank in range(world_size):
        slot_expert[rank, :local_experts] = torch.arange(
            rank * local_experts, (rank + 1) * local_experts, dtype=torch.int32
        )
    slot_expert[:, local_experts:] = expected.replica_expert

    synthetic_expert = expected.exec_rank * num_slots + expected.exec_slot
    per_source_exec_expert = torch.stack(
        [
            torch.bincount(synthetic_expert[src].flatten().to(torch.int64), minlength=world_size * num_slots)
            for src in range(world_size)
        ]
    ).to(torch.int32)
    per_source_rank = expected.is_token_in_exec_rank.sum(dim=1).to(torch.int32)
    per_source_rdma_rank = expected.is_token_in_exec_rank.view(
        world_size, topk.size(1), world_size // ranks_per_server, ranks_per_server
    ).any(dim=-1).sum(dim=1).to(torch.int32)

    alloc_prefix = expected.alloc.to(torch.int64).cumsum(dim=1)
    flat_expert = topk.flatten().to(torch.int64)
    flat_rank = expected.exec_rank.flatten().to(torch.int64)
    flat_slot = expected.exec_slot.flatten().to(torch.int64)
    before = torch.where(
        flat_rank == 0,
        0,
        alloc_prefix[flat_expert, torch.clamp(flat_rank - 1, min=0)],
    )
    local_offset = (
        expected.cu_seqlens[flat_rank, flat_slot].to(torch.int64)
        + expected.route_ordinal.flatten().to(torch.int64)
        - before
    )
    num_servers = world_size // ranks_per_server
    nvs = (
        num_servers * topk.size(1) * topk.size(2)
        + (token_padding - 1) * num_slots
    )
    route_dst = (flat_rank * nvs + local_offset).view_as(topk).to(torch.int32)

    exact = {
        "route_dst": route_dst,
        "exec_rank": expected.exec_rank,
        "exec_slot": expected.exec_slot,
        "is_token_in_rank": expected.is_token_in_exec_rank,
        "slot_count": expected.tokens_per_exec_slot,
        "slot_begin": expected.cu_seqlens[:, :-1],
        "replica_expert": expected.replica_expert,
        "slot_expert": slot_expert,
        "num_tokens_per_rank": per_source_rank,
        "num_tokens_per_rdma_rank": per_source_rdma_rank,
        "num_tokens_per_exec_expert": per_source_exec_expert,
    }
    for name, value in exact.items():
        torch.testing.assert_close(actual[name].cpu(), value, rtol=0, atol=0)
    assert actual["nvs"] == nvs


def test_server_local_cuda_matches_cpu_oracle() -> None:
    workload = make_routing_workload(
        "server_preserving_skew",
        world_size=16,
        num_tokens=256,
        topk=8,
        local_experts=16,
        ranks_per_server=8,
        bias_ratio=1.0,
        seed=1234,
    )
    assert_server_local_cuda_matches_cpu_oracle(workload.topk_experts.contiguous())


def test_server_local_cuda_handles_one_sided_server_load() -> None:
    topk = (
        torch.arange(8, dtype=torch.int64)
        .view(1, 1, 8)
        .expand(16, 64, 8)
        .contiguous()
    )
    assert_server_local_cuda_matches_cpu_oracle(topk)
