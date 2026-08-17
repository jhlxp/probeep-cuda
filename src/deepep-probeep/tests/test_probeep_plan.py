"""SM90 device-contract tests for the standalone ProbeEP CUDA planner.

The production path exchanges compact histograms and runs from persistent
Buffer storage. This diagnostic binding accepts all rank routes on one GPU
so planner invariants can be checked without pretending that RDMA was tested.
No planner or admission decision is implemented in this test.
"""

from __future__ import annotations

from collections import defaultdict

import pytest
import torch

deep_ep_cpp = pytest.importorskip("deep_ep_cpp")


WORLD = 16
SERVERS = 2
EXPERTS = 256
TOPK = 8
CHUNK_BYTES = 4 * 1024 * 1024
EXPERT_BYTES = 84 * 1024 * 1024


def _require_sm90() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("the ProbeEP extension is built for SM90")


def _server_imbalanced_routes(tokens: int = 64, world: int = WORLD) -> torch.Tensor:
    token = torch.arange(world * tokens, dtype=torch.int64).view(world, tokens, 1)
    experts_per_server = EXPERTS // (world // 8)
    hot_lane = torch.arange(6, dtype=torch.int64).view(1, 1, 6)
    cold_lane = torch.arange(2, dtype=torch.int64).view(1, 1, 2)
    server_zero = (token * 6 + hot_lane) % experts_per_server
    server_one = experts_per_server + (token * 2 + cold_lane) % experts_per_server
    return torch.cat((server_zero, server_one), dim=2).contiguous().cuda()


def _plan(
    routes: torch.Tensor,
    budget_bytes: int,
    *,
    learned_total_bytes: int = 0,
):
    budgets = torch.full(
        (routes.size(0),), budget_bytes, dtype=torch.int64, device="cuda"
    )
    return deep_ep_cpp.plan_probeep(
        routes, budgets, learned_total_bytes=learned_total_bytes
    )


def _count(plan, index: int) -> int:
    return int(plan["plan_counts"][index].item())


def test_compute_plan_is_network_independent_and_admission_is_atomic() -> None:
    _require_sm90()
    routes = _server_imbalanced_routes()
    open_plan = _plan(routes, 64 * 1024 * 1024)
    closed_plan = _plan(routes, 0)
    torch.cuda.synchronize()

    open_intents = _count(open_plan, 3)
    closed_intents = _count(closed_plan, 3)
    assert open_intents == closed_intents > 0
    torch.testing.assert_close(
        open_plan["compute_intents"][:open_intents],
        closed_plan["compute_intents"][:closed_intents],
        rtol=0,
        atol=0,
    )
    assert _count(closed_plan, 0) == 0
    assert _count(closed_plan, 1) == 0
    assert _count(open_plan, 0) > 0
    assert _count(open_plan, 1) > 0

    chunks = open_plan["chunk_table"][: _count(open_plan, 1)].cpu()
    grouped = defaultdict(list)
    for row in chunks.tolist():
        grouped[(row[0], row[5])].append(row)
    assert grouped
    expected_chunks = EXPERT_BYTES // CHUNK_BYTES
    for rows in grouped.values():
        assert len(rows) == expected_chunks
        assert sorted(row[3] for row in rows) == list(range(expected_chunks))
        assert sum(row[9] for row in rows) == EXPERT_BYTES


def test_compute_intents_select_hot_experts_first() -> None:
    """The immutable compute planner must expose hot-first migration intents."""

    _require_sm90()
    routes = _server_imbalanced_routes(tokens=256)
    plan = _plan(routes, 64 * 1024 * 1024)
    torch.cuda.synchronize()
    count = _count(plan, 3)
    assert count > 0
    intents = plan["compute_intents"][:count].cpu()
    totals = torch.bincount(routes.flatten().cpu(), minlength=EXPERTS)
    selected_loads = [int(totals[int(row[0])]) for row in intents.tolist()]
    assert selected_loads == sorted(selected_loads, reverse=True)


def test_exactly_balanced_gate_does_not_create_migration_work() -> None:
    _require_sm90()
    tokens = 64
    token = torch.arange(
        WORLD * tokens, dtype=torch.int64, device="cuda"
    ).view(WORLD, tokens, 1)
    lane = torch.arange(TOPK, dtype=torch.int64, device="cuda").view(
        1, 1, TOPK
    )
    routes = ((token * TOPK + lane) % EXPERTS).contiguous()
    plan = _plan(routes, 64 * 1024 * 1024)
    counts = plan["plan_counts"].cpu().tolist()
    assert counts[3] == 0
    assert counts[0] == 0
    assert counts[1] == 0
    torch.testing.assert_close(
        plan["server_padded_load_after"],
        plan["server_padded_load_before"],
        rtol=0,
        atol=0,
    )


def test_raw_balanced_padding_imbalance_triggers_compute_refinement() -> None:
    """Equal raw server load must not hide grouped-GEMM padding skew."""

    _require_sm90()
    # Both servers own exactly 64 routes.  Server 0 spreads them over sixteen
    # four-row experts (16*8 padded rows); server 1 uses eight eight-row experts
    # (8*8 padded rows).  Every token still has eight unique TopK experts.
    server_zero = torch.arange(16, dtype=torch.int64).repeat(4)
    server_one = (128 + torch.arange(8, dtype=torch.int64)).repeat(8)
    routes = torch.cat((server_zero, server_one)).view(WORLD, 1, TOPK).cuda()
    open_plan = _plan(routes, 64 * 1024 * 1024)
    closed_plan = _plan(routes, 0)
    torch.cuda.synchronize()

    before_raw = open_plan["server_load_before"].cpu().tolist()
    after_raw = open_plan["server_load_after"].cpu().tolist()
    before_padded = open_plan["server_padded_load_before"].cpu().tolist()
    after_padded = open_plan["server_padded_load_after"].cpu().tolist()
    assert before_raw == [64, 64]
    assert before_padded == [128, 64]
    assert max(after_padded) < max(before_padded)
    assert after_raw != before_raw
    assert _count(open_plan, 3) > 0
    assert _count(open_plan, 0) > 0

    # Candidate generation remains compute-only even when the NIC window is
    # closed; only admission is suppressed.
    count = _count(open_plan, 3)
    assert _count(closed_plan, 3) == count
    torch.testing.assert_close(
        open_plan["compute_intents"][:count],
        closed_plan["compute_intents"][:count],
        rtol=0,
        atol=0,
    )
    assert _count(closed_plan, 0) == 0


def test_two_stage_balance_pair_waterfill_and_endpoint_caps() -> None:
    _require_sm90()
    routes = _server_imbalanced_routes()
    plan = _plan(routes, 64 * 1024 * 1024)
    torch.cuda.synchronize()

    counts = plan["plan_counts"].cpu().tolist()
    assert counts[2] == 0  # physical replica slot overflow
    assert counts[6] == 0  # negative allocation
    assert counts[7] == 0  # expert-route conservation failure
    assert counts[8] == 1  # compute greedy converged
    before = plan["server_padded_load_before"]
    after = plan["server_padded_load_after"]
    assert int(after.max()) < int(before.max())

    allocation = plan["alloc"]
    expected = torch.bincount(routes.flatten(), minlength=EXPERTS)
    torch.testing.assert_close(allocation.sum(1), expected, rtol=0, atol=0)
    assert int(plan["slot_count"].sum()) == WORLD * 64 * TOPK

    tx = plan["assigned_tx_bytes"] + plan["dispatch_tx_bytes"]
    rx = plan["assigned_rx_bytes"] + plan["dispatch_rx_bytes"]
    cap = plan["endpoint_total_cap_bytes"]
    assert bool(torch.all(tx <= cap))
    assert bool(torch.all(rx <= cap))

    chunks = plan["chunk_table"][: counts[1]].cpu()
    expected_pair = torch.zeros_like(plan["pair_load_bytes"].cpu())
    for row in chunks.tolist():
        expected_pair[row[4], row[5], row[10]] += row[9]
    torch.testing.assert_close(
        plan["pair_load_bytes"].cpu(), expected_pair, rtol=0, atol=0
    )
    for source in range(SERVERS):
        for destination in range(SERVERS):
            if source == destination:
                continue
            rails = expected_pair[source, destination]
            if int(rails.sum()) > 0:
                assert int(rails.max() - rails.min()) <= CHUNK_BYTES


def test_controller_matches_global_sample_formula() -> None:
    _require_sm90()
    mib = 1024 * 1024
    device = torch.device("cuda")
    compute = torch.full((WORLD,), 4_000_000, dtype=torch.int64, device=device)
    network = torch.full((WORLD,), 2_000_000, dtype=torch.int64, device=device)
    dispatch_tx = torch.full((WORLD,), 4 * mib, dtype=torch.int64, device=device)
    dispatch_rx = dispatch_tx.clone()
    migration_tx = torch.full((WORLD,), 16 * mib, dtype=torch.int64, device=device)
    migration_rx = migration_tx.clone()
    result = deep_ep_cpp.probeep_controller(
        compute,
        network,
        dispatch_tx,
        dispatch_rx,
        migration_tx,
        migration_rx,
        200.0,
        0.90,
        0,
        True,
    )
    assert result["migration_budget_bytes"].cpu().tolist() == [32 * mib] * WORLD
    assert result["summary"].cpu().tolist() == [
        4_000_000,
        2_000_000,
        20 * mib,
        36 * mib,
        100_000_000,
        36 * mib,
    ]


def test_controller_zero_byte_sample_uses_explicit_fallback() -> None:
    """A timed but traffic-free window is not a bandwidth observation."""

    _require_sm90()
    mib = 1024 * 1024
    device = torch.device("cuda")
    compute = torch.full((WORLD,), 4_000_000, dtype=torch.int64, device=device)
    network = torch.full((WORLD,), 2_000_000, dtype=torch.int64, device=device)
    zeros = torch.zeros((WORLD,), dtype=torch.int64, device=device)
    result = deep_ep_cpp.probeep_controller(
        compute,
        network,
        zeros,
        zeros,
        zeros,
        zeros,
        200.0,
        0.90,
        13 * mib,
        True,
    )
    assert result["migration_budget_bytes"].cpu().tolist() == [13 * mib] * WORLD
    assert result["summary"].cpu().tolist() == [
        4_000_000,
        2_000_000,
        0,
        0,
        0,
        13 * mib,
    ]


def test_stale_learned_window_cannot_crop_dispatch_baseline() -> None:
    """A changed layer may have more mandatory Dispatch than the old window."""

    _require_sm90()
    plan = _plan(
        _server_imbalanced_routes(),
        0,
        learned_total_bytes=1,
    )
    torch.cuda.synchronize()
    baseline = torch.maximum(
        plan["dispatch_tx_bytes"], plan["dispatch_rx_bytes"]
    )
    torch.testing.assert_close(
        plan["endpoint_total_cap_bytes"], baseline, rtol=0, atol=0
    )
    assert _count(plan, 0) == 0
    assert _count(plan, 1) == 0


def test_dispatch_admission_uses_server_deduplicated_safe_bound() -> None:
    """TopK occurrences to one remote server are one DeepEP wire token."""

    _require_sm90()
    tokens = 64
    routes = torch.empty(
        (WORLD, tokens, TOPK), dtype=torch.int64, device="cuda"
    )
    lanes = torch.arange(TOPK, dtype=torch.int64, device="cuda").view(1, TOPK)
    routes[:8] = 128 + lanes
    routes[8:] = lanes
    plan = _plan(routes.contiguous(), 0)
    torch.cuda.synchronize()

    wire_bytes = (7168 + (7168 // 128) * 4 + 8 + TOPK * 8 + 15) // 16 * 16
    ceiling = tokens * wire_bytes
    # With two servers, every source and same-lane destination relay carries at
    # most one de-duplicated payload per token, not TopK payloads.
    assert plan["dispatch_tx_bytes"].cpu().tolist() == [ceiling] * WORLD
    assert plan["dispatch_rx_bytes"].cpu().tolist() == [ceiling] * WORLD
    torch.testing.assert_close(
        plan["endpoint_total_cap_bytes"],
        torch.full_like(plan["endpoint_total_cap_bytes"], ceiling),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("world", [16, 32, 64, 128])
def test_runtime_topologies_converge_without_slot_or_conservation_error(
    world: int,
) -> None:
    _require_sm90()
    routes = _server_imbalanced_routes(tokens=64, world=world)
    plan = _plan(routes, 64 * 1024 * 1024)
    counts = plan["plan_counts"].cpu().tolist()
    assert counts[3] > 0
    assert counts[2] == 0
    assert counts[6] == 0
    assert counts[7] == 0
    assert counts[8] == 1
    assert int(plan["alloc"].sum()) == world * 64 * TOPK


def test_dsv3_4096_tokens_per_rank_planner_contract() -> None:
    _require_sm90()
    routes = _server_imbalanced_routes(tokens=4096)
    plan = _plan(routes, 64 * 1024 * 1024)
    counts = plan["plan_counts"].cpu().tolist()
    assert counts[2] == 0
    assert counts[6] == 0
    assert counts[7] == 0
    assert counts[8] == 1
    assert int(plan["alloc"].sum()) == WORLD * 4096 * TOPK
    assert int(plan["slot_count"].sum()) == WORLD * 4096 * TOPK
    assert bool(torch.all(plan["exec_rank"] >= 0))
    assert bool(torch.all(plan["exec_rank"] < WORLD))


def test_four_server_directed_pair_tables_are_independent() -> None:
    _require_sm90()
    world = 32
    servers = world // 8
    plan = _plan(
        _server_imbalanced_routes(tokens=64, world=world),
        64 * 1024 * 1024,
    )
    chunk_count = _count(plan, 1)
    chunks = plan["chunk_table"][:chunk_count].cpu()
    expected_pair = torch.zeros_like(plan["pair_load_bytes"].cpu())
    active_pairs = set()
    for row in chunks.tolist():
        expected_pair[row[4], row[5], row[10]] += row[9]
        active_pairs.add((row[4], row[5]))
    assert len(active_pairs) >= 2
    torch.testing.assert_close(
        plan["pair_load_bytes"].cpu(), expected_pair, rtol=0, atol=0
    )
    for source in range(servers):
        for destination in range(servers):
            rails = expected_pair[source, destination]
            if int(rails.sum()) > 0:
                assert int(rails.max() - rails.min()) <= CHUNK_BYTES


def test_multiserver_admission_revisits_temporarily_blocked_intents() -> None:
    """A direct final intent may improve only after a later hot intent.

    The compute planner's internal path is monotonic, but coalescing it into
    final home-to-destination moves changes replay dependencies. A one-pass
    admission used to defer one intent in this deterministic four-server case
    even with an open network budget. Stable revisit must recover the complete
    compute plan without weakening the padded-load objective.
    """

    _require_sm90()
    world = 32
    tokens = 128
    generator = torch.Generator(device="cuda")
    generator.manual_seed(1034)
    logits = torch.randn(
        EXPERTS, generator=generator, device="cuda"
    ) * 2.4
    keys = torch.rand(
        (world, tokens, EXPERTS), generator=generator, device="cuda"
    ).log() / (-logits.exp().view(1, 1, -1))
    routes = keys.topk(TOPK, dim=2).indices.contiguous()
    budget = torch.full(
        (world,), 64 * 1024 * 1024, dtype=torch.int64, device="cuda"
    )
    plan = deep_ep_cpp.plan_probeep(
        routes,
        budget,
        expert_weight_bytes=1024 * 1024,
        weight_chunk_bytes=1024 * 1024,
    )
    counts = plan["plan_counts"].cpu().tolist()
    assert counts[3] > 1
    assert counts[0] == counts[3]
    assert counts[4] == 0
    assert int(plan["server_padded_load_after"].max()) < int(
        plan["server_padded_load_before"].max()
    )


def test_multiserver_tied_hot_and_cold_servers_do_not_stall() -> None:
    """Equal extrema still require progress toward the balanced solution.

    Servers 0 and 1 are equally hot while servers 2--7 are empty.  Moving one
    expert cannot immediately lower the global max or spread because the other
    tied extrema remain.  The sum-of-squares tie-break must admit that first
    monotonic step; the old two-component objective produced zero intents.
    """

    _require_sm90()
    world = 64
    tokens = 128
    token = torch.arange(
        world * tokens, dtype=torch.int64, device="cuda"
    ).view(world, tokens, 1)
    lane = torch.arange(TOPK, dtype=torch.int64, device="cuda").view(
        1, 1, TOPK
    )
    routes = ((token * TOPK + lane) % 64).contiguous()
    budget = torch.full(
        (world,), 64 * 1024 * 1024, dtype=torch.int64, device="cuda"
    )
    plan = deep_ep_cpp.plan_probeep(
        routes,
        budget,
        expert_weight_bytes=1024 * 1024,
        weight_chunk_bytes=1024 * 1024,
    )
    counts = plan["plan_counts"].cpu().tolist()
    before = plan["server_padded_load_before"].cpu()
    after = plan["server_padded_load_after"].cpu()
    assert before.tolist() == [32768, 32768, 0, 0, 0, 0, 0, 0]
    assert counts[3] > 0
    assert counts[0] > 0
    assert int(after.max()) < int(before.max())
    assert int(after.max() - after.min()) < int(before.max() - before.min())
    assert counts[2] == 0
    assert counts[6] == 0
    assert counts[7] == 0
    assert counts[8] == 1


def test_second_stage_packing_and_route_lowering_are_exact() -> None:
    """Validate packing itself, not only the inter-server aggregate."""
    _require_sm90()
    tokens = 128
    routes = _server_imbalanced_routes(tokens=tokens)
    first = _plan(routes, 64 * 1024 * 1024)
    second = _plan(routes, 64 * 1024 * 1024)
    torch.cuda.synchronize()

    # Stable radix ordering is only an implementation optimization: it must
    # still produce one deterministic physical placement and lowering.
    for name in (
        "alloc",
        "slot_count",
        "slot_begin",
        "slot_expert",
        "exec_rank",
        "exec_slot",
        "route_dst",
    ):
        torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)

    slot_count = first["slot_count"]
    rank_padded = (((slot_count + 7) // 8) * 8).sum(1)
    for server in range(SERVERS):
        server_padded = rank_padded[
            server * 8 : (server + 1) * 8
        ]
        assert int(server_padded.max() - server_padded.min()) <= 8

    exec_rank = first["exec_rank"]
    exec_slot = first["exec_slot"]
    slot_expert = first["slot_expert"]
    routed_expert = slot_expert[exec_rank, exec_slot]
    torch.testing.assert_close(
        routed_expert, routes.to(torch.int32), rtol=0, atol=0
    )

    local_row = first["route_dst"] % int(first["nvs"])
    route_begin = first["slot_begin"][exec_rank, exec_slot]
    route_count = slot_count[exec_rank, exec_slot]
    assert bool(torch.all(local_row >= route_begin))
    assert bool(torch.all(local_row < route_begin + route_count))
    assert int(torch.unique(first["route_dst"]).numel()) == routes.numel()

    expected_layout = torch.zeros_like(first["is_token_in_rank"])
    expected_layout.scatter_(2, exec_rank, True)
    torch.testing.assert_close(
        first["is_token_in_rank"], expected_layout, rtol=0, atol=0
    )
    torch.testing.assert_close(
        first["num_tokens_per_rank"],
        expected_layout.sum(1, dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    expected_server = expected_layout.view(
        WORLD, tokens, SERVERS, 8
    ).any(3).sum(1, dtype=torch.int32)
    torch.testing.assert_close(
        first["num_tokens_per_rdma_rank"],
        expected_server,
        rtol=0,
        atol=0,
    )

    slots = slot_count.size(1)
    expected_exec_counts = torch.stack(
        [
            torch.bincount(
                (exec_rank[source] * slots + exec_slot[source]).flatten(),
                minlength=WORLD * slots,
            ).to(torch.int32)
            for source in range(WORLD)
        ]
    )
    torch.testing.assert_close(
        first["num_tokens_per_exec_expert"],
        expected_exec_counts,
        rtol=0,
        atol=0,
    )

    # init+intent, admission, packing and finalization are all visible so the
    # benchmark cannot accidentally omit second-stage packing cost.
    assert bool(torch.all(first["plan_counts"][9:13] > 0))


def test_random_unique_topk_fuzz_preserves_planner_invariants() -> None:
    """Catch placement/lowering failures hidden by one synthetic skew."""
    _require_sm90()
    tokens = 64
    budget = torch.full(
        (WORLD,), 64 * 1024 * 1024, dtype=torch.int64, device="cuda"
    )
    for seed in range(32):
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed)
        routes = torch.randn(
            (WORLD, tokens, EXPERTS),
            generator=generator,
            device="cuda",
        ).topk(TOPK, dim=2).indices.contiguous()
        plan = deep_ep_cpp.plan_probeep(routes, budget)
        counts = plan["plan_counts"].cpu().tolist()
        assert counts[2] == 0
        assert counts[6] == 0
        assert counts[7] == 0
        assert counts[8] == 1
        assert int(plan["alloc"].sum()) == WORLD * tokens * TOPK
        assert int(plan["slot_count"].sum()) == WORLD * tokens * TOPK
        assert int(torch.unique(plan["route_dst"]).numel()) == routes.numel()
        torch.testing.assert_close(
            plan["slot_expert"][plan["exec_rank"], plan["exec_slot"]],
            routes.to(torch.int32),
            rtol=0,
            atol=0,
        )
