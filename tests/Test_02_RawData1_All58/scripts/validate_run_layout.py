#!/usr/bin/env python3
"""Validate Test 02 raw evidence and the algorithm-first layer report."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import defaultdict
from pathlib import Path


REPORT_NAME = "raw_data1_layers_00_57_by_algorithm_rounds_11_20_mean"
REPORT_SCHEMA = "probeep.raw_data1.algorithm_layers.v1"
METHOD_SLUGS = {
    "nccl",
    "deepep",
    "deepep_moonep",
    "ultraep_hybridep",
    "probeep",
}
RAW_FILES = (
    "manifest.json",
    "benchmark_status.jsonl",
    "correctness.jsonl",
    "iterations.csv",
    "rank_samples.csv",
    "expert_samples.csv",
    "rank_expert_samples.csv",
    "rdma_path_load.csv",
    "probeep_plan_summary.jsonl",
)
OBSOLETE_ARTIFACTS = (
    "result.json",
    "配置.json",
    "测试报告.md",
    "visualization_bundle.zip",
    "figures",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--kind", required=True, choices=("single", "test"))
    return parser.parse_args()


def require(root: Path, relatives: tuple[str, ...]) -> None:
    missing = [name for name in relatives if not (root / name).is_file()]
    if missing:
        raise ValueError(f"{root}: missing {', '.join(missing)}")


def reject_obsolete(root: Path) -> None:
    found = [name for name in OBSOLETE_ARTIFACTS if (root / name).exists()]
    if found:
        raise ValueError(f"{root}: obsolete aggregate artifacts remain: {', '.join(found)}")


def json_lines(path: Path) -> list[dict[str, object]]:
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        records.append(record)
    return records


def csv_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def csv_fields(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as stream:
        return set(csv.DictReader(stream).fieldnames or ())


def validate_raw(
    benchmark_dir: Path, *, require_fresh_visualization_raw: bool = False
) -> tuple[int, int]:
    raw = benchmark_dir / "raw"
    require(raw, RAW_FILES)
    manifest = json.loads((raw / "manifest.json").read_text(encoding="utf-8"))
    cases = manifest.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError("raw manifest has no cases")
    case_count = len(cases)
    config = manifest.get("config", {})
    world_size = int(config.get("world_size", 0))
    if case_count != 290 or world_size != 16:
        raise ValueError(
            f"Test 02 requires 290 cases and 16 ranks, got {case_count} and {world_size}"
        )

    statuses = json_lines(raw / "benchmark_status.jsonl")
    correctness = json_lines(raw / "correctness.jsonl")
    if len(statuses) != case_count or len(correctness) != case_count:
        raise ValueError(
            "status/correctness count does not match manifest: "
            f"status={len(statuses)}, correctness={len(correctness)}, cases={case_count}"
        )
    if any(row.get("status") != "PASS" for row in statuses):
        raise ValueError("benchmark contains a non-PASS case")
    if any(row.get("passed") is not True for row in correctness):
        raise ValueError("correctness contains a failed case")

    expected_csv_counts = {
        "iterations.csv": case_count * 10,
        "rank_samples.csv": case_count * 10 * world_size,
        "expert_samples.csv": case_count * 256,
        "rank_expert_samples.csv": case_count * world_size * 256,
    }
    for name, expected in expected_csv_counts.items():
        actual = csv_count(raw / name)
        if actual != expected:
            raise ValueError(f"{name}: rows={actual}, expected={expected}")
    microbatch_files = (
        raw / "microbatch_rank_samples.csv",
        raw / "microbatch_timeline.csv",
    )
    if require_fresh_visualization_raw:
        require(raw, tuple(path.name for path in microbatch_files))
    if any(path.exists() for path in microbatch_files):
        require(raw, tuple(path.name for path in microbatch_files))
        expected_microbatch_rows = case_count * 10 * world_size * 2
        microbatch_rows = read_csv(microbatch_files[0])
        actual_microbatch_rows = len(microbatch_rows)
        if actual_microbatch_rows != expected_microbatch_rows:
            raise ValueError(
                "microbatch_rank_samples.csv: "
                f"rows={actual_microbatch_rows}, expected={expected_microbatch_rows}"
            )
        rank_rows = read_csv(raw / "rank_samples.csv")
        rank_index = {
            tuple(
                row[field]
                for field in (
                    "system", "balance", "workload", "repeat", "iteration",
                    "global_rank",
                )
            ): row
            for row in rank_rows
        }
        microbatch_groups: dict[tuple[str, ...], dict[int, dict[str, str]]] = defaultdict(dict)
        for row in microbatch_rows:
            key = tuple(
                row[field]
                for field in (
                    "system", "balance", "workload", "repeat", "iteration",
                    "global_rank",
                )
            )
            microbatch = int(row["microbatch"])
            if microbatch not in (0, 1) or microbatch in microbatch_groups[key]:
                raise ValueError("microbatch rank log has an invalid/duplicate MB identity")
            if int(row["padded_rows"]) < int(row["exec_load"]):
                raise ValueError("microbatch padded rows are smaller than raw execution rows")
            microbatch_groups[key][microbatch] = row
        if set(microbatch_groups) != set(rank_index):
            raise ValueError("microbatch rank log and combined rank log cover different cases")
        for key, by_microbatch in microbatch_groups.items():
            if set(by_microbatch) != {0, 1}:
                raise ValueError(f"microbatch rank log is incomplete for {key}")
            combined = rank_index[key]
            if sum(int(row["home_load"]) for row in by_microbatch.values()) != int(combined["home_load"]):
                raise ValueError("MB0+MB1 home rows do not equal the combined rank row")
            if sum(int(row["exec_load"]) for row in by_microbatch.values()) != int(combined["exec_load"]):
                raise ValueError("MB0+MB1 execution rows do not equal the combined rank row")
            if key[0] in {"nccl", "deepep"} and any(
                int(row["home_load"]) != int(row["exec_load"])
                for row in by_microbatch.values()
            ):
                raise ValueError(f"{key[0]} baseline changed expert placement")
        microbatch_totals: dict[tuple[str, ...], list[int]] = defaultdict(lambda: [0, 0])
        for row in microbatch_rows:
            key = tuple(
                row[field]
                for field in (
                    "system", "balance", "workload", "repeat", "iteration",
                    "microbatch",
                )
            )
            microbatch_totals[key][0] += int(row["home_load"])
            microbatch_totals[key][1] += int(row["exec_load"])
        if any(values != [262_144, 262_144] for values in microbatch_totals.values()):
            raise ValueError("a microbatch does not conserve 262,144 expert-route rows")
        # Every method records A0, A1, W+D0, W+D1, E0, C0, E1 and C1.
        # ProbeEP additionally records the post-combine observation producer;
        # the controller itself is fused into the next balanced_dispatch and
        # must not be represented as a fabricated standalone interval.
        expected_timeline_rows = 58 * 10 * world_size * (4 * 8 + 9)
        actual_timeline_rows = csv_count(microbatch_files[1])
        if actual_timeline_rows != expected_timeline_rows:
            raise ValueError(
                "microbatch_timeline.csv: "
                f"rows={actual_timeline_rows}, expected={expected_timeline_rows}"
            )
        timeline_rows = read_csv(microbatch_files[1])
        timeline_keys = set()
        timeline_stage_sets: dict[tuple[str, ...], set[tuple[str, str, str]]] = {}
        for row in timeline_rows:
            start = float(row["start_ms"])
            end = float(row["end_ms"])
            duration = float(row["duration_ms"])
            if start < 0.0 or end < start:
                raise ValueError(
                    "microbatch_timeline.csv contains a non-monotonic interval"
                )
            if abs((end - start) - duration) > 1e-5:
                raise ValueError(
                    "microbatch_timeline.csv duration does not equal end-start"
                )
            key = tuple(
                row[field]
                for field in (
                    "system",
                    "balance",
                    "workload",
                    "repeat",
                    "iteration",
                    "global_rank",
                    "microbatch",
                    "logical_stream",
                    "stage",
                )
            )
            if key in timeline_keys:
                raise ValueError(
                    "microbatch_timeline.csv contains a duplicate stage interval"
                )
            timeline_keys.add(key)
            invocation = tuple(
                row[field]
                for field in (
                    "system",
                    "balance",
                    "workload",
                    "repeat",
                    "iteration",
                    "global_rank",
                )
            )
            timeline_stage_sets.setdefault(invocation, set()).add(
                (row["stage"], row["microbatch"], row["logical_stream"])
            )
        if require_fresh_visualization_raw:
            baseline_stages = {
                ("attention_or_gate", "0", "compute"),
                ("attention_or_gate", "1", "compute"),
                ("weight_dispatch", "0", "communication"),
                ("weight_dispatch", "1", "communication"),
                ("expert_mlp", "0", "compute"),
                ("combine", "0", "communication"),
                ("expert_mlp", "1", "compute"),
                ("combine", "1", "communication"),
            }
            probe_stages = baseline_stages | {
                ("observation_prepare", "-1", "compute"),
            }
            for invocation, stages in timeline_stage_sets.items():
                expected = probe_stages if invocation[0] == "probeep" else baseline_stages
                if stages != expected:
                    raise ValueError(
                        f"microbatch timeline stages for {invocation} are incomplete"
                    )
            intervals_by_invocation: dict[
                tuple[str, ...], dict[tuple[str, int, str], tuple[float, float]]
            ] = defaultdict(dict)
            for row in timeline_rows:
                invocation = tuple(
                    row[field]
                    for field in (
                        "system", "balance", "workload", "repeat", "iteration",
                        "global_rank",
                    )
                )
                intervals_by_invocation[invocation][
                    (row["stage"], int(row["microbatch"]), row["logical_stream"])
                ] = (float(row["start_ms"]), float(row["end_ms"]))

            def before(intervals, predecessor, successor) -> bool:
                return intervals[predecessor][1] <= intervals[successor][0] + 1e-4

            a0 = ("attention_or_gate", 0, "compute")
            a1 = ("attention_or_gate", 1, "compute")
            d0 = ("weight_dispatch", 0, "communication")
            d1 = ("weight_dispatch", 1, "communication")
            e0 = ("expert_mlp", 0, "compute")
            e1 = ("expert_mlp", 1, "compute")
            c0 = ("combine", 0, "communication")
            c1 = ("combine", 1, "communication")
            base_edges = (
                (a0, a1), (a0, d0), (a1, e0), (d0, e0),
                (a1, d1), (d0, d1), (e0, e1), (d1, e1),
                (e0, c0), (d1, c0), (e1, c1), (c0, c1),
            )
            for invocation, intervals in intervals_by_invocation.items():
                edges = list(base_edges)
                if invocation[0] in {"deepep_moonep", "ultraep"}:
                    edges.append((e0, d1))
                if invocation[0] == "probeep":
                    edges.append((c1, ("observation_prepare", -1, "compute")))
                for predecessor, successor in edges:
                    if not before(intervals, predecessor, successor):
                        raise ValueError(
                            "microbatch timeline violates the implemented CUDA-event DAG: "
                            f"{invocation}: {predecessor} -> {successor}"
                        )
    if require_fresh_visualization_raw:
        require(
            raw,
            (
                "probeep_observation_samples.csv",
                "probeep_weight_chunks.csv",
            ),
        )
        observation_rows = read_csv(raw / "probeep_observation_samples.csv")
        # Layer 00 has no previous MoE layer and uses the runtime's explicit
        # bootstrap budget.  Every later layer consumes two same-round device
        # observations (A/M) produced by exactly Layer L-1.
        expected_observations = 57 * 10 * 2 * world_size
        if len(observation_rows) != expected_observations:
            raise ValueError(
                "probeep_observation_samples.csv: "
                f"rows={len(observation_rows)}, expected={expected_observations}"
            )
        kinds = {row["compute_name"] for row in observation_rows}
        if kinds != {"attention", "moe"}:
            raise ValueError("ProbeEP observation log must contain independent A/M rows")
        for row in observation_rows:
            if int(row["compute_ns"]) <= 0 or int(row["network_ns"]) <= 0:
                raise ValueError(
                    "ProbeEP A/M observation must contain positive independent "
                    "compute and release-to-network-done windows"
                )
            if row["producer_phase"] != "measured":
                raise ValueError("measured ProbeEP dispatch consumed non-measured feedback")
            consumer = int(row["consumer_iteration"])
            producer = int(row["producer_iteration"])
            producer_layer = int(row["producer_layer_id"])
            consumer_layer = int(row["consumer_layer_id"])
            compute_kind = int(row["compute_kind"])
            dispatch_microbatch = int(row["dispatch_microbatch"])
            overlap_microbatch = int(row["overlap_microbatch"])
            if producer != consumer:
                raise ValueError("ProbeEP feedback did not preserve the measured round")
            if producer_layer + 1 != consumer_layer or consumer_layer <= 0:
                raise ValueError("ProbeEP feedback did not come from the previous layer")
            if row["workload"] != f"raw_data1_layer_{consumer_layer:02d}":
                raise ValueError("ProbeEP observation consumer layer/workload mismatch")
            if int(row["producer_repeat"]) != int(row["consumer_repeat"]):
                raise ValueError("ProbeEP feedback crossed benchmark repeats")
            if (
                dispatch_microbatch != compute_kind
                or overlap_microbatch != 1 - compute_kind
            ):
                raise ValueError("ProbeEP A/M chain has incorrect microbatch identities")

        plans = json_lines(raw / "probeep_plan_summary.jsonl")
        if len(plans) != 58 * 10 * 2:
            raise ValueError(
                f"probeep_plan_summary.jsonl: rows={len(plans)}, expected=1160"
            )
        admitted_by_kind = {0: 0, 1: 0}
        strict_improvements_by_kind = {0: 0, 1: 0}
        versions_by_invocation: dict[tuple[str, int, int], set[int]] = defaultdict(set)
        for plan in plans:
            counts = plan.get("plan_counts", [])
            chunks = plan.get("chunk_table", [])
            invariants = plan.get("invariants", {})
            cycles = plan.get("planner_phase_cycles", {})
            if len(counts) != 13 or len(chunks) != int(counts[1]):
                raise ValueError("ProbeEP plan counters/chunk table are inconsistent")
            if (
                int(invariants.get("placement_or_capacity_error", 1)) != 0
                or int(invariants.get("negative_count", 1)) != 0
                or int(invariants.get("prefix_mismatch_count", 1)) != 0
                or int(invariants.get("planning_done", 0)) != 1
            ):
                raise ValueError("ProbeEP production planner invariant failed")
            if set(cycles) != {
                "compute_intent",
                "network_admission",
                "server_local_packing",
                "finalization",
            }:
                raise ValueError("ProbeEP planner phase-cycle telemetry is incomplete")
            kind = int(plan.get("dispatch_compute_kind", -1))
            if kind not in admitted_by_kind:
                raise ValueError("ProbeEP plan has an invalid A/M compute kind")
            consumer_layer = int(str(plan["workload"]).rsplit("_", 1)[1])
            feedback_source = str(plan.get("feedback_source", ""))
            feedback_layer = int(plan.get("feedback_producer_layer_id", -2))
            feedback_iteration = int(
                plan.get("feedback_producer_iteration", -2)
            )
            if consumer_layer == 0:
                if feedback_source != "bootstrap" or feedback_layer != -1:
                    raise ValueError("ProbeEP Layer 00 must use explicit bootstrap")
            elif (
                feedback_source != "previous_layer_observation"
                or feedback_layer != consumer_layer - 1
                or feedback_iteration != int(plan["iteration"])
            ):
                raise ValueError("ProbeEP plan did not consume same-round previous-layer feedback")
            if (
                int(plan.get("feedback_dispatch_microbatch", -1)) != kind
                or int(plan.get("feedback_overlap_microbatch", -1)) != 1 - kind
            ):
                raise ValueError("ProbeEP plan crossed Attention/MoE microbatch chains")
            if plan.get("weight_cache_mode") != "cold":
                raise ValueError("formal ProbeEP paper run must use cold weight cache")
            version = int(plan.get("expert_weight_version", -1))
            if version <= 0:
                raise ValueError("cold ProbeEP plan has no positive weight version")
            versions_by_invocation[
                (str(plan["workload"]), int(plan["repeat"]), int(plan["iteration"]))
            ].add(version)
            exact_before = [int(value) for value in plan.get("server_load_before", [])]
            exact_after = [int(value) for value in plan.get("server_load_after", [])]
            before = [int(value) for value in plan.get("server_padded_load_before", [])]
            after = [int(value) for value in plan.get("server_padded_load_after", [])]
            if (
                len(exact_before) != 2
                or len(exact_after) != 2
                or sum(exact_before) != sum(exact_after)
                or len(before) != 2
                or len(after) != 2
            ):
                raise ValueError("ProbeEP server-load vectors are invalid")
            if int(counts[0]) > 0:
                admitted_by_kind[kind] += 1
                if max(after) > max(before) or max(after) - min(after) > max(before) - min(before):
                    raise ValueError("ProbeEP admitted placement worsened server balance")
                if max(after) < max(before) or max(after) - min(after) < max(before) - min(before):
                    strict_improvements_by_kind[kind] += 1
            if int(counts[1]) > 0 and int(plan.get("weight_transfer_required", 0)) != 1:
                raise ValueError("cold ProbeEP plan suppressed a required Weight transfer")
        if any(len(versions) != 1 for versions in versions_by_invocation.values()):
            raise ValueError("A/M plans from one iteration must use one common layer version")
        if any(value == 0 for value in admitted_by_kind.values()):
            raise ValueError("ProbeEP did not admit experts for both Attention and MoE plans")
        if any(value == 0 for value in strict_improvements_by_kind.values()):
            raise ValueError("ProbeEP did not improve cross-server balance for both A/M plans")

        rail_rows = read_csv(raw / "rdma_path_load.csv")
        expected_rail_rows = case_count * 10 * 2 * world_size
        if len(rail_rows) != expected_rail_rows:
            raise ValueError(
                "rdma_path_load.csv: "
                f"rows={len(rail_rows)}, expected={expected_rail_rows}"
            )
        systems = {(row["system"], row["balance"]) for row in rail_rows}
        if len(systems) != 5:
            raise ValueError("rdma_path_load.csv does not cover all five algorithms")
        rail_keys = set()
        rail_paths_by_invocation: dict[tuple[str, ...], set[tuple[int, int]]] = defaultdict(set)
        for row in rail_rows:
            if row["schema_version"] != "4":
                raise ValueError("fresh rail telemetry must use raw schema v4")
            dispatch_bytes = int(row["dispatch_bytes"])
            weight_bytes = int(row["weight_bytes"])
            compute_kind = int(row["dispatch_compute_kind"])
            expected_compute_name = "attention" if compute_kind == 0 else "moe"
            if (
                compute_kind not in (0, 1)
                or row["dispatch_compute_name"] != expected_compute_name
                or int(row["microbatch"]) != compute_kind
            ):
                raise ValueError("rail row crossed Attention/MoE microbatch identities")
            if int(row["tx_bytes"]) != dispatch_bytes + weight_bytes:
                raise ValueError("rail TX is not Dispatch + Weight")
            if int(row["rx_bytes"]) != int(row["tx_bytes"]):
                raise ValueError("rail TX/RX byte conservation failed")
            if not row["traffic_source"] or not row["dispatch_unit_name"]:
                raise ValueError("rail row has no runtime provenance")
            if dispatch_bytes != int(row["dispatch_units"]) * int(row["dispatch_bytes_per_unit"]):
                raise ValueError("rail Dispatch bytes disagree with runtime units")
            if int(row["source_rank"]) % 8 != int(row["destination_rank"]) % 8:
                raise ValueError("rail endpoints do not have matching local GPU indices")
            rail = int(row["source_rank"]) % 8
            if (
                int(row["physical_nic"]) != rail // 2
                or int(row["subrail"]) != rail % 2
                or abs(float(row["rail_bandwidth_gbps"]) - 200.0) > 1e-6
                or abs(float(row["physical_nic_bandwidth_gbps"]) - 400.0) > 1e-6
            ):
                raise ValueError("rail row violates the 4x400G -> 8x200G topology")
            if row["system"] == "probeep":
                if row["weight_cache_mode"] != "cold" or int(row["expert_weight_version"]) <= 0:
                    raise ValueError("ProbeEP rail row lost cold-cache/version identity")
            elif row["weight_cache_mode"] != "not_applicable" or int(row["expert_weight_version"]) != -1:
                raise ValueError("baseline rail row carries ProbeEP cache metadata")
            key = tuple(
                row[field]
                for field in (
                    "system",
                    "balance",
                    "workload",
                    "repeat",
                    "iteration",
                    "microbatch",
                    "source_rank",
                    "destination_rank",
                )
            )
            if key in rail_keys:
                raise ValueError("rdma_path_load.csv contains a duplicate rail row")
            rail_keys.add(key)
            invocation = tuple(
                row[field]
                for field in (
                    "system", "balance", "workload", "repeat", "iteration",
                    "microbatch",
                )
            )
            source_rank = int(row["source_rank"])
            destination_rank = int(row["destination_rank"])
            expected_path_id = (
                ((source_rank // 8) * 2 + destination_rank // 8) * 8
                + source_rank % 8
            )
            if int(row["path_id"]) != expected_path_id:
                raise ValueError("rail row path_id does not encode its directed path")
            rail_paths_by_invocation[invocation].add((source_rank, destination_rank))
        expected_directed_paths = {
            *((rank, rank + 8) for rank in range(8)),
            *((rank + 8, rank) for rank in range(8)),
        }
        if any(paths != expected_directed_paths for paths in rail_paths_by_invocation.values()):
            raise ValueError("rail telemetry must retain all 16 directed paths per microbatch")
        expected_invocations = case_count * 10 * 2
        if len(rail_paths_by_invocation) != expected_invocations:
            raise ValueError(
                "rail telemetry invocation count is incomplete: "
                f"{len(rail_paths_by_invocation)} != {expected_invocations}"
            )

        chunk_rows = read_csv(raw / "probeep_weight_chunks.csv")
        chunk_sums: dict[tuple[str, ...], int] = defaultdict(int)
        chunk_counts: dict[tuple[str, int, int, int], int] = defaultdict(int)
        chunk_keys = set()
        for row in chunk_rows:
            if int(row["chunk_bytes"]) <= 0:
                raise ValueError("ProbeEP weight chunk must have positive bytes")
            compute_kind = int(row["dispatch_compute_kind"])
            if (
                compute_kind not in (0, 1)
                or row["dispatch_compute_name"] != ("attention" if compute_kind == 0 else "moe")
            ):
                raise ValueError("ProbeEP Weight chunk crossed A/M identities")
            if int(row["source_rank"]) % 8 != int(row["rail"]):
                raise ValueError("ProbeEP weight chunk rail/source rank mismatch")
            if int(row["destination_rank"]) % 8 != int(row["rail"]):
                raise ValueError("ProbeEP weight chunk rail/destination rank mismatch")
            rail = int(row["rail"])
            if (
                int(row["physical_nic"]) != rail // 2
                or int(row["subrail"]) != rail % 2
                or abs(float(row["rail_bandwidth_gbps"]) - 200.0) > 1e-6
                or abs(float(row["physical_nic_bandwidth_gbps"]) - 400.0) > 1e-6
                or row["weight_cache_mode"] != "cold"
                or int(row["expert_weight_version"]) <= 0
            ):
                raise ValueError("ProbeEP Weight chunk has invalid rail/cache metadata")
            if (
                int(row["source_rank"]) // 8 != int(row["source_server"])
                or int(row["destination_rank"]) // 8 != int(row["destination_server"])
            ):
                raise ValueError("ProbeEP Weight chunk server/rank identity mismatch")
            chunk_key = tuple(
                row[field]
                for field in (
                    "workload", "repeat", "iteration", "dispatch_compute_kind",
                    "chunk_ordinal",
                )
            )
            if chunk_key in chunk_keys:
                raise ValueError("ProbeEP Weight chunk ordinal is duplicated")
            chunk_keys.add(chunk_key)
            chunk_counts[(
                row["workload"], int(row["repeat"]), int(row["iteration"]),
                compute_kind,
            )] += 1
            if int(row["transfer_required"]) != 0:
                key = tuple(
                    row[field]
                    for field in (
                        "workload",
                        "repeat",
                        "iteration",
                        "dispatch_compute_kind",
                        "source_rank",
                        "destination_rank",
                    )
                )
                chunk_sums[key] += int(row["chunk_bytes"])
        rail_weight_sums: dict[tuple[str, ...], int] = defaultdict(int)
        for row in rail_rows:
            if row["system"] != "probeep":
                continue
            key = tuple(
                row[field]
                for field in (
                    "workload",
                    "repeat",
                    "iteration",
                    "dispatch_compute_kind",
                    "source_rank",
                    "destination_rank",
                )
            )
            rail_weight_sums[key] += int(row["weight_bytes"])
        if chunk_sums != {
            key: value for key, value in rail_weight_sums.items() if value != 0
        }:
            raise ValueError("ProbeEP chunk bytes do not conserve rail Weight bytes")

        probe_transport: dict[tuple[str, int, int, int], dict[str, list[int]]] = {}
        for row in rail_rows:
            if row["system"] != "probeep":
                continue
            key = (
                row["workload"], int(row["repeat"]), int(row["iteration"]),
                int(row["dispatch_compute_kind"]),
            )
            vectors = probe_transport.setdefault(key, {
                "dispatch_tx": [0] * world_size,
                "dispatch_rx": [0] * world_size,
                "weight_tx": [0] * world_size,
                "weight_rx": [0] * world_size,
            })
            source_rank = int(row["source_rank"])
            destination_rank = int(row["destination_rank"])
            vectors["dispatch_tx"][source_rank] += int(row["dispatch_bytes"])
            vectors["dispatch_rx"][destination_rank] += int(row["dispatch_bytes"])
            vectors["weight_tx"][source_rank] += int(row["weight_bytes"])
            vectors["weight_rx"][destination_rank] += int(row["weight_bytes"])

        plan_keys = set()
        for plan in plans:
            kind = int(plan["dispatch_compute_kind"])
            key = (
                str(plan["workload"]), int(plan["repeat"]),
                int(plan["iteration"]), kind,
            )
            if key in plan_keys:
                raise ValueError("ProbeEP plan has a duplicate A/M invocation")
            plan_keys.add(key)
            if chunk_counts.get(key, 0) != int(plan["plan_counts"][1]):
                raise ValueError("ProbeEP plan chunk count disagrees with chunk CSV")
            transport = probe_transport.get(key)
            if transport is None:
                raise ValueError("ProbeEP plan has no matching rail telemetry")
            transfer_required = int(plan["weight_transfer_required"])
            expected_tx = [int(value) * transfer_required for value in plan["assigned_tx_bytes"]]
            expected_rx = [int(value) * transfer_required for value in plan["assigned_rx_bytes"]]
            if transport["weight_tx"] != expected_tx:
                raise ValueError("ProbeEP plan TX admission disagrees with rail Weight bytes")
            if transport["weight_rx"] != expected_rx:
                raise ValueError("ProbeEP plan RX admission disagrees with rail Weight bytes")
        expected_plan_keys = {
            (str(case["workload"]), int(case.get("repeat", 0)), iteration, kind)
            for case in cases if case.get("variant") == "probeep"
            for iteration in range(10) for kind in (0, 1)
        }
        if plan_keys != expected_plan_keys:
            raise ValueError("ProbeEP plan A/M invocation set is incomplete")

        observation_keys = set()
        for row in observation_rows:
            consumer_key = (
                row["workload"], int(row["consumer_repeat"]),
                int(row["consumer_iteration"]), int(row["compute_kind"]),
                int(row["global_rank"]),
            )
            if consumer_key in observation_keys:
                raise ValueError("ProbeEP observation has a duplicate rank sample")
            observation_keys.add(consumer_key)
            producer_key = (
                f"raw_data1_layer_{int(row['producer_layer_id']):02d}",
                int(row["producer_repeat"]), int(row["producer_iteration"]),
                int(row["compute_kind"]),
            )
            transport = probe_transport.get(producer_key)
            if transport is None:
                raise ValueError("ProbeEP observation has no producer rail telemetry")
            rank = int(row["global_rank"])
            expected = (
                transport["dispatch_tx"][rank], transport["dispatch_rx"][rank],
                transport["weight_tx"][rank], transport["weight_rx"][rank],
            )
            actual = (
                int(row["dispatch_tx_bytes"]), int(row["dispatch_rx_bytes"]),
                int(row["migration_tx_bytes"]), int(row["migration_rx_bytes"]),
            )
            if actual != expected:
                raise ValueError(
                    "ProbeEP observation bytes disagree with the producer Layer rail log"
                )
    return case_count, world_size


def validate_zip(report_dir: Path, zip_path: Path) -> None:
    expected = sorted(
        str(path.relative_to(report_dir))
        for path in report_dir.rglob("*")
        if path.is_file()
    )
    with zipfile.ZipFile(zip_path) as archive:
        actual = sorted(name for name in archive.namelist() if not name.endswith("/"))
        if actual != expected:
            raise ValueError("report ZIP members do not exactly match the report directory")
        broken = archive.testzip()
        if broken is not None:
            raise ValueError(f"corrupt ZIP member: {broken}")


def validate_report(report_dir: Path, *, require_complete_raw: bool = False) -> None:
    require(report_dir, ("algorithm_comparison.html", "README.md", "manifest.json"))
    payload = json.loads((report_dir / "manifest.json").read_text(encoding="utf-8"))
    if payload.get("schema") != REPORT_SCHEMA:
        raise ValueError(f"unexpected report schema: {payload.get('schema')}")
    if payload.get("layers") != list(range(58)):
        raise ValueError("Test 02 report must contain Layer 00..57 separately")
    if payload.get("physical_rounds") != list(range(11, 21)):
        raise ValueError("report must use physical Round 11..20")
    if payload.get("warmup_rounds") != list(range(1, 11)):
        raise ValueError("report must exclude physical Round 1..10 warmup")
    if int(payload.get("num_ranks", 0)) != 16:
        raise ValueError("Test 02 report must contain 16 ranks")
    telemetry_contract = payload.get("telemetry_contract", {})
    if require_complete_raw and telemetry_contract.get("fresh_visualization_raw") is not True:
        raise ValueError("formal report is not marked as complete raw schema v4 evidence")

    methods = payload.get("methods", [])
    if not isinstance(methods, list) or {item.get("slug") for item in methods} != METHOD_SLUGS:
        raise ValueError("report must contain exactly the five Test 02 algorithms")
    for method in methods:
        slug = str(method["slug"])
        expected_layer_pages = [
            f"algorithms/{slug}/layers/layer_{layer:02d}.html"
            for layer in range(58)
        ]
        if method.get("layer_pages") != expected_layer_pages:
            raise ValueError(f"{slug}: manifest must list 58 independent layer pages")
        method_dir = report_dir / "algorithms" / slug
        require(method_dir, ("algorithm_dashboard.html",))
        dashboard = (method_dir / "algorithm_dashboard.html").read_text(
            encoding="utf-8"
        )
        if dashboard.count("layers/layer_") != 58:
            raise ValueError(f"{slug}: algorithm dashboard must link 58 layer pages")
        layer_dir = method_dir / "layers"
        expected_layer_files = {
            f"layer_{layer:02d}.html" for layer in range(58)
        }
        actual_layer_files = {path.name for path in layer_dir.glob("layer_*.html")}
        if actual_layer_files != expected_layer_files:
            raise ValueError(f"{slug}: layer page set is incomplete")
        for layer in range(58):
            layer_page = (layer_dir / f"layer_{layer:02d}.html").read_text(
                encoding="utf-8"
            )
            evidence_marker = (
                "本层证据：完整 raw schema v4"
                if telemetry_contract.get("fresh_visualization_raw") is True
                else "本层证据：历史 raw"
            )
            if evidence_marker not in layer_page:
                raise ValueError(
                    f"{slug}/layer_{layer:02d}.html mislabels its raw evidence"
                )
            layer_ids = [
                candidate
                for candidate in range(58)
                if f'id="layer-{candidate:02d}"' in layer_page
            ]
            if layer_ids != [layer]:
                raise ValueError(
                    f"{slug}/layer_{layer:02d}.html mixes layer sections: {layer_ids}"
                )
            if "双 microbatch / 双 stream 事件依赖 DAG" not in layer_page:
                raise ValueError(
                    f"{slug}/layer_{layer:02d}.html has no dual-microbatch DAG"
                )
            for dag_contract in (
                "同一 stream 提交顺序",
                "跨 stream CUDA-event wait",
                "跨层反馈严格分链",
            ):
                if dag_contract not in layer_page:
                    raise ValueError(
                        f"{slug}/layer_{layer:02d}.html misses DAG contract: "
                        f"{dag_contract}"
                    )
            if slug == "probeep":
                for chain_label in ("A[L−1,r]", "M[L−1,r]"):
                    if chain_label not in layer_page:
                        raise ValueError(
                            f"probeep/layer_{layer:02d}.html merges A/M feedback"
                        )
                if "A/M Obs" in layer_page:
                    raise ValueError(
                        f"probeep/layer_{layer:02d}.html contains merged A/M node"
                    )
            timeline_position = layer_page.find("严格时间戳双 stream 流水线")
            mb1_position = layer_page.find("Microbatch 1（runtime MB0）")
            mb2_position = layer_page.find("Microbatch 2（runtime MB1）")
            if not (0 <= timeline_position < mb1_position < mb2_position):
                raise ValueError(
                    f"{slug}/layer_{layer:02d}.html must order timeline, MB1, MB2"
                )
            if "rail-pair-combined-" in layer_page or "rail-path-combined-" in layer_page:
                raise ValueError(
                    f"{slug}/layer_{layer:02d}.html mixes two microbatches in rail charts"
                )
            if slug == "probeep":
                if layer_page.count("Expert placement / Weight chunks") != 2:
                    raise ValueError(
                        f"probeep/layer_{layer:02d}.html must show one expert "
                        "placement/Weight panel per microbatch"
                    )
                if "柱宽严格等于 Token Dispatch bytes + Expert Weight bytes" not in layer_page:
                    raise ValueError(
                        f"probeep/layer_{layer:02d}.html does not state additive rail bytes"
                    )
            data_dir = layer_dir / "data" / f"layer_{layer:02d}"
            require(
                data_dir,
                (
                    "result.json",
                    "summary.csv",
                    "measured_rounds.csv",
                    "rank_load.csv",
                    "microbatch_rank_load.csv",
                ),
            )
            layer_result = json.loads(
                (data_dir / "result.json").read_text(encoding="utf-8")
            )
            if layer_result.get("schema") != "probeep.raw_data1.algorithm_single_layer.v1":
                raise ValueError(f"{slug} layer {layer:02d}: unexpected result schema")
            if layer_result.get("method", {}).get("slug") != slug:
                raise ValueError(f"{slug} layer {layer:02d}: method identity mismatch")
            if int(layer_result.get("layer", {}).get("layer", -1)) != layer:
                raise ValueError(f"{slug} layer {layer:02d}: result mixes another layer")
            layer_payload = layer_result["layer"]
            if require_complete_raw:
                expected_timeline = 10 * 16 * (9 if slug == "probeep" else 8)
                if len(layer_payload.get("timeline", [])) != expected_timeline:
                    raise ValueError(
                        f"{slug} layer {layer:02d}: incomplete measured timeline"
                    )
                microbatches = layer_payload.get("microbatches", [])
                if len(microbatches) != 2 or not all(
                    item.get("home_available") and item.get("available")
                    for item in microbatches
                ):
                    raise ValueError(
                        f"{slug} layer {layer:02d}: incomplete per-microbatch load"
                    )
            expected_counts = {
                "summary.csv": 1,
                "measured_rounds.csv": 10,
                "rank_load.csv": 16,
                "microbatch_rank_load.csv": 32,
            }
            for name, expected in expected_counts.items():
                actual = csv_count(data_dir / name)
                if actual != expected:
                    raise ValueError(
                        f"{slug} layer {layer:02d}/{name}: rows={actual}, expected={expected}"
                    )
            if slug == "probeep" or require_complete_raw:
                require(
                    data_dir,
                    ("rdma_path_load.csv",),
                )
                if csv_count(data_dir / "rdma_path_load.csv") != 320:
                    raise ValueError(
                        f"{slug} layer {layer:02d}: expected 320 per-round rail rows"
                    )
                required_rail_fields = {
                    "schema_version", "iteration", "physical_round",
                    "dispatch_compute_kind", "dispatch_compute_name", "microbatch",
                    "physical_nic", "subrail", "rail_bandwidth_gbps",
                    "physical_nic_bandwidth_gbps", "dispatch_units",
                    "dispatch_unit_name", "dispatch_bytes_per_unit", "traffic_source",
                    "dispatch_bytes", "weight_bytes", "tx_bytes", "rx_bytes",
                }
                if not required_rail_fields.issubset(csv_fields(data_dir / "rdma_path_load.csv")):
                    raise ValueError(
                        f"{slug} layer {layer:02d}: exported rail CSV lost provenance fields"
                    )
            if slug == "probeep" and require_complete_raw:
                require(
                    data_dir,
                    (
                        "probeep_plan_summary.jsonl",
                        "probeep_weight_chunks.csv",
                    ),
                )
                observation_path = data_dir / "probeep_observation_samples.csv"
                if layer == 0:
                    if observation_path.exists() and csv_count(observation_path) != 0:
                        raise ValueError("probeep Layer 00 must use bootstrap, not fake feedback")
                elif (
                    not observation_path.is_file()
                    or csv_count(observation_path) != 320
                ):
                    raise ValueError(
                        f"probeep layer {layer:02d}: expected 320 previous-layer A/M rank observations"
                    )
                if len(json_lines(data_dir / "probeep_plan_summary.jsonl")) != 20:
                    raise ValueError(
                        f"probeep layer {layer:02d}: expected 20 A/M production plans"
                    )
                required_chunk_fields = {
                    "physical_nic", "subrail", "rail_bandwidth_gbps",
                    "physical_nic_bandwidth_gbps", "weight_cache_mode",
                    "expert_weight_version", "transfer_required",
                }
                if not required_chunk_fields.issubset(
                    csv_fields(data_dir / "probeep_weight_chunks.csv")
                ):
                    raise ValueError(
                        f"probeep layer {layer:02d}: exported chunk CSV lost NIC/cache fields"
                    )
                if "不画伪时间线" in layer_page:
                    raise ValueError(
                        f"probeep layer {layer:02d}: fresh report omitted measured timeline"
                    )

    serialized = json.dumps(payload, ensure_ascii=False)
    forbidden = ("mean_layer_p99", "pooled_p99", "probeep.multinode.load_profile.v1")
    present = [name for name in forbidden if name in serialized]
    if present:
        raise ValueError(f"report contains obsolete aggregate fields: {present}")
    validate_zip(report_dir, report_dir.parent / f"{report_dir.name}.zip")


def validate_runner_status(raw: Path, num_servers: int) -> None:
    for node_rank in range(num_servers):
        schedule = raw / f"formal_schedule_node_{node_rank}.jsonl"
        runner_status = raw / f"runner_status_node_{node_rank}.jsonl"
        require(raw, (schedule.name, runner_status.name))
        if not any(row.get("state") == "PASS" for row in json_lines(runner_status)):
            raise ValueError(f"{runner_status.name}: no PASS runner record")


def validate_nsys(run_dir: Path, num_servers: int) -> None:
    targets = (
        "deepep-smoke",
        "deepep-moonep-smoke",
        "probeep-forward",
        "probeep-expert-io",
        "probeep-backward",
        "ultraep-smoke",
        "formal-pipeline",
    )
    for target in targets:
        root = run_dir / "nsys" / target / "nsys"
        if not root.is_dir():
            raise ValueError(f"missing Nsight Systems target: {target}")
        if len(list(root.glob("*.nsys-rep"))) < num_servers:
            raise ValueError(f"{target}: missing per-node .nsys-rep")
        if len(list(root.glob("*.sqlite"))) < num_servers:
            raise ValueError(f"{target}: missing per-node .sqlite")
        if target == "formal-pipeline":
            overlap = list(root.glob("*-overlap.json"))
            if len(overlap) < num_servers:
                raise ValueError("formal-pipeline: missing per-node overlap JSON")
            for path in overlap:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if int(payload.get("measurement_iterations", 0)) <= 0:
                    raise ValueError(f"{path}: no measured NVTX iteration")
                variants = {
                    row.get("variant") for row in payload.get("variant_summary", [])
                }
                expected = {
                    "nccl",
                    "deepep",
                    "deepep_moonep_on",
                    "ultraep_hybridep",
                    "probeep",
                }
                if variants != expected:
                    raise ValueError(
                        f"{path}: formal pipeline variants={sorted(variants)}, "
                        f"expected={sorted(expected)}"
                    )


def main() -> None:
    args = arguments()
    run_dir = args.run_dir.resolve()
    if args.kind == "single":
        require(run_dir, ("launch.env", "readiness.json"))
        if not (run_dir / "logs").is_dir():
            raise ValueError("single benchmark run must contain logs/")
        reject_obsolete(run_dir)
        _, world_size = validate_raw(run_dir)
        validate_runner_status(run_dir / "raw", world_size // 8)
        validate_report(run_dir / "reprocessed" / REPORT_NAME)
    else:
        for directory in ("workload", "setup", "correctness", "benchmark", "nsys", "artifacts"):
            if not (run_dir / directory).is_dir():
                raise ValueError(f"test run must contain {directory}/")
        reject_obsolete(run_dir / "artifacts")
        _, world_size = validate_raw(
            run_dir / "benchmark", require_fresh_visualization_raw=True
        )
        require(run_dir / "benchmark", ("launch.env", "readiness.json"))
        if not (run_dir / "benchmark/logs").is_dir():
            raise ValueError("benchmark must contain logs/")
        validate_runner_status(run_dir / "benchmark/raw", world_size // 8)
        validate_nsys(run_dir, world_size // 8)
        validate_report(
            run_dir / "artifacts" / REPORT_NAME, require_complete_raw=True
        )
    print(f"PASS {args.kind}: {run_dir}")


if __name__ == "__main__":
    main()
