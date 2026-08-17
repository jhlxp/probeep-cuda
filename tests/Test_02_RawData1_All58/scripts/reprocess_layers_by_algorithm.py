#!/usr/bin/env python3
"""Reprocess RawData1 logs into algorithm-first, layer-by-layer dashboards.

Statistics contract:
- one case is one algorithm on one RawData1 layer;
- physical rounds 1..10 are warmup and are not logged here;
- measured iterations 0..9 are displayed as physical rounds 11..20;
- latency is averaged only inside that layer over the ten measured iterations;
- layers are never averaged together in the visual report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
from collections import defaultdict
from pathlib import Path
import shutil
import statistics
import zipfile

import numpy as np

from raw_data1 import describe, realize_exact_topk


METHODS = (
    ("nccl", "na", "NCCL", "nccl", "原始 expert placement；不做专家均衡。"),
    ("deepep", "na", "DeepEP", "deepep", "官方分层数据面；不改变 expert placement，不做专家均衡。"),
    ("deepep_moonep", "on", "DeepEP-MoonEP", "deepep_moonep", "MoonEP 思路接入 DeepEP 数据面，只做服务器内均衡。"),
    ("ultraep", "hybridep", "UltraEP + HybridEP", "ultraep_hybridep", "UltraEP runtime + 固定官方 HybridEP 数据面。"),
    ("probeep", "server_first", "ProbeEP", "probeep", "Attention/MoE 独立探测，先跨服务器均衡，再做服务器内二次均衡。"),
)
EXPECTED_MEASURED_ITERATIONS = tuple(range(10))
ROUND_OFFSET = 11
SERVER_SIZE = 8
NUM_RANKS = 16
NUM_EXPERTS = 256
GLOBAL_ASSIGNMENTS = 524_288
# Matches probeep::kMaxServers.  The production planner encodes an admitted
# placement as expert_id * kMaxServers + destination_server.
PROBEEP_PLAN_SERVER_STRIDE = 16
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROBE_CHUNK_EXPORT_FIELDS = [
    "schema_version", "run_id", "slurm_job_id", "benchmark_scope",
    "runner_mode", "system", "balance", "direction", "workload",
    "bias_ratio", "seed", "repeat", "iteration", "routing_sha256",
    "dispatch_compute_kind", "dispatch_compute_name", "chunk_ordinal",
    "expert_id", "replica_id", "seed_rank", "expert_chunk_index",
    "source_server", "destination_server", "source_rank", "destination_rank",
    "physical_nic", "subrail", "rail_bandwidth_gbps",
    "physical_nic_bandwidth_gbps", "weight_cache_mode",
    "expert_weight_version",
    "expert_offset_bytes", "chunk_bytes", "rail", "source_path_offset_bytes",
    "destination_path_offset_bytes", "transfer_required",
]
RAIL_EXPORT_FIELDS = [
    "schema_version", "run_id", "slurm_job_id", "benchmark_scope",
    "runner_mode", "system", "balance", "direction", "workload",
    "bias_ratio", "seed", "repeat", "iteration", "physical_round",
    "routing_sha256", "dispatch_compute_kind", "dispatch_compute_name",
    "microbatch", "path_id", "physical_nic", "subrail",
    "rail_bandwidth_gbps", "physical_nic_bandwidth_gbps",
    "weight_cache_mode", "expert_weight_version", "source_rank",
    "destination_rank", "chunk_count", "dispatch_units",
    "dispatch_unit_name", "dispatch_bytes_per_unit", "traffic_source",
    "dispatch_bytes", "weight_bytes", "tx_bytes", "rx_bytes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--layers",
        default="all",
        help="Layer selector: all, a comma list such as 0,7,15, or ranges such as 0-19.",
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def parse_layers(value: str, available: list[int]) -> list[int]:
    if value == "all":
        return available
    selected: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            raw_start, raw_end = item.split("-", 1)
            start, end = int(raw_start), int(raw_end)
            if start > end:
                raise ValueError(f"bad layer range: {item}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(item))
    unknown = sorted(selected - set(available))
    if unknown:
        raise ValueError(f"layers not present in raw log: {unknown}")
    return sorted(selected)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_csv_optional(path: Path) -> list[dict[str, str]]:
    return read_csv(path) if path.is_file() else []


def read_jsonl_optional(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def p50(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def ratio(values: list[float]) -> float:
    total = sum(values)
    return (max(values) * len(values) / total) if total else 0.0


def vector_mean(vectors: list[list[int]]) -> list[float]:
    return [mean([float(vector[i]) for vector in vectors]) for i in range(len(vectors[0]))]


def vectors_stable(vectors: list[list[int]]) -> bool:
    return all(vector == vectors[0] for vector in vectors[1:])


def server_sums(values: list[float]) -> list[float]:
    if len(values) % SERVER_SIZE:
        raise ValueError(f"rank count {len(values)} is not divisible by {SERVER_SIZE}")
    return [sum(values[i : i + SERVER_SIZE]) for i in range(0, len(values), SERVER_SIZE)]


def workload_name(layer: int) -> str:
    return f"raw_data1_layer_{layer:02d}"


def js(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def accept_row(row: dict[str, str]) -> bool:
    return (
        row.get("direction") == "forward"
        and row.get("benchmark_scope") == "full_moe_grouped"
        and row.get("runner_mode") == "dual_microbatch_ht"
    )


def index_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], list[dict[str, str]]]:
    indexed: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if accept_row(row):
            indexed[(row["system"], row["balance"], row["workload"])].append(row)
    return indexed


def available_layers(iteration_rows: list[dict[str, str]]) -> list[int]:
    layers = set()
    for row in iteration_rows:
        workload = row.get("workload", "")
        if accept_row(row) and workload.startswith("raw_data1_layer_"):
            layers.add(int(workload.rsplit("_", 1)[1]))
    return sorted(layers)


def round_record(row: dict[str, str]) -> dict[str, object]:
    iteration = int(row["iteration"])
    return {
        "iteration": iteration,
        "round": iteration + ROUND_OFFSET,
        "e2e_ms": float(row["e2e_max_ms"]),
        "dispatch_ms": float(row["dispatch_max_ms"]),
        "expert_ms": float(row["expert_compute_max_ms"]),
        "combine_ms": float(row["combine_max_ms"]),
        "plan_ms": float(row["plan_max_ms"]),
        "count_exchange_ms": float(row["count_exchange_max_ms"]),
        "layout_ms": float(row["layout_materialize_max_ms"]),
        "weight_ms": float(row["weight_prefetch_max_ms"]),
        "tokens_per_second": float(row["tokens_per_second"]),
    }


def summarize_rounds(rounds: list[dict[str, object]]) -> dict[str, float]:
    values = [float(row["e2e_ms"]) for row in rounds]
    return {
        "e2e_mean_ms": mean(values),
        "e2e_p50_ms": p50(values),
        "e2e_min_ms": min(values),
        "e2e_max_ms": max(values),
        "dispatch_mean_ms": mean([float(row["dispatch_ms"]) for row in rounds]),
        "expert_mean_ms": mean([float(row["expert_ms"]) for row in rounds]),
        "combine_mean_ms": mean([float(row["combine_ms"]) for row in rounds]),
        "plan_mean_ms": mean([float(row["plan_ms"]) for row in rounds]),
        "layout_mean_ms": mean([float(row["layout_ms"]) for row in rounds]),
        "weight_mean_ms": mean([float(row["weight_ms"]) for row in rounds]),
    }


def extract_rank_vectors(
    rows: list[dict[str, str]],
    field: str,
    *,
    expected_total: int = GLOBAL_ASSIGNMENTS,
) -> list[list[int]]:
    by_iteration: dict[int, dict[int, int]] = defaultdict(dict)
    for row in rows:
        by_iteration[int(row["iteration"])][int(row["global_rank"])] = int(row[field])
    if tuple(sorted(by_iteration)) != EXPECTED_MEASURED_ITERATIONS:
        raise ValueError(f"{field}: expected measured iterations 0..9, got {sorted(by_iteration)}")
    vectors: list[list[int]] = []
    for iteration in EXPECTED_MEASURED_ITERATIONS:
        rank_map = by_iteration[iteration]
        ranks = sorted(rank_map)
        if ranks != list(range(NUM_RANKS)):
            raise ValueError(f"{field}: iteration {iteration} ranks are {ranks}")
        vector = [rank_map[rank] for rank in ranks]
        if sum(vector) != expected_total:
            raise ValueError(f"{field}: assignment conservation failed in iteration {iteration}")
        vectors.append(vector)
    return vectors


def expert_matrix(rows: list[dict[str, str]], label: str) -> tuple[int, list[int], list[int]]:
    if not rows:
        raise ValueError(f"{label}: missing rank-expert rows")
    expert_iteration = max(int(row["iteration"]) for row in rows)
    selected = [row for row in rows if int(row["iteration"]) == expert_iteration]
    matrix = [0] * (NUM_RANKS * NUM_EXPERTS)
    padded = [0] * (NUM_RANKS * NUM_EXPERTS)
    for row in selected:
        rank = int(row["global_rank"])
        expert = int(row["expert_id"])
        matrix[rank * NUM_EXPERTS + expert] = int(row["raw_rows"])
        padded[rank * NUM_EXPERTS + expert] = int(row["padded_rows"])
    if sum(matrix) != GLOBAL_ASSIGNMENTS:
        raise ValueError(f"{label}: rank-expert assignment conservation failed")
    return expert_iteration + ROUND_OFFSET, matrix, padded


def rail_records(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row in rows:
        compute_kind = int(row.get("dispatch_compute_kind") or 1)
        compute_name = row.get("dispatch_compute_name") or (
            "attention" if compute_kind == 0 else "moe"
        )
        if compute_kind not in (0, 1):
            raise ValueError("rail telemetry has an invalid dispatch compute kind")
        if compute_name != ("attention" if compute_kind == 0 else "moe"):
            raise ValueError("rail telemetry crossed Attention/MoE identities")
        if row.get("microbatch", "") not in ("", str(compute_kind)):
            raise ValueError("rail telemetry microbatch does not match compute kind")
        source_rank = int(row["source_rank"])
        destination_rank = int(row["destination_rank"])
        if source_rank % SERVER_SIZE != destination_rank % SERVER_SIZE:
            raise ValueError("rail telemetry is not a same-local-index RDMA path")
        rail = source_rank % SERVER_SIZE
        physical_nic = int(row.get("physical_nic") or rail // 2)
        subrail = int(row.get("subrail") or rail % 2)
        records.append(
            {
                "schema_version": int(row.get("schema_version") or 0),
                "run_id": row.get("run_id", ""),
                "slurm_job_id": row.get("slurm_job_id", ""),
                "benchmark_scope": row.get("benchmark_scope", ""),
                "runner_mode": row.get("runner_mode", ""),
                "system": row.get("system", ""),
                "balance": row.get("balance", ""),
                "direction": row.get("direction", ""),
                "workload": row.get("workload", ""),
                "bias_ratio": row.get("bias_ratio", ""),
                "seed": int(row.get("seed") or 0),
                "repeat": int(row.get("repeat") or 0),
                "iteration": int(row["iteration"]),
                "physical_round": int(row["iteration"]) + ROUND_OFFSET,
                "routing_sha256": row.get("routing_sha256", ""),
                "dispatch_compute_kind": compute_kind,
                "dispatch_compute_name": compute_name,
                "round": int(row["iteration"]) + ROUND_OFFSET,
                "compute_kind": compute_kind,
                "compute": compute_name,
                "microbatch": compute_kind,
                "path_id": int(row["path_id"]),
                "source_rank": source_rank,
                "destination_rank": destination_rank,
                "src": source_rank,
                "dst": destination_rank,
                "physical_nic": physical_nic,
                "subrail": subrail,
                "rail_bandwidth_gbps": float(
                    row.get("rail_bandwidth_gbps") or 0.0
                ),
                "physical_nic_bandwidth_gbps": float(
                    row.get("physical_nic_bandwidth_gbps") or 0.0
                ),
                "weight_cache_mode": row.get("weight_cache_mode", "not_recorded"),
                "expert_weight_version": int(
                    row.get("expert_weight_version") or -1
                ),
                "chunk_count": int(row["chunk_count"]),
                "chunks": int(row["chunk_count"]),
                "dispatch_units": int(row.get("dispatch_units") or 0),
                "dispatch_unit_name": row.get("dispatch_unit_name", "not_recorded"),
                "dispatch_bytes_per_unit": int(
                    row.get("dispatch_bytes_per_unit") or 0
                ),
                "traffic_source": row.get("traffic_source", "not_recorded"),
                "dispatch_bytes": int(row["dispatch_bytes"]),
                "weight_bytes": int(row["weight_bytes"]),
                "tx_bytes": int(row["tx_bytes"]),
                "rx_bytes": int(row["rx_bytes"]),
            }
        )
    return records


def rail_profile(records: list[dict[str, object]]) -> dict[str, object]:
    """Build Poseidon-style directed pair and per-rail byte summaries."""

    if not records:
        return {"server_pairs": [], "rails": []}
    numeric_fields = (
        "chunks",
        "dispatch_bytes",
        "weight_bytes",
        "tx_bytes",
        "rx_bytes",
    )
    for row in records:
        if int(row["tx_bytes"]) != int(row["dispatch_bytes"]) + int(
            row["weight_bytes"]
        ):
            raise ValueError("rail TX bytes do not equal Dispatch + Expert Weight")
        if int(row["tx_bytes"]) != int(row["rx_bytes"]):
            raise ValueError("rail TX/RX byte conservation failed")

    def aggregate(
        source_records: list[dict[str, object]],
        key_fields: tuple[str, ...],
    ) -> list[dict[str, object]]:
        per_round: dict[tuple[object, ...], dict[str, int]] = {}
        identities: dict[tuple[object, ...], dict[str, object]] = {}
        for row in source_records:
            identity = tuple(row[field] for field in key_fields)
            identities[identity] = {field: row[field] for field in key_fields}
            round_key = (*identity, int(row["round"]))
            bucket = per_round.setdefault(
                round_key,
                {field: 0 for field in numeric_fields},
            )
            for field in numeric_fields:
                bucket[field] += int(row[field])
        grouped: dict[tuple[object, ...], list[dict[str, int]]] = defaultdict(list)
        for round_key, values in per_round.items():
            grouped[round_key[:-1]].append(values)
        summaries = []
        for identity, round_values in grouped.items():
            if len(round_values) != len(EXPECTED_MEASURED_ITERATIONS):
                raise ValueError(
                    f"rail {identity}: expected ten measured rounds, got {len(round_values)}"
                )
            summary = dict(identities[identity])
            for field in numeric_fields:
                values = [item[field] for item in round_values]
                summary[f"mean_{field}"] = mean([float(value) for value in values])
                summary[f"min_{field}"] = min(values)
                summary[f"max_{field}"] = max(values)
            summary["round_count"] = len(round_values)
            summaries.append(summary)
        return summaries

    enriched = []
    for row in records:
        item = dict(row)
        item["source_server"] = int(item["src"]) // SERVER_SIZE
        item["destination_server"] = int(item["dst"]) // SERVER_SIZE
        item["rail"] = int(item["src"]) % SERVER_SIZE
        item.setdefault("physical_nic", int(item["rail"]) // 2)
        item.setdefault("subrail", int(item["rail"]) % 2)
        item.setdefault("rail_bandwidth_gbps", 0.0)
        item.setdefault("physical_nic_bandwidth_gbps", 0.0)
        enriched.append(item)
    pair_rows = aggregate(
        enriched,
        ("compute", "source_server", "destination_server"),
    )
    rail_rows = aggregate(
        enriched,
        (
            "compute",
            "source_server",
            "destination_server",
            "rail",
            "physical_nic",
            "subrail",
            "rail_bandwidth_gbps",
            "physical_nic_bandwidth_gbps",
            "path_id",
            "src",
            "dst",
        )
    )
    order = {"attention": 0, "moe": 1}
    pair_rows.sort(
        key=lambda row: (
            order.get(str(row["compute"]), 99),
            int(row["source_server"]),
            int(row["destination_server"]),
        )
    )
    rail_rows.sort(
        key=lambda row: (
            order.get(str(row["compute"]), 99),
            int(row["source_server"]),
            int(row["destination_server"]),
            int(row["rail"]),
        )
    )
    return {
        "server_pairs": pair_rows,
        "rails": rail_rows,
    }


def attach_weight_components(
    profile: dict[str, object], chunks: list[dict[str, str]]
) -> dict[str, object]:
    """Attach exact per-expert mean bytes without changing byte accounting.

    A cache-hit plan still contains a logical chunk table, but contributes zero
    bytes to the measured rail.  Only transfer_required rows become colored
    byte segments; planned/cache-hit experts are rendered in the placement
    table instead.
    """

    pair_totals: dict[tuple[str, int, int, int], int] = defaultdict(int)
    rail_totals: dict[tuple[str, int, int, int, int], int] = defaultdict(int)
    for row in chunks:
        compute_kind = int(row["dispatch_compute_kind"])
        compute = row["dispatch_compute_name"]
        if compute_kind not in (0, 1) or compute != (
            "attention" if compute_kind == 0 else "moe"
        ):
            raise ValueError("Weight chunk crossed Attention/MoE identities")
        source_server = int(row["source_server"])
        destination_server = int(row["destination_server"])
        source_rank = int(row["source_rank"])
        destination_rank = int(row["destination_rank"])
        rail = int(row["rail"])
        if (
            source_rank // SERVER_SIZE != source_server
            or destination_rank // SERVER_SIZE != destination_server
            or source_rank % SERVER_SIZE != rail
            or destination_rank % SERVER_SIZE != rail
        ):
            raise ValueError("Weight chunk server/rank/rail identity is inconsistent")
        chunk_bytes = int(row["chunk_bytes"])
        if chunk_bytes <= 0 or int(row["transfer_required"]) not in (0, 1):
            raise ValueError("Weight chunk has invalid byte/admission fields")
        if int(row["transfer_required"]) == 0:
            continue
        expert = int(row["expert_id"])
        pair_totals[(compute, source_server, destination_server, expert)] += chunk_bytes
        rail_totals[(compute, source_server, destination_server, rail, expert)] += chunk_bytes

    rounds = float(len(EXPECTED_MEASURED_ITERATIONS))
    for row in profile["server_pairs"]:
        components = [
            {"expert_id": expert, "mean_bytes": total / rounds}
            for (compute, source, destination, expert), total in pair_totals.items()
            if compute == row["compute"]
            and source == int(row["source_server"])
            and destination == int(row["destination_server"])
        ]
        components.sort(key=lambda item: int(item["expert_id"]))
        row["weight_components"] = components
    for row in profile["rails"]:
        components = [
            {"expert_id": expert, "mean_bytes": total / rounds}
            for (compute, source, destination, rail, expert), total in rail_totals.items()
            if compute == row["compute"]
            and source == int(row["source_server"])
            and destination == int(row["destination_server"])
            and rail == int(row["rail"])
        ]
        components.sort(key=lambda item: int(item["expert_id"]))
        row["weight_components"] = components

    if chunks:
        for row in [*profile["server_pairs"], *profile["rails"]]:
            component_bytes = sum(
                float(item["mean_bytes"])
                for item in row["weight_components"]
            )
            if abs(component_bytes - float(row["mean_weight_bytes"])) > 0.5:
                raise ValueError(
                    "per-expert Weight components disagree with aggregate rail bytes"
                )
    return profile


def admitted_placements(plan: dict[str, object]) -> list[dict[str, int]]:
    explicit = plan.get("admitted_placements")
    if isinstance(explicit, list):
        return [
            {
                "expert_id": int(item["expert_id"]),
                "destination_server": int(item["destination_server"]),
            }
            for item in explicit
        ]
    return [
        {
            "expert_id": int(encoded) // PROBEEP_PLAN_SERVER_STRIDE,
            "destination_server": int(encoded) % PROBEEP_PLAN_SERVER_STRIDE,
        }
        for encoded in plan.get("admitted_experts", [])
    ]


def microbatch_rank_payload(
    rows: list[dict[str, str]],
    *,
    label: str,
    system: str,
    materialized_reference: list[dict[str, object]],
    fallback_plan_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Return exact MB0/MB1 rank load when the raw log contains it.

    The 2026-08-17 run predates microbatch_rank_samples.csv.  Its immutable
    materialized route can still be reconstructed byte-for-byte from RawData1
    and accepted only after the .npy SHA-256 matches the recorded case SHA.
    ProbeEP execution rows additionally come from the completed production
    plan handles; the server-local baselines use their exact capacity rule.
    """

    expected_total = GLOBAL_ASSIGNMENTS // 2
    output = []
    for microbatch in range(2):
        selected = [
            row for row in rows if int(row["microbatch"]) == microbatch
        ]
        home_vectors: list[list[int]] = []
        execution_vectors: list[list[int]] = []
        source = "unavailable"
        if selected:
            if len(selected) != len(EXPECTED_MEASURED_ITERATIONS) * NUM_RANKS:
                raise ValueError(
                    f"{label} MB{microbatch}: expected 160 rank rows, got {len(selected)}"
                )
            home_vectors = extract_rank_vectors(
                selected,
                "home_load",
                expected_total=expected_total,
            )
            execution_vectors = extract_rank_vectors(
                selected,
                "exec_load",
                expected_total=expected_total,
            )
            source = "microbatch_rank_samples.csv"
        else:
            reference = materialized_reference[microbatch]
            home_vector = [int(value) for value in reference["home"]]
            if len(home_vector) != NUM_RANKS or sum(home_vector) != expected_total:
                raise ValueError(f"{label} MB{microbatch}: invalid materialized home load")
            home_vectors = [home_vector for _ in EXPECTED_MEASURED_ITERATIONS]
            source = "SHA-verified materialized RawData1 route"
        if not execution_vectors and fallback_plan_rows:
            matching = sorted(
                (
                    row
                    for row in fallback_plan_rows
                    if int(row["dispatch_compute_kind"]) == microbatch
                ),
                key=lambda row: int(row["iteration"]),
            )
            if len(matching) == len(EXPECTED_MEASURED_ITERATIONS):
                execution_vectors = [
                    [int(value) for value in row["rank_load_after"]]
                    for row in matching
                ]
                for iteration, vector in enumerate(execution_vectors):
                    if len(vector) != NUM_RANKS or sum(vector) != expected_total:
                        raise ValueError(
                            f"{label} MB{microbatch} iteration {iteration}: invalid fallback plan load"
                        )
                source = "probeep_plan_summary.jsonl execution-only recovery"
        if not execution_vectors and system in {"nccl", "deepep"}:
            execution_vectors = [
                list(home_vectors[0]) for _ in EXPECTED_MEASURED_ITERATIONS
            ]
            source += "; no-balance execution"
        if not execution_vectors and system == "deepep_moonep":
            server_local = [
                int(value)
                for value in materialized_reference[microbatch]["server_local"]
            ]
            if len(server_local) != NUM_RANKS or sum(server_local) != expected_total:
                raise ValueError(f"{label} MB{microbatch}: invalid server-local load")
            execution_vectors = [
                server_local for _ in EXPECTED_MEASURED_ITERATIONS
            ]
            source += "; exact server-local capacity"
        output.append(
            {
                "microbatch": microbatch,
                "observation": "attention" if microbatch == 0 else "moe",
                "available": bool(execution_vectors),
                "home_available": bool(home_vectors),
                "home": vector_mean(home_vectors) if home_vectors else [],
                "execution": (
                    vector_mean(execution_vectors) if execution_vectors else []
                ),
                "home_stable": vectors_stable(home_vectors) if home_vectors else None,
                "execution_stable": (
                    vectors_stable(execution_vectors)
                    if execution_vectors
                    else None
                ),
                "source": source,
            }
        )
    return output


def materialized_microbatch_references(
    iteration_index: dict[tuple[str, str, str], list[dict[str, str]]],
    layers: list[int],
) -> dict[int, list[dict[str, object]]]:
    """Rebuild the exact immutable .npy routes and split them into MB0/MB1."""

    selector = "raw_data1_all" if max(layers, default=-1) >= 20 else "raw_data1_eval20"
    summary, scaled = describe(
        PROJECT_ROOT / "workload/raw_data1",
        selector,
        NUM_RANKS,
        4096,
        8,
    )
    by_layer_counts = {
        int(layer): counts
        for layer, counts in zip(summary["selected_layer_ids"], scaled)
    }
    output: dict[int, list[dict[str, object]]] = {}
    for layer in layers:
        workload = workload_name(layer)
        case_rows = iteration_index.get(("deepep", "na", workload), [])
        hashes = {row["routing_sha256"] for row in case_rows}
        if len(hashes) != 1:
            raise ValueError(f"layer {layer}: missing immutable route SHA")
        routes = realize_exact_topk(
            by_layer_counts[layer],
            NUM_RANKS * 4096,
            8,
        ).reshape(NUM_RANKS, 4096, 8)
        serialized = io.BytesIO()
        np.save(serialized, routes, allow_pickle=False)
        digest = hashlib.sha256(serialized.getvalue()).hexdigest()
        recorded = next(iter(hashes))
        if digest != recorded:
            raise ValueError(
                f"layer {layer}: reconstructed route SHA {digest} != raw {recorded}"
            )
        microbatches = []
        for start, end in ((0, 2048), (2048, 4096)):
            expert_counts = np.bincount(
                routes[:, start:end, :].astype(np.int64).ravel(),
                minlength=NUM_EXPERTS,
            )
            home = expert_counts.reshape(NUM_RANKS, NUM_EXPERTS // NUM_RANKS).sum(1)
            server_local: list[int] = []
            for server in range(NUM_RANKS // SERVER_SIZE):
                begin = server * SERVER_SIZE
                total = int(home[begin : begin + SERVER_SIZE].sum())
                base, remainder = divmod(total, SERVER_SIZE)
                server_local.extend(
                    base + int(local_rank < remainder)
                    for local_rank in range(SERVER_SIZE)
                )
            microbatches.append(
                {
                    "home": [int(value) for value in home],
                    "server_local": server_local,
                    "routing_sha256": digest,
                }
            )
        output[layer] = microbatches
    return output


def microbatch_timeline_payload(
    rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    records = []
    for row in rows:
        records.append(
            {
                "round": int(row["iteration"]) + ROUND_OFFSET,
                "rank": int(row["global_rank"]),
                "microbatch": int(row["microbatch"]),
                "stream": row["logical_stream"],
                "stage": row["stage"],
                "start_ms": float(row["start_ms"]),
                "end_ms": float(row["end_ms"]),
                "duration_ms": float(row["duration_ms"]),
            }
        )
    return records


def make_payload(run_dir: Path, layers: list[int]) -> dict[str, object]:
    raw_dir = run_dir / "raw"
    iteration_rows_all = read_csv(raw_dir / "iterations.csv")
    iteration_index = index_rows(iteration_rows_all)
    rank_index = index_rows(read_csv(raw_dir / "rank_samples.csv"))
    expert_index = index_rows(read_csv(raw_dir / "rank_expert_samples.csv"))
    rail_index = index_rows(read_csv(raw_dir / "rdma_path_load.csv"))
    microbatch_rank_index = index_rows(
        read_csv_optional(raw_dir / "microbatch_rank_samples.csv")
    )
    timeline_index = index_rows(
        read_csv_optional(raw_dir / "microbatch_timeline.csv")
    )
    observation_index = index_rows(
        read_csv_optional(raw_dir / "probeep_observation_samples.csv")
    )
    weight_chunk_index = index_rows(
        read_csv_optional(raw_dir / "probeep_weight_chunks.csv")
    )
    schema_versions = sorted({int(row["schema_version"]) for row in iteration_rows_all})
    telemetry_contract = {
        "schema_versions": schema_versions,
        "microbatch_rank_samples": (raw_dir / "microbatch_rank_samples.csv").is_file(),
        "microbatch_timeline": (raw_dir / "microbatch_timeline.csv").is_file(),
        "probeep_observations": (raw_dir / "probeep_observation_samples.csv").is_file(),
        "probeep_weight_chunks": (raw_dir / "probeep_weight_chunks.csv").is_file(),
        "nsys_reports": any((run_dir / "nsys").rglob("*.nsys-rep"))
        if (run_dir / "nsys").is_dir()
        else False,
    }
    telemetry_contract["fresh_visualization_raw"] = (
        schema_versions == [4]
        and all(
            bool(telemetry_contract[field])
            for field in (
                "microbatch_rank_samples",
                "microbatch_timeline",
                "probeep_observations",
                "probeep_weight_chunks",
            )
        )
    )
    probe_plan_index: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in read_jsonl_optional(raw_dir / "probeep_plan_summary.jsonl"):
        probe_plan_index[str(row["workload"])].append(row)
    materialized_references = materialized_microbatch_references(
        iteration_index,
        layers,
    )
    methods: list[dict[str, object]] = []

    for system, balance, label, slug, semantics in METHODS:
        method_layers: list[dict[str, object]] = []
        for layer in layers:
            workload = workload_name(layer)
            key = (system, balance, workload)
            iter_rows = sorted(iteration_index.get(key, []), key=lambda row: int(row["iteration"]))
            rank_rows = rank_index.get(key, [])
            expert_rows = expert_index.get(key, [])
            if len(iter_rows) != 10 or len(rank_rows) != 160:
                raise ValueError(
                    f"{label} layer {layer}: expected 10 iteration and 160 rank rows, "
                    f"got {len(iter_rows)} and {len(rank_rows)}"
                )
            if tuple(int(row["iteration"]) for row in iter_rows) != EXPECTED_MEASURED_ITERATIONS:
                raise ValueError(f"{label} layer {layer}: expected measured iterations 0..9")
            seeds = {row["seed"] for row in iter_rows}
            hashes = {row["routing_sha256"] for row in iter_rows}
            if len(seeds) != 1 or len(hashes) != 1:
                raise ValueError(f"{label} layer {layer}: seed or routing hash changed across measured iterations")
            home_vectors = extract_rank_vectors(rank_rows, "home_load")
            raw_exec_vectors = extract_rank_vectors(rank_rows, "exec_load")
            raw_exec_mismatch_iterations = [
                iteration
                for iteration, (home, execution) in enumerate(zip(home_vectors, raw_exec_vectors))
                if home != execution
            ]
            exec_vectors = raw_exec_vectors
            execution_source = "raw exec_load"
            if system == "deepep" and raw_exec_mismatch_iterations:
                raise ValueError(
                    f"{label} layer {layer}: DeepEP baseline changed placement in iterations "
                    f"{raw_exec_mismatch_iterations}"
                )
            if system == "nccl" and raw_exec_mismatch_iterations:
                exec_vectors = home_vectors
                execution_source = "home_load corrected for no-balance NCCL baseline"
            home = vector_mean(home_vectors)
            execution = vector_mean(exec_vectors)
            expert_round, matrix, padded_matrix = expert_matrix(expert_rows, f"{label} layer {layer}")
            rounds = [round_record(row) for row in iter_rows]
            layer_rails = rail_records(rail_index.get(key, []))
            microbatch_ranks = microbatch_rank_payload(
                microbatch_rank_index.get(key, []),
                label=f"{label} layer {layer}",
                system=system,
                materialized_reference=materialized_references[layer],
                fallback_plan_rows=(
                    probe_plan_index.get(workload, [])
                    if system == "probeep"
                    else []
                ),
            )
            if not all(item["home_available"] for item in microbatch_ranks):
                raise ValueError(f"{label} layer {layer}: MB0/MB1 home load is incomplete")
            split_home = [
                float(microbatch_ranks[0]["home"][rank])
                + float(microbatch_ranks[1]["home"][rank])
                for rank in range(NUM_RANKS)
            ]
            if any(abs(left - right) > 1e-6 for left, right in zip(split_home, home)):
                raise ValueError(f"{label} layer {layer}: MB home rows do not sum to combined")
            if all(item["available"] for item in microbatch_ranks):
                split_execution = [
                    float(microbatch_ranks[0]["execution"][rank])
                    + float(microbatch_ranks[1]["execution"][rank])
                    for rank in range(NUM_RANKS)
                ]
                if any(
                    abs(left - right) > 1e-6
                    for left, right in zip(split_execution, execution)
                ):
                    raise ValueError(
                        f"{label} layer {layer}: MB execution rows do not sum to combined"
                    )
            timeline = microbatch_timeline_payload(
                timeline_index.get(key, [])
            )
            plans = (
                sorted(
                    probe_plan_index.get(workload, []),
                    key=lambda row: (
                        int(row["iteration"]),
                        int(row["dispatch_compute_kind"]),
                    ),
                )
                if system == "probeep"
                else []
            )
            weight_chunks = weight_chunk_index.get(key, [])
            layer_rail_profile = attach_weight_components(
                rail_profile(layer_rails), weight_chunks
            )
            method_layers.append(
                {
                    "layer": layer,
                    "workload": workload,
                    "seed": next(iter(seeds)),
                    "routing_sha256": next(iter(hashes)),
                    "home": home,
                    "execution": execution,
                    "home_stable": vectors_stable(home_vectors),
                    "execution_stable": vectors_stable(exec_vectors),
                    "raw_execution_stable": vectors_stable(raw_exec_vectors),
                    "raw_exec_mismatch_iterations": raw_exec_mismatch_iterations,
                    "execution_source": execution_source,
                    "home_rank_ratio": ratio(home),
                    "execution_rank_ratio": ratio(execution),
                    "home_server_ratio": ratio(server_sums(home)),
                    "execution_server_ratio": ratio(server_sums(execution)),
                    "rounds": rounds,
                    "summary": summarize_rounds(rounds),
                    "expert_round": expert_round,
                    "expert_matrix": matrix,
                    "padded_matrix": padded_matrix,
                    "rails": layer_rails,
                    "rail_profile": layer_rail_profile,
                    "microbatches": microbatch_ranks,
                    "timeline": timeline,
                    "observations": observation_index.get(key, []),
                    "plans": plans,
                    "weight_chunks": weight_chunks,
                    "telemetry_contract": telemetry_contract,
                }
            )
        methods.append(
            {
                "system": system,
                "balance": balance,
                "label": label,
                "slug": slug,
                "semantics": semantics,
                "layers": method_layers,
            }
        )
    for layer_index, layer in enumerate(layers):
        identities = {
            (
                method["layers"][layer_index]["seed"],
                method["layers"][layer_index]["routing_sha256"],
            )
            for method in methods
        }
        if len(identities) != 1:
            raise ValueError(
                f"layer {layer}: algorithms did not consume one identical seed/route SHA"
            )
    return {
        "schema": "probeep.raw_data1.algorithm_layers.v1",
        "run_id": run_dir.name,
        "layers": layers,
        "physical_rounds": list(range(11, 21)),
        "warmup_rounds": list(range(1, 11)),
        "num_ranks": NUM_RANKS,
        "num_servers": NUM_RANKS // SERVER_SIZE,
        "server_size": SERVER_SIZE,
        "experts": NUM_EXPERTS,
        "tokens_per_rank": 4096,
        "topk": 8,
        "global_assignments": GLOBAL_ASSIGNMENTS,
        "telemetry_contract": telemetry_contract,
        "methods": methods,
        "statistics_contract": (
            "algorithm first; every selected layer is shown separately; latency and rank load "
            "are averaged only across the ten measured iterations inside that layer"
        ),
    }


STYLE = r"""
:root{color-scheme:light;font-family:Inter,"Noto Sans SC",Arial,sans-serif}
*{box-sizing:border-box}body{margin:0;background:#f4f6f8;color:#17212b}
header{padding:18px 22px;background:#fff;border-bottom:1px solid #ced5de}
h1{margin:0;font-size:22px}h2{margin:0;font-size:17px}h3{margin:0 0 10px;font-size:14px}
.subtitle{margin-top:8px;color:#536170;font-size:13px;line-height:1.6}
main{padding:14px;display:grid;gap:12px}.panel{background:#fff;border:1px solid #c8d0da;padding:15px}
.back{display:inline-block;margin-bottom:10px;color:#1d5e9e;text-decoration:none;font-size:13px}
.metrics{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px}
.metric{background:#eef2f5;border-left:3px solid #1d5e9e;padding:10px}.metric b{display:block;font-size:20px}.metric span{font-size:12px;color:#536170}
.grid2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.chart{width:100%;min-height:280px}
.note{font-size:12px;color:#536170;line-height:1.6;margin:8px 0 0}.small{font-size:11px;color:#6c7987}
svg text{font-family:Inter,"Noto Sans SC",Arial,sans-serif}.tablewrap{overflow:auto;max-height:460px}
table{width:100%;border-collapse:collapse;font-size:12px}th,td{padding:7px 9px;border-bottom:1px solid #e1e6eb;text-align:right;white-space:nowrap}
th{position:sticky;top:0;background:#eef2f5;color:#485563}th:first-child,td:first-child{text-align:left}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:#536170;margin:8px 0}.swatch{display:inline-block;width:10px;height:10px;margin-right:5px}
.transport-chart{overflow-x:auto;margin-top:10px}.transport-chart svg{display:block;width:100%;min-width:920px;border:1px solid #d8dee6;background:#fff}
.transport-details{margin-top:10px;border-top:1px solid #d8dee6}.transport-details>summary{cursor:pointer;padding:10px 0;color:#536170}
.pipeline-svg{display:block;width:100%;min-width:900px;border:1px solid #d8dee6;background:#fff}.pipeline-scroll{overflow-x:auto;margin-top:10px}
.pipeline-controls{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin:10px 0;font-size:12px;color:#536170}.pipeline-controls select{margin-left:6px;padding:4px 7px;border:1px solid #aeb8c3;background:#fff;color:#17212b}
.empty{padding:28px;background:#f7f9fb;border:1px dashed #aeb8c3;color:#536170;line-height:1.7}
.microbatch-block{display:grid;gap:12px;padding:12px;border:2px solid #aeb8c3;background:#e9eef3}.microbatch-title{background:#24384a;color:#fff;border-color:#24384a}.microbatch-title .note{color:#d8e2eb}
.algorithm{min-height:88px;padding:15px 16px;background:#fff;border:1px solid #c8d0da;color:#17212b;text-decoration:none;display:flex;align-items:center;justify-content:space-between;gap:20px}
.algorithm:hover{border-color:#718397;background:#fafbfc}.algorithm strong{display:block;margin-bottom:7px}.algorithm p{margin:0;color:#536170;font-size:13px}.arrow{color:#1d5e9e;font-size:22px}
.layer{scroll-margin-top:10px}.layer-title{display:flex;justify-content:space-between;gap:14px;align-items:baseline;flex-wrap:wrap}
.layer-title a{font-size:12px;color:#1d5e9e;text-decoration:none}.unstable{color:#a04700;font-weight:700}.stable{color:#25632f;font-weight:700}
.warning{border-left:5px solid #b45309;background:#fff8e8}.pass{border-left:5px solid #238636;background:#f0fff4}
@media(max-width:1000px){.metrics{grid-template-columns:repeat(2,1fr)}.grid2{grid-template-columns:1fr}}
"""


CHART_JS = r"""
function node(tag,attrs={}){const n=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v);return n}
function clearSvg(id){const s=document.getElementById(id);s.innerHTML='';s.setAttribute('viewBox','0 0 920 290');return s}
function textAt(s,x,y,t,a='middle',size=11,fill='#536170'){const n=node('text',{x,y,'text-anchor':a,'font-size':size,fill});n.textContent=t;s.appendChild(n)}
function fmtK(v){return (v/1000).toFixed(v>=100000?0:1)+'k'}
function rankBars(id,values,color){const s=clearSvg(id),W=920,H=290,l=56,r=12,t=18,b=42,iw=W-l-r,ih=H-t-b,max=Math.max(...values)*1.12,bw=iw/values.length;
 [0,.25,.5,.75,1].forEach(q=>{const y=t+ih*(1-q);s.appendChild(node('line',{x1:l,y1:y,x2:W-r,y2:y,stroke:'#dce2e8'}));textAt(s,l-7,y+4,Math.round(max*q).toLocaleString(),'end')});
 values.forEach((v,i)=>{const h=ih*v/max,x=l+i*bw+bw*.14,y=t+ih-h;s.appendChild(node('rect',{x,y,width:bw*.72,height:h,fill:color}));textAt(s,x+bw*.36,y-4,fmtK(v),'middle',10,'#17212b');textAt(s,x+bw*.36,H-20,'R'+i)});
 s.appendChild(node('line',{x1:l,y1:t+ih,x2:W-r,y2:t+ih,stroke:'#718397'}));s.appendChild(node('line',{x1:l+8*bw,y1:t,x2:l+8*bw,y2:t+ih,stroke:'#718397','stroke-dasharray':'5 5'}));}
function lineChart(id,series){const s=clearSvg(id),W=920,H=290,l=55,r=20,t=18,b=42,iw=W-l-r,ih=H-t-b,all=series.flatMap(x=>x.values),mn=Math.min(...all),mx=Math.max(...all),pad=Math.max((mx-mn)*.12,.5),lo=Math.max(0,mn-pad),hi=mx+pad,colors=['#1d5e9e','#ea580c','#16a34a','#7c3aed'];
 [0,.25,.5,.75,1].forEach(q=>{const y=t+ih*(1-q);s.appendChild(node('line',{x1:l,y1:y,x2:W-r,y2:y,stroke:'#dce2e8'}));textAt(s,l-7,y+4,(lo+(hi-lo)*q).toFixed(1),'end')});
 const X=i=>l+iw*i/9,Y=v=>t+ih*(hi-v)/(hi-lo);series.forEach((line,j)=>{const pts=line.values.map((v,i)=>X(i)+','+Y(v)).join(' ');s.appendChild(node('polyline',{points:pts,fill:'none',stroke:colors[j], 'stroke-width':2.3}));line.values.forEach((v,i)=>{s.appendChild(node('circle',{cx:X(i),cy:Y(v),r:3,fill:colors[j]}))})});
 for(let i=0;i<10;i++)textAt(s,X(i),H-20,String(i+11));textAt(s,15,16,'ms','start',11);}
function fmtBytes(v){const n=Number(v);if(n>=1024**3)return (n/1024**3).toFixed(2)+' GiB';if(n>=1024**2)return (n/1024**2).toFixed(2)+' MiB';if(n>=1024)return (n/1024).toFixed(2)+' KiB';return Math.round(n)+' B'}
function expertColor(expert){return `hsl(${(Number(expert)*137.508)%360} 65% 45%)`}
function directedLoadChart(id,rows,labelFor){const s=document.getElementById(id);s.innerHTML='';const rowH=25,W=1080,H=32+Math.max(1,rows.length)*rowH+28,l=300,r=104,t=25,iw=W-l-r,max=Math.max(1,...rows.map(x=>Number(x.mean_tx_bytes)||0));s.setAttribute('viewBox',`0 0 ${W} ${H}`);s.setAttribute('role','img');s.setAttribute('aria-label','Directed RDMA logical rail byte load');
 [0,.25,.5,.75,1].forEach(q=>{const x=l+iw*q;s.appendChild(node('line',{x1:x,y1:t-8,x2:x,y2:H-22,stroke:'#e3e7eb'}));textAt(s,x,14,fmtBytes(max*q),'middle',10)});
 rows.forEach((row,i)=>{const y=t+i*rowH,label=labelFor(row);textAt(s,l-8,y+15,label,'end',10,'#42505e');let offset=0;const dispatch=Number(row.mean_dispatch_bytes)||0;if(dispatch>0){const w=dispatch/max*iw,rect=node('rect',{x:l+offset,y:y+3,width:w,height:16,fill:'#e28200'}),tip=node('title');tip.textContent=`${label} | Token Dispatch ${fmtBytes(dispatch)}`;rect.appendChild(tip);s.appendChild(rect);offset+=w}const components=Array.isArray(row.weight_components)?row.weight_components:[];if(components.length){for(const item of components){const value=Number(item.mean_bytes)||0,w=value/max*iw;if(value<=0)continue;const rect=node('rect',{x:l+offset,y:y+3,width:w,height:16,fill:expertColor(item.expert_id)}),tip=node('title');tip.textContent=`${label} | Expert E${item.expert_id} Weight ${fmtBytes(value)} | actual transferred bytes`;rect.appendChild(tip);s.appendChild(rect);offset+=w}}else{const value=Number(row.mean_weight_bytes)||0,w=value/max*iw;if(value>0){const rect=node('rect',{x:l+offset,y:y+3,width:w,height:16,fill:'#1098a3'}),tip=node('title');tip.textContent=`${label} | Expert Weight ${fmtBytes(value)} | historical aggregate has no expert ids`;rect.appendChild(tip);s.appendChild(rect);offset+=w}}textAt(s,Math.min(l+offset+6,W-r+5),y+15,fmtBytes(row.mean_tx_bytes),'start',10,'#293746')});}
function pipelineDag(id,slug,layer){const s=document.getElementById(id),W=1400,H=430;s.innerHTML='';s.setAttribute('viewBox',`0 0 ${W} ${H}`);s.setAttribute('role','img');s.setAttribute('aria-label','Two-stream CUDA event dependency DAG');const shared=['deepep_moonep','ultraep_hybridep'].includes(slug),probe=slug==='probeep',bootstrap=Number(layer)===0;const lanes={feedback:30,compute:122,communication:252};
 const defs=node('defs'),marker=node('marker',{id:`arrow-${id}`,viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:6,markerHeight:6,orient:'auto-start-reverse'});marker.appendChild(node('path',{d:'M 0 0 L 10 5 L 0 10 z',fill:'#3f4c59'}));defs.appendChild(marker);s.appendChild(defs);
 for(const [name,y] of Object.entries(lanes)){const label=name==='compute'?'Compute/default stream':(name==='communication'?'Communication stream':'Persistent feedback state');textAt(s,142,y+26,label,'end',12,'#293746');s.appendChild(node('line',{x1:160,y1:y+25,x2:1370,y2:y+25,stroke:'#c7d0d9'}))}
 const boxes=[];function box(key,label,stream,x,w,color,mb){const y=lanes[stream],r=node('rect',{x,y,width:w,height:50,fill:color,rx:2}),tip=node('title');tip.textContent=`${label} | ${stream} | ${mb<0?'shared':'MB'+mb}`;r.appendChild(tip);s.appendChild(r);textAt(s,x+w/2,y+30,label,'middle',11,'#fff');boxes.push({key,x,y,w});}
 const x0=probe?270:175,pwd=(probe||shared)?'P/W/D':'P/D';if(probe){box('prev_a',bootstrap?'A bootstrap':'A[L−1,r]','feedback',145,105,'#7657b5',-1);box('prev_m',bootstrap?'M bootstrap':'M[L−1,r]','feedback',475,105,'#256f54',-1)}box('a0','A/G0','compute',x0,82,'#7657b5',0);box('a1','A/G1','compute',x0+145,82,'#7657b5',1);box('d0',pwd+'0','communication',x0+145,112,'#e28200',0);const e0x=x0+345;box('e0','E0','compute',e0x,82,'#2f74b5',0);box('d1',pwd+'1','communication',e0x,112,'#1098a3',1);const e1x=e0x+220;box('e1','E1','compute',e1x,82,'#256f54',1);const c0x=e0x+205;box('c0','C0','communication',c0x,72,'#ca4057',0);const c1x=e1x+190;box('c1','C1','communication',c1x,72,'#8a4a9c',1);if(probe){box('obs_a','A[L,r]=A1∥D0','feedback',c1x+105,112,'#7657b5',-1);box('obs_m','M[L,r]=E0∥D1','feedback',c1x+227,112,'#256f54',-1)}
 const by=Object.fromEntries(boxes.map(x=>[x.key,x]));function edge(a,b,kind='event'){const A=by[a],B=by[b],x1=A.x+A.w,x2=B.x,y1=A.y+25,y2=B.y+25,stroke=kind==='order'?'#788592':'#3f4c59',dash=kind==='next'?'6 5':(kind==='host'?'2 3':null),p=node('path',{d:`M${x1},${y1} C${x1+30},${y1} ${x2-30},${y2} ${x2},${y2}`,fill:'none',stroke,'stroke-width':kind==='event'?1.8:1.2,'marker-end':`url(#arrow-${id})`});if(dash)p.setAttribute('stroke-dasharray',dash);s.insertBefore(p,defs.nextSibling)}
 edge('a0','a1','order');edge('a1','e0','order');edge('e0','e1','order');edge('d0','d1','order');edge('d1','c0','order');edge('c0','c1','order');edge('a0','d0');edge('d0','e0');edge('a1','d1');edge('d1','e1');edge('e0','c0');edge('e1','c1');if(shared)edge('e0','d1');if(probe){edge('prev_a','d0','next');edge('prev_m','d1','next');edge('a1','obs_a','next');edge('d0','obs_a','next');edge('e0','obs_m','next');edge('d1','obs_m','next');edge('c1','obs_a','host');edge('c1','obs_m','host')}
 textAt(s,165,345,'Wavefront: A0 → (A1 ∥ W+D0) → (E0 ∥ W+D1) → E1；MoonEP/UltraEP 单 replica bank 额外要求 E0 → W+D1','start',11,'#293746');textAt(s,165,365,'灰线=同一 stream 提交顺序；深色=跨 stream CUDA-event wait；虚线=跨 Layer feedback/同类 observation 数据源；点线=C1 后物化 feedback','start',11,'#536170');textAt(s,165,385,bootstrap?'Layer 0：A/M 两条链分别使用显式 bootstrap，不伪造上一层 observation':'跨层反馈严格分链：A[L−1,r] 只进入 MB0 P/W/D；M[L−1,r] 只进入 MB1 P/W/D；两者不可替代','start',11,'#536170');textAt(s,165,405,'A[L,r]=A1 Attention ∥ MB0 W+D；M[L,r]=MB0 Expert ∥ MB1 W+D；P/W/D 为融合主路径','start',11,'#536170');}
function measuredTimeline(id,rows,roundSelectId,rankSelectId){const s=document.getElementById(id),roundSelect=document.getElementById(roundSelectId),rankSelect=document.getElementById(rankSelectId);const rounds=[...new Set(rows.map(x=>x.round))].sort((a,b)=>a-b),ranks=[...new Set(rows.map(x=>x.rank))].sort((a,b)=>a-b);for(const v of rounds)roundSelect.add(new Option(`Round ${v}`,v));for(const v of ranks)rankSelect.add(new Option(`R${v}`,v));function draw(){const selected=rows.filter(x=>x.round===Number(roundSelect.value)&&x.rank===Number(rankSelect.value)),W=1120,H=250,l=150,r=30,t=35,lanes={compute:65,communication:150},end=Math.max(1,...selected.map(x=>x.end_ms)),iw=W-l-r;s.innerHTML='';s.setAttribute('viewBox',`0 0 ${W} ${H}`);for(const [name,y] of Object.entries(lanes)){textAt(s,l-12,y+19,name==='compute'?'Compute stream':'Communication stream','end',11,'#293746');s.appendChild(node('line',{x1:l,y1:y+18,x2:W-r,y2:y+18,stroke:'#d8dee6'}))}[0,.25,.5,.75,1].forEach(q=>{const x=l+iw*q;s.appendChild(node('line',{x1:x,y1:t-8,x2:x,y2:H-35,stroke:'#e6eaee'}));textAt(s,x,H-16,(end*q).toFixed(2)+' ms','middle',10)});const colors={0:'#2f74b5',1:'#109873','-1':'#596575'};selected.forEach(row=>{const y=lanes[row.stream],x=l+iw*row.start_ms/end,w=Math.max(2,iw*(row.end_ms-row.start_ms)/end),rect=node('rect',{x,y,width:w,height:36,fill:colors[row.microbatch]||'#596575'}),label=`${row.microbatch<0?'shared':'MB'+row.microbatch} ${row.stage}`;const tip=node('title');tip.textContent=`${label} | ${row.start_ms.toFixed(4)}-${row.end_ms.toFixed(4)} ms | ${row.duration_ms.toFixed(4)} ms`;rect.appendChild(tip);s.appendChild(rect);if(w>68)textAt(s,x+w/2,y+23,label,'middle',10,'#fff')});}roundSelect.onchange=draw;rankSelect.onchange=draw;draw();}
"""


def page_header(title: str, subtitle: str) -> str:
    return (
        "<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        f"<title>{html.escape(title)}</title><style>{STYLE}</style></head><body>"
        f"<header><h1>{html.escape(title)}</h1><div class=\"subtitle\">{html.escape(subtitle)}</div></header>"
    )


def write_root(path: Path, payload: dict[str, object]) -> None:
    cards = []
    for method in payload["methods"]:
        layers = method["layers"]
        cards.append(
            f"""<a class="algorithm" href="algorithms/{method['slug']}/algorithm_dashboard.html"><div><strong>{html.escape(str(method['label']))}</strong><p>{html.escape(str(method['semantics']))}</p><p class="small">{len(layers)} 个独立 Layer 页面；页面之间不聚合。</p></div><span class="arrow">→</span></a>"""
        )
    doc = page_header(
        "RawData1 五算法入口",
        "根页只做算法入口；进入算法后选择 Layer，再进入该 Layer 的独立页面。统计只在同一层的 Round 11–20 内求均值。",
    )
    contract = payload["telemetry_contract"]
    if contract["fresh_visualization_raw"]:
        evidence = (
            "<section class=\"panel pass\"><h2>证据状态：完整 raw schema v4</h2>"
            "<p class=\"note\">逐 microbatch rank rows、CUDA-event 时间线、A/M observation 与逐专家 Weight chunk 均来自本次运行；"
            "报告不会从合计值反推缺失数据。</p></section>"
        )
    else:
        missing = ", ".join(
            field
            for field in (
                "microbatch_rank_samples",
                "microbatch_timeline",
                "probeep_observations",
                "probeep_weight_chunks",
            )
            if not contract[field]
        )
        evidence = (
            "<section class=\"panel warning\"><h2>证据状态：历史 raw，不满足当前正式验收口径</h2>"
            f"<p class=\"note\">缺失：{html.escape(missing or 'raw schema v4')}。"
            "本报告只展示可守恒验证的字段；重建值会标出来源，缺失时间线、observation 或逐专家 chunk 不会被补造。"
            "该历史 run 不能替代修复后的 H20 多机重跑。</p></section>"
        )
    doc += (
        f"""<main>{evidence}<section class="panel"><h2>固定口径</h2><p class="note">2 台 H20 × 8 GPU，EP16，E256，4096 tokens/rank，TopK=8；每层每轮 {payload['global_assignments']:,} expert-route rows。warmup Round 1–10 丢弃；正式计时 Round 11–20 作为同一 case 的重复测量。</p></section>{''.join(cards)}</main></body></html>"""
    )
    path.write_text(doc, encoding="utf-8")


def layer_table(layers: list[dict[str, object]]) -> str:
    cards = []
    for layer in layers:
        layer_id = f"{int(layer['layer']):02d}"
        cards.append(
            f"""<a class="algorithm" href="layers/layer_{layer_id}.html"><div><strong>Layer {layer_id}</strong><p class="small">单层独立数据：严格时间线/DAG、Microbatch 1、Microbatch 2 与 Round 11–20。</p></div><span class="arrow">→</span></a>"""
        )
    return "".join(cards)


def round_table(rounds: list[dict[str, object]]) -> str:
    rows = []
    for row in rounds:
        rows.append(
            "<tr>"
            f"<td>Round {row['round']}</td>"
            f"<td>{row['e2e_ms']:.6f}</td>"
            f"<td>{row['dispatch_ms']:.6f}</td>"
            f"<td>{row['expert_ms']:.6f}</td>"
            f"<td>{row['combine_ms']:.6f}</td>"
            f"<td>{row['plan_ms']:.6f}</td>"
            f"<td>{row['layout_ms']:.6f}</td>"
            f"<td>{row['weight_ms']:.6f}</td>"
            f"<td>{row['tokens_per_second']:.3f}</td>"
            "</tr>"
        )
    return (
        "<div class=\"tablewrap\"><table><thead><tr><th>物理轮次</th><th>E2E ms</th>"
        "<th>Dispatch ms</th><th>Expert ms</th><th>Combine ms</th><th>Plan ms</th>"
        "<th>Layout ms</th><th>Weight ms</th><th>tokens/s</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def format_bytes(value: float) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.2f} {unit}" if unit != "B" else f"{amount:.0f} B"
        amount /= 1024
    raise AssertionError("unreachable")


def pipeline_section(
    layer: dict[str, object], prefix: str, slug: str
) -> str:
    timeline = layer["timeline"]
    measured = ""
    if timeline:
        measured = (
            "<section class=\"panel\"><h2>严格时间戳双 stream 流水线</h2>"
            "<p class=\"note\">每个 start_ms/end_ms 都由 CUDA event 实测，并相对本 iteration 的 E2E 起点；"
            "选择本 Layer 的 Round 和 rank。时间条按实测时间缩放，不使用 phase 均值拼接。"
            "kernel 级嵌套与硬件计数器仍以同次运行保存的 nsys report 为准。</p>"
            f"<div class=\"pipeline-controls\"><label>Round<select id=\"timeline-round-{prefix}\"></select></label>"
            f"<label>Rank<select id=\"timeline-rank-{prefix}\"></select></label></div>"
            f"<div class=\"pipeline-scroll\"><svg id=\"timeline-{prefix}\" class=\"pipeline-svg\"></svg></div></section>"
        )
    else:
        nsys_note = (
            "本次 run 另有 nsys report；kernel 级时间请直接查看对应 report。"
            if layer["telemetry_contract"]["nsys_reports"]
            else "本次 run 也没有保存 nsys report。"
        )
        measured = (
            "<section class=\"panel\"><h2>严格时间戳双 stream 流水线</h2>"
            "<div class=\"empty\">本次历史 run 没有 microbatch_timeline.csv；"
            "因此不画伪时间线。新 runner 已按 Poseidon DAG_TASK_START/DAG_TASK_DONE 的同一口径，"
            f"补充逐 Layer、逐 Round、逐 rank 的 start_ms/end_ms。{nsys_note}</div></section>"
        )
    return (
        measured
        + "<section class=\"panel\"><h2>双 microbatch / 双 stream 事件依赖 DAG</h2>"
        "<p class=\"note\">同一 Layer 内，MB0 与 MB1 共用一个 compute stream 和一个 communication stream。"
        "灰线是同 stream 提交顺序，深色曲线是跨 stream event wait，点线是本轮完成/host 边界，"
        "虚线是 Obs[k]→Ctrl[k+1] 的跨轮 feedback。DAG 不冒充时间轴；严格时长只看上面的实测时间线。</p>"
        f"<div class=\"pipeline-scroll\"><svg id=\"dag-{prefix}\" class=\"pipeline-svg\"></svg></div></section>"
    )


def rail_microbatch_section(
    layer: dict[str, object], prefix: str, microbatch: int
) -> str:
    display_index = microbatch + 1
    compute = "attention" if microbatch == 0 else "moe"
    if not layer["rails"]:
        return (
            f"<section class=\"panel\"><h2>Microbatch {display_index} · RDMA rail/NIC 负载</h2>"
            "<div class=\"empty\">本次 raw log 没有保存该算法的逐 microbatch rail telemetry；"
            "不从合计值反推，也不复用其他算法的数据。</div></section>"
        )
    profile = layer["rail_profile"]
    layer_id = f"{int(layer['layer']):02d}"
    pair_rows = [row for row in profile["server_pairs"] if row["compute"] == compute]
    rail_rows = [row for row in profile["rails"] if row["compute"] == compute]
    mean_weight_bytes = sum(float(row["mean_weight_bytes"]) for row in pair_rows)
    has_exact_expert_colors = any(row.get("weight_components") for row in rail_rows)
    weight_note = (
        f"当前 microbatch 每轮平均 Expert Weight={format_bytes(mean_weight_bytes)}；"
        + (
            "每个实际传输专家使用独立颜色。"
            if has_exact_expert_colors
            else "历史 raw 只有聚合字节，使用青色段且不伪造 expert→rail 映射。"
        )
        if mean_weight_bytes > 0
        else "当前 microbatch 的实际 weight_bytes=0；admitted/cache-hit expert 在上方 placement 表显示，不混入传输柱。"
    )

    def table_rows(rows: list[dict[str, object]], *, per_rail: bool) -> str:
        rendered = []
        for row in rows:
            rail_cap = float(row.get("rail_bandwidth_gbps", 0.0))
            rail_cells = (
                f"<td>Rail {row['rail']}</td>"
                f"<td>NIC {row['physical_nic']} / subrail {row['subrail']}</td>"
                f"<td>{f'{rail_cap:.0f} Gbps' if rail_cap > 0 else 'not recorded'}</td>"
                f"<td>{row['path_id']}</td>"
                f"<td>R{row['src']}</td><td>R{row['dst']}</td>"
                if per_rail
                else ""
            )
            rendered.append(
                "<tr>"
                f"<td>S{row['source_server']} → S{row['destination_server']}</td>"
                f"{rail_cells}"
                f"<td>{format_bytes(float(row['mean_dispatch_bytes']))}</td>"
                f"<td>{format_bytes(float(row['mean_weight_bytes']))}</td>"
                f"<td>{format_bytes(float(row['mean_tx_bytes']))}</td>"
                f"<td>{format_bytes(float(row['mean_rx_bytes']))}</td>"
                f"<td>{format_bytes(float(row['min_tx_bytes']))}–{format_bytes(float(row['max_tx_bytes']))}</td>"
                f"<td>{float(row['mean_chunks']):.2f} / {int(row['max_chunks'])}</td>"
                "</tr>"
            )
        return "".join(rendered)

    pair_table = (
        "<div class=\"tablewrap\"><table class=\"transport-table\"><thead><tr>"
        "<th>Direction</th><th>Dispatch mean</th><th>Expert Weight mean</th>"
        "<th>TX mean</th><th>RX mean</th><th>TX min–max</th><th>Chunks mean/max</th>"
        "</tr></thead><tbody>"
        + table_rows(pair_rows, per_rail=False)
        + "</tbody></table></div>"
    )
    rail_table = (
        "<div class=\"tablewrap\"><table class=\"transport-table\"><thead><tr>"
        "<th>Direction</th><th>Rail</th><th>Physical NIC / subrail</th><th>Rail cap</th>"
        "<th>Path ID</th><th>Source rank</th><th>Destination rank</th>"
        "<th>Dispatch mean</th><th>Expert Weight mean</th><th>TX mean</th><th>RX mean</th>"
        "<th>TX min–max</th><th>Chunks mean/max</th></tr></thead><tbody>"
        + table_rows(rail_rows, per_rail=True)
        + "</tbody></table></div>"
    )
    return (
        f"<section class=\"panel\"><h2>Microbatch {display_index} · Directed server-pair 负载</h2>"
        "<p class=\"note\">只包含当前 microbatch。每根柱是本 Layer Round 11–20 的每轮平均发送 bytes；"
        "柱宽严格等于 Token Dispatch bytes + Expert Weight bytes；Expert 段从 Token 段末尾继续，绝不覆盖。"
        f"{weight_note}</p>"
        "<div class=\"legend\"><span><i class=\"swatch\" style=\"background:#e28200\"></i>Token Dispatch bytes</span>"
        "<span><i class=\"swatch\" style=\"background:#1098a3\"></i>Expert Weight（新 raw 逐专家着色；历史 raw 为青色聚合）</span></div>"
        f"<div class=\"transport-chart\"><svg id=\"rail-pair-{prefix}-{microbatch}\"></svg></div>"
        f"<details class=\"transport-details\"><summary>Directed server-pair data table</summary>{pair_table}</details></section>"
        f"<section class=\"panel\"><h2>Microbatch {display_index} · Per-rail/NIC directed 负载</h2>"
        "<p class=\"note\">只包含当前 microbatch。每台服务器 4 个 400-Gbps 物理 NIC；每个物理口拆成两个 200-Gbps subrail，"
        "Rail=源 rank 的 server-local index，NIC=floor(Rail/2)。Path ID 是 ProbeEP runtime 的有向路径编号。"
        "bytes 来自算法/runtime telemetry，不冒充 mlx5 硬件计数；没有 active time 时不反推实测 Gbps。</p>"
        "<div class=\"legend\"><span><i class=\"swatch\" style=\"background:#e28200\"></i>Token Dispatch bytes</span>"
        "<span><i class=\"swatch\" style=\"background:#1098a3\"></i>Expert Weight（新 raw 逐专家着色；历史 raw 为青色聚合）</span></div>"
        f"<div class=\"transport-chart\"><svg id=\"rail-path-{prefix}-{microbatch}\"></svg></div>"
        f"<details class=\"transport-details\"><summary>Per-rail directed data table</summary>{rail_table}</details>"
        f"<p class=\"note\"><a href=\"data/layer_{layer_id}/rdma_path_load.csv\">下载本 Layer 的逐 Round rail raw CSV</a></p></section>"
    )


def microbatch_block(
    layer: dict[str, object], prefix: str, record: dict[str, object]
) -> str:
    microbatch = int(record["microbatch"])
    display_index = microbatch + 1
    observation = (
        "Attention observation"
        if str(record["observation"]) == "attention"
        else "MoE observation"
    )
    if record["home_available"]:
        home_panel = (
            f"<div class=\"panel\"><h2>Microbatch {display_index} · 原始 home expert rows</h2>"
            f"<svg id=\"mb-home-{prefix}-{microbatch}\" class=\"chart\"></svg></div>"
        )
    else:
        home_panel = (
            f"<div class=\"panel\"><h2>Microbatch {display_index} · 原始 home expert rows</h2>"
            "<div class=\"empty\">历史 raw log 没有保存该 microbatch 的原始 rank rows；不从合计值反推。</div></div>"
        )
    if record["available"]:
        execution_panel = (
            f"<div class=\"panel\"><h2>Microbatch {display_index} · 调度后 grouped-GEMM rows</h2>"
            f"<svg id=\"mb-exec-{prefix}-{microbatch}\" class=\"chart\"></svg>"
            f"<p class=\"note\">source={html.escape(str(record['source']))}。</p></div>"
        )
    else:
        execution_panel = (
            f"<div class=\"panel\"><h2>Microbatch {display_index} · 调度后 grouped-GEMM rows</h2>"
            "<div class=\"empty\">历史 raw log 没有保存该算法的逐 microbatch execution rows；"
            "不能从两个 microbatch 的合计值唯一反推。新 runner 已补充独立记录。</div></div>"
        )
    return (
        "<section class=\"microbatch-block\">"
        f"<div class=\"panel microbatch-title\"><h2>Microbatch {display_index}（runtime MB{microbatch}）</h2>"
        f"<p class=\"note\">当前规划链={observation}；2048 tokens/rank，TopK=8，"
        "全局 262,144 expert-route rows。本区块不混入另一个 microbatch。</p></div>"
        f"<div class=\"grid2\">{home_panel}{execution_panel}</div>"
        f"{probe_microbatch_control_section(layer, microbatch)}"
        f"{rail_microbatch_section(layer, prefix, microbatch)}"
        "</section>"
    )


def microbatch_section(layer: dict[str, object], prefix: str) -> str:
    return "".join(
        microbatch_block(layer, prefix, record)
        for record in layer["microbatches"]
    )


def probe_microbatch_control_section(
    layer: dict[str, object], microbatch: int
) -> str:
    compute_name = "attention" if microbatch == 0 else "moe"
    display_index = microbatch + 1
    observations = [
        row
        for row in layer.get("observations", [])
        if row["compute_name"] == compute_name
    ]
    plans = [
        row
        for row in layer.get("plans", [])
        if int(row["dispatch_compute_kind"]) == microbatch
    ]
    chunks = [
        row
        for row in layer.get("weight_chunks", [])
        if int(row["dispatch_compute_kind"]) == microbatch
    ]
    if not observations and not plans and not chunks:
        return ""

    observation_groups: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in observations:
        observation_groups[int(row["iteration"])].append(row)
    plan_index = {
        int(row["iteration"]): row
        for row in plans
    }
    control_rows = []
    for iteration in EXPECTED_MEASURED_ITERATIONS:
        samples = observation_groups.get(iteration, [])
        plan = plan_index.get(iteration, {})
        counts = list(plan.get("plan_counts", []))
        cycles = dict(plan.get("planner_phase_cycles", {}))
        placements = admitted_placements(plan)
        placement_text = ", ".join(
            f"E{item['expert_id']}→S{item['destination_server']}"
            for item in placements
        ) or "—"
        deferred_text = ", ".join(
            f"E{int(expert)}" for expert in plan.get("deferred_experts", [])
        ) or "—"
        transfer_required = int(plan.get("weight_transfer_required", 0))
        producer_layers = sorted({
            int(row["producer_layer_id"])
            for row in samples
            if str(row.get("producer_layer_id", "")).strip()
        })
        if producer_layers:
            producer_text = ", ".join(
                f"Layer {value:02d}" for value in producer_layers
            )
        elif samples:
            producer_text = "legacy / provenance unavailable"
        else:
            producer_text = (
                "bootstrap" if int(layer["layer"]) == 0 else "missing"
            )
        cache_state = (
            "cache miss / transfer"
            if placements and transfer_required
            else ("cache hit / 0 B" if placements else "no placement / 0 B")
        )
        before = " / ".join(f"{int(value):,}" for value in plan.get("server_load_before", [])) or "—"
        after = " / ".join(f"{int(value):,}" for value in plan.get("server_load_after", [])) or "—"
        control_rows.append(
            "<tr>"
            f"<td>Round {iteration + ROUND_OFFSET}</td>"
            f"<td>{producer_text}</td>"
            f"<td>{before}</td><td>{after}</td>"
            f"<td>{html.escape(placement_text)}</td>"
            f"<td>{html.escape(deferred_text)}</td>"
            f"<td>{cache_state}</td>"
            f"<td>{int(counts[1]) if len(counts) > 1 else len(plan.get('chunk_table', []))}</td>"
            f"<td>{sum(int(value) for value in cycles.values()):,}</td>"
            f"<td>{max((int(row['compute_ns']) for row in samples), default=0):,}</td>"
            f"<td>{max((int(row['network_ns']) for row in samples), default=0):,}</td>"
            "</tr>"
        )

    expert_groups: dict[tuple[int, int, int, int], dict[str, int]] = defaultdict(
        lambda: {"planned_bytes": 0, "actual_bytes": 0, "chunks": 0}
    )
    for row in chunks:
        key = (
            int(row["iteration"]),
            int(row["rail"]),
            int(row["expert_id"]),
            int(row["destination_server"]),
        )
        chunk_bytes = int(row["chunk_bytes"])
        expert_groups[key]["planned_bytes"] += chunk_bytes
        expert_groups[key]["actual_bytes"] += (
            chunk_bytes * int(row["transfer_required"])
        )
        expert_groups[key]["chunks"] += 1
    expert_rows = []
    for (iteration, rail, expert, destination), values in sorted(
        expert_groups.items(), key=lambda item: (item[0][0], item[0][1:])
    ):
        hue = (expert * 137.508) % 360
        expert_rows.append(
            "<tr>"
            f"<td>Round {iteration + ROUND_OFFSET}</td>"
            f"<td>Rail {rail}</td><td>S{destination}</td>"
            f"<td><i class=\"swatch\" style=\"background:hsl({hue:.1f} 65% 48%)\"></i> E{expert}</td>"
            f"<td>{values['chunks']}</td>"
            f"<td>{format_bytes(values['planned_bytes'])}</td>"
            f"<td>{format_bytes(values['actual_bytes'])}</td>"
            "</tr>"
        )
    layer_id = f"{int(layer['layer']):02d}"
    observation_link = (
        f"<a href=\"data/layer_{layer_id}/probeep_observation_samples.csv\">A/M raw CSV</a> · "
        if observations
        else ""
    )
    had_weight_bytes = any(
        row["compute"] == compute_name and int(row["weight_bytes"]) > 0
        for row in layer["rails"]
    )
    placements_exist = any(admitted_placements(plan) for plan in plans)
    empty_chunk_text = (
        "历史 raw 只有 aggregate Weight bytes，缺逐 expert chunk table；可显示 admitted expert，但不能补造其 rail/chunk 颜色。"
        if had_weight_bytes and not chunks
        else (
            "历史 raw 未保存逐 chunk 表；admitted placement 命中缓存，本轮实际 Weight bytes=0。"
            if placements_exist and not chunks
            else "本 microbatch 没有 expert admission，也没有 Weight 传输。"
        )
    )
    chunk_panel = (
        f"<section class=\"panel\"><h2>Microbatch {display_index} · Expert placement / Weight chunks</h2>"
        "<p class=\"note\">admitted placement 与实际 Weight 传输分开显示。cache hit 时专家仍参与跨服务器均衡，"
        "但 actual bytes 必须是 0；不能把逻辑专家画进 activation bytes。表中每行只属于一个正式 Round，"
        "不把十轮累计值伪装成单轮流量；rail 图再对这十轮取同层均值。</p>"
        "<div class=\"tablewrap\"><table><thead><tr><th>Round</th><th>Rail</th><th>Destination</th>"
        "<th>Expert</th><th>Chunk records</th><th>Planned bytes</th><th>Actual bytes</th></tr></thead><tbody>"
        + ("".join(expert_rows) if expert_rows else f"<tr><td colspan=\"7\">{empty_chunk_text}</td></tr>")
        + "</tbody></table></div>"
        f"<p class=\"note\"><a href=\"data/layer_{layer_id}/probeep_weight_chunks.csv\">下载逐 chunk raw CSV</a></p></section>"
    )
    return (
        f"<section class=\"panel\"><h2>Microbatch {display_index} · previous-layer {compute_name} feedback → CUDA planner</h2>"
        "<p class=\"note\">MB0 Dispatch 消费上一层的 Attention-chain（MB1 Attention overlap），MB1 Dispatch 消费上一层的 MoE-chain（MB0 Expert overlap）。"
        "Layer 00 明确使用 bootstrap，不伪造 observation。server load、admitted placement、cache 状态与 planner cycles 均来自 production handle；"
        "不根据均衡后的柱图反推专家。</p>"
        "<div class=\"tablewrap\"><table><thead><tr><th>Round</th><th>Feedback source</th>"
        "<th>Server load before</th><th>Server load after</th><th>Admitted placement</th>"
        "<th>Deferred</th><th>Weight state</th><th>Planned chunks</th><th>Planner cycles</th>"
        "<th>compute max ns</th><th>network max ns</th>"
        "</tr></thead><tbody>"
        + (
            "".join(control_rows)
            if plans
            else "<tr><td colspan=\"11\">本次 raw 未保存当前 microbatch 的 production plan。</td></tr>"
        )
        + "</tbody></table></div>"
        f"<p class=\"note\">{observation_link}"
        f"<a href=\"data/layer_{layer_id}/probeep_plan_summary.jsonl\">plan raw JSONL</a></p></section>"
        + chunk_panel
    )


def layer_section(layer: dict[str, object], slug: str) -> str:
    layer_id = f"{int(layer['layer']):02d}"
    prefix = f"{slug}-{layer_id}"
    summary = layer["summary"]
    stable_text = "稳定" if layer["execution_stable"] else "有变化"
    stable_class = "stable" if layer["execution_stable"] else "unstable"
    raw_mismatch_note = ""
    if layer["raw_exec_mismatch_iterations"]:
        raw_mismatch_note = (
            f" raw exec_load 与 home_load 不一致的 iteration："
            f"{','.join(str(x) for x in layer['raw_exec_mismatch_iterations'])}；"
            f"本页 execution load 来源为 {html.escape(str(layer['execution_source']))}。"
        )
    contract = layer["telemetry_contract"]
    if contract["fresh_visualization_raw"]:
        evidence = (
            '<section class="panel pass"><h2>本层证据：完整 raw schema v4</h2>'
            '<p class="note">microbatch rank、CUDA-event 时间线、A/M observation、rail 与逐专家 '
            'Weight chunk 均来自同一次运行，并已通过跨文件守恒检查。</p></section>'
        )
    else:
        missing = ", ".join(
            field
            for field in (
                "microbatch_rank_samples",
                "microbatch_timeline",
                "probeep_observations",
                "probeep_weight_chunks",
            )
            if not contract[field]
        )
        evidence = (
            '<section class="panel warning"><h2>本层证据：历史 raw，不可作为正式 H20 结果</h2>'
            f'<p class="note">缺失：{html.escape(missing or "raw schema v4")}。'
            '本页只展示 raw 中存在且可守恒验证的内容；不补造时间戳、observation 或专家 chunk。'
            '</p></section>'
        )
    return f"""
{evidence}
<section class="panel layer" id="layer-{layer_id}">
  <div class="layer-title"><h2>Layer {layer_id}</h2><a href="../algorithm_dashboard.html">返回 Layer 目录</a></div>
  <div class="metrics">
    <div class="metric"><b>{float(layer['home_rank_ratio']):.4f}</b><span>MB0+MB1 输入 rank max/mean</span></div>
    <div class="metric"><b>{float(layer['execution_rank_ratio']):.4f}</b><span>MB0+MB1 执行 rank max/mean</span></div>
    <div class="metric"><b>{float(layer['execution_server_ratio']):.4f}</b><span>MB0+MB1 执行 server max/mean</span></div>
    <div class="metric"><b>{summary['e2e_mean_ms']:.3f} ms</b><span>Round 11-20 E2E mean</span></div>
    <div class="metric"><b class="{stable_class}">{stable_text}</b><span>十轮 execution layout</span></div>
  </div>
  <p class="note">本层 routing SHA-256：<code>{html.escape(str(layer['routing_sha256']))}</code>。load 柱图是同层十轮的 rank-wise mean；若 layout 稳定，它等价于任意一轮的精确 load。{raw_mismatch_note}</p>
</section>
{pipeline_section(layer, prefix, slug)}
{microbatch_section(layer, prefix)}
<section class="panel">
  <h2>Layer {layer_id} · Round 11-20 原始时延</h2>
  <div class="legend"><span><i class="swatch" style="background:#1d5e9e"></i>E2E</span><span><i class="swatch" style="background:#ea580c"></i>Dispatch</span><span><i class="swatch" style="background:#16a34a"></i>Expert compute</span><span><i class="swatch" style="background:#7c3aed"></i>Combine</span></div>
  <svg id="latency-{prefix}" class="chart"></svg>
  <p class="note">E2E mean={summary['e2e_mean_ms']:.6f} ms，p50={summary['e2e_p50_ms']:.6f} ms，min-max={summary['e2e_min_ms']:.6f}-{summary['e2e_max_ms']:.6f} ms。phase 可能 overlap，不能相加还原 E2E。</p>
</section>
<section class="panel"><h2>Layer {layer_id} · 十轮明细</h2>{round_table(layer['rounds'])}</section>
"""


def write_method(path: Path, payload: dict[str, object], method: dict[str, object]) -> None:
    layers = list(method["layers"])
    doc = page_header(
        f"{method['label']} · {len(layers)} 层目录",
        f"{method['semantics']} 选择一个 Layer 后进入该层的独立页面。",
    )
    doc += (
        f"""<main><a class="back" href="../../algorithm_comparison.html">← 返回五算法入口</a>
<section class="panel"><h2>Layer 目录</h2><p class="note">{len(layers)} 个页面彼此独立；不在目录页展示或聚合性能数值。</p></section>
{layer_table(layers)}</main></body></html>"""
    )
    path.write_text(doc, encoding="utf-8")


def write_layer(path: Path, method: dict[str, object], layer: dict[str, object]) -> None:
    slug = str(method["slug"])
    layer_id = f"{int(layer['layer']):02d}"
    prefix = f"{slug}-{layer_id}"
    browser_layer = {
        key: value
        for key, value in layer.items()
        if key not in {"expert_matrix", "padded_matrix"}
    }
    doc = page_header(
        f"{method['label']} · Layer {layer_id}",
        "本页只包含这一个 Layer；Round 11–20 是该 Layer 的十次正式重复测量。",
    )
    doc += f"""<main>{layer_section(layer, slug)}</main><script>{CHART_JS}
const layer={js(browser_layer)};
const prefix={js(prefix)};
for (const mb of layer.microbatches) {{
  if (mb.home_available) rankBars(`mb-home-${{prefix}}-${{mb.microbatch}}`, mb.home, '#737d8d');
  if (mb.available) rankBars(`mb-exec-${{prefix}}-${{mb.microbatch}}`, mb.execution, mb.microbatch===0?'#2f74b5':'#109873');
}}
pipelineDag('dag-'+prefix, {js(slug)}, Number(layer.layer));
if (layer.timeline.length) measuredTimeline(
  'timeline-'+prefix,
  layer.timeline,
  'timeline-round-'+prefix,
  'timeline-rank-'+prefix
);
lineChart('latency-'+prefix, [
  {{values:layer.rounds.map(x=>x.e2e_ms)}},
  {{values:layer.rounds.map(x=>x.dispatch_ms)}},
  {{values:layer.rounds.map(x=>x.expert_ms)}},
  {{values:layer.rounds.map(x=>x.combine_ms)}}
]);
if(layer.rails.length) {{
  for (const mb of [0,1]) {{
    const compute=mb===0?'attention':'moe', display=mb+1;
    const pairs=layer.rail_profile.server_pairs.filter(x=>x.compute===compute);
    const paths=layer.rail_profile.rails.filter(x=>x.compute===compute);
    directedLoadChart(`rail-pair-${{prefix}}-${{mb}}`, pairs,
      x=>`Microbatch ${{display}} · S${{x.source_server}}→S${{x.destination_server}}`);
    directedLoadChart(`rail-path-${{prefix}}-${{mb}}`, paths,
      x=>`MB${{display}} · S${{x.source_server}}→S${{x.destination_server}} / NIC${{x.physical_nic}}.${{x.subrail}} / Rail${{x.rail}} / R${{x.src}}→R${{x.dst}}`);
  }}
}}
</script></body></html>"""
    path.write_text(doc, encoding="utf-8")


def write_bundle(output: Path, payload: dict[str, object]) -> Path:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    root = output / "algorithm_comparison.html"
    write_root(root, payload)
    manifest = {
        key: value
        for key, value in payload.items()
        if key != "methods"
    }
    manifest["methods"] = [
        {
            "system": method["system"],
            "balance": method["balance"],
            "label": method["label"],
            "slug": method["slug"],
            "semantics": method["semantics"],
            "layer_pages": [
                f"algorithms/{method['slug']}/layers/layer_{int(layer['layer']):02d}.html"
                for layer in method["layers"]
            ],
        }
        for method in payload["methods"]
    ]
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for method in payload["methods"]:
        method_dir = output / "algorithms" / str(method["slug"])
        method_dir.mkdir(parents=True)
        write_method(method_dir / "algorithm_dashboard.html", payload, method)
        layer_dir = method_dir / "layers"
        layer_dir.mkdir()
        for layer in method["layers"]:
            layer_id = f"{int(layer['layer']):02d}"
            write_layer(
                layer_dir / f"layer_{layer_id}.html",
                method,
                layer,
            )
            data_dir = layer_dir / "data" / f"layer_{layer_id}"
            data_dir.mkdir(parents=True)
            summary = layer["summary"]
            (data_dir / "result.json").write_text(
                json.dumps(
                    {
                        "schema": "probeep.raw_data1.algorithm_single_layer.v1",
                        "run_id": payload["run_id"],
                        "method": {
                            "system": method["system"],
                            "balance": method["balance"],
                            "label": method["label"],
                            "slug": method["slug"],
                            "semantics": method["semantics"],
                        },
                        "layer": layer,
                        "physical_rounds": payload["physical_rounds"],
                        "warmup_rounds": payload["warmup_rounds"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            write_csv(
                data_dir / "summary.csv",
                [{
                    "layer": layer["layer"],
                    "home_rank_max_mean": f"{layer['home_rank_ratio']:.8f}",
                    "execution_rank_max_mean": f"{layer['execution_rank_ratio']:.8f}",
                    "execution_server_max_mean": f"{layer['execution_server_ratio']:.8f}",
                    "execution_layout_stable": layer["execution_stable"],
                    "raw_execution_layout_stable": layer["raw_execution_stable"],
                    "raw_exec_mismatch_iteration_count": len(layer["raw_exec_mismatch_iterations"]),
                    "execution_source": layer["execution_source"],
                    "e2e_mean_ms": f"{summary['e2e_mean_ms']:.8f}",
                    "e2e_p50_ms": f"{summary['e2e_p50_ms']:.8f}",
                    "e2e_min_ms": f"{summary['e2e_min_ms']:.8f}",
                    "e2e_max_ms": f"{summary['e2e_max_ms']:.8f}",
                    "dispatch_mean_ms": f"{summary['dispatch_mean_ms']:.8f}",
                    "expert_mean_ms": f"{summary['expert_mean_ms']:.8f}",
                    "combine_mean_ms": f"{summary['combine_mean_ms']:.8f}",
                }],
                [
                    "layer",
                    "home_rank_max_mean",
                    "execution_rank_max_mean",
                    "execution_server_max_mean",
                    "execution_layout_stable",
                    "raw_execution_layout_stable",
                    "raw_exec_mismatch_iteration_count",
                    "execution_source",
                    "e2e_mean_ms",
                    "e2e_p50_ms",
                    "e2e_min_ms",
                    "e2e_max_ms",
                    "dispatch_mean_ms",
                    "expert_mean_ms",
                    "combine_mean_ms",
                ],
            )
            round_rows = []
            for row in layer["rounds"]:
                record = dict(row)
                record["layer"] = layer["layer"]
                round_rows.append(record)
            write_csv(
                data_dir / "measured_rounds.csv",
                round_rows,
                [
                    "layer",
                    "iteration",
                    "round",
                    "e2e_ms",
                    "dispatch_ms",
                    "expert_ms",
                    "combine_ms",
                    "plan_ms",
                    "count_exchange_ms",
                    "layout_ms",
                    "weight_ms",
                    "tokens_per_second",
                ],
            )
            rank_rows = []
            for rank in range(NUM_RANKS):
                rank_rows.append(
                    {
                        "layer": layer["layer"],
                        "rank": rank,
                        "server": rank // SERVER_SIZE,
                        "home_rows_mean": f"{layer['home'][rank]:.6f}",
                        "execution_rows_mean": f"{layer['execution'][rank]:.6f}",
                    }
                )
            write_csv(
                data_dir / "rank_load.csv",
                rank_rows,
                ["layer", "rank", "server", "home_rows_mean", "execution_rows_mean"],
            )
            microbatch_rows = []
            for microbatch in layer["microbatches"]:
                if not (
                    microbatch["home_available"] or microbatch["available"]
                ):
                    continue
                for rank in range(NUM_RANKS):
                    microbatch_rows.append(
                        {
                            "layer": layer["layer"],
                            "microbatch": microbatch["microbatch"],
                            "observation": microbatch["observation"],
                            "rank": rank,
                            "server": rank // SERVER_SIZE,
                            "home_rows_mean": (
                                f"{microbatch['home'][rank]:.6f}"
                                if microbatch["home_available"]
                                else ""
                            ),
                            "execution_rows_mean": (
                                f"{microbatch['execution'][rank]:.6f}"
                                if microbatch["available"]
                                else ""
                            ),
                            "source": microbatch["source"],
                        }
                    )
            if microbatch_rows:
                write_csv(
                    data_dir / "microbatch_rank_load.csv",
                    microbatch_rows,
                    [
                        "layer",
                        "microbatch",
                        "observation",
                        "rank",
                        "server",
                        "home_rows_mean",
                        "execution_rows_mean",
                        "source",
                    ],
                )
            if layer["timeline"]:
                timeline_rows = []
                for row in layer["timeline"]:
                    record = dict(row)
                    record["layer"] = layer["layer"]
                    timeline_rows.append(record)
                write_csv(
                    data_dir / "microbatch_timeline.csv",
                    timeline_rows,
                    [
                        "layer",
                        "round",
                        "rank",
                        "microbatch",
                        "stream",
                        "stage",
                        "start_ms",
                        "end_ms",
                        "duration_ms",
                    ],
                )
            if layer["rails"]:
                rail_export_fields = ["layer", *RAIL_EXPORT_FIELDS]
                rail_rows = []
                for row in layer["rails"]:
                    record = {field: row.get(field, "") for field in RAIL_EXPORT_FIELDS}
                    record["layer"] = layer["layer"]
                    rail_rows.append(record)
                write_csv(
                    data_dir / "rdma_path_load.csv",
                    rail_rows,
                    rail_export_fields,
                )
            if layer["observations"]:
                write_csv(
                    data_dir / "probeep_observation_samples.csv",
                    list(layer["observations"]),
                    list(layer["observations"][0].keys()),
                )
            if method["slug"] == "probeep":
                write_csv(
                    data_dir / "probeep_weight_chunks.csv",
                    list(layer["weight_chunks"]),
                    PROBE_CHUNK_EXPORT_FIELDS,
                )
            if layer["plans"]:
                (data_dir / "probeep_plan_summary.jsonl").write_text(
                    "".join(
                        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                        for row in layer["plans"]
                    ),
                    encoding="utf-8",
                )
    (output / "README.md").write_text(
        "# RawData1 algorithm-first layer report\n\n"
        f"- input run: `{payload['run_id']}`\n"
        f"- evidence: `{'complete raw schema v4' if payload['telemetry_contract']['fresh_visualization_raw'] else 'historical/incomplete raw; not formal evidence'}`\n"
        f"- layers: `{payload['layers'][0]}`..`{payload['layers'][-1]}` ({len(payload['layers'])} layers)\n"
        "- warmup: physical Round 1..10, not included\n"
        "- measured: physical Round 11..20, ten repeated measurements per algorithm/layer case\n"
        "- latency: mean/min/max are computed only across the ten measured iterations inside one layer\n"
        "- pipeline: strict start/end timestamps first, then the event-dependency DAG; absent timestamps are never inferred\n"
        "- load: Microbatch 1 and Microbatch 2 are rendered as independent original/execution and rail blocks; no combined bars\n"
        "- traffic: Dispatch activation and Expert Weight chunks are independently colored and byte-stacked\n"
        "- navigation: `algorithm_comparison.html` -> algorithm -> one standalone page per layer\n"
        "- data: every layer owns `layers/data/layer_XX/`; no cross-layer performance CSV is generated\n",
        encoding="utf-8",
    )
    zip_path = output.parent / f"{output.name}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for file in sorted(output.rglob("*")):
            if file.is_file():
                archive.write(file, file.relative_to(output))
    with zipfile.ZipFile(zip_path) as archive:
        broken = archive.testzip()
        if broken is not None:
            raise RuntimeError(f"corrupt ZIP member: {broken}")
    return zip_path


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    available = available_layers(read_csv(run_dir / "raw" / "iterations.csv"))
    layers = parse_layers(args.layers, available)
    output = args.output_dir
    if output is None:
        first, last = layers[0], layers[-1]
        output = run_dir / "reprocessed" / f"raw_data1_layers_{first:02d}_{last:02d}_by_algorithm_rounds_11_20_mean"
    else:
        output = output.resolve()
    payload = make_payload(run_dir, layers)
    zip_path = write_bundle(output, payload)
    print(f"wrote {output / 'algorithm_comparison.html'}")
    print(f"wrote {zip_path}")
    unstable = []
    for method in payload["methods"]:
        for layer in method["layers"]:
            if not layer["execution_stable"]:
                unstable.append(f"{method['label']} layer {layer['layer']:02d}")
    if unstable:
        print("execution layout changed across measured iterations:")
        for item in unstable:
            print(f"  - {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
