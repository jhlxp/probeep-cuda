"""Versioned on-disk schemas shared by ProbeEP benchmark runners."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


SCHEMA_VERSION = 4

CASE_FIELDS = (
    "schema_version",
    "run_id",
    "slurm_job_id",
    "benchmark_scope",
    "runner_mode",
    "system",
    "balance",
    "direction",
    "workload",
    "bias_ratio",
    "seed",
    "repeat",
    "iteration",
    "routing_sha256",
)

ITERATION_FIELDS = CASE_FIELDS + (
    "global_tokens",
    "global_assignments",
    "expert_maxvio",
    "rank_maxvio_before",
    "rank_maxvio_after",
    "node_maxvio",
    "plan_max_ms",
    "count_exchange_max_ms",
    "layout_materialize_max_ms",
    "weight_prefetch_max_ms",
    "dispatch_max_ms",
    "expert_compute_max_ms",
    "combine_max_ms",
    "grad_reduce_max_ms",
    "e2e_max_ms",
    "tokens_per_second",
    "assignments_per_second",
)

RANK_SAMPLE_FIELDS = CASE_FIELDS + (
    "global_rank",
    "node_rank",
    "local_rank",
    "home_load",
    "exec_load",
    "replica_count",
    "moved_assignments",
    "prefetch_bytes",
    "valid_recv_rows",
    "padding_rows",
    "plan_ms",
    "count_exchange_ms",
    "layout_materialize_ms",
    "weight_prefetch_ms",
    "dispatch_ms",
    "expert_compute_ms",
    "combine_ms",
    "grad_reduce_ms",
    "e2e_ms",
)

EXPERT_SAMPLE_FIELDS = CASE_FIELDS + (
    "expert_id",
    "home_rank",
    "home_server",
    "receive_rows",
)

RANK_EXPERT_SAMPLE_FIELDS = CASE_FIELDS + (
    "global_rank",
    "node_rank",
    "local_rank",
    "expert_id",
    "raw_rows",
    "padded_rows",
)

MICROBATCH_RANK_SAMPLE_FIELDS = CASE_FIELDS + (
    "microbatch",
    "global_rank",
    "node_rank",
    "local_rank",
    "home_load",
    "exec_load",
    "padded_rows",
)

MICROBATCH_TIMELINE_FIELDS = CASE_FIELDS + (
    "global_rank",
    "node_rank",
    "local_rank",
    "microbatch",
    "logical_stream",
    "stage",
    "start_ms",
    "end_ms",
    "duration_ms",
)

PROBEEP_OBSERVATION_SAMPLE_FIELDS = CASE_FIELDS + (
    "producer_phase",
    "producer_iteration",
    "producer_layer_id",
    "producer_repeat",
    "consumer_iteration",
    "consumer_layer_id",
    "consumer_repeat",
    "compute_kind",
    "compute_name",
    "dispatch_microbatch",
    "overlap_microbatch",
    "global_rank",
    "node_rank",
    "local_rank",
    "compute_ns",
    "network_ns",
    "dispatch_tx_bytes",
    "dispatch_rx_bytes",
    "migration_tx_bytes",
    "migration_rx_bytes",
    "controller_alpha",
    "rdma_path_bandwidth_gbps",
)

PROBEEP_WEIGHT_CHUNK_FIELDS = CASE_FIELDS + (
    "dispatch_compute_kind",
    "dispatch_compute_name",
    "chunk_ordinal",
    "expert_id",
    "replica_id",
    "seed_rank",
    "expert_chunk_index",
    "source_server",
    "destination_server",
    "source_rank",
    "destination_rank",
    "physical_nic",
    "subrail",
    "rail_bandwidth_gbps",
    "physical_nic_bandwidth_gbps",
    "weight_cache_mode",
    "expert_weight_version",
    "expert_offset_bytes",
    "chunk_bytes",
    "rail",
    "source_path_offset_bytes",
    "destination_path_offset_bytes",
    "transfer_required",
)

RUN_SUMMARY_FIELDS = (
    "schema_version",
    "run_id",
    "benchmark_scope",
    "runner_mode",
    "system",
    "balance",
    "direction",
    "workload",
    "bias_ratio",
    "seed",
    "repeat",
    "num_iterations",
    "e2e_p99_ms",
    "e2e_max_ms",
    "throughput_at_p99_tokens_s",
    "planner_share_median_pct",
    "rank_maxvio_before_median",
    "rank_maxvio_after_median",
)

SUMMARY_FIELDS = (
    "schema_version",
    "run_id",
    "benchmark_scope",
    "runner_mode",
    "system",
    "balance",
    "direction",
    "workload",
    "bias_ratio",
    "num_runs",
    "num_iterations",
    "e2e_p99_ms",
    "e2e_max_ms",
    "throughput_at_p99_tokens_s",
    "speedup_vs_official",
    "overhead_vs_official_pct",
    "planner_share_median_pct",
    "expert_maxvio_median",
    "rank_maxvio_before_median",
    "rank_maxvio_after_median",
    "node_maxvio_median",
)


def append_csv_rows(
    path: str | Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output.exists()
    with output.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def write_csv_rows(
    path: str | Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: str | Path, record: Mapping[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_manifest(path: str | Path, manifest: Mapping[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": SCHEMA_VERSION, **manifest}
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
