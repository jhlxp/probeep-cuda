#!/usr/bin/env python3
"""CPU/static tests for the H20 launcher contract; no GPU or RDMA is used."""

from __future__ import annotations

import json
import hashlib
import importlib.util
import os
import subprocess
import sys
import re
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load_reprocessor():
    path = HERE / "reprocess_layers_by_algorithm.py"
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("test01_reprocessor", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_formal_benchmark():
    formal = HERE / "formal"
    path = formal / "run_benchmark.py"
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(formal))
    spec = importlib.util.spec_from_file_location("test01_formal_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def base_environment(tmp_path: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "NODE_RANK": "0",
            "NNODES": "2",
            "GPUS_PER_NODE": "8",
            "MASTER_ADDR": "h20-node-01",
            "MASTER_PORT": "29500",
            "PROBEEP_RUN_ID": "static-test",
            "PROBEEP_RUN_DIR": str(tmp_path),
            "NUM_EXPERTS": "256",
        }
    )
    return environment


def run_runner(tmp_path: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(HERE / "runner.py"), *arguments],
        cwd=ROOT,
        env=base_environment(tmp_path),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def test_readiness_reports_real_feedback_gap(tmp_path: Path) -> None:
    result = run_runner(tmp_path, "readiness")
    assert result.returncode == 0, result.stdout
    payload = json.loads((tmp_path / "readiness.json").read_text())
    assert payload["implemented"]["slurm_and_manual_launch"] is True
    assert payload["blocked"]["real_dynamic_observation_closed_loop"] is True
    assert payload["formal_performance_ready"] is False


def test_probeep_full_dry_run_has_three_real_cuda_tests(tmp_path: Path) -> None:
    result = run_runner(tmp_path, "probeep-full", "--dry-run")
    assert result.returncode == 0, result.stdout
    assert "test_balanced_internode.py" in result.stdout
    assert "test_balanced_expert_io.py" in result.stdout
    assert "test_balanced_backward.py" in result.stdout
    statuses = [json.loads(line) for line in (tmp_path / "raw/runner_status_node_0.jsonl").read_text().splitlines()]
    assert [item["state"] for item in statuses] == ["DRY_RUN"] * 3


def test_deepep_smoke_uses_dsv3_shape(tmp_path: Path) -> None:
    result = run_runner(tmp_path, "deepep-smoke", "--dry-run")
    assert result.returncode == 0, result.stdout
    assert "--num-tokens 4096" in result.stdout
    assert "--hidden 7168" in result.stdout
    assert "--num-topk 8" in result.stdout
    assert "--num-experts 256" in result.stdout


def test_formal_performance_fails_closed(tmp_path: Path) -> None:
    result = run_runner(tmp_path, "formal-performance")
    assert result.returncode != 0
    assert "formal performance is fail-closed" in result.stdout


def test_private_formal_entrypoint_validates_five_backend_plan(tmp_path: Path) -> None:
    route = tmp_path / "layer_00_topk_idx.npy"
    route.write_bytes(b"immutable-route-contract")
    digest = hashlib.sha256(route.read_bytes()).hexdigest()
    variants = [
        "nccl",
        "deepep",
        "deepep_moonep_on",
        "ultraep_hybridep",
        "probeep",
    ]
    plan = tmp_path / "benchmark_plan.json"
    plan.write_text(json.dumps({
        "schema": "probeep.multinode.benchmark_plan.v1",
        "topology": {"world_size": 16},
        "execution": {"warmup_iters_per_layer": 1, "measure_iters_per_layer": 1},
        "variants": variants,
        "case_count": len(variants),
        "cases": [
            {
                "repeat": 0,
                "layer_id": 0,
                "variant": variant,
                "route_file": str(route),
                "routing_sha256": digest,
            }
            for variant in variants
        ],
    }))
    environment = base_environment(tmp_path)
    environment["PROBEEP_BENCHMARK_PLAN"] = str(plan)
    result = subprocess.run(
        [
            sys.executable,
            str(HERE / "formal_entrypoint.py"),
            "--benchmark-plan",
            str(plan),
            "--validate-only",
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    assert json.loads(result.stdout)["case_count"] == 5


def test_nsys_overlap_analyzer_writes_nonempty_summary(tmp_path: Path) -> None:
    database = tmp_path / "profile.sqlite"
    process = 7 << 24
    with sqlite3.connect(database) as connection:
        connection.execute(
            "create table NVTX_EVENTS(start integer,end integer,text text,globalTid integer)"
        )
        connection.execute(
            "create table CUPTI_ACTIVITY_KIND_KERNEL("
            "start integer,end integer,deviceId integer,streamId integer,globalPid integer)"
        )
        connection.executemany(
            "insert into NVTX_EVENTS values(?,?,?,?)",
            [
                (0, 1000, "probeep/measurement_iteration", process + 1),
                (100, 600, "ubatch1/ht_dispatch", process + 1),
                (250, 450, "attention_or_gate/ubatch0", process + 1),
                (500, 900, "ubatch0/expert_mlp", process + 1),
            ],
        )
        connection.executemany(
            "insert into CUPTI_ACTIVITY_KIND_KERNEL values(?,?,?,?,?)",
            [
                (150, 550, 0, 11, process),
                (280, 430, 0, 22, process),
                (520, 850, 0, 22, process),
            ],
        )
    output_json = tmp_path / "overlap.json"
    output_text = tmp_path / "overlap.txt"
    result = subprocess.run(
        [
            sys.executable,
            str(HERE / "analyze_nsys_overlap.py"),
            str(database),
            "--json-output", str(output_json),
            "--text-output", str(output_text),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    payload = json.loads(output_json.read_text())
    assert payload["measurement_iterations"] == 1
    assert payload["variant_summary"][0]["variant"] == "probeep"
    assert output_text.is_file()


def test_fresh_visualization_evidence_is_fail_closed() -> None:
    benchmark = (HERE / "formal/run_benchmark.py").read_text()
    validator = (HERE / "validate_run_layout.py").read_text()
    helpers = (HERE / "run_helpers.sh").read_text()
    runner = (HERE / "runner.py").read_text()
    for artifact in (
        "microbatch_rank_samples.csv",
        "microbatch_timeline.csv",
        "rdma_path_load.csv",
        "probeep_observation_samples.csv",
        "probeep_weight_chunks.csv",
        "probeep_plan_summary.jsonl",
    ):
        assert artifact in benchmark
        assert artifact in validator
    assert "--max-cases 5" in helpers
    assert "--exclude-variant ultraep_hybridep" not in helpers
    assert 'trace_domains = "cuda,nvtx,osrt"' in runner


def test_grouped_correctness_fingerprint_distinguishes_all_dsv3_experts() -> None:
    benchmark = load_formal_benchmark()
    signs = benchmark.expert_fingerprint_signs(
        torch.arange(256, dtype=torch.int64), 16
    )
    assert signs.shape == (256, 16)
    assert torch.unique(signs, dim=0).size(0) == 256
    assert torch.equal(signs[0], torch.ones(16))


def test_correctness_forward_consumes_previous_layer_feedback() -> None:
    source = (HERE / "formal/run_benchmark.py").read_text()
    assert "correctness_feedback = (" in source
    assert "phase=\"warmup\"" in source
    assert source.count("probe_feedback=correctness_feedback") == 2
    assert "probeep_oracle_summary" not in source


def test_dual_microbatch_wavefront_and_h20_rail_contract_are_explicit() -> None:
    benchmark = (HERE / "formal/run_benchmark.py").read_text()
    validator = (HERE / "validate_run_layout.py").read_text()
    reprocessor = (HERE / "reprocess_layers_by_algorithm.py").read_text()
    assert "A0 -> (A1 || W+D0) -> (E0 || W+D1) -> E1" in benchmark
    assert '("attention_or_gate", 0, "compute")' in benchmark
    assert '("attention_or_gate", 1, "compute")' in benchmark
    assert '("weight_dispatch", 0, "communication")' in benchmark
    assert '("weight_dispatch", 1, "communication")' in benchmark
    assert "attention_starts[1].elapsed_time(network_ends[0])" in benchmark
    assert "expert_starts[0].elapsed_time(network_ends[1])" in benchmark
    assert 'PROBEEP_WEIGHT_CACHE_MODE", "cold"' in benchmark
    assert 'PROBEEP_PHYSICAL_NICS_PER_SERVER", "4"' in benchmark
    assert 'PROBEEP_RAILS_PER_PHYSICAL_NIC", "2"' in benchmark
    assert 'RDMA_PATH_BANDWIDTH_GBPS", "200"' in benchmark
    assert "4x400G -> 8x200G" in validator
    assert "A0 → (A1 ∥ W+D0) → (E0 ∥ W+D1) → E1" in reprocessor


def test_probe_feedback_is_previous_layer_and_microbatch_exact() -> None:
    benchmark = (HERE / "formal/run_benchmark.py").read_text()
    batch = (HERE / "formal/run_benchmark_batch.py").read_text()
    schema = (HERE / "formal/result_schema.py").read_text()
    validator = (HERE / "validate_run_layout.py").read_text()
    assert "_PERSISTENT_PROBE_FEEDBACK_BANK" in benchmark
    assert '_PERSISTENT_PROBE_FEEDBACK_LAYER == current_layer_id - 1' in benchmark
    assert 'phase="measured"' in benchmark
    assert "for compute_kind in (0, 1)" in benchmark
    assert "dispatch_microbatch=compute_kind" in benchmark
    assert "overlap_microbatch=1 - compute_kind" in benchmark
    assert "attention_starts[1].elapsed_time(attention_ends[1])" in benchmark
    assert "attention_starts[1].elapsed_time(network_ends[0])" in benchmark
    assert "expert_starts[0].elapsed_time(expert_ends[0])" in benchmark
    assert "expert_starts[0].elapsed_time(network_ends[1])" in benchmark
    assert 'current_layer_id == 0' in benchmark
    assert "backend.reset_probe_controller(32 * 1024 * 1024)" in benchmark
    assert "reset_balanced_probe_controller" in (
        HERE / "formal/backend.py"
    ).read_text()
    assert "persistent cases must be canonical, contiguous layer order" in batch
    assert '"producer_layer_id"' in schema
    assert '"consumer_layer_id"' in schema
    assert '"dispatch_microbatch"' in schema
    assert '"overlap_microbatch"' in schema
    assert "producer_layer + 1 != consumer_layer" in validator
    reprocessor = (HERE / "reprocess_layers_by_algorithm.py").read_text()
    assert "A[L,r]=A1∥D0" in reprocessor
    assert "M[L,r]=E0∥D1" in reprocessor
    assert "A bootstrap" in reprocessor
    assert "M bootstrap" in reprocessor


def test_dispatch_rx_uses_destination_server_same_lane_endpoint() -> None:
    benchmark = load_formal_benchmark()
    matrix = torch.zeros((16, 16), dtype=torch.int64)
    # Both final destinations are on server 1, but all traffic from R0 must
    # enter through destination relay R8.  R1 traffic must enter through R9.
    matrix[0, 8] = 10
    matrix[0, 9] = 20
    matrix[1, 8] = 7
    tx, rx = benchmark.dispatch_endpoint_bytes(matrix, 8)
    assert tx.tolist() == [30, 7] + [0] * 14
    assert rx.tolist() == [0] * 8 + [30, 7] + [0] * 6
    assert int(tx.sum()) == int(rx.sum()) == int(matrix.sum())


def test_dispatch_feedback_deduplicates_once_per_destination_server() -> None:
    benchmark = load_formal_benchmark()
    server_bytes = torch.zeros((16, 2), dtype=torch.int64)
    # R0 has one physical wire payload for server 1, even if the token is later
    # fanned out to several execution ranks there. R1 uses the next relay lane.
    server_bytes[0, 1] = 100
    server_bytes[1, 1] = 70
    matrix = benchmark.relay_dispatch_matrix(server_bytes, 8)
    assert matrix[0, 8].item() == 100
    assert matrix[1, 9].item() == 70
    assert int(torch.count_nonzero(matrix)) == 2
    tx, rx = benchmark.dispatch_endpoint_bytes(matrix, 8)
    assert tx.tolist() == [100, 70] + [0] * 14
    assert rx.tolist() == [0] * 8 + [100, 70] + [0] * 6


def test_weight_and_grad_expected_counters_are_disjoint() -> None:
    header = (
        ROOT / "src/deepep-probeep/csrc/kernels/probeep_weight_transport.cuh"
    ).read_text()
    source = (
        ROOT / "src/deepep-probeep/csrc/kernels/probeep_weight_transport.cu"
    ).read_text()
    assert "kProbeWeightSignalExpectedOffset" in header
    assert "symmetric + kProbeWeightSignalExpectedOffset" in source
    assert "symmetric + kProbeSignalExpectedOffset" in source


def test_full_duplex_weight_and_grad_staging_banks_are_disjoint() -> None:
    header = (
        ROOT / "src/deepep-probeep/csrc/kernels/probeep_weight_transport.cuh"
    ).read_text()
    source = (
        ROOT / "src/deepep-probeep/csrc/kernels/probeep_weight_transport.cu"
    ).read_text()
    assert "kProbeTxStagingOffset" in header
    assert "kProbeRxStagingOffset" in header
    assert "kProbeRxStagingOffset +" in header
    assert "auto* tx_staging = symmetric + kProbeTxStagingOffset" in source
    assert "auto* rx_staging = symmetric + kProbeRxStagingOffset" in source
    assert "remote_staging = rx_staging" in source
    assert "local_staging = tx_staging" in source


def test_cold_weight_version_is_unique_across_layers(monkeypatch) -> None:
    benchmark = load_formal_benchmark()
    monkeypatch.setenv("PROBEEP_WEIGHT_CACHE_MODE", "cold")
    monkeypatch.setenv("PROBEEP_WEIGHT_BASE_VERSION", "7")
    first = benchmark.set_probe_weight_version(
        1, layer_id=0, phase="measured", iteration=3
    )
    second = benchmark.set_probe_weight_version(
        1, layer_id=1, phase="measured", iteration=3
    )
    assert first != second
    same_layer_next_iteration = benchmark.set_probe_weight_version(
        1, layer_id=1, phase="measured", iteration=4
    )
    assert same_layer_next_iteration == second
    assert os.environ["PROBEEP_WEIGHT_VERSION"] == str(second)
    # The active version is mutable; the case-level seed must not be polluted
    # by a previous layer in the persistent process.
    assert os.environ["PROBEEP_WEIGHT_BASE_VERSION"] == "7"
    third = benchmark.set_probe_weight_version(
        int(os.environ["PROBEEP_WEIGHT_BASE_VERSION"]),
        layer_id=2,
        phase="measured",
        iteration=3,
    )
    assert third < 2**63
    assert third == 7_003_200_001


def test_eval20_formal_plan_passes_static_gate(tmp_path: Path) -> None:
    entrypoint = tmp_path / "formal_backend.py"
    entrypoint.write_text("raise SystemExit('dry-run only')\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema": "probeep.raw_data1.selection.v1",
        "selector": "raw_data1_eval20",
        "world_size": 16,
        "tokens_per_rank": 4096,
        "topk": 8,
        "num_experts": 256,
    }))
    plan = tmp_path / "benchmark_plan.json"
    plan.write_text(json.dumps({
        "schema": "probeep.multinode.benchmark_plan.v1",
        "selector": "raw_data1_eval20",
        "manifest": str(manifest.resolve()),
        "variants": [
            "nccl", "deepep", "deepep_moonep_on",
            "ultraep_hybridep", "probeep",
        ],
        "selected_layer_ids": list(range(20)),
        "topology": {"world_size": 16, "gpus_per_server": 8},
        "model": {
            "num_experts": 256,
            "tokens_per_rank": 4096,
            "topk": 8,
            "hidden": 7168,
            "expert_mode": "grouped_ffn",
        },
        "execution": {"runner_mode": "dual_microbatch_ht", "microbatches": 2},
    }))
    environment = base_environment(tmp_path)
    environment.update({
        "PROBEEP_DYNAMIC_OBSERVATION_MODE": "real",
        "PROBEEP_FORMAL_ENTRYPOINT": str(entrypoint),
        "PROBEEP_WORKLOAD_SELECTOR": "raw_data1_eval20",
        "PROBEEP_WORKLOAD_MANIFEST": str(manifest),
        "PROBEEP_BENCHMARK_PLAN": str(plan),
        "PROBEEP_FORMAL_PAPER_MODE": "0",
    })
    result = subprocess.run(
        [sys.executable, str(HERE / "runner.py"), "formal-performance", "--dry-run"],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    assert str(entrypoint) in result.stdout


def test_hybridep_vendored_tree_lock_matches_head() -> None:
    lock = json.loads((HERE / "source_lock.json").read_text())
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD:src/ultraep/HybridEP"], cwd=ROOT, text=True
    ).strip()
    assert actual == lock["ultraep_hybridep"]["vendored_git_tree"]


def test_slurm_launcher_builds_two_node_allocation(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch_log = tmp_path / "sbatch.args"
    sbatch = fake_bin / "sbatch"
    sbatch.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$FAKE_SBATCH_LOG"\n')
    sbatch.chmod(0o755)
    config = tmp_path / "h20.env"
    config.write_text(
        (HERE / "configs/h20_multinode.env.example").read_text()
        + f"\nexport PROBEEP_RESULTS_ROOT={tmp_path / 'results'}\n"
    )
    environment = os.environ.copy()
    environment.update({
        "PATH": str(fake_bin) + os.pathsep + environment["PATH"],
        "FAKE_SBATCH_LOG": str(sbatch_log),
    })
    result = subprocess.run(
        ["bash", str(HERE / "launch_slurm.sh"), str(config), "preflight"],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    arguments = sbatch_log.read_text().splitlines()
    assert "--nodes=2" in arguments
    assert "--ntasks=2" in arguments
    assert "--ntasks-per-node=1" in arguments
    assert "--gres=gpu:8" in arguments
    match = re.search(r"results: (.+)", result.stdout)
    assert match, result.stdout
    run_dir = Path(match.group(1))
    assert (run_dir / "launch.env").is_file()
    assert all((run_dir / name).is_dir() for name in ("logs", "raw", "preflight", "nsys"))


def test_manual_node_launcher_reaches_runner(tmp_path: Path) -> None:
    config = tmp_path / "h20.env"
    config.write_text(
        (HERE / "configs/h20_multinode.env.example").read_text()
        + "\nexport VENV_DIR=\n"
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    environment = os.environ.copy()
    environment.update({
        "PROBEEP_RUN_ID": "manual-static-test",
        "PROBEEP_RUN_DIR": str(run_dir),
    })
    result = subprocess.run(
        ["bash", str(HERE / "launch_node.sh"), str(config), "readiness", "0", "h20-node-0"],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    assert (run_dir / "readiness.json").is_file()


def test_rail_weight_segments_are_strictly_additive() -> None:
    report = load_reprocessor()
    records = [
        {
            "round": round_id,
            "compute": "moe",
            "path_id": 16,
            "src": 8,
            "dst": 0,
            "chunks": 2,
            "dispatch_bytes": 100,
            "weight_bytes": 60,
            "tx_bytes": 160,
            "rx_bytes": 160,
        }
        for round_id in range(11, 21)
    ]
    chunks = [
        {
            "iteration": str(iteration),
            "dispatch_compute_kind": "1",
            "dispatch_compute_name": "moe",
            "source_server": "1",
            "destination_server": "0",
            "source_rank": "8",
            "destination_rank": "0",
            "rail": "0",
            "expert_id": str(expert),
            "chunk_bytes": "30",
            "transfer_required": "1",
        }
        for iteration in range(10)
        for expert in (147, 245)
    ]
    profile = report.attach_weight_components(report.rail_profile(records), chunks)
    row = profile["rails"][0]
    assert row["weight_components"] == [
        {"expert_id": 147, "mean_bytes": 30.0},
        {"expert_id": 245, "mean_bytes": 30.0},
    ]
    assert row["mean_dispatch_bytes"] + sum(
        item["mean_bytes"] for item in row["weight_components"]
    ) == row["mean_tx_bytes"]


def test_fresh_probe_telemetry_schema_and_byte_merge_are_exact() -> None:
    benchmark = load_formal_benchmark()
    report = load_reprocessor()
    assert tuple(report.PROBE_CHUNK_EXPORT_FIELDS) == tuple(
        benchmark.PROBEEP_WEIGHT_CHUNK_FIELDS
    )
    assert set(report.RAIL_EXPORT_FIELDS) == {
        *benchmark.RDMA_PATH_LOAD_FIELDS,
        "physical_round",
    }
    counts = torch.tensor(
        [1, 1, 0, 1, 0, 1, 0, 0, 1, 10, 11, 12, 13],
        dtype=torch.int64,
    )
    chunk = torch.tensor(
        [[147, 0, 8, 0, 1, 0, 8, 0, 0, 30, 0, 0, 0]],
        dtype=torch.int64,
    )
    assigned_tx = torch.zeros(16, dtype=torch.int64)
    assigned_rx = torch.zeros(16, dtype=torch.int64)
    assigned_tx[8] = 30
    assigned_rx[0] = 30
    pair_load = torch.zeros((2, 2, 8), dtype=torch.int64)
    pair_load[1, 0, 0] = 30
    handle = SimpleNamespace(
        probe_plan_counts=counts,
        probe_admitted_experts=torch.tensor([147 * 16], dtype=torch.int64),
        probe_deferred_experts=torch.zeros(256, dtype=torch.bool),
        probe_chunk_table=chunk,
        probe_assigned_tx_bytes=assigned_tx,
        probe_assigned_rx_bytes=assigned_rx,
        probe_server_load_before=torch.tensor([80, 120], dtype=torch.int64),
        probe_server_load_after=torch.tensor([100, 100], dtype=torch.int64),
        probe_server_padded_load_before=torch.tensor([80, 120], dtype=torch.int64),
        probe_server_padded_load_after=torch.tensor([100, 100], dtype=torch.int64),
        slot_count=torch.zeros((16, 1), dtype=torch.int64),
        probe_migration_budget_snapshot=torch.full((16,), 1_000, dtype=torch.int64),
        probe_endpoint_total_cap_bytes=torch.full((16,), 1_000, dtype=torch.int64),
        probe_dispatch_tx_bytes=torch.zeros(16, dtype=torch.int64),
        probe_dispatch_rx_bytes=torch.zeros(16, dtype=torch.int64),
        probe_compute_intents=torch.tensor([[147, 1, 0, 0]], dtype=torch.int64),
        weight_transfer_required=torch.tensor(1, dtype=torch.int32),
        probe_pair_load_bytes=pair_load,
    )
    dispatch_matrix = torch.zeros((16, 16), dtype=torch.int64)
    dispatch_matrix[8, 0] = 100
    summary, weight_rows, chunk_rows = benchmark.collect_probe_runtime_telemetry(
        SimpleNamespace(handle=handle),
        iteration=3,
        dispatch_compute_kind=1,
        run_id="schema-contract",
        scope="formal",
        runner_mode="dual_microbatch_ht",
        system="probeep",
        balance="server_first",
        workload="raw_data1_layer_01",
        bias_ratio=1.0,
        seed=7,
        repeat=0,
        routing_sha256="route-sha",
        ranks_per_server=8,
        dispatch_matrix_bytes=dispatch_matrix,
        expert_weight_version=1_002_200_004,
        weight_cache_mode="cold",
    )
    assert len(weight_rows) == len(chunk_rows) == 1
    assert set(weight_rows[0]) == set(benchmark.RDMA_PATH_LOAD_FIELDS)
    assert set(chunk_rows[0]) == set(benchmark.PROBEEP_WEIGHT_CHUNK_FIELDS)
    assert weight_rows[0]["path_id"] == 16
    assert weight_rows[0]["dispatch_bytes"] == 100
    assert weight_rows[0]["weight_bytes"] == 30
    assert chunk_rows[0]["source_rank"] == 8
    assert chunk_rows[0]["destination_rank"] == 0
    assert chunk_rows[0]["physical_nic"] == 0
    assert chunk_rows[0]["subrail"] == 0
    assert summary["assigned_tx_bytes"] == assigned_tx.tolist()
    assert summary["assigned_rx_bytes"] == assigned_rx.tolist()

    fixed_dispatch = dict(weight_rows[0])
    fixed_dispatch.update(
        chunk_count=0,
        dispatch_units=1,
        dispatch_unit_name="destination_server_token",
        dispatch_bytes_per_unit=100,
        traffic_source="runtime_dispatch",
        weight_bytes=0,
        tx_bytes=100,
        rx_bytes=100,
    )
    benchmark.merge_probe_weight_telemetry([fixed_dispatch], weight_rows)
    assert fixed_dispatch["chunk_count"] == 1
    assert fixed_dispatch["weight_bytes"] == 30
    assert fixed_dispatch["tx_bytes"] == fixed_dispatch["rx_bytes"] == 130
