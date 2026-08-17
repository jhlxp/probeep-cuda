from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("benchmark_plan.py")
SPEC = importlib.util.spec_from_file_location("probeep_tests_benchmark_plan", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark_plan = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark_plan)


def make_manifest(root: Path, selector: str, layer_ids: list[int]) -> Path:
    layers = []
    for layer_id in layer_ids:
        route = root / f"layer_{layer_id:02d}_topk_idx.npy"
        route.write_bytes(f"contract-route-{layer_id}".encode())
        layers.append(
            {
                "layer_id": layer_id,
                "path": route.name,
                "sha256": benchmark_plan.file_sha256(route),
                "shape": [16, 4096, 8],
                "dtype": "int16",
                "expert_counts": [2048] * 256,
            }
        )
    manifest = {
        "schema": "probeep.raw_data1.selection.v1",
        "selector": selector,
        "selected_layer_ids": layer_ids,
        "selected_layer_count": len(layer_ids),
        "num_experts": 256,
        "world_size": 16,
        "experts_per_rank": 16,
        "tokens_per_rank": 4096,
        "topk": 8,
        "source_sha256": "contract-source",
        "layers": layers,
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_eval20_plan_is_exact(tmp_path: Path) -> None:
    layer_ids = list(range(20))
    manifest = make_manifest(tmp_path, "raw_data1_eval20", layer_ids)
    plan = benchmark_plan.build_plan(
        manifest,
        expected_selector="raw_data1_eval20",
        expected_layer_ids=layer_ids,
        warmup_iters=10,
        measure_iters=10,
        repeats=1,
        paper_eligible=False,
    )
    assert plan["case_count"] == 100
    assert plan["paper_eligible"] is False
    assert plan["variants"] == list(benchmark_plan.VARIANTS)


def test_route_digest_mismatch_fails_closed(tmp_path: Path) -> None:
    layer_ids = list(range(20))
    manifest = make_manifest(tmp_path, "raw_data1_eval20", layer_ids)
    (tmp_path / "layer_00_topk_idx.npy").write_bytes(b"changed")
    try:
        benchmark_plan.load_manifest(
            manifest,
            expected_selector="raw_data1_eval20",
            expected_layer_ids=layer_ids,
        )
    except benchmark_plan.BenchmarkPlanError as error:
        assert "SHA-256 mismatch" in str(error)
    else:
        raise AssertionError("a changed route tensor must invalidate the plan")
