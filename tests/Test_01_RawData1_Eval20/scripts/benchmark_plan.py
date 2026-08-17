#!/usr/bin/env python3
"""Validate a materialized workload and build an immutable multi-node case plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "probeep.multinode.benchmark_plan.v1"
WORKLOAD_SCHEMA = "probeep.raw_data1.selection.v1"
VARIANTS = (
    "nccl",
    "deepep",
    "deepep_moonep_on",
    "ultraep_hybridep",
    "probeep",
)


class BenchmarkPlanError(ValueError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(
    manifest_path: Path,
    *,
    expected_selector: str,
    expected_layer_ids: list[int],
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkPlanError(f"cannot read workload manifest: {error}") from error

    expected = {
        "schema": WORKLOAD_SCHEMA,
        "selector": expected_selector,
        "selected_layer_ids": expected_layer_ids,
        "selected_layer_count": len(expected_layer_ids),
        "num_experts": 256,
        "tokens_per_rank": 4096,
        "topk": 8,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise BenchmarkPlanError(
                f"manifest {key}={payload.get(key)!r}, expected {value!r}"
            )

    world_size = payload.get("world_size")
    if (
        not isinstance(world_size, int)
        or world_size < 16
        or world_size % 8
        or 256 % world_size
    ):
        raise BenchmarkPlanError(
            "manifest world_size must describe at least two 8-GPU servers "
            "and divide E256"
        )

    layers = payload.get("layers")
    if not isinstance(layers, list) or len(layers) != len(expected_layer_ids):
        raise BenchmarkPlanError("manifest layer records do not match the selector")
    base = manifest_path.parent
    for expected_layer, layer in zip(expected_layer_ids, layers):
        if not isinstance(layer, dict) or layer.get("layer_id") != expected_layer:
            raise BenchmarkPlanError("manifest layer order is not canonical")
        route_path = base / str(layer.get("path", ""))
        if not route_path.is_file():
            raise BenchmarkPlanError(f"missing route tensor: {route_path}")
        if layer.get("shape") != [world_size, 4096, 8] or layer.get("dtype") != "int16":
            raise BenchmarkPlanError(f"layer {expected_layer}: invalid route shape/dtype")
        if file_sha256(route_path) != layer.get("sha256"):
            raise BenchmarkPlanError(f"layer {expected_layer}: route SHA-256 mismatch")
        counts = layer.get("expert_counts")
        if not isinstance(counts, list) or len(counts) != 256:
            raise BenchmarkPlanError(f"layer {expected_layer}: invalid expert histogram")
        if sum(int(value) for value in counts) != world_size * 4096 * 8:
            raise BenchmarkPlanError(f"layer {expected_layer}: route total mismatch")
    return payload


def build_plan(
    manifest_path: Path,
    *,
    expected_selector: str,
    expected_layer_ids: list[int],
    warmup_iters: int,
    measure_iters: int,
    repeats: int,
    paper_eligible: bool,
) -> dict[str, Any]:
    if min(warmup_iters, measure_iters, repeats) <= 0:
        raise BenchmarkPlanError("warmup, measure and repeats must be positive")
    manifest = load_manifest(
        manifest_path,
        expected_selector=expected_selector,
        expected_layer_ids=expected_layer_ids,
    )
    layers = {int(item["layer_id"]): item for item in manifest["layers"]}
    cases = []
    for repeat in range(repeats):
        for layer_id in expected_layer_ids:
            layer = layers[layer_id]
            for variant in VARIANTS:
                cases.append(
                    {
                        "repeat": repeat,
                        "layer_id": layer_id,
                        "variant": variant,
                        "route_file": str((manifest_path.parent / layer["path"]).resolve()),
                        "routing_sha256": layer["sha256"],
                    }
                )
    return {
        "schema": SCHEMA,
        "selector": expected_selector,
        "paper_eligible": paper_eligible,
        "manifest": str(manifest_path.resolve()),
        "source_sha256": manifest["source_sha256"],
        "topology": {
            "world_size": manifest["world_size"],
            "gpus_per_server": 8,
            "num_servers": manifest["world_size"] // 8,
            "experts_per_rank": manifest["experts_per_rank"],
        },
        "model": {
            "num_experts": 256,
            "tokens_per_rank": 4096,
            "topk": 8,
            "hidden": 7168,
            "expert_mode": "grouped_ffn",
        },
        "execution": {
            "runner_mode": "dual_microbatch_ht",
            "microbatches": 2,
            "warmup_iters_per_layer": warmup_iters,
            "measure_iters_per_layer": measure_iters,
            "repeats": repeats,
            "rank_reduction": "max",
        },
        "variants": list(VARIANTS),
        "selected_layer_ids": expected_layer_ids,
        "case_count": len(cases),
        "cases": cases,
    }


def run_cli(
    *,
    expected_selector: str,
    expected_layer_ids: list[int],
    paper_eligible: bool,
) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-plan", type=Path, required=True)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--measure-iters", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    plan = build_plan(
        args.manifest,
        expected_selector=expected_selector,
        expected_layer_ids=expected_layer_ids,
        warmup_iters=args.warmup_iters,
        measure_iters=args.measure_iters,
        repeats=args.repeats,
        paper_eligible=paper_eligible,
    )
    if args.output_plan.exists():
        raise BenchmarkPlanError(f"output plan already exists: {args.output_plan}")
    args.output_plan.parent.mkdir(parents=True, exist_ok=True)
    args.output_plan.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: plan[key] for key in (
        "schema", "selector", "paper_eligible", "case_count", "selected_layer_ids"
    )}, ensure_ascii=False, indent=2))
