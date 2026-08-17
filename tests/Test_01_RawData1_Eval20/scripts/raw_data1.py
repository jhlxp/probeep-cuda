#!/usr/bin/env python3
"""Validate and materialize exact TopK routes for categorized multi-node tests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import deque
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = REPO_ROOT / "workload/raw_data1"
PLACEMENT_FILE = "DSV3_32x8_256_unique.json"
NUM_EXPERTS = 256
EXPECTED_LAYERS = 58


class WorkloadError(ValueError):
    pass


def source_digest(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def selector_layers(selector: str, *, num_layers: int = EXPECTED_LAYERS) -> list[int]:
    if selector == "raw_data1_all":
        return list(range(num_layers))
    if selector == "raw_data1_eval20":
        if num_layers < 20:
            raise WorkloadError("raw_data1_eval20 requires at least 20 layers")
        return list(range(20))
    prefix = "raw_data1_layer_"
    if selector.startswith(prefix):
        try:
            layer = int(selector[len(prefix) :])
        except ValueError as error:
            raise WorkloadError(f"invalid layer selector: {selector}") from error
        if not 0 <= layer < num_layers:
            raise WorkloadError(f"layer {layer} is outside [0, {num_layers})")
        return [layer]
    raise WorkloadError(
        "selector must be raw_data1_all, raw_data1_eval20, or raw_data1_layer_<id>"
    )


def load_counts(data_dir: Path) -> tuple[np.ndarray, str]:
    data_dir = data_dir.resolve()
    placement_path = data_dir / PLACEMENT_FILE
    if not placement_path.is_file():
        raise WorkloadError(f"missing placement: {placement_path}")
    placement = json.loads(placement_path.read_text(encoding="utf-8"))
    layers = placement.get("layer_list")
    if placement.get("moe_layer_count") != EXPECTED_LAYERS or not isinstance(layers, list):
        raise WorkloadError("raw_data1 placement must contain exactly 58 MoE layers")
    if len(layers) != EXPECTED_LAYERS:
        raise WorkloadError(f"expected 58 layer entries, found {len(layers)}")

    counts = np.empty((EXPECTED_LAYERS, NUM_EXPERTS), dtype=np.int64)
    csv_paths = []
    for storage_rank in range(32):
        path = data_dir / f"decode_{storage_rank}.csv"
        if not path.is_file():
            raise WorkloadError(f"missing receive counts: {path}")
        csv_paths.append(path)
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        if len(rows) != EXPECTED_LAYERS:
            raise WorkloadError(f"{path}: expected 58 rows, found {len(rows)}")
        for layer_id, row in enumerate(rows):
            if len(row) != 8:
                raise WorkloadError(f"{path}:{layer_id + 1}: expected 8 experts")
            try:
                values = [int(value) for value in row]
            except ValueError as error:
                raise WorkloadError(f"{path}:{layer_id + 1}: counts must be integers") from error
            if any(value < 0 for value in values):
                raise WorkloadError(f"{path}:{layer_id + 1}: negative expert count")
            start = storage_rank * 8
            counts[layer_id, start : start + 8] = values

    for layer_id, layer in enumerate(layers):
        if layer.get("layer_id") != layer_id:
            raise WorkloadError("placement layer IDs must be contiguous")
        devices = layer.get("device_list")
        if not isinstance(devices, list) or len(devices) != 32:
            raise WorkloadError(f"layer {layer_id}: expected 32 storage devices")
        expert_ids = [
            expert
            for device in sorted(devices, key=lambda item: item["device_id"])
            for expert in device.get("device_expert", [])
        ]
        if expert_ids != list(range(NUM_EXPERTS)):
            raise WorkloadError(f"layer {layer_id}: placement is not canonical E256")

    return counts, source_digest([placement_path, *csv_paths])


def largest_remainder(values: np.ndarray, total: int) -> np.ndarray:
    if values.ndim != 1 or np.any(values < 0):
        raise WorkloadError("expert counts must be a non-negative vector")
    source_total = int(values.sum())
    if source_total <= 0 or total <= 0:
        raise WorkloadError("source and target route totals must be positive")
    numerators = [int(value) * total for value in values.tolist()]
    result = np.asarray([value // source_total for value in numerators], dtype=np.int64)
    remainder = total - int(result.sum())
    order = sorted(
        range(values.size),
        key=lambda expert: (-(numerators[expert] % source_total), expert),
    )
    if remainder:
        result[np.asarray(order[:remainder], dtype=np.int64)] += 1
    if int(result.sum()) != total:
        raise WorkloadError("largest-remainder scaling lost routes")
    return result


def realize_exact_topk(expert_counts: np.ndarray, num_tokens: int, topk: int) -> np.ndarray:
    """Realize a deterministic simple token-expert graph with exact degrees."""

    counts = np.asarray(expert_counts, dtype=np.int64)
    if counts.shape != (NUM_EXPERTS,):
        raise WorkloadError(f"expected {NUM_EXPERTS} expert counts")
    if int(counts.sum()) != num_tokens * topk:
        raise WorkloadError("expert counts do not equal num_tokens*topk")
    if int(counts.max(initial=0)) > num_tokens:
        raise WorkloadError("an expert count exceeds the unique-token capacity")

    routes = np.full((num_tokens, topk), -1, dtype=np.int16)
    buckets = [deque() for _ in range(topk + 1)]
    buckets[0].extend(range(num_tokens))
    expert_order = sorted(range(NUM_EXPERTS), key=lambda expert: (-int(counts[expert]), expert))

    for expert in expert_order:
        remaining = int(counts[expert])
        selected: list[tuple[int, int]] = []
        for load in range(topk):
            bucket = buckets[load]
            take = min(remaining, len(bucket))
            for _ in range(take):
                selected.append((load, bucket.popleft()))
            remaining -= take
            if remaining == 0:
                break
        if remaining:
            raise WorkloadError(f"cannot realize expert {expert} without duplicate TopK entries")
        for load, token in selected:
            routes[token, load] = expert
            buckets[load + 1].append(token)

    if len(buckets[topk]) != num_tokens or np.any(routes < 0):
        raise WorkloadError("route realization did not fill every token")
    sorted_routes = np.sort(routes.astype(np.int32), axis=1)
    if topk > 1 and np.any(sorted_routes[:, 1:] == sorted_routes[:, :-1]):
        raise WorkloadError("route realization produced duplicate experts in one token")
    observed = np.bincount(routes.astype(np.int64).ravel(), minlength=NUM_EXPERTS)
    if not np.array_equal(observed, counts):
        raise WorkloadError("route realization changed the expert histogram")
    return routes


def describe(
    data_dir: Path,
    selector: str,
    world_size: int,
    tokens_per_rank: int,
    topk: int,
) -> tuple[dict[str, object], np.ndarray]:
    if min(world_size, tokens_per_rank, topk) <= 0:
        raise WorkloadError("world size, tokens/rank and TopK must be positive")
    if NUM_EXPERTS % world_size:
        raise WorkloadError(f"E256 is not divisible by world size {world_size}")
    counts, digest = load_counts(data_dir)
    layer_ids = selector_layers(selector, num_layers=counts.shape[0])
    routes_per_layer = world_size * tokens_per_rank * topk
    scaled = np.stack(
        [largest_remainder(counts[layer_id], routes_per_layer) for layer_id in layer_ids]
    )
    max_mean = scaled.max(axis=1) / scaled.mean(axis=1)
    summary = {
        "schema": "probeep.raw_data1.selection.v1",
        "selector": selector,
        "source_dir": str(data_dir.resolve()),
        "source_sha256": digest,
        "source_layers": EXPECTED_LAYERS,
        "selected_layer_ids": layer_ids,
        "selected_layer_count": len(layer_ids),
        "num_experts": NUM_EXPERTS,
        "world_size": world_size,
        "experts_per_rank": NUM_EXPERTS // world_size,
        "tokens_per_rank": tokens_per_rank,
        "topk": topk,
        "global_tokens_per_layer": world_size * tokens_per_rank,
        "routes_per_layer": routes_per_layer,
        "topk_weight_policy": "uniform_1_over_topk",
        "routing_policy": "exact_histogram_lowest_token_load_stable_expert_id",
        "layer_expert_max_mean": [float(value) for value in max_mean],
        "paper_eligible": selector == "raw_data1_all",
    }
    return summary, scaled


def materialize(
    data_dir: Path,
    selector: str,
    output_dir: Path,
    world_size: int,
    tokens_per_rank: int,
    topk: int,
) -> dict[str, object]:
    if output_dir.exists():
        raise WorkloadError(f"output directory already exists: {output_dir}")
    summary, scaled = describe(data_dir, selector, world_size, tokens_per_rank, topk)
    output_dir.mkdir(parents=True)
    layer_files = []
    num_tokens = world_size * tokens_per_rank
    for layer_id, counts in zip(summary["selected_layer_ids"], scaled):
        routes = realize_exact_topk(counts, num_tokens, topk)
        routes = routes.reshape(world_size, tokens_per_rank, topk)
        path = output_dir / f"layer_{int(layer_id):02d}_topk_idx.npy"
        np.save(path, routes, allow_pickle=False)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        layer_files.append({
            "layer_id": int(layer_id),
            "path": path.name,
            "sha256": digest,
            "shape": list(routes.shape),
            "dtype": str(routes.dtype),
            "expert_counts": [int(value) for value in counts],
        })
    manifest = {**summary, "layers": layer_files}
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "materialize"):
        child = subparsers.add_parser(name)
        child.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
        child.add_argument("--selector", default="raw_data1_all")
        child.add_argument("--world-size", type=int, default=16)
        child.add_argument("--tokens-per-rank", type=int, default=4096)
        child.add_argument("--topk", type=int, default=8)
        if name == "materialize":
            child.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.command == "inspect":
        payload, _ = describe(
            args.data_dir, args.selector, args.world_size, args.tokens_per_rank, args.topk
        )
    else:
        payload = materialize(
            args.data_dir,
            args.selector,
            args.output_dir,
            args.world_size,
            args.tokens_per_rank,
            args.topk,
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
