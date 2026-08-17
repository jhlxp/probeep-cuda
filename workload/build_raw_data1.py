#!/usr/bin/env python3
"""Build the 58-layer, 256-expert, 4096-token/rank raw_data1 dataset.

The Poseidon input has 32 devices x 9 physical slots but only 256 logical
expert IDs.  The placement does not mark which occurrence of a duplicated
expert is the primary.  For every layer this tool therefore keeps exactly one
occurrence per expert (the occurrence with the largest receive count by
default), pools the counts of the other 32 occurrences, and redistributes that
pool over all 256 retained experts in proportion to their retained counts.

The output uses a canonical 32 x 8 storage layout: source device r contains
logical experts [8*r, 8*r+8).  After removing replicas, a second
largest-remainder apportionment scales every layer to
model_ranks * tokens_per_rank * topk expert rows.  This preserves the observed
per-layer distribution without stochastic sampling.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable


DEFAULT_PLACEMENT = "ET_4+4_32_9_gsm8k_r1_2k_2k_0417_al_0.json"
DEFAULT_OUTPUT_PLACEMENT = "DSV3_32x8_256_unique.json"


class ConversionError(ValueError):
    """Raised when the source dataset violates the conversion contract."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--placement", default=DEFAULT_PLACEMENT)
    parser.add_argument("--output-placement", default=DEFAULT_OUTPUT_PLACEMENT)
    parser.add_argument("--model-ranks", type=int, default=16)
    parser.add_argument("--tokens-per-rank", type=int, default=4096)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument(
        "--primary-policy",
        choices=("max-receive", "first"),
        default="max-receive",
        help=(
            "how to retain one occurrence when an expert is replicated; "
            "max-receive is stable and loses the least observed primary load"
        ),
    )
    return parser.parse_args()


def _source_digest(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_source(
    source_dir: Path, placement_name: str
) -> tuple[dict[str, object], list[list[list[int]]], list[Path]]:
    placement_path = source_dir / placement_name
    if not placement_path.is_file():
        raise ConversionError(f"missing placement JSON: {placement_path}")
    try:
        root = json.loads(placement_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConversionError(f"invalid placement JSON: {placement_path}") from exc

    layers = root.get("layer_list") if isinstance(root, dict) else None
    if not isinstance(layers, list) or not layers:
        raise ConversionError("placement JSON needs a non-empty layer_list")
    if root.get("moe_layer_count") != len(layers):
        raise ConversionError("moe_layer_count disagrees with layer_list")

    first_devices = layers[0].get("device_list")
    if not isinstance(first_devices, list) or not first_devices:
        raise ConversionError("layer 0 needs a non-empty device_list")
    num_devices = len(first_devices)
    if num_devices != 32:
        raise ConversionError(f"expected 32 source devices, found {num_devices}")

    csv_paths = [source_dir / f"decode_{rank}.csv" for rank in range(num_devices)]
    receives: list[list[list[int]]] = []
    for rank, csv_path in enumerate(csv_paths):
        if not csv_path.is_file():
            raise ConversionError(f"missing receive CSV: {csv_path}")
        rows: list[list[int]] = []
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row_index, row in enumerate(csv.reader(handle), start=1):
                if len(row) != 9:
                    raise ConversionError(
                        f"{csv_path}:{row_index}: expected 9 slots, found {len(row)}"
                    )
                try:
                    values = [int(value) for value in row]
                except ValueError as exc:
                    raise ConversionError(
                        f"{csv_path}:{row_index}: counts must be integers"
                    ) from exc
                if any(value < 0 for value in values):
                    raise ConversionError(
                        f"{csv_path}:{row_index}: counts must be non-negative"
                    )
                rows.append(values)
        if len(rows) != len(layers):
            raise ConversionError(
                f"{csv_path}: expected {len(layers)} layers, found {len(rows)}"
            )
        receives.append(rows)

    for layer_id, layer in enumerate(layers):
        if layer.get("layer_id") != layer_id:
            raise ConversionError("layer IDs must be contiguous")
        devices = layer.get("device_list")
        if not isinstance(devices, list) or len(devices) != num_devices:
            raise ConversionError(f"layer {layer_id}: expected 32 devices")
        by_rank: list[list[int] | None] = [None] * num_devices
        for device in devices:
            rank = device.get("device_id") if isinstance(device, dict) else None
            slots = device.get("device_expert") if isinstance(device, dict) else None
            if not isinstance(rank, int) or not 0 <= rank < num_devices:
                raise ConversionError(f"layer {layer_id}: invalid device_id")
            if by_rank[rank] is not None:
                raise ConversionError(f"layer {layer_id}: duplicate device_id {rank}")
            if not isinstance(slots, list) or len(slots) != 9:
                raise ConversionError(
                    f"layer {layer_id}, device {rank}: expected 9 expert slots"
                )
            if any(not isinstance(expert, int) for expert in slots):
                raise ConversionError("expert IDs must be integers")
            by_rank[rank] = slots
        ids = [expert for slots in by_rank for expert in (slots or [])]
        if set(ids) != set(range(256)):
            raise ConversionError(
                f"layer {layer_id}: expert IDs must cover exactly [0, 256)"
            )
        if len(ids) - len(set(ids)) != 32:
            raise ConversionError(
                f"layer {layer_id}: expected exactly 32 redundant slots"
            )

    return root, receives, [placement_path, *csv_paths]


def _largest_remainder(weights: list[int], total: int) -> list[int]:
    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ConversionError("cannot redistribute against zero primary load")
    numerators = [weight * total for weight in weights]
    allocation = [numerator // weight_sum for numerator in numerators]
    remainder = total - sum(allocation)
    order = sorted(
        range(len(weights)),
        key=lambda expert: (-(numerators[expert] % weight_sum), expert),
    )
    for expert in order[:remainder]:
        allocation[expert] += 1
    if sum(allocation) != total:
        raise ConversionError("largest-remainder apportionment lost counts")
    return allocation


def _normalize_layer(
    layer: dict[str, object],
    receives: list[list[list[int]]],
    layer_id: int,
    primary_policy: str,
) -> tuple[list[int], dict[str, object]]:
    devices = sorted(layer["device_list"], key=lambda item: item["device_id"])
    occurrences: list[list[tuple[int, int, int]]] = [[] for _ in range(256)]
    original_total = 0
    for device in devices:
        rank = device["device_id"]
        for slot, expert in enumerate(device["device_expert"]):
            count = receives[rank][layer_id][slot]
            occurrences[expert].append((rank, slot, count))
            original_total += count

    primary = [0] * 256
    redundant_pool = 0
    replicated_experts = 0
    for expert, copies in enumerate(occurrences):
        if not copies:
            raise ConversionError(f"layer {layer_id}: missing expert {expert}")
        if len(copies) > 1:
            replicated_experts += 1
        if primary_policy == "max-receive":
            # max() plus negative coordinates gives stable lowest-rank/slot ties.
            kept = max(copies, key=lambda item: (item[2], -item[0], -item[1]))
        else:
            kept = min(copies, key=lambda item: (item[0], item[1]))
        primary[expert] = kept[2]
        redundant_pool += sum(item[2] for item in copies if item != kept)

    redistributed = _largest_remainder(primary, redundant_pool)
    normalized = [base + extra for base, extra in zip(primary, redistributed)]
    if sum(normalized) != original_total:
        raise ConversionError(f"layer {layer_id}: normalization lost counts")
    return normalized, {
        "layer_id": layer_id,
        "original_total": original_total,
        "retained_primary_total": sum(primary),
        "redistributed_replica_total": redundant_pool,
        "normalized_total": sum(normalized),
        "replicated_logical_experts": replicated_experts,
        "removed_physical_slots": 32,
    }


def _canonical_placement(num_layers: int) -> dict[str, object]:
    return {
        "moe_layer_count": num_layers,
        "layer_list": [
            {
                "layer_id": layer_id,
                "device_count": 32,
                "device_list": [
                    {
                        "device_id": rank,
                        "device_expert": list(range(rank * 8, (rank + 1) * 8)),
                    }
                    for rank in range(32)
                ],
            }
            for layer_id in range(num_layers)
        ],
    }


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise ConversionError(f"output directory already exists: {output_dir}")
    if min(args.model_ranks, args.tokens_per_rank, args.topk) <= 0:
        raise ConversionError("model-ranks, tokens-per-rank and topk must be positive")
    target_rows_per_layer = args.model_ranks * args.tokens_per_rank * args.topk

    root, receives, source_files = _load_source(source_dir, args.placement)
    normalized_layers: list[list[int]] = []
    layer_stats: list[dict[str, object]] = []
    for layer_id, layer in enumerate(root["layer_list"]):
        normalized, stats = _normalize_layer(
            layer, receives, layer_id, args.primary_policy
        )
        scaled = _largest_remainder(normalized, target_rows_per_layer)
        stats["scaled_total"] = sum(scaled)
        normalized_layers.append(scaled)
        layer_stats.append(stats)

    output_dir.mkdir(parents=True)
    placement = _canonical_placement(len(normalized_layers))
    placement_path = output_dir / args.output_placement
    placement_path.write_text(
        json.dumps(placement, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    csv_paths: list[Path] = []
    for rank in range(32):
        csv_path = output_dir / f"decode_{rank}.csv"
        csv_paths.append(csv_path)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            start = rank * 8
            writer.writerows(layer[start : start + 8] for layer in normalized_layers)

    output_digest = _source_digest([placement_path, *csv_paths])
    source_digest = _source_digest(source_files)
    layer_zero = layer_stats[0]
    readme = f"""# raw_data1

由 `workload/raw_data` 通过 `workload/build_raw_data1.py` 生成。它不是原始物理
placement 的复制品，而是去除 32 个冗余物理槽后的 256-专家 workload。

| 项目 | 输入 | raw_data1 |
|---|---:|---:|
| layers | {len(normalized_layers)} | {len(normalized_layers)} |
| devices | 32 | 32 |
| slots/device | 9 | 8 |
| physical slots/layer | 288 | 256 |
| unique logical experts/layer | 256 | 256 |
| duplicate expert IDs/layer | 32 extra slots | 0 |

每层先为每个 logical expert 保留接收量最大的一个物理副本；其余 32 个物理槽的
token count 汇成池，再按 256 个保留主槽的接收量比例用最大余数法回填。随后把该层
分布缩放到 `{args.model_ranks} ranks × {args.tokens_per_rank} tokens/rank × TopK
{args.topk} = {target_rows_per_layer:,}` expert rows。输出 JSON 的 32 个 storage ranks
固定保存专家 `[8r, 8r+8)`；这只是 canonical 文件分块。运行时按
`256 / model_ranks` 专家/GPU 重新分组；EP16 为 16 experts/GPU。

| 转换项 | 值 |
|---|---|
| primary policy | `{args.primary_policy}` |
| redistribution | `proportional_to_retained_primary_load` |
| integer apportionment | `largest_remainder_stable_expert_id` |
| scaling | deterministic largest remainder；无随机采样 |
| model ranks | {args.model_ranks} |
| tokens/rank | {args.tokens_per_rank:,} |
| TopK | {args.topk} |
| output rows/layer | {target_rows_per_layer:,} |
| Layer 0 retained primary | {layer_zero['retained_primary_total']:,} |
| Layer 0 redistributed replicas | {layer_zero['redistributed_replica_total']:,} |
| Layer 0 pre-scale total | {layer_zero['normalized_total']:,} |
| Layer 0 output total | {layer_zero['scaled_total']:,} |

原始 33 个运行时文件（JSON+CSV）tree SHA-256：

```text
{source_digest}
```

输出 33 个运行时文件（JSON+CSV，不含 README/plot）tree SHA-256：

```text
{output_digest}
```

`plot/` 只保存这份 `raw_data1` 派生出来的负载图，不混放其他 workload 的图片。
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    print(json.dumps({
        "output_dir": str(output_dir),
        "layers": len(normalized_layers),
        "logical_experts": 256,
        "slots_per_device": 8,
        "model_ranks": args.model_ranks,
        "tokens_per_rank": args.tokens_per_rank,
        "topk": args.topk,
        "rows_per_layer": target_rows_per_layer,
        "source_tree_sha256": source_digest,
        "tree_sha256": output_digest,
        "all_layer_pre_scale_totals_conserved": all(
            layer["original_total"] == layer["normalized_total"]
            for layer in layer_stats
        ),
        "all_layer_output_totals_match": all(
            layer["scaled_total"] == target_rows_per_layer
            for layer in layer_stats
        ),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
