#!/usr/bin/env python3
"""Build raw_data2 from the original DSV3 receive trace.

raw_data2 keeps the raw_data1 runtime storage contract:

* 94 sample rows by default, matching Qwen3-235B-A22B layer count
* 256 logical experts
* canonical 32 x 8 storage layout
* every sample scaled to model_ranks * tokens_per_rank * topk rows

The benchmark meaning is different from raw_data1: each output row is a
prefill-batch hotspot sample for a Qwen3-235B-style MoE layer slot.  The source
raw_data rows are used as empirical hotspot templates because the checked-in
trace has no explicit Qwen3 per-layer x per-batch tensor.  After
raw_data1-style deduplication and scaling, every sample is capped so expert
max/mean stays in the requested range.  The default range is 8..14; the source
trace is already above 8 on every row, so the generator only compresses rows
whose expert-level max/mean exceeds 14.  Finally, sample order is
deterministically shuffled to avoid benchmark code depending on the original
template order.  If --num-samples is larger than the available empirical
templates, extra samples are deterministic reuses of shuffled templates; they
are workload stress samples, not a claimed Qwen routing trace.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workload.build_raw_data1 import (  # noqa: E402
    DEFAULT_OUTPUT_PLACEMENT,
    DEFAULT_PLACEMENT,
    ConversionError,
    _canonical_placement,
    _largest_remainder,
    _load_source,
    _normalize_layer,
    _source_digest,
)


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
        "--num-samples",
        type=int,
        default=94,
        help="number of output fixed-layer/prefill-batch samples; Qwen3-235B uses 94 layers",
    )
    parser.add_argument(
        "--primary-policy",
        choices=("max-receive", "first"),
        default="max-receive",
    )
    parser.add_argument("--min-max-mean", type=float, default=8.0)
    parser.add_argument("--max-max-mean", type=float, default=14.0)
    parser.add_argument("--shuffle-seed", type=int, default=20260815)
    return parser.parse_args()


def _bounded_distribute(
    *,
    weights: list[int],
    capacities: list[int],
    total: int,
) -> list[int]:
    if total < 0:
        raise ConversionError("cannot distribute a negative total")
    if sum(capacities) < total:
        raise ConversionError("cap is too low to preserve sample total")
    allocation = [0] * len(weights)
    remaining = total
    active = {index for index, capacity in enumerate(capacities) if capacity > 0}

    while remaining > 0:
        if not active:
            raise ConversionError("no capacity left while redistributing tail load")
        denominator = sum(weights[index] for index in active)
        if denominator <= 0:
            denominator = sum(capacities[index] for index in active)
            current_weights = capacities
        else:
            current_weights = weights

        grants = [0] * len(weights)
        fractional_order: list[tuple[int, int]] = []
        for index in active:
            numerator = int(current_weights[index]) * remaining
            grant = min(capacities[index], numerator // denominator)
            grants[index] = grant
            if grant < capacities[index]:
                fractional_order.append((numerator % denominator, index))

        granted = sum(grants)
        for index, grant in enumerate(grants):
            if grant:
                allocation[index] += grant
                capacities[index] -= grant
        remaining -= granted

        if remaining > 0 and fractional_order:
            for _, index in sorted(fractional_order, key=lambda item: (-item[0], item[1])):
                if remaining == 0:
                    break
                if capacities[index] <= 0:
                    continue
                allocation[index] += 1
                capacities[index] -= 1
                remaining -= 1

        active = {index for index in active if capacities[index] > 0}

    return allocation


def _cap_layer_max_mean(
    counts: list[int],
    *,
    min_ratio: float,
    max_ratio: float,
) -> tuple[list[int], dict[str, float | int | bool]]:
    total = sum(counts)
    if total <= 0:
        raise ConversionError("cannot cap an empty sample")
    mean = total / len(counts)
    before = max(counts) / mean
    if before < min_ratio:
        raise ConversionError(
            f"source sample max/mean {before:.4f} is below requested minimum {min_ratio}"
        )

    cap = int(max_ratio * total // len(counts))
    if cap <= 0:
        raise ConversionError("max-max-mean cap is too low")
    base = [min(value, cap) for value in counts]
    excess = total - sum(base)
    if excess:
        capacities = [cap - value for value in base]
        allocation = _bounded_distribute(
            weights=counts,
            capacities=capacities,
            total=excess,
        )
        capped = [value + extra for value, extra in zip(base, allocation)]
    else:
        capped = base

    if sum(capped) != total:
        raise ConversionError("tail capping changed sample total")
    after = max(capped) / mean
    if not (min_ratio <= after <= max_ratio + 1e-12):
        raise ConversionError(
            f"tail capping produced max/mean {after:.4f}, outside "
            f"[{min_ratio}, {max_ratio}]"
        )
    return capped, {
        "before_expert_max_mean": before,
        "after_expert_max_mean": after,
        "cap_rows": cap,
        "capped_experts": sum(1 for value in counts if value > cap),
        "redistributed_rows": excess,
        "changed": bool(excess),
    }


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise ConversionError(f"output directory already exists: {output_dir}")
    if min(args.model_ranks, args.tokens_per_rank, args.topk, args.num_samples) <= 0:
        raise ConversionError(
            "model-ranks, tokens-per-rank, topk and num-samples must be positive"
        )
    if args.min_max_mean < 1 or args.max_max_mean < args.min_max_mean:
        raise ConversionError("need 1 <= min-max-mean <= max-max-mean")

    target_rows_per_layer = args.model_ranks * args.tokens_per_rank * args.topk
    root, receives, source_files = _load_source(source_dir, args.placement)

    rawdata1_layers: list[list[int]] = []
    rawdata1_stats: list[dict[str, object]] = []
    cap_stats: list[dict[str, float | int | bool]] = []
    for layer_id, layer in enumerate(root["layer_list"]):
        normalized, stats = _normalize_layer(
            layer, receives, layer_id, args.primary_policy
        )
        scaled = _largest_remainder(normalized, target_rows_per_layer)
        capped, cap_stat = _cap_layer_max_mean(
            scaled,
            min_ratio=args.min_max_mean,
            max_ratio=args.max_max_mean,
        )
        stats["scaled_total"] = sum(scaled)
        rawdata1_layers.append(capped)
        rawdata1_stats.append(stats)
        cap_stats.append(cap_stat)

    generator = random.Random(args.shuffle_seed)
    template_indices: list[int] = []
    while len(template_indices) < args.num_samples:
        cycle = list(range(len(rawdata1_layers)))
        generator.shuffle(cycle)
        template_indices.extend(cycle)
    template_indices = template_indices[: args.num_samples]
    output_layers = [rawdata1_layers[index] for index in template_indices]

    output_dir.mkdir(parents=True)
    placement = _canonical_placement(len(output_layers))
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
            writer.writerows(layer[start : start + 8] for layer in output_layers)

    output_digest = _source_digest([placement_path, *csv_paths])
    source_digest = _source_digest(source_files)
    before_ratios = [float(item["before_expert_max_mean"]) for item in cap_stats]
    after_ratios_unshuffled = [float(item["after_expert_max_mean"]) for item in cap_stats]
    after_ratios = [after_ratios_unshuffled[index] for index in template_indices]
    changed_layers = sum(1 for item in cap_stats if item["changed"])
    total_redistributed = sum(
        int(cap_stats[index]["redistributed_rows"]) for index in template_indices
    )
    layer_map = ", ".join(
        f"{new}->{old}" for new, old in enumerate(template_indices)
    )
    readme = f"""# raw_data2

由 `workload/raw_data` 通过 `workload/build_raw_data2.py` 生成。它先执行
`raw_data1` 同款去冗余与缩放，再把每个 prefill-batch sample 的 expert-level
`max/mean` 控制到 `{args.min_max_mean:g}..{args.max_max_mean:g}`，最后按固定
seed 打散 sample 顺序。

注意：底层 JSON 仍使用 `layer_list` 字段，这是为了复用现有 loader 和 benchmark
代码；对 `raw_data2` 来说，`decode_*` 的每一行语义是一个 Qwen3-235B-style
MoE layer slot / prefill batch hotspot sample，不是 DSV3 的第 N 个 MoE layer。
本地只有 58 个经验模板；输出 94 个样本时，额外样本是固定 seed 的模板复用，不声称
是真实 Qwen3 routing trace。

| 项目 | raw_data2 |
|---|---:|
| prefill batch samples | {len(output_layers)} |
| target model layer count | {args.num_samples} |
| logical experts | 256 |
| storage layout | 32 × 8 |
| model ranks | {args.model_ranks} |
| tokens/rank | {args.tokens_per_rank:,} |
| TopK | {args.topk} |
| output rows/sample | {target_rows_per_layer:,} |
| primary policy | `{args.primary_policy}` |
| tail cap range | `{args.min_max_mean:g}..{args.max_max_mean:g}` |
| shuffled sample seed | {args.shuffle_seed} |
| empirical source templates | {len(rawdata1_layers)} |
| capped source templates | {changed_layers} |
| redistributed rows across all samples | {total_redistributed:,} |

| max/mean 指标 | min | mean | max |
|---|---:|---:|---:|
| before tail cap | {min(before_ratios):.4f} | {sum(before_ratios) / len(before_ratios):.4f} | {max(before_ratios):.4f} |
| after tail cap + shuffle | {min(after_ratios):.4f} | {sum(after_ratios) / len(after_ratios):.4f} | {max(after_ratios):.4f} |

原始 33 个运行时文件（JSON+CSV）tree SHA-256：

```text
{source_digest}
```

输出 33 个运行时文件（JSON+CSV，不含 README/plot）tree SHA-256：

```text
{output_digest}
```

Sample template map (`new_sample -> source_template`):

```text
{layer_map}
```

`plot/` 只保存这份 `raw_data2` 派生出来的 prefill-batch 负载图，不混放其他
workload 的图片。
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "batch_samples": len(output_layers),
                "target_model_layers": args.num_samples,
                "logical_experts": 256,
                "slots_per_device": 8,
                "model_ranks": args.model_ranks,
                "tokens_per_rank": args.tokens_per_rank,
                "topk": args.topk,
                "rows_per_sample": target_rows_per_layer,
                "source_tree_sha256": source_digest,
                "tree_sha256": output_digest,
                "max_mean_min": min(after_ratios),
                "max_mean_mean": sum(after_ratios) / len(after_ratios),
                "max_mean_max": max(after_ratios),
                "capped_source_layers": changed_layers,
                "shuffle_seed": args.shuffle_seed,
                "source_templates": len(rawdata1_layers),
                "all_layer_pre_scale_totals_conserved": all(
                    layer["original_total"] == layer["normalized_total"]
                    for layer in rawdata1_stats
                ),
                "all_layer_output_totals_match": all(
                    sum(layer) == target_rows_per_layer
                    for layer in output_layers
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
