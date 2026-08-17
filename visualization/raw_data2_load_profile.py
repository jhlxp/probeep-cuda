#!/usr/bin/env python3
"""Plot the 94-sample raw_data2 expert/rank load profile."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workload.gate import RawReceiveDataset


DEFAULT_DATA_DIR = ROOT / "workload/raw_data2"
DEFAULT_PLACEMENT = DEFAULT_DATA_DIR / "DSV3_32x8_256_unique.json"
DEFAULT_PLOT_DIR = DEFAULT_DATA_DIR / "plot"
DEFAULT_OUTPUT = DEFAULT_PLOT_DIR / "qwen3_235b_94_batch_load.png"
DEFAULT_SVG = DEFAULT_PLOT_DIR / "qwen3_235b_94_batch_load.svg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--placement-json", type=Path, default=DEFAULT_PLACEMENT)
    parser.add_argument("--num-model-ranks", type=int, default=16)
    parser.add_argument("--ranks-per-server", type=int, default=8)
    parser.add_argument("--tokens-per-rank", type=int, default=4096)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--svg", type=Path, default=DEFAULT_SVG)
    return parser.parse_args()


def style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color="#d8dee9", linewidth=0.8, alpha=0.65)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#334155")


def main() -> int:
    args = parse_args()
    dataset = RawReceiveDataset.load(args.placement_json)
    if dataset.num_logical_experts != 256:
        raise ValueError(
            f"raw_data2 must contain 256 experts, found {dataset.num_logical_experts}"
        )
    if args.num_model_ranks <= 0 or 256 % args.num_model_ranks:
        raise ValueError("num-model-ranks must be a positive divisor of 256")
    if (
        args.ranks_per_server <= 0
        or args.num_model_ranks % args.ranks_per_server
    ):
        raise ValueError("ranks-per-server must divide num-model-ranks")

    expected_rows = args.num_model_ranks * args.tokens_per_rank * args.topk
    expert_rows = np.asarray(dataset.logical_loads, dtype=np.int64)
    totals = expert_rows.sum(axis=1)
    if not np.all(totals == expected_rows):
        raise ValueError(
            "raw_data2 sample totals disagree with ranks*tokens*topk: "
            f"expected {expected_rows}, observed {sorted(set(totals.tolist()))}"
        )

    num_samples = dataset.num_layers
    experts_per_rank = dataset.num_logical_experts // args.num_model_ranks
    rank_rows = expert_rows.reshape(
        num_samples, args.num_model_ranks, experts_per_rank
    ).sum(axis=2)
    server_rows = rank_rows.reshape(
        num_samples,
        args.num_model_ranks // args.ranks_per_server,
        args.ranks_per_server,
    ).sum(axis=2)

    samples = np.arange(num_samples)
    expert_share = expert_rows / totals[:, None] * 100
    rank_share = rank_rows / totals[:, None] * 100
    expert_imbalance = expert_rows.max(axis=1) / expert_rows.mean(axis=1)
    rank_imbalance = rank_rows.max(axis=1) / rank_rows.mean(axis=1)
    server_imbalance = server_rows.max(axis=1) / server_rows.mean(axis=1)

    figure = plt.figure(figsize=(16, 9.5), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(3.2, 1.2))
    expert_axis = figure.add_subplot(grid[0, 0])
    expert_line_axis = figure.add_subplot(grid[1, 0], sharex=expert_axis)
    rank_axis = figure.add_subplot(grid[0, 1])
    rank_line_axis = figure.add_subplot(grid[1, 1], sharex=rank_axis)

    expert_colors = plt.get_cmap("turbo")(np.linspace(0.02, 0.98, 256))
    expert_axis.stackplot(
        samples,
        expert_share.T,
        colors=expert_colors,
        linewidth=0,
    )
    expert_axis.set_title("256 Experts · Scaled Receive Share", fontsize=18)
    expert_axis.set_ylabel("Expert load share (%)", fontsize=14)
    expert_axis.set_ylim(0, 100)
    expert_axis.tick_params(axis="x", labelbottom=False)
    style_axis(expert_axis)

    expert_line_axis.plot(
        samples, expert_imbalance, color="#19396c", linewidth=2.4
    )
    expert_line_axis.fill_between(
        samples, 1, expert_imbalance, color="#93c5fd", alpha=0.22
    )
    expert_line_axis.set_ylabel("Expert max/mean", fontsize=13)
    expert_line_axis.set_xlabel("Prefill batch index", fontsize=15)
    expert_line_axis.set_xlim(0, num_samples - 1)
    style_axis(expert_line_axis)

    rank_colors = plt.get_cmap("tab20")(
        np.linspace(0, 1, args.num_model_ranks)
    )
    rank_axis.stackplot(
        samples,
        rank_share.T,
        colors=rank_colors,
        labels=[f"R{rank}" for rank in range(args.num_model_ranks)],
        linewidth=0.15,
        edgecolor="white",
    )
    rank_axis.set_title(
        f"{args.num_model_ranks} Ranks · Static {experts_per_rank}-expert Placement",
        fontsize=18,
    )
    rank_axis.set_ylabel("Rank load share (%)", fontsize=14)
    rank_axis.set_ylim(0, 100)
    rank_axis.tick_params(axis="x", labelbottom=False)
    rank_axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=8,
        frameon=False,
        fontsize=9,
        columnspacing=0.8,
        handlelength=1.1,
    )
    style_axis(rank_axis)

    server_count = args.num_model_ranks // args.ranks_per_server
    rank_line_axis.plot(
        samples,
        rank_imbalance,
        color="#19396c",
        linewidth=2.4,
        label=f"{args.num_model_ranks}-rank max/mean",
    )
    rank_line_axis.plot(
        samples,
        server_imbalance,
        color="#e76f51",
        linewidth=2.2,
        label=f"{server_count}-server max/mean",
    )
    rank_line_axis.axhline(1, color="#64748b", linewidth=0.9, linestyle="--")
    rank_line_axis.set_ylabel("Imbalance", fontsize=13)
    rank_line_axis.set_xlabel("Prefill batch index", fontsize=15)
    rank_line_axis.set_xlim(0, num_samples - 1)
    rank_line_axis.legend(loc="upper right", frameon=False, fontsize=10)
    style_axis(rank_line_axis)

    figure.suptitle(
        "Qwen3-235B-style Fixed-Layer Prefill Batch Hotspots · "
        f"{args.tokens_per_rank} Tokens/Rank · TopK={args.topk}",
        fontsize=22,
        fontweight="bold",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    if args.svg:
        args.svg.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.svg, bbox_inches="tight")
    plt.close(figure)

    print(
        f"samples={num_samples} experts={dataset.num_logical_experts} "
        f"rows/sample={expected_rows} "
        f"expert_max_mean=[{expert_imbalance.min():.4f},"
        f"{expert_imbalance.max():.4f}] output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
