#!/usr/bin/env python3
"""Plot all 58 scaled DSV3 expert/rank load shares and imbalance."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workload.gate import RawReceiveDataset, load_full_dsv3_moe_trace


DEFAULT_PLACEMENT = ROOT / "workload/raw_data1/DSV3_32x8_256_unique.json"
DEFAULT_PLOT_DIR = ROOT / "workload/raw_data1/plot"
DEFAULT_OUTPUT = DEFAULT_PLOT_DIR / "full_dsv3_58_layer_load.png"
DEFAULT_SVG = DEFAULT_PLOT_DIR / "full_dsv3_58_layer_load.svg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--placement-json", type=Path, default=DEFAULT_PLACEMENT)
    parser.add_argument("--num-model-ranks", type=int, default=16)
    parser.add_argument("--ranks-per-server", type=int, default=8)
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
    trace = load_full_dsv3_moe_trace(
        RawReceiveDataset.load(args.placement_json),
        num_model_ranks=args.num_model_ranks,
        ranks_per_logical_server=args.ranks_per_server,
    )
    layers = np.arange(trace.num_layers)
    expert_share = trace.expert_rows / trace.expert_rows.sum(axis=1, keepdims=True) * 100
    rank_share = (
        trace.static_rank_rows
        / trace.static_rank_rows.sum(axis=1, keepdims=True)
        * 100
    )
    expert_imbalance = (
        trace.expert_rows.max(axis=1) / trace.expert_rows.mean(axis=1)
    )
    rank_imbalance = (
        trace.static_rank_rows.max(axis=1) / trace.static_rank_rows.mean(axis=1)
    )
    server_imbalance = (
        trace.static_server_rows.max(axis=1)
        / trace.static_server_rows.mean(axis=1)
    )

    figure = plt.figure(figsize=(16, 9.5), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(3.2, 1.2))
    expert_axis = figure.add_subplot(grid[0, 0])
    expert_line_axis = figure.add_subplot(grid[1, 0], sharex=expert_axis)
    rank_axis = figure.add_subplot(grid[0, 1])
    rank_line_axis = figure.add_subplot(grid[1, 1], sharex=rank_axis)

    expert_colors = plt.get_cmap("turbo")(np.linspace(0.02, 0.98, 256))
    expert_axis.stackplot(
        layers,
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
        layers, expert_imbalance, color="#19396c", linewidth=2.4
    )
    expert_line_axis.fill_between(
        layers, 1, expert_imbalance, color="#93c5fd", alpha=0.22
    )
    expert_line_axis.set_ylabel("Expert max/mean", fontsize=13)
    expert_line_axis.set_xlabel("MoE layer index", fontsize=15)
    expert_line_axis.set_xlim(0, trace.num_layers - 1)
    style_axis(expert_line_axis)

    rank_count = trace.static_rank_rows.shape[1]
    server_count = trace.static_server_rows.shape[1]
    rank_colors = plt.get_cmap("tab20")(np.linspace(0, 1, rank_count))
    rank_axis.stackplot(
        layers,
        rank_share.T,
        colors=rank_colors,
        labels=[f"R{rank}" for rank in range(rank_count)],
        linewidth=0.15,
        edgecolor="white",
    )
    rank_axis.axhline(50, color="#334155", linewidth=0.9, linestyle="--")
    rank_axis.set_title(
        f"{rank_count} Ranks · Static {256 // rank_count}-expert Placement",
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

    rank_line_axis.plot(
        layers,
        rank_imbalance,
        color="#19396c",
        linewidth=2.4,
        label=f"{rank_count}-rank max/mean",
    )
    rank_line_axis.plot(
        layers,
        server_imbalance,
        color="#e76f51",
        linewidth=2.2,
        label=f"{server_count}-server max/mean",
    )
    rank_line_axis.axhline(1, color="#64748b", linewidth=0.9, linestyle="--")
    rank_line_axis.set_ylabel("Imbalance", fontsize=13)
    rank_line_axis.set_xlabel("MoE layer index", fontsize=15)
    rank_line_axis.set_xlim(0, trace.num_layers - 1)
    rank_line_axis.legend(loc="upper right", frameon=False, fontsize=10)
    style_axis(rank_line_axis)

    figure.suptitle(
        "DSV3 58-layer MoE Distribution · 4096 Tokens/Rank · TopK=8",
        fontsize=22,
        fontweight="bold",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    if args.svg:
        args.svg.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.svg, bbox_inches="tight")
    plt.close(figure)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
