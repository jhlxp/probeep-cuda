"""Reproducible CUDA benchmark for the complete ProbeEP planning path.

The timed diagnostic call includes histogram construction, compute planning,
network admission, server-local packing, finalization and route lowering.  It
therefore cannot accidentally report the greedy planner while omitting pack.
The production distributed path replaces the diagnostic all-rank histogram
with one segmented local histogram plus compact IPC/RDMA count exchange.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import deep_ep_cpp


PHASES = ("intent", "admission", "packing", "finalization")


def make_server_imbalanced_routes(world: int, tokens: int) -> torch.Tensor:
    if world not in (16, 32, 64, 128):
        raise ValueError("world must be one of 16, 32, 64, 128")
    token = torch.arange(
        world * tokens, dtype=torch.int64, device="cuda"
    ).view(world, tokens, 1)
    experts_per_server = 256 // (world // 8)
    hot_lane = torch.arange(6, dtype=torch.int64, device="cuda").view(
        1, 1, 6
    )
    cold_lane = torch.arange(2, dtype=torch.int64, device="cuda").view(
        1, 1, 2
    )
    return torch.cat(
        (
            (token * 6 + hot_lane) % experts_per_server,
            experts_per_server
            + (token * 2 + cold_lane) % experts_per_server,
        ),
        dim=2,
    ).contiguous()


def percentile(values: list[float], fraction: float) -> float:
    return values[round((len(values) - 1) * fraction)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", type=int, default=16)
    parser.add_argument("--tokens-per-rank", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--budget-mib", type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.iterations <= 0 or args.warmup < 0:
        raise ValueError("iterations must be positive and warmup nonnegative")

    routes = make_server_imbalanced_routes(
        args.world, args.tokens_per_rank
    )
    budgets = torch.full(
        (args.world,),
        args.budget_mib * 1024 * 1024,
        dtype=torch.int64,
        device="cuda",
    )
    for _ in range(args.warmup):
        plan = deep_ep_cpp.plan_probeep(routes, budgets)
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(args.iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        plan = deep_ep_cpp.plan_probeep(routes, budgets)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()

    counts = plan["plan_counts"].cpu().tolist()
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    phase_cycles = dict(zip(PHASES, counts[9:13], strict=True))
    # clock_rate is kHz, so cycles / clock_rate is milliseconds.  This is an
    # attribution estimate; CUDA-event/nsys operator time remains normative.
    phase_ms_at_advertised_clock = {
        phase: cycles / properties.clock_rate
        for phase, cycles in phase_cycles.items()
    }
    result = {
        "device": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "world": args.world,
        "servers": args.world // 8,
        "tokens_per_rank": args.tokens_per_rank,
        "topk": 8,
        "experts": 256,
        "budget_mib_per_endpoint": args.budget_mib,
        "timed_scope": (
            "diagnostic histogram + intent + admission + packing + "
            "finalization + route lowering"
        ),
        "latency_ms": {
            "minimum": samples[0],
            "p50": statistics.median(samples),
            "p95": percentile(samples, 0.95),
            "p99": percentile(samples, 0.99),
        },
        "phase_cycles": phase_cycles,
        "phase_ms_at_advertised_clock": phase_ms_at_advertised_clock,
        "phase_ms_sum": sum(phase_ms_at_advertised_clock.values()),
        "compute_intents": counts[3],
        "admitted_experts": counts[0],
        "weight_chunks": counts[1],
        "invariants": {
            "slot_overflow": counts[2],
            "negative_allocation": counts[6],
            "conservation_failure": counts[7],
            "planner_converged": counts[8],
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
