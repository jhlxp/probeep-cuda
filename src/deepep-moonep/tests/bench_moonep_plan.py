"""Microbenchmark for the standalone CUDA planner binding.

This measures the test binding, including its output allocations.  The
production Buffer API reuses the same kernels with ring-buffered output, so
the number is a conservative upper bound for the hot-path planner.
"""

from pathlib import Path
import argparse
import statistics
import sys

import torch

import deep_ep_cpp


PROBEEP_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROBEEP_ROOT))

from test.correctness.workloads import make_routing_workload  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--bias-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    workload = make_routing_workload(
        "server_preserving_skew",
        num_tokens=args.tokens,
        bias_ratio=args.bias_ratio,
        seed=args.seed,
    )
    topk = workload.topk_experts.cuda()

    for _ in range(args.warmup):
        deep_ep_cpp.plan_server_local(topk)
    torch.cuda.synchronize()

    samples_us = []
    for _ in range(args.iterations):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        deep_ep_cpp.plan_server_local(topk)
        end.record()
        end.synchronize()
        samples_us.append(begin.elapsed_time(end) * 1000.0)

    ordered = sorted(samples_us)
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
    print(
        f"planner tokens={args.tokens} routes={topk.numel()} "
        f"median_us={statistics.median(samples_us):.3f} p95_us={p95:.3f} "
        f"min_us={min(samples_us):.3f} max_us={max(samples_us):.3f}"
    )


if __name__ == "__main__":
    main()
