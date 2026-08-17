#!/usr/bin/env python3
"""Build the fixed 20-layer multi-node ProbeEP benchmark plan."""

from __future__ import annotations

import importlib.util
from pathlib import Path


PLAN_MODULE = Path(__file__).resolve().parent / "scripts/benchmark_plan.py"
SPEC = importlib.util.spec_from_file_location("probeep_benchmark_plan", PLAN_MODULE)
assert SPEC is not None and SPEC.loader is not None
benchmark_plan = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark_plan)


if __name__ == "__main__":
    benchmark_plan.run_cli(
        expected_selector="raw_data1_eval20",
        expected_layer_ids=list(range(20)),
        paper_eligible=False,
    )
