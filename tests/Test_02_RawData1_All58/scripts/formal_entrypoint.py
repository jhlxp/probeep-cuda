#!/usr/bin/env python3
"""Execute one immutable RawData1 benchmark plan on every allocated H20 node."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
WORKER = HERE / "formal/run_benchmark_batch.py"
VARIANTS = (
    "nccl",
    "deepep",
    "deepep_moonep_on",
    "ultraep_hybridep",
    "probeep",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.resolve().read_text(encoding="utf-8"))
    if plan.get("schema") != "probeep.multinode.benchmark_plan.v1":
        raise ValueError("unsupported benchmark plan schema")
    if tuple(plan.get("variants", ())) != VARIANTS:
        raise ValueError("benchmark plan does not contain the canonical five variants")
    cases = plan.get("cases")
    if not isinstance(cases, list) or len(cases) != plan.get("case_count"):
        raise ValueError("benchmark plan case_count is inconsistent")
    expected_world = int(os.environ["NNODES"]) * int(os.environ["GPUS_PER_NODE"])
    if plan.get("topology", {}).get("world_size") != expected_world:
        raise ValueError("benchmark plan world size does not match the allocation")
    for index, case in enumerate(cases):
        if case.get("variant") not in VARIANTS:
            raise ValueError(f"case {index}: unknown variant")
        route = Path(str(case.get("route_file", ""))).resolve()
        if not route.is_file() or file_sha256(route) != case.get("routing_sha256"):
            raise ValueError(f"case {index}: route file is missing or its digest changed")
    return plan


def append_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def command_for(
    variant: str,
    cases: list[dict[str, Any]],
    plan_path: Path,
    *,
    profile_worker: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nnodes={os.environ['NNODES']}",
        f"--nproc-per-node={os.environ['GPUS_PER_NODE']}",
        f"--node-rank={os.environ['NODE_RANK']}",
        f"--master-addr={os.environ['MASTER_ADDR']}",
        f"--master-port={os.environ['MASTER_PORT']}",
        str(WORKER),
        "--benchmark-plan",
        str(plan_path.resolve()),
        "--variant",
        variant,
    ]
    for case in cases:
        command.extend(
            (
                "--case-key",
                f"{int(case['repeat'])}:{int(case['layer_id'])}",
            )
        )
    if profile_worker:
        command.append("--profile")
    return command


def batch_environment() -> dict[str, str]:
    environment = os.environ.copy()
    python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(ROOT) + (
        os.pathsep + python_path if python_path else ""
    )
    raw_dir = Path(environment["PROBEEP_RUN_DIR"]).resolve() / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    environment.update(
        {
            "PROBEEP_ROOT": str(ROOT),
            "DEEPEP_ROOT": str(ROOT / "src/deepep"),
            "DEEPEP_MOONEP_ROOT": str(ROOT / "src/deepep-moonep"),
            "DEEPEP_PROBEEP_ROOT": str(ROOT / "src/deepep-probeep"),
            "ULTRAEP_HYBRIDEP_ROOT": environment.get(
                "ULTRAEP_HYBRIDEP_ROOT",
                str(ROOT / "build/ultraep-hybridep-e0a5b1d9"),
            ),
            "PROBEEP_RUN_DIR": str(raw_dir),
            "NUM_TOKENS_PER_RANK": os.environ.get("TOKENS_PER_RANK", "4096"),
            "BENCHMARK_RUNNER_MODE": "dual_microbatch_ht",
            "BENCHMARK_ENABLE_ATTENTION_OVERLAP": "1",
            "PROBEEP_ENABLE_ATTENTION_FEEDBACK": "1",
            "EXPERT_MODE": "grouped",
        }
    )
    return environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_plan = os.environ.get("PROBEEP_BENCHMARK_PLAN")
    parser.add_argument(
        "--benchmark-plan",
        type=Path,
        default=Path(default_plan) if default_plan else None,
        required=default_plan is None,
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--max-cases", type=int, default=0, help="diagnostic only; zero executes the complete plan")
    parser.add_argument("--profile-worker", action="store_true")
    parser.add_argument(
        "--include-layer",
        action="append",
        type=int,
        default=[],
        help="diagnostic only; execute cases from the selected layer IDs",
    )
    parser.add_argument(
        "--exclude-variant",
        action="append",
        choices=VARIANTS,
        default=[],
        help="diagnostic profiling only; omit a backend that cannot run under profiler injection",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not WORKER.is_file():
        raise FileNotFoundError(WORKER)
    plan = load_plan(args.benchmark_plan)
    if args.max_cases < 0:
        raise ValueError("--max-cases cannot be negative")
    if (
        args.max_cases
        and not args.profile_worker
        and os.environ.get("PROBEEP_FORMAL_PAPER_MODE", "1") == "1"
    ):
        raise ValueError("paper mode cannot execute a partial benchmark plan")
    if args.validate_only:
        print(json.dumps({"status": "PASS", "case_count": len(plan["cases"]), "variants": list(VARIANTS)}))
        return

    run_dir = Path(os.environ["PROBEEP_RUN_DIR"]).resolve()
    node_rank = int(os.environ["NODE_RANK"])
    status_path = (
        run_dir / "raw" / f"formal_batch_schedule_node_{node_rank}.jsonl"
    )
    excluded = set(args.exclude_variant)
    included_layers = set(args.include_layer)
    cases = [
        case
        for case in plan["cases"]
        if case["variant"] not in excluded
        and (
            not included_layers
            or int(case["layer_id"]) in included_layers
        )
    ]
    if args.max_cases:
        cases = cases[: args.max_cases]
    batches = [
        (variant, [case for case in cases if case["variant"] == variant])
        for variant in VARIANTS
    ]
    batches = [(variant, batch) for variant, batch in batches if batch]
    for index, (variant, batch) in enumerate(batches):
        base = {
            "timestamp_utc": utc_now(),
            "node_rank": node_rank,
            "batch_index": index,
            "variant": variant,
            "case_count": len(batch),
            "case_keys": [
                f"{int(case['repeat'])}:{int(case['layer_id'])}"
                for case in batch
            ],
        }
        append_status(status_path, {**base, "state": "STARTED"})
        started = time.monotonic()
        result = subprocess.run(
            command_for(
                variant,
                batch,
                args.benchmark_plan,
                profile_worker=args.profile_worker,
            ),
            cwd=ROOT,
            env=batch_environment(),
            check=False,
        )
        append_status(
            status_path,
            {
                **base,
                "timestamp_utc": utc_now(),
                "state": "PASS" if result.returncode == 0 else "FAIL",
                "returncode": result.returncode,
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
