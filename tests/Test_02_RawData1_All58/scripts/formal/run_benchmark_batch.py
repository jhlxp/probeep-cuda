#!/usr/bin/env python3
"""Run multiple RawData1 layers in one persistent distributed worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from run_benchmark import cleanup_persistent_runtime, main as run_case


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-plan", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument(
        "--case-key",
        action="append",
        required=True,
        help="selected case as repeat:layer_id",
    )
    parser.add_argument("--profile", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_status(path: Path, payload: dict[str, object]) -> None:
    if int(os.environ["LOCAL_RANK"]) != 0:
        return
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    args = arguments()
    plan = json.loads(args.benchmark_plan.read_text(encoding="utf-8"))
    execution = plan["execution"]
    selected = {
        tuple(int(value) for value in key.split(":", maxsplit=1))
        for key in args.case_key
    }
    cases = [
        case
        for case in plan["cases"]
        if case["variant"] == args.variant
        and (int(case["repeat"]), int(case["layer_id"])) in selected
    ]
    if len(cases) != len(selected):
        raise ValueError("persistent batch case selection is incomplete")
    case_order = [
        (int(case["repeat"]), int(case["layer_id"])) for case in cases
    ]
    for index, (repeat, layer_id) in enumerate(case_order):
        if index == 0:
            if layer_id != 0:
                raise ValueError(
                    "a persistent layer-feedback chain must start at Layer 0"
                )
            continue
        previous_repeat, previous_layer = case_order[index - 1]
        contiguous = repeat == previous_repeat and layer_id == previous_layer + 1
        next_repeat = repeat == previous_repeat + 1 and layer_id == 0
        if not (contiguous or next_repeat):
            raise ValueError(
                "persistent cases must be canonical, contiguous layer order"
            )

    status_path = (
        Path(os.environ["PROBEEP_RUN_DIR"])
        / f"formal_schedule_node_{os.environ['NODE_RANK']}.jsonl"
    )
    try:
        for case_index, case in enumerate(cases):
            layer_id = int(case["layer_id"])
            repeat = int(case["repeat"])
            route_file = Path(case["route_file"]).resolve()
            if file_sha256(route_file) != case["routing_sha256"]:
                raise ValueError(f"layer {layer_id}: routing SHA-256 mismatch")
            os.environ["PROBEEP_ROUTE_FILE"] = str(route_file)
            os.environ["PROBEEP_ROUTING_SHA256"] = str(
                case["routing_sha256"]
            )
            # Keep the case-level seed separate from the active derived
            # version.  Otherwise persistent layers recursively multiply the
            # previous version and can overflow the C++ int64 contract.
            os.environ["PROBEEP_WEIGHT_BASE_VERSION"] = str(
                repeat * 1000 + layer_id + 1
            )
            command = [
                "--variant",
                args.variant,
                "--expert-mode",
                "grouped",
                "--runner-mode",
                "dual_microbatch_ht",
                "--workload",
                f"raw_data1_layer_{layer_id:02d}",
                "--warmup-iters",
                str(execution["warmup_iters_per_layer"]),
                "--measure-iters",
                str(execution["measure_iters_per_layer"]),
                "--repeat",
                str(repeat),
            ]
            if args.profile:
                command.append("--profile")
            base = {
                "timestamp_utc": utc_now(),
                "node_rank": int(os.environ["NODE_RANK"]),
                "case_index": case_index,
                "repeat": repeat,
                "layer_id": layer_id,
                "variant": args.variant,
                "routing_sha256": case["routing_sha256"],
            }
            append_status(status_path, {**base, "state": "STARTED"})
            started = time.monotonic()
            try:
                run_case(command, persistent_runtime=True)
            except BaseException:
                append_status(
                    status_path,
                    {
                        **base,
                        "timestamp_utc": utc_now(),
                        "state": "FAIL",
                        "elapsed_seconds": time.monotonic() - started,
                    },
                )
                raise
            append_status(
                status_path,
                {
                    **base,
                    "timestamp_utc": utc_now(),
                    "state": "PASS",
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
    finally:
        cleanup_persistent_runtime()


if __name__ == "__main__":
    main()
