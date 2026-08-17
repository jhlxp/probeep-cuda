#!/usr/bin/env python3
"""Summarize the formal dual-microbatch timeline from an nsys SQLite export."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


PROCESS_MASK = ~((1 << 24) - 1)
ITERATION = re.compile(
    r"^(nccl|deepep|deepep_moonep_on|ultraep_hybridep|probeep)"
    r"/measurement_iteration$"
)


def table_exists(connection: sqlite3.Connection, name: str) -> bool:
    return connection.execute(
        "select 1 from sqlite_master where type='table' and name=?", (name,)
    ).fetchone() is not None


def columns(connection: sqlite3.Connection, name: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f"pragma table_info({name})")}


def process_key(value: int | None) -> int | None:
    return None if value is None else int(value) & PROCESS_MASK


def merge(windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted((start, end) for start, end in windows if end > start)
    if not ordered:
        return []
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        old_start, old_end = merged[-1]
        if start <= old_end:
            merged[-1] = (old_start, max(old_end, end))
        else:
            merged.append((start, end))
    return merged


def duration(windows: list[tuple[int, int]]) -> int:
    return sum(end - start for start, end in merge(windows))


def intersection(first: list[tuple[int, int]], second: list[tuple[int, int]]) -> int:
    left, right = merge(first), merge(second)
    i = j = total = 0
    while i < len(left) and j < len(right):
        start, end = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        total += max(0, end - start)
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return total


def load_nvtx(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    if not table_exists(connection, "NVTX_EVENTS"):
        raise RuntimeError("SQLite report has no NVTX_EVENTS table")
    available = columns(connection, "NVTX_EVENTS")
    if "text" in available and "textId" in available and table_exists(connection, "StringIds"):
        query = (
            "select n.start,n.end,coalesce(n.text,s.value),n.globalTid "
            "from NVTX_EVENTS n left join StringIds s on n.textId=s.id "
            "where n.end is not null and coalesce(n.text,s.value) is not null"
        )
    elif "text" in available:
        query = (
            "select start,end,text,globalTid from NVTX_EVENTS "
            "where end is not null and text is not null"
        )
    else:
        raise RuntimeError("NVTX_EVENTS has no resolvable text column")
    return [
        {
            "start": int(start),
            "end": int(end),
            "label": str(label),
            "process": process_key(global_tid),
        }
        for start, end, label, global_tid in connection.execute(query)
    ]


def load_kernels(
    connection: sqlite3.Connection,
) -> dict[int | None, list[tuple[int, int, int, int]]]:
    table = "CUPTI_ACTIVITY_KIND_KERNEL"
    required = {"start", "end", "deviceId", "streamId", "globalPid"}
    if not table_exists(connection, table) or not required.issubset(columns(connection, table)):
        return {}
    output: dict[int | None, list[tuple[int, int, int, int]]] = defaultdict(list)
    query = (
        "select start,end,deviceId,streamId,globalPid "
        f"from {table} where end>start"
    )
    for start, end, device, stream, pid in connection.execute(query):
        output[None if pid is None else int(pid)].append(
            (int(start), int(end), int(device), int(stream))
        )
    return output


def event_kernels(
    kernels: dict[int | None, list[tuple[int, int, int, int]]],
    event: dict[str, Any],
) -> list[tuple[int, int, int, int]]:
    return [
        row
        for row in kernels.get(event["process"], ())
        if row[0] < event["end"] and row[1] > event["start"]
    ]


def summarize(path: Path) -> dict[str, Any]:
    with sqlite3.connect(path) as connection:
        events = load_nvtx(connection)
        kernels = load_kernels(connection)
    iterations = []
    for outer in events:
        match = ITERATION.match(outer["label"])
        if match is None:
            continue
        children = [
            item for item in events
            if item["process"] == outer["process"]
            and item["start"] < outer["end"]
            and item["end"] > outer["start"]
            and item is not outer
        ]

        def stage(predicate: Any) -> tuple[list[tuple[int, int]], list[str]]:
            rows = [item for item in children if predicate(item["label"])]
            matched = [kernel for item in rows for kernel in event_kernels(kernels, item)]
            windows = merge([(start, end) for start, end, _, _ in matched])
            streams = sorted({f"gpu{device}:stream{stream}" for _, _, device, stream in matched})
            return windows, streams

        network, network_streams = stage(
            lambda label: label.endswith("/ht_dispatch")
        )
        attention, attention_streams = stage(
            lambda label: label.startswith("attention_or_gate/")
        )
        expert, expert_streams = stage(lambda label: label.endswith("/expert_mlp"))
        combine, combine_streams = stage(lambda label: label.endswith("/ht_combine"))
        feedback, feedback_streams = stage(
            lambda label: label == "probeep/feedback_prepare"
        )
        compute = merge(attention + expert)
        network_ns = duration(network)
        compute_ns = duration(compute)
        overlap_ns = intersection(network, compute)
        iterations.append(
            {
                "variant": match.group(1),
                "process_key": outer["process"],
                "iteration_nvtx_ms": (outer["end"] - outer["start"]) / 1e6,
                "network_gpu_ms": network_ns / 1e6,
                "attention_gpu_ms": duration(attention) / 1e6,
                "expert_gpu_ms": duration(expert) / 1e6,
                "combine_gpu_ms": duration(combine) / 1e6,
                "feedback_prepare_gpu_ms": duration(feedback) / 1e6,
                "network_compute_overlap_ms": overlap_ns / 1e6,
                "network_compute_overlap_percent": (
                    100.0 * overlap_ns / network_ns if network_ns else 0.0
                ),
                "streams": {
                    "network": network_streams,
                    "attention": attention_streams,
                    "expert": expert_streams,
                    "combine": combine_streams,
                    "feedback_prepare": feedback_streams,
                },
            }
        )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in iterations:
        grouped[row["variant"]].append(row)
    variants = [
        {
            "variant": variant,
            "iterations": len(rows),
            "iteration_nvtx_ms": fmean(row["iteration_nvtx_ms"] for row in rows),
            "network_gpu_ms": fmean(row["network_gpu_ms"] for row in rows),
            "attention_gpu_ms": fmean(row["attention_gpu_ms"] for row in rows),
            "expert_gpu_ms": fmean(row["expert_gpu_ms"] for row in rows),
            "combine_gpu_ms": fmean(row["combine_gpu_ms"] for row in rows),
            "feedback_prepare_gpu_ms": fmean(
                row["feedback_prepare_gpu_ms"] for row in rows
            ),
            "network_compute_overlap_percent": fmean(
                row["network_compute_overlap_percent"] for row in rows
            ),
        }
        for variant, rows in sorted(grouped.items())
    ]
    return {
        "schema": "probeep.nsys.overlap.v1",
        "source_sqlite": str(path),
        "measurement_iterations": len(iterations),
        "variant_summary": variants,
        "iterations": iterations,
        "note": "GPU intervals are NVTX/kernel intersections; final dependency judgement uses the .nsys-rep timeline.",
    }


def table(payload: dict[str, Any]) -> str:
    lines = [
        str(payload["note"]),
        "variant iterations iter_ms network_ms attention_ms expert_ms combine_ms feedback_ms overlap%",
    ]
    for row in payload["variant_summary"]:
        lines.append(
            f"{row['variant']:<22} {row['iterations']:>5} "
            f"{row['iteration_nvtx_ms']:>8.3f} {row['network_gpu_ms']:>10.3f} "
            f"{row['attention_gpu_ms']:>12.3f} {row['expert_gpu_ms']:>9.3f} "
            f"{row['combine_gpu_ms']:>10.3f} "
            f"{row['feedback_prepare_gpu_ms']:>11.3f} "
            f"{row['network_compute_overlap_percent']:>8.1f}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--text-output", required=True, type=Path)
    args = parser.parse_args()
    payload = summarize(args.sqlite)
    args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.text_output.write_text(table(payload), encoding="utf-8")
    print(args.json_output)
    print(args.text_output)


if __name__ == "__main__":
    main()
