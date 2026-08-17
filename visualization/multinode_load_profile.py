#!/usr/bin/env python3
"""Create the self-contained multi-node five-algorithm result bundle.

The on-disk contract uses one consolidated ``result.json`` per multi-node
replicate and one ``visualization_bundle.zip`` whose only member is
``load_profile.html``.  Raw CSV/JSONL telemetry may live either in the
replicate root (legacy runner) or in its ``raw/`` subdirectory.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
import zipfile


METHOD_ORDER = (
    ("nccl", "na", "NCCL"),
    ("deepep", "na", "DeepEP"),
    ("deepep_moonep", "on", "MoonEP"),
    ("ultraep", "hybridep", "UltraEP + HybridEP"),
    ("probeep", "server_first", "ProbeEP"),
)
PHASES = (
    "plan_ms",
    "layout_materialize_ms",
    "weight_prefetch_ms",
    "dispatch_ms",
    "expert_compute_ms",
    "combine_ms",
    "grad_reduce_ms",
)
SERVER_SIZE = 8


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--workload", default="server_imbalanced")
    parser.add_argument("--direction", default="forward")
    parser.add_argument("--scope", default="full_moe_grouped")
    parser.add_argument("--runner-mode", default="dual_microbatch_ht")
    parser.add_argument("--zip-output", "--output-zip", dest="zip_output", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--title", default="ProbeEP 多机五算法负载与流水线")
    return parser.parse_args()


def rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def artifact(run_dir: Path, name: str) -> Path:
    """Resolve a replicate artifact without weakening the canonical layout."""
    canonical = run_dir / "raw" / name
    return canonical if canonical.is_file() else run_dir / name


def json_lines(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def center(values: list[float], *, aggregate: bool) -> float:
    if not values:
        return 0.0
    return float(statistics.fmean(values) if aggregate else statistics.median(values))


def selected_workloads(workload: str) -> tuple[set[str], bool]:
    prefix = "raw_data1_layers_"
    if not workload.startswith(prefix):
        return {workload}, False
    parts = workload[len(prefix):].split("_")
    if len(parts) != 2:
        raise ValueError(f"invalid raw_data1 layer aggregate: {workload}")
    start, end = (int(parts[0]), int(parts[1]))
    if start > end:
        raise ValueError(f"invalid raw_data1 layer aggregate: {workload}")
    return {f"raw_data1_layer_{index}" for index in range(start, end + 1)}, True


def aggregate_percentile(
    rows: list[dict[str, str]],
    *,
    workloads: set[str],
    field: str,
    fraction: float,
    aggregate: bool,
) -> float:
    if not aggregate:
        return _percentile([float(row[field]) for row in rows], fraction)
    by_workload: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_workload[row["workload"]].append(float(row[field]))
    return statistics.fmean(
        _percentile(by_workload[workload], fraction)
        for workload in sorted(workloads)
        if by_workload.get(workload)
    )


def aggregate_max(
    rows: list[dict[str, str]],
    *,
    workloads: set[str],
    field: str,
    aggregate: bool,
) -> float:
    if not aggregate:
        return max(float(row[field]) for row in rows)
    by_workload: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_workload[row["workload"]].append(float(row[field]))
    return statistics.fmean(
        max(by_workload[workload])
        for workload in sorted(workloads)
        if by_workload.get(workload)
    )


def mean_plan_list(rows: list[dict[str, object]], field: str) -> list[float]:
    sequences = [
        row.get(field, [])
        for row in rows
        if isinstance(row.get(field, []), list)
    ]
    if not sequences:
        return []
    width = min(len(sequence) for sequence in sequences)
    return [
        statistics.fmean(float(sequence[index]) for sequence in sequences)
        for index in range(width)
    ]


def server_sums(values: list[float], server_size: int = SERVER_SIZE) -> list[float]:
    return [
        sum(values[offset: offset + server_size])
        for offset in range(0, len(values), server_size)
    ]


def select(
    records: list[dict[str, str]],
    *,
    workloads: set[str],
    direction: str,
    scope: str,
    runner_mode: str,
) -> list[dict[str, str]]:
    selected = [
        row for row in records
        if row.get("workload") in workloads
        and row.get("direction") == direction
        and row.get("benchmark_scope") == scope
        # Some pre-fix expert telemetry has the schema column but an empty
        # value.  It is still isolated by run directory and can inherit the
        # explicitly selected report mode.
        and (row.get("runner_mode") or runner_mode) == runner_mode
    ]
    if not selected:
        raise ValueError(
            "no rows match "
            f"workloads={sorted(workloads)}, direction={direction}, "
            f"scope={scope}, runner_mode={runner_mode}"
        )
    return selected


def build_payload(args: argparse.Namespace) -> dict[str, object]:
    for consolidated_name in ("summary.json", "result.json"):
        consolidated = args.run_dir / consolidated_name
        if consolidated.is_file():
            loaded = json.loads(consolidated.read_text(encoding="utf-8"))
            if loaded.get("schema") == "probeep.multinode.load_profile.v1":
                loaded["title"] = args.title
                return loaded

    workloads, aggregate = selected_workloads(args.workload)
    rank_rows = select(
        rows(artifact(args.run_dir, "rank_samples.csv")),
        workloads=workloads,
        direction=args.direction,
        scope=args.scope,
        runner_mode=args.runner_mode,
    )
    iteration_rows = select(
        rows(artifact(args.run_dir, "iterations.csv")),
        workloads=workloads,
        direction=args.direction,
        scope=args.scope,
        runner_mode=args.runner_mode,
    )
    routing_sha256: dict[str, str] = {}
    for workload in sorted(workloads):
        hashes = {
            row.get("routing_sha256", "")
            for row in rank_rows
            if row.get("workload") == workload
        }
        hashes.discard("")
        if len(hashes) != 1:
            raise ValueError(
                f"{workload}: expected exactly one shared routing SHA, got {len(hashes)}"
            )
        routing_sha256[workload] = next(iter(hashes))
    rank_expert_path = artifact(args.run_dir, "rank_expert_samples.csv")
    rank_expert_rows = (
        select(
            rows(rank_expert_path),
            workloads=workloads,
            direction=args.direction,
            scope=args.scope,
            runner_mode=args.runner_mode,
        )
        if rank_expert_path.is_file()
        else []
    )
    methods = []
    world_size = 0
    for system, balance, label in METHOD_ORDER:
        ranks = [
            row for row in rank_rows
            if row.get("system") == system and row.get("balance") == balance
        ]
        iterations = [
            row for row in iteration_rows
            if row.get("system") == system and row.get("balance") == balance
        ]
        if not ranks or not iterations:
            continue
        by_rank: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in ranks:
            by_rank[int(row["global_rank"])].append(row)
        rank_ids = sorted(by_rank)
        if rank_ids != list(range(len(rank_ids))):
            raise ValueError(
                f"{label}: expected contiguous node-major rank samples"
            )
        world_size = max(world_size, len(rank_ids))
        home = [
            center([float(x["home_load"]) for x in by_rank[r]], aggregate=aggregate)
            for r in rank_ids
        ]
        execute = [
            center([float(x["exec_load"]) for x in by_rank[r]], aggregate=aggregate)
            for r in rank_ids
        ]
        padded = [
            center(
                [float(x["exec_load"]) + float(x["padding_rows"]) for x in by_rank[r]],
                aggregate=aggregate,
            )
            for r in rank_ids
        ]
        server_home = server_sums(home)
        server_execute = server_sums(execute)
        server_padded = server_sums(padded)
        rank_raw_max_mean = max(execute) / statistics.fmean(execute)
        rank_compute_max_mean = max(padded) / statistics.fmean(padded)
        phases = {
            name: center(
                [float(row[name.replace("_ms", "_max_ms")]) for row in iterations],
                aggregate=aggregate,
            )
            for name in PHASES
        }
        method_rank_expert = [
            row for row in rank_expert_rows
            if row.get("system") == system and row.get("balance") == balance
        ]
        rank_expert_raw: list[list[float]] = []
        rank_expert_padded: list[list[float]] = []
        if method_rank_expert:
            grouped: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
            for row in method_rank_expert:
                grouped[(int(row["global_rank"]), int(row["expert_id"]))].append(row)
            expert_ids = sorted({key[1] for key in grouped})
            rank_expert_raw = [
                [
                    center(
                        [float(item["raw_rows"]) for item in grouped.get((rank, expert), [])],
                        aggregate=aggregate,
                    )
                    for expert in expert_ids
                ]
                for rank in rank_ids
            ]
            rank_expert_padded = [
                [
                    center(
                        [
                            float(item.get("padded_rows", item["raw_rows"]))
                            for item in grouped.get((rank, expert), [])
                        ],
                        aggregate=aggregate,
                    )
                    for expert in expert_ids
                ]
                for rank in rank_ids
            ]
        methods.append(
            {
                "system": system,
                "balance": balance,
                "label": label,
                "home": home,
                "execute": execute,
                "padded": padded,
                "server_home": server_home,
                "server_execute": server_execute,
                "server_padded": server_padded,
                "rank_raw_max_mean": rank_raw_max_mean,
                "rank_compute_max_mean": rank_compute_max_mean,
                "rank_max_mean": rank_compute_max_mean,
                "server_max_mean": max(server_execute) / statistics.fmean(server_execute),
                "p99_ms": aggregate_percentile(
                    iterations,
                    workloads=workloads,
                    field="e2e_max_ms",
                    fraction=0.99,
                    aggregate=aggregate,
                ),
                "max_ms": aggregate_max(
                    iterations,
                    workloads=workloads,
                    field="e2e_max_ms",
                    aggregate=aggregate,
                ),
                "phases": phases,
                "rank_expert_raw": rank_expert_raw,
                "rank_expert_padded": rank_expert_padded,
            }
        )

    if len(methods) != len(METHOD_ORDER):
        present = {(item["system"], item["balance"]) for item in methods}
        missing = [
            label for system, balance, label in METHOD_ORDER
            if (system, balance) not in present
        ]
        raise ValueError(f"five-algorithm result is incomplete: {', '.join(missing)}")
    if any(not item["rank_expert_raw"] for item in methods):
        raise ValueError(
            "rank_expert_samples.csv must cover all five algorithms"
        )

    rdma_path_csv = artifact(args.run_dir, "rdma_path_load.csv")
    rdma_paths = []
    if rdma_path_csv.is_file():
        rdma_path_rows = [
            row for row in rows(rdma_path_csv)
            if row.get("workload") in workloads
            and row.get("direction") == args.direction
            and row.get("benchmark_scope") == args.scope
            and row.get("runner_mode", "sync_single") == args.runner_mode
            and row.get("system") == "probeep"
        ]
        by_path: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in rdma_path_rows:
            by_path[int(row["path_id"])].append(row)
        for path_id in sorted(by_path):
            group = by_path[path_id]
            paths = sorted(
                {
                    (int(row["source_rank"]), int(row["destination_rank"]))
                    for row in group
                }
            )
            rdma_paths.append(
                {
                    "path_id": path_id,
                    "path": ", ".join(f"R{src}→R{dst}" for src, dst in paths),
                    "tx_bytes": center(
                        [float(row["tx_bytes"]) for row in group],
                        aggregate=aggregate,
                    ),
                    "rx_bytes": center(
                        [float(row["rx_bytes"]) for row in group],
                        aggregate=aggregate,
                    ),
                    "chunk_count": center(
                        [float(row["chunk_count"]) for row in group],
                        aggregate=aggregate,
                    ),
                    "dispatch_bytes": center(
                        [float(row.get("dispatch_bytes", 0)) for row in group],
                        aggregate=aggregate,
                    ),
                    "weight_bytes": center(
                        [float(row.get("weight_bytes", row["tx_bytes"])) for row in group],
                        aggregate=aggregate,
                    ),
                    "active_time_us": center(
                        [float(row["active_time_us"]) for row in group],
                        aggregate=aggregate,
                    ) if group[0].get("active_time_us") not in (None, "") else None,
                }
            )
            current = rdma_paths[-1]
            active_time_us = current["active_time_us"]
            current["measured_gbps"] = (
                max(float(current["tx_bytes"]), float(current["rx_bytes"]))
                * 8.0
                / float(active_time_us)
                / 1e3
                if active_time_us
                else None
            )

    plans = [
        row
        for row in json_lines(artifact(args.run_dir, "probeep_plan_summary.jsonl"))
        if row.get("workload") in workloads
        # Pre-runner telemetry did not carry runner_mode.  It still belongs
        # to the explicitly selected run directory, so inherit the report's
        # requested mode instead of silently dropping the production plan.
        and row.get("runner_mode", args.runner_mode) == args.runner_mode
    ]
    plan_summary: dict[str, object] = {}
    if plans:
        latest = max(plans, key=lambda row: int(row.get("iteration", -1)))
        source = latest.get("source", "")
        if aggregate:
            source = f"{source}_mean_layers" if source else "mean_layers"
        plan_summary = {
            "observations": len(plans),
            "iteration": int(latest.get("iteration", -1)),
            "source": source,
            "server_load_before": mean_plan_list(plans, "server_load_before")
            if aggregate else latest.get("server_load_before", []),
            "server_load_after": mean_plan_list(plans, "server_load_after")
            if aggregate else latest.get("server_load_after", []),
            "server_padded_load_before": mean_plan_list(
                plans, "server_padded_load_before"
            ) if aggregate else latest.get("server_padded_load_before", []),
            "server_padded_load_after": mean_plan_list(
                plans, "server_padded_load_after"
            ) if aggregate else latest.get("server_padded_load_after", []),
            "migration_budget_bytes": mean_plan_list(
                plans, "migration_budget_bytes"
            ) if aggregate else latest.get("migration_budget_bytes", []),
            "admitted_count": center(
                [len(row.get("admitted_experts", [])) for row in plans],
                aggregate=aggregate,
            ),
            "deferred_count": center(
                [len(row.get("deferred_experts", [])) for row in plans],
                aggregate=aggregate,
            ),
        }
    if not plan_summary:
        raise ValueError("probeep_plan_summary.jsonl is required")

    experts = []
    expert_path = artifact(args.run_dir, "expert_samples.csv")
    if expert_path.is_file():
        expert_rows = select(
            rows(expert_path),
            workloads=workloads,
            direction=args.direction,
            scope=args.scope,
            runner_mode=args.runner_mode,
        )
        selected_system = next(
            (
                (system, balance)
                for system, balance, _ in METHOD_ORDER
                if any(
                    row.get("system") == system
                    and row.get("balance") == balance
                    for row in expert_rows
                )
            ),
            None,
        )
        if selected_system is not None:
            by_expert: dict[int, list[float]] = defaultdict(list)
            for row in expert_rows:
                if (row.get("system"), row.get("balance")) == selected_system:
                    by_expert[int(row["expert_id"])].append(
                        float(row["receive_rows"])
                    )
            if sorted(by_expert) == list(range(256)):
                experts = [
                    center(by_expert[index], aggregate=aggregate)
                    for index in range(256)
                ]
    if len(experts) != 256:
        raise ValueError("expert_samples.csv must contain experts 0..255")
    if not rdma_paths:
        raise ValueError("rdma_path_load.csv must contain ProbeEP RDMA paths")

    manifest_path = artifact(args.run_dir, "manifest.json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    config = manifest.get("config", {}) if isinstance(manifest, dict) else {}
    if not world_size and isinstance(config, dict):
        world_size = int(config.get("world_size", 0) or 0)
    server_size = int(config.get("gpus_per_server", SERVER_SIZE) or SERVER_SIZE)
    return {
        "schema": "probeep.multinode.load_profile.v1",
        "title": args.title,
        "run_dir": str(args.run_dir.resolve()),
        "workload": args.workload,
        "direction": args.direction,
        "scope": args.scope,
        "runner_mode": args.runner_mode,
        "routing_sha256": routing_sha256,
        "methods": methods,
        "rdma_paths": rdma_paths,
        "plan": plan_summary,
        "nsys_overlap": None,
        "experts": experts,
        "manifest": manifest,
        "world_size": world_size,
        "server_size": server_size,
        "num_servers": (world_size + server_size - 1) // server_size
        if world_size
        else 0,
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = (len(ordered) - 1) * fraction
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    weight = index - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def html(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>ProbeEP multi-node</title>
<style>
body{margin:0;background:#f4f6f8;color:#18222d;font:14px system-ui,-apple-system,sans-serif}header{padding:22px 28px;background:#fff;border-bottom:1px solid #d7dde4}h1{margin:0 0 7px;font-size:24px}header p{margin:0;color:#607080}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:14px;padding:14px}.panel{background:#fff;border:1px solid #d7dde4;padding:16px;overflow:auto}.panel h2{font-size:16px;margin:0 0 6px}.note{color:#667788;margin-bottom:10px}.metrics{display:flex;gap:18px;flex-wrap:wrap;margin:10px 0}.metric b{font-size:18px;display:block}svg{width:100%;min-width:500px;height:auto}.axis{stroke:#718096;stroke-width:1}.gridline{stroke:#dce2e8;stroke-width:1}.divider{stroke:#667788;stroke-width:2;stroke-dasharray:5 5}.label{fill:#526171;font-size:11px}.value{fill:#22303c;font-size:10px;font-weight:600}table{border-collapse:collapse;width:100%}th,td{padding:7px;border-bottom:1px solid #e2e7ec;text-align:right}th:first-child,td:first-child{text-align:left}.empty{padding:24px;color:#8a5560;background:#fff5f5}
</style></head><body><header><h1 id="title"></h1><p id="subtitle"></p></header><main class="grid" id="root"></main>
<script>const DATA=__DATA__;
const root=document.getElementById('root');document.getElementById('title').textContent=DATA.title;const topo=`${DATA.world_size||'?'} ranks · ${DATA.num_servers||'?'} servers × ${DATA.server_size||8} GPUs`;document.getElementById('subtitle').textContent=`${DATA.workload} · ${DATA.scope} · ${DATA.runner_mode} · ${DATA.direction} · ${topo} · node-major ranks`;
const colors=['#6B7280','#477DB3','#E68A19','#7A5CB8','#3CA56C'];
function panel(title,note){const p=document.createElement('section');p.className='panel';p.innerHTML=`<h2>${title}</h2><div class="note">${note}</div>`;root.appendChild(p);return p}
function bars(values,color,maxValue,serverDivider=true,paddedValues=null){const w=720,h=260,l=45,r=12,t=20,b=38,plotW=w-l-r,plotH=h-t-b;let s=`<svg viewBox="0 0 ${w} ${h}">`;for(let i=0;i<=4;i++){const y=t+plotH*i/4;const v=maxValue*(1-i/4);s+=`<line class="gridline" x1="${l}" y1="${y}" x2="${w-r}" y2="${y}"/><text class="label" x="${l-5}" y="${y+4}" text-anchor="end">${Math.round(v).toLocaleString()}</text>`}const ideal=values.reduce((a,b)=>a+b,0)/values.length;if(serverDivider){const iy=t+plotH-ideal/maxValue*plotH;s+=`<line class="divider" x1="${l}" y1="${iy}" x2="${w-r}" y2="${iy}"/><text class="label" x="${w-r-2}" y="${iy-4}" text-anchor="end">raw ideal ${Math.round(ideal).toLocaleString()}</text>`}values.forEach((v,i)=>{const total=paddedValues?paddedValues[i]:v,padding=Math.max(0,total-v),bw=plotW/values.length*.68,x=l+(i+.16)*plotW/values.length,rawH=v/maxValue*plotH,padH=padding/maxValue*plotH,rawY=t+plotH-rawH,padY=rawY-padH;s+=`<rect x="${x}" y="${rawY}" width="${bw}" height="${rawH}" fill="${color}"/>`;if(padH)s+=`<rect x="${x}" y="${padY}" width="${bw}" height="${padH}" fill="${color}" opacity=".35"/>`;s+=`<text class="value" x="${x+bw/2}" y="${Math.max(11,padY-4)}" text-anchor="middle">${(total/1000).toFixed(1)}k</text><text class="label" x="${x+bw/2}" y="${h-17}" text-anchor="middle">R${i}</text>`});if(serverDivider){const step=DATA.server_size||8;for(let cut=step;cut<values.length;cut+=step){const x=l+plotW*cut/values.length;s+=`<line class="divider" x1="${x}" y1="${t}" x2="${x}" y2="${h-b}"/>`}}return s+'</svg>'}
function expertBars(values){const w=720,h=230,l=45,r=12,t=18,b=30,plotW=w-l-r,plotH=h-t-b,maxValue=Math.max(1,...values);let s=`<svg viewBox="0 0 ${w} ${h}">`;for(let i=0;i<=4;i++){const y=t+plotH*i/4,v=maxValue*(1-i/4);s+=`<line class="gridline" x1="${l}" y1="${y}" x2="${w-r}" y2="${y}"/><text class="label" x="${l-5}" y="${y+4}" text-anchor="end">${Math.round(v).toLocaleString()}</text>`}values.forEach((v,i)=>{const bw=plotW/values.length,x=l+i*bw,bh=v/maxValue*plotH,y=t+plotH-bh;s+=`<rect x="${x}" y="${y}" width="${Math.max(.8,bw)}" height="${bh}" fill="#3CA56C"/>`});[0,63,127,191,255].forEach(i=>{const x=l+(i+.5)*plotW/256;s+=`<text class="label" x="${x}" y="${h-9}" text-anchor="middle">E${i}</text>`});return s+'</svg>'}
if(!DATA.methods.length){panel('结果缺失','没有找到匹配的五算法样本').innerHTML+='<div class="empty">先完成多机五算法 benchmark，再生成报告。</div>'}
const maxLoad=Math.max(1,...DATA.methods.flatMap(m=>m.padded));DATA.methods.forEach((m,i)=>{const p=panel(`${m.label} · ${m.execute.length}-rank MoE 负载`,`实色为 dispatch 后 raw grouped-GEMM rows，浅色为 padding；柱顶是实际计算 rows。`);p.innerHTML+=`<div class="metrics"><span class="metric"><b>${m.rank_raw_max_mean.toFixed(4)}</b>raw max/mean</span><span class="metric"><b>${m.rank_compute_max_mean.toFixed(4)}</b>compute max/mean</span><span class="metric"><b>${m.server_max_mean.toFixed(4)}</b>server raw max/mean</span><span class="metric"><b>${m.p99_ms.toFixed(3)} ms</b>P99 E2E</span></div>`+bars(m.execute,colors[i%colors.length],maxLoad,true,m.padded)});
const exact=panel('MoE 负载精确数据','每格为 raw / padded grouped-GEMM rows。');let et='<table><thead><tr><th>算法</th>';for(let r=0;r<(DATA.world_size||0);r++)et+=`<th>R${r}</th>`;et+='<th>raw max/mean</th><th>compute max/mean</th><th>server raw max/mean</th><th>P99 E2E</th></tr></thead><tbody>';DATA.methods.forEach(m=>{et+=`<tr><td>${m.label}</td>`;m.execute.forEach((v,r)=>et+=`<td>${Math.round(v).toLocaleString()} / ${Math.round(m.padded[r]).toLocaleString()}</td>`);et+=`<td>${m.rank_raw_max_mean.toFixed(4)}</td><td>${m.rank_compute_max_mean.toFixed(4)}</td><td>${m.server_max_mean.toFixed(4)}</td><td>${m.p99_ms.toFixed(3)} ms</td></tr>`});exact.innerHTML+=et+'</tbody></table>';
const ep=panel('256 个路由专家的接收负载','五算法消费同一份 routing；该图画输入 receive rows，不把副本执行误写成新 expert 请求。');if(DATA.experts.length)ep.innerHTML+=expertBars(DATA.experts);else ep.innerHTML+='<div class="empty">当前 run 没有 expert_samples.csv。</div>';
const sp=panel('服务器负载',`每个柱子是同一 server 内 ${DATA.server_size||8} 个 rank 的执行 routes 之和。`);let table='<table><thead><tr><th>算法</th>';for(let s=0;s<(DATA.num_servers||0);s++)table+=`<th>server${s} before</th>`;for(let s=0;s<(DATA.num_servers||0);s++)table+=`<th>server${s} after</th>`;table+='</tr></thead><tbody>';DATA.methods.forEach(m=>{table+=`<tr><td>${m.label}</td>`;m.server_home.forEach(v=>table+=`<td>${Math.round(v).toLocaleString()}</td>`);m.server_execute.forEach(v=>table+=`<td>${Math.round(v).toLocaleString()}</td>`);table+='</tr>'});sp.innerHTML+=table+'</tbody></table>';
const rp=panel('真实 RDMA path 负载','总字节=真实 Dispatch 路由字节+production CUDA Weight chunk 字节；goodput 只由实测 active time 反推。');if(DATA.rdma_paths.length){const maxPath=Math.max(1,...DATA.rdma_paths.map(x=>x.tx_bytes));rp.innerHTML+=bars(DATA.rdma_paths.map(x=>x.tx_bytes),'#7A5CB8',maxPath,false);let rt='<table><thead><tr><th>path id</th><th>路径</th><th>chunks</th><th>Dispatch MiB</th><th>Weight MiB</th><th>总 TX MiB</th><th>总 RX MiB</th><th>active μs</th><th>goodput Gbps</th></tr></thead><tbody>';DATA.rdma_paths.forEach(x=>rt+=`<tr><td>${x.path_id}</td><td>${x.path}</td><td>${x.chunk_count}</td><td>${(x.dispatch_bytes/1048576).toFixed(2)}</td><td>${(x.weight_bytes/1048576).toFixed(2)}</td><td>${(x.tx_bytes/1048576).toFixed(2)}</td><td>${(x.rx_bytes/1048576).toFixed(2)}</td><td>${x.active_time_us==null?'—':x.active_time_us.toFixed(3)}</td><td>${x.measured_gbps==null?'—':x.measured_gbps.toFixed(2)}</td></tr>`);rp.innerHTML+=rt+'</tbody></table>'}else rp.innerHTML+='<div class="empty">当前 run 没有 rdma_path_load.csv。</div>';
const bp=panel('CUDA 流水线分段','每项是 iteration 中跨 rank 最大时间的 median；总和不等于 E2E 时表示存在 overlap。');let bt='<table><thead><tr><th>算法</th>'+Object.keys(DATA.methods[0]?.phases||{}).map(x=>`<th>${x.replace('_ms','')}</th>`).join('')+'</tr></thead><tbody>';DATA.methods.forEach(m=>bt+=`<tr><td>${m.label}</td>`+Object.values(m.phases).map(x=>`<td>${x.toFixed(3)}</td>`).join('')+'</tr>');bp.innerHTML+=bt+'</tbody></table>';
const pp=panel('ProbeEP production plan','只读 BalancedHandle telemetry；CPU oracle 不进入本面板。');if(Object.keys(DATA.plan).length){pp.innerHTML+=`<div class="metrics"><span class="metric"><b>${DATA.plan.admitted_count}</b>admitted experts</span><span class="metric"><b>${DATA.plan.deferred_count}</b>deferred experts</span><span class="metric"><b>${DATA.plan.observations}</b>observations</span></div><pre>${JSON.stringify(DATA.plan,null,2)}</pre>`}else pp.innerHTML+='<div class="empty">当前 run 没有 probeep_plan_summary.jsonl。</div>';
</script></body></html>""".replace("__DATA__", encoded)


def main() -> None:
    args = arguments()
    payload = build_payload(args)
    json_output = args.json_output
    if json_output is None and artifact(args.run_dir, "rank_samples.csv").is_file():
        json_output = args.run_dir / "result.json"
    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    output = args.zip_output or args.run_dir / "visualization_bundle.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        archive.writestr("load_profile.html", html(payload).encode("utf-8"))
    with zipfile.ZipFile(output) as archive:
        if archive.namelist() != ["load_profile.html"]:
            raise RuntimeError(
                "visualization ZIP must contain only load_profile.html"
            )
        broken = archive.testzip()
        if broken is not None:
            raise RuntimeError(f"corrupt ZIP member: {broken}")
    if json_output is not None:
        print(json_output)
    print(output)


if __name__ == "__main__":
    main()
