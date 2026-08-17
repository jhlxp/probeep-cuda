#!/usr/bin/env python3
"""Node-local dispatcher for H20 multi-node build, correctness and profiling suites."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
SUITE_DIR = Path(__file__).resolve().parent
ULTRAEP_HYBRIDEP_ROOT = Path(
    os.environ.get(
        "ULTRAEP_HYBRIDEP_ROOT",
        ROOT / "build/ultraep-hybridep-e0a5b1d9",
    )
).resolve()
SOURCE_ROOTS = {
    "deepep": ROOT / "src/deepep",
    "deepep_moonep": ROOT / "src/deepep-moonep",
    "deepep_probeep": ROOT / "src/deepep-probeep",
    "ultraep": ULTRAEP_HYBRIDEP_ROOT,
    "hybridep": ULTRAEP_HYBRIDEP_ROOT / "HybridEP",
}
NODE_SPAWN_SUITES = {
    "deepep-smoke",
    "deepep-moonep-smoke",
    "probeep-forward",
    "probeep-feedback-synthetic",
    "probeep-expert-io",
    "probeep-backward",
}
MEASURED_OBSERVATION_MODES = {"benchmark_cuda_events", "real"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def integer_env(name: str, default: int | None = None) -> int:
    value = os.environ.get(name)
    if value is None:
        if default is None:
            raise ValueError(f"missing environment variable {name}")
        return default
    return int(value)


@dataclass(frozen=True)
class Context:
    node_rank: int
    num_nodes: int
    gpus_per_node: int
    master_addr: str
    master_port: int
    run_dir: Path
    run_id: str

    @classmethod
    def from_environment(cls) -> "Context":
        node_rank = integer_env("NODE_RANK", integer_env("SLURM_PROCID", 0))
        num_nodes = integer_env("NNODES", 2)
        gpus = integer_env("GPUS_PER_NODE", 8)
        master_addr = os.environ.get("MASTER_ADDR") or os.environ.get("PROBEEP_RENDEZVOUS_ADDR")
        if not master_addr:
            raise ValueError("MASTER_ADDR or PROBEEP_RENDEZVOUS_ADDR is required")
        run_dir = Path(os.environ["PROBEEP_RUN_DIR"]).resolve()
        return cls(
            node_rank=node_rank,
            num_nodes=num_nodes,
            gpus_per_node=gpus,
            master_addr=master_addr,
            master_port=integer_env("MASTER_PORT", 29500),
            run_dir=run_dir,
            run_id=os.environ["PROBEEP_RUN_ID"],
        )

    def node_environment(self, *, port_offset: int = 0, source: Path | None = None) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update(
            {
                "PROBEEP_ROOT": str(ROOT),
                "WORLD_SIZE": str(self.num_nodes),
                "RANK": str(self.node_rank),
                "LOCAL_WORLD_SIZE": str(self.gpus_per_node),
                "MASTER_ADDR": self.master_addr,
                "MASTER_PORT": str(self.master_port + port_offset),
                "TORCH_CUDA_ARCH_LIST": environment.get("TORCH_CUDA_ARCH_LIST", "9.0"),
            }
        )
        if source is not None:
            previous = environment.get("PYTHONPATH")
            environment["PYTHONPATH"] = str(source) + (os.pathsep + previous if previous else "")
        return environment


@dataclass(frozen=True)
class Step:
    name: str
    argv: list[str]
    cwd: Path
    environment: dict[str, str]

    def display(self) -> str:
        return f"cd {shlex.quote(str(self.cwd))} && {shlex.join(self.argv)}"


def python_step(
    context: Context,
    name: str,
    source_key: str,
    script: str,
    arguments: list[str],
    *,
    port_offset: int,
) -> Step:
    source = SOURCE_ROOTS[source_key]
    return Step(
        name=name,
        argv=[sys.executable, str(source / script), *arguments],
        cwd=source,
        environment=context.node_environment(port_offset=port_offset, source=source),
    )


def torchrun_prefix(context: Context, *, port_offset: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nnodes={context.num_nodes}",
        f"--nproc-per-node={context.gpus_per_node}",
        f"--node-rank={context.node_rank}",
        f"--master-addr={context.master_addr}",
        f"--master-port={context.master_port + port_offset}",
    ]


def preflight_step(context: Context) -> Step:
    return Step(
        name="preflight",
        argv=[*torchrun_prefix(context, port_offset=0), str(SUITE_DIR / "preflight.py")],
        cwd=ROOT,
        environment=context.node_environment(),
    )


def build_steps(context: Context) -> list[Step]:
    if context.node_rank != 0:
        return []
    environment = context.node_environment()
    environment["TORCH_CUDA_ARCH_LIST"] = "9.0"
    steps = []
    for name in ("deepep", "deepep_moonep", "deepep_probeep"):
        source = SOURCE_ROOTS[name]
        steps.append(Step(f"build-{name}", [sys.executable, "setup.py", "build_ext", "--inplace"], source, environment))
    steps.extend(build_ultraep_hybridep_steps(context))
    return steps


def build_ultraep_hybridep_steps(context: Context) -> list[Step]:
    if context.node_rank != 0:
        return []
    environment = context.node_environment()
    environment["ULTRAEP_SOURCE_ROOT"] = str(ROOT / "src/ultraep")
    environment["ULTRAEP_HYBRIDEP_ROOT"] = str(ULTRAEP_HYBRIDEP_ROOT)
    return [Step(
        "build-ultraep-hybridep",
        ["bash", str(SUITE_DIR / "build_ultraep_hybridep.sh")],
        ROOT,
        environment,
    )]


def import_steps(context: Context) -> list[Step]:
    checks = []
    for name in ("deepep", "deepep_moonep", "deepep_probeep"):
        source = SOURCE_ROOTS[name]
        code = (
            "from pathlib import Path; import deep_ep, deep_ep_cpp; "
            f"root=Path({str(source)!r}).resolve(); p=Path(deep_ep.__file__).resolve(); "
            "assert p.is_relative_to(root), (p, root); print(p); print(deep_ep_cpp.__file__)"
        )
        checks.append(Step(
            f"import-{name}", [sys.executable, "-c", code], source,
            context.node_environment(source=source),
        ))
    ultra = SOURCE_ROOTS["ultraep"]
    checks.append(Step(
        "import-ultraep",
        [sys.executable, "-c", "from pathlib import Path; import ultra_ep; print(Path(ultra_ep.__file__).resolve()); import ultra_ep._C; print(ultra_ep._C.__file__)"],
        ultra,
        context.node_environment(source=ultra),
    ))
    hybrid = SOURCE_ROOTS["hybridep"]
    hybrid_code = (
        "import importlib.util; from pathlib import Path; import hybrid_ep_cpp; "
        f"root=Path({str(hybrid)!r}).resolve(); "
        "extension=Path(hybrid_ep_cpp.__file__).resolve(); "
        "assert extension.is_relative_to(root), (extension, root); "
        "adapter=root/'deep_ep/hybrid_ep_buffer.py'; "
        "spec=importlib.util.spec_from_file_location('_probeep_hybrid_ep_buffer', adapter); "
        "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
        "print(adapter); print(extension)"
    )
    checks.append(Step(
        "import-hybridep",
        [sys.executable, "-c", hybrid_code],
        hybrid,
        context.node_environment(source=hybrid),
    ))
    return checks


def suite_steps(context: Context, suite: str, extra: list[str]) -> list[Step]:
    smoke_tokens = os.environ.get("PROBEEP_SMOKE_TOKENS", "256")
    expert_tokens = os.environ.get("PROBEEP_EXPERT_IO_TOKENS", "64")
    backward_tokens = os.environ.get("PROBEEP_BACKWARD_TOKENS", "64")
    seed = os.environ.get("PROBEEP_TEST_SEED", "20260813")
    rounds = os.environ.get("PROBEEP_TEST_ROUNDS", "2")
    gpus = str(context.gpus_per_node)

    current_ep16_only = {
        "deepep-moonep-smoke", "probeep-forward", "probeep-feedback-synthetic",
        "probeep-expert-io", "probeep-backward", "probeep-full",
    }
    if suite in current_ep16_only and (context.num_nodes != 2 or context.gpus_per_node != 8):
        raise RuntimeError(
            f"{suite} currently uses the EP16 source oracle and requires 2 nodes x 8 GPUs; "
            "parameterize the source test before using a larger topology"
        )

    if suite == "preflight":
        return [preflight_step(context)]
    if suite == "build-extensions":
        return build_steps(context)
    if suite == "build-ultraep-hybridep":
        return build_ultraep_hybridep_steps(context)
    if suite == "import-smoke":
        return import_steps(context)
    if suite == "deepep-smoke":
        return [python_step(context, suite, "deepep", "tests/test_internode.py", [
            "--num-processes", gpus,
            "--num-tokens", os.environ.get("TOKENS_PER_RANK", "4096"),
            "--hidden", os.environ.get("HIDDEN", "7168"),
            "--num-topk", os.environ.get("TOPK", "8"),
            "--num-experts", os.environ.get("NUM_EXPERTS", "256"),
            "--skip-benchmark",
            *extra,
        ], port_offset=10)]
    if suite == "deepep-moonep-smoke":
        return [python_step(context, suite, "deepep_moonep", "tests/test_balanced_internode.py", [
            "--num-processes", gpus, "--num-tokens", smoke_tokens,
            "--seed", seed, "--rounds", rounds, *extra,
        ], port_offset=20)]
    if suite in ("probeep-forward", "probeep-feedback-synthetic"):
        return [python_step(context, suite, "deepep_probeep", "tests/test_balanced_internode.py", [
            "--num-processes", gpus, "--num-tokens", smoke_tokens,
            "--seed", seed, "--rounds", rounds, *extra,
        ], port_offset=30)]
    if suite == "probeep-expert-io":
        return [python_step(context, suite, "deepep_probeep", "tests/test_balanced_expert_io.py", [
            "--num-processes", gpus, "--num-tokens", expert_tokens, *extra,
        ], port_offset=40)]
    if suite == "probeep-backward":
        return [python_step(context, suite, "deepep_probeep", "tests/test_balanced_backward.py", [
            "--num-processes", gpus, "--num-tokens", backward_tokens, "--seed", seed, *extra,
        ], port_offset=50)]
    if suite == "probeep-full":
        steps: list[Step] = []
        for name in ("probeep-forward", "probeep-expert-io", "probeep-backward"):
            steps.extend(suite_steps(context, name, []))
        return steps
    if suite == "ultraep-smoke":
        environment = context.node_environment(port_offset=60, source=ROOT)
        environment["ULTRAEP_HYBRIDEP_ROOT"] = str(ULTRAEP_HYBRIDEP_ROOT)
        return [Step(
            suite,
            [
                *torchrun_prefix(context, port_offset=60),
                str(SUITE_DIR / "formal/run_benchmark.py"),
                "--variant", "ultraep_hybridep",
                "--expert-mode", "grouped",
                "--runner-mode", "sync_single",
                "--workload", "server_preserving_skew",
                "--warmup-iters", "1",
                "--measure-iters", "1",
                *extra,
            ],
            ROOT,
            environment,
        )]
    if suite == "formal-performance":
        return formal_performance_steps(context, extra)
    raise ValueError(f"unknown suite {suite!r}")


def formal_performance_steps(context: Context, extra: list[str]) -> list[Step]:
    blockers = []
    if os.environ.get("PROBEEP_DYNAMIC_OBSERVATION_MODE") not in MEASURED_OBSERVATION_MODES:
        blockers.append(
            "PROBEEP_DYNAMIC_OBSERVATION_MODE must select measured CUDA-event observations"
        )
    raw_entry = os.environ.get("PROBEEP_FORMAL_ENTRYPOINT")
    if not raw_entry:
        blockers.append("PROBEEP_FORMAL_ENTRYPOINT is unset")
    elif not Path(raw_entry).is_file():
        blockers.append(f"formal entrypoint does not exist: {raw_entry}")
    selector = os.environ.get("PROBEEP_WORKLOAD_SELECTOR", "raw_data1_all")
    if selector not in ("raw_data1_eval20", "raw_data1_all") and not selector.startswith("raw_data1_layer_"):
        blockers.append(f"unsupported workload selector: {selector}")
    raw_manifest = os.environ.get("PROBEEP_WORKLOAD_MANIFEST")
    manifest_payload: dict[str, Any] | None = None
    if not raw_manifest:
        blockers.append("PROBEEP_WORKLOAD_MANIFEST is unset")
    elif not Path(raw_manifest).is_file():
        blockers.append(f"workload manifest does not exist: {raw_manifest}")
    else:
        try:
            manifest_payload = json.loads(Path(raw_manifest).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            blockers.append(f"invalid workload manifest: {error}")
    if manifest_payload is not None:
        expected = {
            "schema": "probeep.raw_data1.selection.v1",
            "selector": selector,
            "world_size": context.num_nodes * context.gpus_per_node,
            "tokens_per_rank": integer_env("TOKENS_PER_RANK", 4096),
            "topk": integer_env("TOPK", 8),
            "num_experts": integer_env("NUM_EXPERTS", 256),
        }
        for key, value in expected.items():
            if manifest_payload.get(key) != value:
                blockers.append(f"workload manifest {key}={manifest_payload.get(key)!r}, expected {value!r}")
    benchmark_plan = os.environ.get("PROBEEP_BENCHMARK_PLAN")
    plan_payload: dict[str, Any] | None = None
    if not benchmark_plan:
        blockers.append("PROBEEP_BENCHMARK_PLAN is unset")
    elif not Path(benchmark_plan).is_file():
        blockers.append(f"benchmark plan does not exist: {benchmark_plan}")
    else:
        try:
            plan_payload = json.loads(Path(benchmark_plan).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            blockers.append(f"invalid benchmark plan: {error}")
    if plan_payload is not None:
        if selector == "raw_data1_all":
            expected_layers = list(range(58))
        elif selector == "raw_data1_eval20":
            expected_layers = list(range(20))
        else:
            expected_layers = [int(selector.removeprefix("raw_data1_layer_"))]
        expected_plan = {
            "schema": "probeep.multinode.benchmark_plan.v1",
            "selector": selector,
            "manifest": str(Path(raw_manifest).resolve()) if raw_manifest else None,
            "variants": [
                "nccl", "deepep", "deepep_moonep_on",
                "ultraep_hybridep", "probeep",
            ],
            "selected_layer_ids": expected_layers,
        }
        for key, value in expected_plan.items():
            if plan_payload.get(key) != value:
                blockers.append(
                    f"benchmark plan {key}={plan_payload.get(key)!r}, expected {value!r}"
                )
        topology = plan_payload.get("topology", {})
        model = plan_payload.get("model", {})
        execution = plan_payload.get("execution", {})
        plan_contract = {
            "world_size": context.num_nodes * context.gpus_per_node,
            "gpus_per_server": context.gpus_per_node,
        }
        for key, value in plan_contract.items():
            if topology.get(key) != value:
                blockers.append(f"benchmark plan topology.{key} must be {value}")
        for key, value in {
            "num_experts": integer_env("NUM_EXPERTS", 256),
            "tokens_per_rank": integer_env("TOKENS_PER_RANK", 4096),
            "topk": integer_env("TOPK", 8),
            "hidden": integer_env("HIDDEN", 7168),
            "expert_mode": "grouped_ffn",
        }.items():
            if model.get(key) != value:
                blockers.append(f"benchmark plan model.{key} must be {value!r}")
        if execution.get("runner_mode") != "dual_microbatch_ht":
            blockers.append("benchmark plan must use dual_microbatch_ht")
        if execution.get("microbatches") != 2:
            blockers.append("benchmark plan must use exactly two microbatches")
    if os.environ.get("PROBEEP_FORMAL_PAPER_MODE", "1") == "1" and selector != "raw_data1_all":
        blockers.append("paper mode requires raw_data1_all (all 58 layers)")
    if blockers:
        raise RuntimeError(
            "formal performance is fail-closed because its immutable execution contract is incomplete: "
            + "; ".join(blockers)
        )
    assert raw_entry is not None
    environment = context.node_environment(port_offset=70)
    environment.update({
        "BENCHMARK_RUNNER_MODE": "dual_microbatch_ht",
        "BENCHMARK_ENABLE_ATTENTION_OVERLAP": "1",
        "WORKLOADS": selector,
        "PROBEEP_WORKLOAD_SELECTOR": selector,
        "PROBEEP_WORKLOAD_MANIFEST": str(raw_manifest),
        "PROBEEP_BENCHMARK_PLAN": str(benchmark_plan),
        "VARIANTS": "nccl,deepep,deepep_moonep_on,ultraep_hybridep,probeep",
    })
    return [Step(
        "formal-performance",
        [sys.executable, raw_entry, *extra],
        ROOT,
        environment,
    )]


def readiness(context: Context) -> dict[str, Any]:
    real_feedback = (
        os.environ.get("PROBEEP_DYNAMIC_OBSERVATION_MODE")
        in MEASURED_OBSERVATION_MODES
    )
    formal_entry = os.environ.get("PROBEEP_FORMAL_ENTRYPOINT", "")
    workload_manifest = os.environ.get("PROBEEP_WORKLOAD_MANIFEST", "")
    workload_selector = os.environ.get("PROBEEP_WORKLOAD_SELECTOR", "raw_data1_all")
    workload_ready = bool(workload_manifest and Path(workload_manifest).is_file())
    benchmark_plan = os.environ.get("PROBEEP_BENCHMARK_PLAN", "")
    benchmark_plan_ready = bool(benchmark_plan and Path(benchmark_plan).is_file())
    paper_selector_ready = (
        os.environ.get("PROBEEP_FORMAL_PAPER_MODE", "1") != "1"
        or workload_selector == "raw_data1_all"
    )
    return {
        "schema": "probeep.h20.execution_readiness.v1",
        "created_at_utc": utc_now(),
        "topology": {
            "num_nodes": context.num_nodes,
            "gpus_per_node": context.gpus_per_node,
            "world_size": context.num_nodes * context.gpus_per_node,
        },
        "workload": {
            "selector": workload_selector,
            "manifest": workload_manifest or None,
            "manifest_exists": workload_ready,
            "paper_mode": os.environ.get("PROBEEP_FORMAL_PAPER_MODE", "1") == "1",
            "benchmark_plan": benchmark_plan or None,
            "benchmark_plan_exists": benchmark_plan_ready,
        },
        "implemented": {
            "slurm_and_manual_launch": True,
            "distributed_preflight": True,
            "extension_build": True,
            "isolated_import_provenance": True,
            "deepep_internode_smoke": True,
            "probeep_forward": True,
            "probeep_weight_and_gradient_io": True,
            "probeep_backward": True,
            "ultraep_hybridep_smoke": True,
            "formal_five_backend_scheduler": True,
            "benchmark_cuda_event_observation_producer": True,
            "nsys_wrapper": True,
        },
        "blocked": {
            "real_dynamic_observation_closed_loop": not real_feedback,
            "formal_raw_data1_five_backend_entrypoint": not bool(formal_entry and Path(formal_entry).is_file()),
            "materialized_raw_data1_manifest": not workload_ready,
            "categorized_benchmark_plan": not benchmark_plan_ready,
            "paper_selector_is_full_58_layers": not paper_selector_ready,
            "multi_node_results": True,
        },
        "formal_performance_ready": bool(
            real_feedback and formal_entry and Path(formal_entry).is_file()
            and workload_ready and benchmark_plan_ready and paper_selector_ready
        ),
        "note": (
            "benchmark_cuda_events uses measured Attention/MoE CUDA windows and "
            "feeds the completed observation to the next fused ProbeEP dispatch; "
            "synthetic observation remains correctness-only."
        ),
    }


def append_status(context: Context, payload: dict[str, Any]) -> None:
    raw_dir = context.run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"runner_status_node_{context.node_rank}.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def execute(context: Context, suite: str, steps: list[Step], *, dry_run: bool) -> None:
    for step in steps:
        started = time.monotonic()
        base = {"timestamp_utc": utc_now(), "suite": suite, "step": step.name, "node_rank": context.node_rank, "command": step.display()}
        append_status(context, {**base, "state": "STARTED" if not dry_run else "DRY_RUN"})
        print(f"[{step.name}] {step.display()}", flush=True)
        if dry_run:
            continue
        result = subprocess.run(step.argv, cwd=step.cwd, env=step.environment, check=False)
        status = {
            **base,
            "timestamp_utc": utc_now(),
            "state": "PASS" if result.returncode == 0 else "FAIL",
            "returncode": result.returncode,
            "elapsed_seconds": time.monotonic() - started,
        }
        append_status(context, status)
        if result.returncode != 0:
            raise SystemExit(result.returncode)


def parse_arguments() -> argparse.Namespace:
    suites = sorted(NODE_SPAWN_SUITES | {
        "readiness", "preflight", "build-extensions", "build-ultraep-hybridep",
        "import-smoke", "probeep-full",
        "ultraep-smoke", "formal-performance", "nsys",
    })
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=suites)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--target-suite",
        choices=sorted(
            NODE_SPAWN_SUITES | {"ultraep-smoke", "formal-performance"}
        ),
        default="probeep-forward",
    )
    arguments, suite_args = parser.parse_known_args()
    arguments.suite_args = suite_args
    return arguments


def main() -> None:
    args = parse_arguments()
    context = Context.from_environment()
    for directory in (context.run_dir / "logs", context.run_dir / "raw", context.run_dir / "preflight", context.run_dir / "nsys"):
        directory.mkdir(parents=True, exist_ok=True)

    ready = readiness(context)
    if context.node_rank == 0:
        (context.run_dir / "readiness.json").write_text(json.dumps(ready, ensure_ascii=False, indent=2) + "\n")
    if args.suite == "readiness":
        print(json.dumps(ready, ensure_ascii=False, indent=2))
        return

    steps = suite_steps(context, args.target_suite if args.suite == "nsys" else args.suite, args.suite_args)
    if args.suite == "nsys":
        wrapped = []
        trace_domains = "cuda,nvtx,osrt"
        for step in steps:
            output = context.run_dir / "nsys" / f"{step.name}-node-{context.node_rank}"
            report = Path(f"{output}.nsys-rep")
            sqlite = Path(f"{output}.sqlite")
            wrapped.append(Step(
                f"nsys-{step.name}",
                [
                    "nsys", "profile", f"--trace={trace_domains}", "--sample=none",
                    "--cpuctxsw=none", "--trace-fork-before-exec=false", "--force-overwrite=true",
                    "-o", str(output), *step.argv,
                ],
                step.cwd,
                step.environment,
            ))
            wrapped.append(Step(
                f"nsys-export-{step.name}",
                ["nsys", "export", "--type=sqlite", "--force-overwrite=true", f"--output={sqlite}", str(report)],
                step.cwd,
                step.environment,
            ))
            for report_name in ("nvtx_gpu_proj_sum", "cuda_gpu_kern_sum", "cuda_api_sum", "cuda_gpu_mem_time_sum"):
                wrapped.append(Step(
                    f"nsys-stats-{report_name}-{step.name}",
                    [
                        "nsys", "stats", f"--report={report_name}", "--format=column",
                        "--force-overwrite=true", f"--output={output}-{report_name}", str(sqlite),
                    ],
                    step.cwd,
                    step.environment,
                ))
            if args.target_suite == "formal-performance":
                wrapped.append(Step(
                    f"nsys-overlap-{step.name}",
                    [
                        sys.executable,
                        str(SUITE_DIR / "analyze_nsys_overlap.py"),
                        str(sqlite),
                        "--json-output", f"{output}-overlap.json",
                        "--text-output", f"{output}-overlap.txt",
                    ],
                    step.cwd,
                    step.environment,
                ))
        steps = wrapped
    execute(context, args.suite, steps, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
