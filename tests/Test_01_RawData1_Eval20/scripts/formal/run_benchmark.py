#!/usr/bin/env python3
"""Full multi-node forward benchmark for official baselines and ProbeEP.

All variants consume the same deterministic BF16 input, FP8 block scales,
top-k routes and gate weights.  The default identity expert isolates the full
FP8 dispatch/BF16 combine transport.  ``--expert-mode grouped`` inserts the
shared three-GEMM gated FFN; official DeepEP additionally performs the local
assignment regroup that its transport layout requires.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as functional

from invariants import max_violation, plan_statistics
from planning_reference import PlanningReference, plan_server_local
from probeep_reference import ProbeConfig, plan_probeep
from workloads import make_routing_workload
from backend import (
    BackendUnavailable,
    DispatchResult,
    RuntimeBackend,
    root_for_variant,
)
from grouped_ffn import (
    GroupedFFNLayout,
    make_grouped_ffn_layout,
    pad_grouped_assignment_rows,
    run_grouped_ffn_stage,
    run_grouped_ffn_stage_out,
)
from result_schema import (
    EXPERT_SAMPLE_FIELDS,
    ITERATION_FIELDS,
    MICROBATCH_RANK_SAMPLE_FIELDS,
    MICROBATCH_TIMELINE_FIELDS,
    PROBEEP_OBSERVATION_SAMPLE_FIELDS,
    PROBEEP_WEIGHT_CHUNK_FIELDS,
    RANK_EXPERT_SAMPLE_FIELDS,
    RANK_SAMPLE_FIELDS,
    SCHEMA_VERSION,
    append_csv_rows,
    append_jsonl,
    write_manifest,
)
from workload.gate.raw_receive import runtime_tree_sha256


PHASE_NAMES = (
    "plan_ms",
    "layout_materialize_ms",
    "dispatch_ms",
    "weight_prefetch_ms",
    "expert_compute_ms",
    "combine_ms",
    "e2e_ms",
)

# Must match probeep::kMaxServers in csrc/probeep_topology.hpp.  Production
# admitted placement ids use expert_id * 16 + destination_server.
PROBEEP_PLAN_SERVER_STRIDE = 16

# Fixed row order for the measured dual-microbatch CUDA-event timeline.  The
# records are diagnostic and never replace the max-rank E2E timing contract.
MICROBATCH_TIMELINE_STAGES = (
    ("attention_or_gate", 0, "compute"),
    ("attention_or_gate", 1, "compute"),
    ("weight_dispatch", 0, "communication"),
    ("weight_dispatch", 1, "communication"),
    ("expert_mlp", 0, "compute"),
    ("combine", 0, "communication"),
    ("expert_mlp", 1, "compute"),
    ("combine", 1, "communication"),
    # build_probe_feedback() is issued by the caller on the current/default
    # compute stream after this iteration has completed.  Do not label it as
    # communication-stream work merely because it contains collectives.
    ("observation_prepare", -1, "compute"),
)

RDMA_PATH_LOAD_FIELDS = (
    "schema_version",
    "run_id",
    "slurm_job_id",
    "benchmark_scope",
    "runner_mode",
    "system",
    "balance",
    "direction",
    "workload",
    "bias_ratio",
    "seed",
    "repeat",
    "iteration",
    "routing_sha256",
    "dispatch_compute_kind",
    "dispatch_compute_name",
    "microbatch",
    "path_id",
    "physical_nic",
    "subrail",
    "rail_bandwidth_gbps",
    "physical_nic_bandwidth_gbps",
    "weight_cache_mode",
    "expert_weight_version",
    "source_rank",
    "destination_rank",
    "chunk_count",
    "dispatch_units",
    "dispatch_unit_name",
    "dispatch_bytes_per_unit",
    "traffic_source",
    "dispatch_bytes",
    "weight_bytes",
    "tx_bytes",
    "rx_bytes",
)

# The identity path contains FP8 block dequantization followed by at most eight
# weighted BF16 route reductions. Grouped mode additionally contains three
# BF16 tensor-core reductions and SiLU. These tolerances cover those rounding
# points without hiding a misplaced route, weight, or expert output.
CORRECTNESS_TOLERANCES = {
    "identity": (0.02, 0.02),
    "grouped": (0.05, 0.03),
}
CORRECTNESS_CHUNK_ROWS = 128
EXPERT_FINGERPRINT_BITS = 16
EXPERT_FINGERPRINT_GAIN = 128.0


@dataclass(frozen=True)
class GroupedWeights:
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor
    expert_offset: int


@dataclass(frozen=True)
class AttentionProbeState:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor


@dataclass(frozen=True)
class ExpertMaterialization:
    layout: GroupedFFNLayout
    source_rows: torch.Tensor | None
    route_weights: torch.Tensor | None
    physical_layout: GroupedFFNLayout | None = None


@dataclass(frozen=True)
class ForwardSample:
    output: torch.Tensor
    local_ms: dict[str, float]
    valid_rows: int
    handle: object
    handles: tuple[object, ...] = ()
    dispatches: tuple[DispatchResult, ...] = ()
    # (compute_kind, compute_ms, matching W+D network_ms).  The formal dual
    # runner records A against microbatch-0 communication and MoE against
    # microbatch-1 communication in the same HT window.
    probe_windows_ms: tuple[tuple[int, float, float], ...] = ()
    local_rank_expert_raw: torch.Tensor | None = None
    local_rank_expert_padded: torch.Tensor | None = None
    local_microbatch_rank_expert_raw: torch.Tensor | None = None
    local_microbatch_rank_expert_padded: torch.Tensor | None = None
    # (stage, microbatch, logical_stream, start_ms, end_ms), relative to this
    # iteration's e2e_start CUDA event.  microbatch=-1 is shared control work.
    timeline_intervals_ms: tuple[
        tuple[str, int, str, float, float], ...
    ] = ()
    # Retained only until the caller records ProbeEP's post-combine observation
    # producer.  This allows its start/end to share the exact same CUDA-event
    # time origin instead of being appended from a duration estimate.
    timeline_origin_event: object | None = None


@dataclass(frozen=True)
class ProbeFeedbackDevice:
    compute_ns: torch.Tensor
    network_ns: torch.Tensor
    dispatch_tx_bytes: torch.Tensor
    dispatch_rx_bytes: torch.Tensor
    migration_tx_bytes: torch.Tensor
    migration_rx_bytes: torch.Tensor
    dispatch_matrix_bytes: torch.Tensor
    compute_kind: int
    # The communication belongs to dispatch_microbatch, while its masking
    # compute window belongs to the other microbatch in the fixed HT wavefront.
    dispatch_microbatch: int
    overlap_microbatch: int


@dataclass(frozen=True)
class ProbeFeedbackSet:
    updates: tuple[ProbeFeedbackDevice, ...]
    dispatch_compute_kind: int
    producer_phase: str
    producer_iteration: int
    producer_layer_id: int
    producer_repeat: int


@dataclass(frozen=True)
class MicrobatchInput:
    x_fp8: torch.Tensor
    x_scales: torch.Tensor
    topk_idx: torch.Tensor
    topk_weights: torch.Tensor
    balanced_execution_rows: int
    balanced_home_execution_rows: int


_PERSISTENT_BACKEND: RuntimeBackend | None = None
_PERSISTENT_GROUPED_WEIGHTS: GroupedWeights | None = None
_PERSISTENT_VARIANT: str | None = None
# The paper runner executes layers in canonical order inside one persistent
# worker.  Feedback is banked by phase/round/compute-kind so Layer L consumes
# the matching completed observation from Layer L-1.  Splitting kind 0/1 in
# the key makes it impossible for the Attention and MoE chains to overwrite
# or substitute for each other.
_PERSISTENT_PROBE_FEEDBACK_BANK: dict[
    tuple[str, int, int], ProbeFeedbackDevice
] = {}
_PERSISTENT_PROBE_FEEDBACK_LAYER: int | None = None
_PERSISTENT_PROBE_FEEDBACK_REPEAT: int | None = None


@dataclass(frozen=True)
class ForwardCorrectness:
    passed: bool
    mismatch_count: int
    element_count: int
    max_abs_error: float
    max_scaled_error: float
    rtol: float
    atol: float

    def as_dict(self) -> dict[str, int | float | bool]:
        return {
            "passed": self.passed,
            "mismatch_count": self.mismatch_count,
            "element_count": self.element_count,
            "max_abs_error": self.max_abs_error,
            "max_scaled_error": self.max_scaled_error,
            "rtol": self.rtol,
            "atol": self.atol,
        }


@contextlib.contextmanager
def nvtx_range(enabled: bool, name: str):
    """Emit benchmark-only NVTX ranges without touching the hot-path API."""

    if enabled:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        required=True,
        choices=(
            "nccl",
            "deepep",
            "deepep_moonep_on",
            "ultraep_hybridep",
            "probeep",
        ),
    )
    parser.add_argument(
        "--expert-mode",
        choices=("identity", "grouped"),
        default=os.getenv("EXPERT_MODE", "identity"),
    )
    parser.add_argument(
        "--workload", default=os.getenv("ROUTING_MODE", "server_preserving_skew")
    )
    parser.add_argument(
        "--bias-ratio", type=float, default=float(os.getenv("BIAS_RATIO", "1"))
    )
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED", "1234")))
    parser.add_argument(
        "--warmup-iters", type=int, default=int(os.getenv("WARMUP_ITERS", "50"))
    )
    parser.add_argument(
        "--measure-iters",
        type=int,
        default=int(os.getenv("MEASURE_ITERS", "100")),
    )
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument(
        "--runner-mode",
        choices=("dual_microbatch_ht", "sync_single"),
        default=os.getenv("BENCHMARK_RUNNER_MODE", "dual_microbatch_ht"),
        help=(
            "Formal performance mode is dual_microbatch_ht: vLLM DBO-style "
            "two-microbatch scheduling on the DeepEP high-throughput backend. "
            "sync_single is only for correctness/diagnosis."
        ),
    )
    parser.add_argument("--profile", action="store_true")
    return parser.parse_args(argv)


def variant_identity(variant: str) -> tuple[str, str]:
    if variant == "nccl":
        return "nccl", "na"
    if variant == "deepep":
        return "deepep", "na"
    if variant == "deepep_moonep_on":
        return "deepep_moonep", "on"
    if variant == "ultraep_hybridep":
        return "ultraep", "hybridep"
    if variant == "probeep":
        return "probeep", "server_first"
    raise ValueError(f"unknown variant: {variant}")


def benchmark_scope(expert_mode: str) -> str:
    return f"full_moe_{expert_mode}"


def validate_backend_expert_mode(backend: object, expert_mode: str) -> None:
    """Reject benchmark combinations whose expert semantics are incomplete."""

    if (
        expert_mode == "grouped"
        and getattr(backend, "probeep_hybrid", False)
        and not getattr(backend, "dynamic_expert_weights_ready", False)
    ):
        raise RuntimeError(
            "ProbeEP grouped FFN is disabled: the physical token route is "
            "connected to HybridEP, but logical expert weight/grad "
            "materialization is not yet connected to the registered "
            "cross-server transport. Identity mode may be used only for "
            "dispatch/combine diagnostics."
        )


def balanced_row_counts(
    num_tokens: int,
    topk: int,
    token_padding: int,
    execution_slots: int,
    num_servers: int,
) -> tuple[int, int]:
    padding_rows = (token_padding - 1) * execution_slots
    return (
        num_tokens * topk + padding_rows,
        num_servers * num_tokens * topk + padding_rows,
    )


def git_revision(root: str) -> str:
    vendored = vendored_revision(root)
    if vendored:
        return vendored
    resolved = str(Path(root).resolve())
    revision = subprocess.check_output(
        ["git", "-c", f"safe.directory={resolved}",
         "-C", resolved, "rev-parse", "HEAD"],
        text=True,
    ).strip()
    working_tree = subprocess.check_output(
        ["git", "-c", f"safe.directory={resolved}",
         "-C", resolved, "status", "--porcelain"],
        text=True,
    )
    return revision + ("+dirty" if working_tree else "")


def git_revision_optional(root: str | None) -> str:
    if not root:
        return ""
    if not Path(root).exists():
        return ""
    return git_revision(root)


def vendored_revision(root: str) -> str | None:
    probeep_root = os.getenv("PROBEEP_ROOT")
    if not probeep_root:
        return None
    root_path = Path(root).resolve()
    project_root = Path(probeep_root).resolve()
    try:
        relative = root_path.relative_to(project_root).as_posix()
    except ValueError:
        return None
    versions_path = project_root / "src" / "VENDORED_VERSIONS.md"
    if not versions_path.is_file():
        return None
    for line in versions_path.read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        path_cell = cells[0].strip("`")
        version_cell = cells[2].strip("`")
        if path_cell == relative and version_cell:
            return version_cell
    return None


def quantize_fp8_blocks(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, hidden = x.shape
    blocks = x.view(rows, hidden // 128, 128)
    amax = blocks.abs().float().amax(dim=2).clamp_(1e-4)
    scales = amax / 448.0
    fp8 = (blocks * (448.0 / amax).unsqueeze(2)).to(
        torch.float8_e4m3fn
    ).view_as(x)
    # Match DeepEP's supported token-contiguous, column-major scale layout.
    return fp8, scales.T.contiguous().T


def dequantize_fp8_blocks(
    x_fp8: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    rows, hidden = x_fp8.shape
    output = x_fp8.view(rows, hidden // 128, 128).to(torch.bfloat16)
    output.mul_(scales.to(torch.bfloat16).contiguous().unsqueeze(2))
    return output.view(rows, hidden)


def shared_expert_reference_chunk(
    x_fp8: torch.Tensor,
    scales: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    expert_mode: str,
    grouped_weights: GroupedWeights | None,
) -> torch.Tensor:
    """Mathematical per-source-token oracle for the benchmark workload.

    Gate/up and the non-fingerprint columns of down are shared, while the first
    16 output columns encode the global expert ID with exact BF16 sign flips.
    This keeps the oracle to one FFN evaluation per source token while making a
    wrong logical expert, replica, or owner mapping observable. Using identical
    expert weights here would let those routing bugs pass with ``mismatch=0``.
    """

    dequantized = dequantize_fp8_blocks(x_fp8, scales)
    if expert_mode == "identity":
        expert_output = dequantized
    else:
        if grouped_weights is None:
            raise RuntimeError("grouped correctness reference requires weights")
        gate = dequantized @ grouped_weights.gate[0]
        up = dequantized @ grouped_weights.up[0]
        expert_output = (functional.silu(gate) * up) @ grouped_weights.down[0]
    weighted = expert_output.float() * topk_weights.sum(
        dim=1, keepdim=True, dtype=torch.float32
    )
    if expert_mode == "grouped":
        bits = min(EXPERT_FINGERPRINT_BITS, weighted.size(1))
        route_signs = expert_fingerprint_signs(topk_idx, bits)
        local_base = torch.full_like(
            topk_idx[:, :1], grouped_weights.expert_offset
        )
        local_signs = expert_fingerprint_signs(local_base, bits)
        relative_signs = route_signs * local_signs
        fingerprint_weights = (
            topk_weights.float().unsqueeze(2) * relative_signs
        ).sum(dim=1)
        weighted[:, :bits] = (
            expert_output[:, :bits].float() * fingerprint_weights
        )
    return weighted


@torch.no_grad()
def check_forward_correctness(
    output: torch.Tensor,
    x_fp8: torch.Tensor,
    x_scales: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    expert_mode: str,
    grouped_weights: GroupedWeights | None,
    chunk_rows: int = CORRECTNESS_CHUNK_ROWS,
) -> ForwardCorrectness:
    """Compare one untimed full-MoE output with a chunked local oracle."""

    if output.shape != x_fp8.shape:
        raise AssertionError(
            f"forward output shape {tuple(output.shape)} != {tuple(x_fp8.shape)}"
        )
    rtol, atol = CORRECTNESS_TOLERANCES[expert_mode]
    mismatches = torch.zeros((), dtype=torch.int64, device=output.device)
    max_abs = torch.zeros((), dtype=torch.float32, device=output.device)
    max_scaled = torch.zeros((), dtype=torch.float32, device=output.device)

    for begin in range(0, output.size(0), chunk_rows):
        end = min(begin + chunk_rows, output.size(0))
        expected = shared_expert_reference_chunk(
            x_fp8[begin:end],
            x_scales[begin:end],
            topk_idx[begin:end],
            topk_weights[begin:end],
            expert_mode,
            grouped_weights,
        )
        actual = output[begin:end].float()
        abs_error = (actual - expected).abs()
        allowed = atol + rtol * expected.abs()
        if expert_mode == "grouped":
            bits = min(EXPERT_FINGERPRINT_BITS, expected.size(1))
            allowed[:, :bits].add_(atol * (EXPERT_FINGERPRINT_GAIN - 1.0))
        mismatches.add_(((abs_error > allowed) | ~torch.isfinite(abs_error)).sum())
        finite_abs_error = torch.nan_to_num(
            abs_error, nan=float("inf"), posinf=float("inf")
        )
        max_abs = torch.maximum(max_abs, finite_abs_error.max())
        max_scaled = torch.maximum(max_scaled, (finite_abs_error / allowed).max())

    mismatch_count = int(mismatches.item())
    return ForwardCorrectness(
        passed=mismatch_count == 0,
        mismatch_count=mismatch_count,
        element_count=output.numel(),
        max_abs_error=float(max_abs.item()),
        max_scaled_error=float(max_scaled.item()),
        rtol=rtol,
        atol=atol,
    )


def make_input(
    rank: int, num_tokens: int, hidden: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed * 104729 + rank)
    bf16 = torch.randn(
        (num_tokens, hidden),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    fp8, scales = quantize_fp8_blocks(bf16)
    return bf16, fp8, scales


def expert_fingerprint_signs(
    expert_ids: torch.Tensor, num_bits: int = EXPERT_FINGERPRINT_BITS
) -> torch.Tensor:
    """Return a collision-free +/-1 code for up to 2**num_bits experts."""

    if num_bits <= 0:
        return torch.empty((*expert_ids.shape, 0), device=expert_ids.device)
    bit = torch.arange(num_bits, dtype=torch.int64, device=expert_ids.device)
    encoded = (expert_ids.to(torch.int64).unsqueeze(-1) >> bit) & 1
    return torch.where(encoded == 0, 1.0, -1.0)


def make_grouped_weights(
    slots: int,
    hidden: int,
    intermediate: int,
    seed: int,
    expert_offset: int,
) -> GroupedWeights:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed + 0xF17E)

    def matrix(rows: int, columns: int) -> torch.Tensor:
        base = torch.randn(
            (rows, columns),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        ).mul_(0.01)
        return base.unsqueeze(0).expand(slots, -1, -1).contiguous()

    gate = matrix(hidden, intermediate)
    up = matrix(hidden, intermediate)
    down = matrix(intermediate, hidden)
    bits = min(EXPERT_FINGERPRINT_BITS, hidden)
    expert_ids = torch.arange(
        expert_offset,
        expert_offset + slots,
        dtype=torch.int64,
        device="cuda",
    )
    signs = expert_fingerprint_signs(expert_ids, bits).to(down.dtype)
    signs.mul_(EXPERT_FINGERPRINT_GAIN)
    down[:, :, :bits].mul_(signs.unsqueeze(1))
    return GroupedWeights(
        gate=gate,
        up=up,
        down=down,
        expert_offset=expert_offset,
    )


def make_attention_probe_state(
    microbatch_tokens: int, hidden: int, seed: int
) -> AttentionProbeState:
    """Allocate the DSV3 MLA attention-core benchmark producer.

    This is intentionally an independent compute producer: it supplies the
    scheduler boundary paired with ``W+D[0]`` and never reuses MoE timing.  The
    default dimensions are DeepSeek-V3's 128 attention heads, 128 non-RoPE plus
    64 RoPE Q/K dimensions, and 128 value dimensions.  Q/K/V projections are
    outside the communication-overlap window; the SDPA/MLA attention core is
    the measured producer.
    """

    tokens = int(
        os.getenv("PROBEEP_ATTENTION_TOKENS", str(microbatch_tokens))
    )
    heads = int(os.getenv("PROBEEP_ATTENTION_HEADS", "128"))
    qk_head_dim = int(os.getenv("PROBEEP_ATTENTION_QK_HEAD_DIM", "192"))
    value_head_dim = int(os.getenv("PROBEEP_ATTENTION_V_HEAD_DIM", "128"))
    if min(tokens, heads, qk_head_dim, value_head_dim) <= 0:
        raise ValueError("ProbeEP attention probe dimensions must be positive")
    if os.getenv("PROBEEP_FORMAL_PAPER_MODE", "1") == "1" and (
        tokens != microbatch_tokens
        or heads != 128
        or qk_head_dim != 192
        or value_head_dim != 128
    ):
        raise ValueError(
            "paper mode requires the DSV3 MLA producer: microbatch tokens, "
            "128 heads, Q/K dim 192, V dim 128"
        )
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed + 0xA77E)
    qk_shape = (1, heads, tokens, qk_head_dim)
    value_shape = (1, heads, tokens, value_head_dim)
    q = torch.randn(qk_shape, dtype=torch.bfloat16, device="cuda", generator=generator)
    k = torch.randn(qk_shape, dtype=torch.bfloat16, device="cuda", generator=generator)
    v = torch.randn(value_shape, dtype=torch.bfloat16, device="cuda", generator=generator)
    return AttentionProbeState(q=q, k=k, v=v)


def run_attention_overlap_probe(
    state: AttentionProbeState | None,
    *,
    profile: bool,
    name: str,
) -> torch.Tensor | None:
    """Run benchmark-only pre-MoE compute that can cover HT communication."""

    if state is None:
        return None
    with nvtx_range(profile, name):
        return functional.scaled_dot_product_attention(
            state.q, state.k, state.v, is_causal=True
        )


def wait_event(event: object | None) -> None:
    """Wait for a DeepEP EventOverlap if the call returned one."""

    if event is not None and hasattr(event, "current_stream_wait"):
        event.current_stream_wait()


def probe_weight_cache_mode() -> str:
    """Return the explicit paper benchmark weight-cache contract."""

    mode = os.getenv("PROBEEP_WEIGHT_CACHE_MODE", "cold").strip().lower()
    if mode not in {"cold", "steady"}:
        raise ValueError("PROBEEP_WEIGHT_CACHE_MODE must be cold or steady")
    return mode


def set_probe_weight_version(
    base_version: int, *, layer_id: int, phase: str, iteration: int
) -> int:
    """Materialize one layer/version identity before an invocation.

    ``cold`` represents a complete multi-layer wavefront whose small plan ring
    cannot retain all DSV3 layers: every invocation receives a new version and
    therefore pays real Weight transport.  ``steady`` intentionally measures a
    same-layer cache hit after warmup and is supplementary only.
    """

    mode = probe_weight_cache_mode()
    if mode == "steady":
        version = base_version
    else:
        phase_offset = {"warmup": 0, "correctness": 100_000, "measured": 200_000}
        if phase not in phase_offset:
            raise ValueError(f"invalid ProbeEP weight-version phase: {phase}")
        if layer_id < -1:
            raise ValueError("ProbeEP layer id must be -1 or non-negative")
        # A persistent worker keeps the three physical replica banks across all
        # RawData1 layers.  Layer identity must therefore be part of the model
        # version; phase/round alone can alias a different layer's parameters.
        version = (
            base_version * 1_000_000_000
            + (layer_id + 1) * 1_000_000
            + phase_offset[phase]
            + iteration
            + 1
        )
    os.environ["PROBEEP_WEIGHT_VERSION"] = str(version)
    return version


def probe_rail_topology(ranks_per_server: int) -> tuple[int, int, float, float]:
    """Validate 4x400G physical NICs split into eight 200G GPU rails."""

    physical_nics = int(os.getenv("PROBEEP_PHYSICAL_NICS_PER_SERVER", "4"))
    rails_per_nic = int(os.getenv("PROBEEP_RAILS_PER_PHYSICAL_NIC", "2"))
    physical_gbps = float(
        os.getenv("PROBEEP_PHYSICAL_NIC_BANDWIDTH_GBPS", "400")
    )
    rail_gbps = float(
        os.getenv(
            "PROBEEP_RDMA_PATH_BANDWIDTH_GBPS",
            os.getenv("RDMA_PATH_BANDWIDTH_GBPS", "200"),
        )
    )
    if physical_nics <= 0 or rails_per_nic <= 0:
        raise ValueError("ProbeEP physical NIC/rail counts must be positive")
    if physical_nics * rails_per_nic != ranks_per_server:
        raise ValueError(
            "ProbeEP topology must map exactly one logical rail to each local "
            f"GPU: {physical_nics} NICs x {rails_per_nic} != {ranks_per_server}"
        )
    if abs(rail_gbps * rails_per_nic - physical_gbps) > 1e-6:
        raise ValueError(
            "ProbeEP rail bandwidth does not conserve physical NIC capacity: "
            f"{rail_gbps} x {rails_per_nic} != {physical_gbps} Gbps"
        )
    return physical_nics, rails_per_nic, physical_gbps, rail_gbps


def materialize_grouped_layout(
    backend: RuntimeBackend,
    dispatched: DispatchResult,
    rank: int,
    local_experts: int,
    balanced_execution_rows: int,
    balanced_home_execution_rows: int = 0,
) -> ExpertMaterialization:
    if getattr(backend, "probeep_hybrid", False):
        del rank, local_experts, balanced_execution_rows, balanced_home_execution_rows
        padded_counts = dispatched.handle.local_padded_tokens_per_expert.to(
            device=dispatched.exec_x.device, dtype=torch.int32, non_blocking=True
        ).contiguous()
        cu_seqlens = torch.cat(
            (padded_counts.new_zeros(1), padded_counts.cumsum(0))
        )
        layout = make_grouped_ffn_layout(
            dispatched.exec_x.size(0),
            cu_seqlens=cu_seqlens,
            slot_count=padded_counts,
        )
        return ExpertMaterialization(layout, None, None)

    if getattr(backend, "variant", "") == "ultraep_hybridep":
        counts = dispatched.exec_counts.to(
            device=dispatched.exec_x.device,
            dtype=torch.int32,
        )
        cu_seqlens = torch.cat(
            (counts.new_zeros(1), counts.cumsum(dim=0, dtype=torch.int32))
        )
        layout = make_grouped_ffn_layout(
            dispatched.exec_x.size(0),
            cu_seqlens=cu_seqlens,
            slot_count=counts,
        )
        return ExpertMaterialization(
            layout, None, dispatched.handle.route_weights
        )

    if backend.balanced:
        if dispatched.exec_counts is None:
            raise RuntimeError("balanced dispatch did not return slot counts")
        slot_begin = dispatched.handle.slot_begin[rank]
        layout = make_grouped_ffn_layout(
            balanced_execution_rows,
            slot_begin=slot_begin,
            slot_count=dispatched.exec_counts,
            runtime_end=True,
        )
        physical_layout = None
        if backend.variant == "probeep":
            replica_slots = layout.num_slots - local_experts
            plan_slots = 3
            plan_slot = int(dispatched.handle.slot)
            home_end = layout.offsets[local_experts - 1 : local_experts]
            final_end = layout.offsets[-1:]
            empty_before = plan_slot * replica_slots
            empty_after = (plan_slots - plan_slot - 1) * replica_slots
            offsets = torch.cat(
                (
                    layout.offsets[:local_experts],
                    home_end.expand(empty_before),
                    layout.offsets[local_experts:],
                    final_end.expand(empty_after),
                )
            ).contiguous()
            begins = torch.cat(
                (
                    layout.slot_begin[:local_experts],
                    home_end.expand(empty_before),
                    layout.slot_begin[local_experts:],
                    final_end.expand(empty_after),
                )
            ).contiguous()
            counts = torch.cat(
                (
                    layout.slot_count[:local_experts],
                    layout.slot_count.new_zeros(empty_before),
                    layout.slot_count[local_experts:],
                    layout.slot_count.new_zeros(empty_after),
                )
            ).contiguous()
            physical_layout = GroupedFFNLayout(
                offsets=offsets,
                slot_begin=begins,
                slot_count=counts,
                num_rows=layout.num_rows,
            )
        return ExpertMaterialization(
            layout, None, None, physical_layout=physical_layout
        )

    if dispatched.recv_topk_idx is None or dispatched.recv_topk_weights is None:
        raise RuntimeError("official dispatch did not return route metadata")
    recv_idx = dispatched.recv_topk_idx
    recv_weights = dispatched.recv_topk_weights
    rows = torch.arange(
        recv_idx.size(0), device=recv_idx.device, dtype=torch.int64
    )
    rows = rows[:, None].expand_as(recv_idx).reshape(-1)
    flat_idx = recv_idx.reshape(-1)
    valid = flat_idx >= 0
    source_rows = rows[valid]
    expert = flat_idx[valid]
    route_weights = recv_weights.reshape(-1)[valid]
    order = torch.argsort(expert, stable=True)
    source_rows = source_rows[order]
    route_weights = route_weights[order]
    expert = expert[order]
    counts = torch.bincount(expert, minlength=local_experts).to(torch.int32)
    source_rows, route_weights, cu_seqlens = pad_grouped_assignment_rows(
        source_rows, route_weights, counts
    )
    layout = make_grouped_ffn_layout(
        source_rows.numel(), cu_seqlens=cu_seqlens, slot_count=counts
    )
    return ExpertMaterialization(layout, source_rows, route_weights)


def local_rank_expert_rows(
    backend: RuntimeBackend,
    dispatched: DispatchResult,
    materialized: ExpertMaterialization,
    *,
    rank: int,
    local_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lower the actual physical grouped layout to a local E256 histogram."""

    num_experts = int(backend.num_experts)
    raw = torch.zeros(num_experts, dtype=torch.int64, device=dispatched.exec_x.device)
    padded = torch.zeros_like(raw)
    if getattr(backend, "probeep_hybrid", False):
        experts = backend.execution_experts(dispatched).to(
            device=raw.device, dtype=torch.int64
        )
        raw_counts = backend.execution_raw_counts(dispatched).to(
            device=raw.device, dtype=torch.int64
        )
        padded_counts = dispatched.handle.local_padded_tokens_per_expert.to(
            device=raw.device, dtype=torch.int64
        )
    elif getattr(backend, "variant", "") == "ultraep_hybridep":
        experts = backend.execution_experts(dispatched).to(
            device=raw.device, dtype=torch.int64
        )
        raw_counts = backend.execution_raw_counts(dispatched).to(
            device=raw.device, dtype=torch.int64
        )
        padded_counts = dispatched.exec_counts.to(
            device=raw.device, dtype=torch.int64
        )
    elif backend.balanced:
        experts = dispatched.handle.slot_expert[rank].to(torch.int64)
        raw_counts = materialized.layout.slot_count.to(torch.int64)
        padded_counts = materialized.layout.offsets.to(torch.int64) - materialized.layout.slot_begin.to(torch.int64)
    else:
        experts = (
            torch.arange(local_experts, device=raw.device, dtype=torch.int64)
            + rank * local_experts
        )
        raw_counts = materialized.layout.slot_count.to(torch.int64)
        padded_counts = materialized.layout.offsets.to(torch.int64) - materialized.layout.slot_begin.to(torch.int64)
    valid = experts >= 0
    raw.scatter_add_(0, experts[valid], raw_counts[valid])
    padded.scatter_add_(0, experts[valid], padded_counts[valid])
    return raw, padded


def run_identity_expert(
    backend: RuntimeBackend,
    dispatched: DispatchResult,
    balanced_execution_rows: int,
) -> torch.Tensor:
    exec_x = dispatched.exec_x
    exec_scales = dispatched.exec_scales
    if getattr(backend, "probeep_hybrid", False):
        return dequantize_fp8_blocks(exec_x, exec_scales)
    if getattr(backend, "variant", "") == "ultraep_hybridep":
        output = dequantize_fp8_blocks(exec_x, exec_scales)
        output.mul_(
            dispatched.handle.route_weights.to(torch.bfloat16).unsqueeze(1)
        )
        dispatched.handle.exec_y.copy_(output)
        return dispatched.handle.exec_y
    if backend.balanced:
        exec_x = exec_x[:balanced_execution_rows]
        exec_scales = exec_scales[:balanced_execution_rows]
    output = dequantize_fp8_blocks(exec_x, exec_scales)
    if backend.balanced:
        dispatched.handle.exec_y[:balanced_execution_rows].copy_(output)
        return dispatched.handle.exec_y
    if not backend.balanced:
        if dispatched.recv_topk_idx is None or dispatched.recv_topk_weights is None:
            raise RuntimeError("official dispatch did not return route metadata")
        token_weight = torch.where(
            dispatched.recv_topk_idx >= 0,
            dispatched.recv_topk_weights,
            torch.zeros_like(dispatched.recv_topk_weights),
        ).sum(dim=1)
        output.mul_(token_weight.to(torch.bfloat16).unsqueeze(1))
    return output


def run_grouped_expert(
    backend: RuntimeBackend,
    dispatched: DispatchResult,
    materialized: ExpertMaterialization,
    weights: GroupedWeights,
    local_experts: int,
    balanced_execution_rows: int,
) -> torch.Tensor:
    exec_x = dispatched.exec_x
    exec_scales = dispatched.exec_scales
    if getattr(backend, "probeep_hybrid", False):
        dequantized = dequantize_fp8_blocks(exec_x, exec_scales)
        gate, up, down = backend.grouped_weights_for(dispatched)
        return run_grouped_ffn_stage(
            dequantized,
            materialized.layout,
            gate,
            up,
            down,
        )
    if getattr(backend, "variant", "") == "ultraep_hybridep":
        dequantized = dequantize_fp8_blocks(exec_x, exec_scales)
        grouped_y = run_grouped_ffn_stage(
            dequantized,
            materialized.layout,
            weights.gate,
            weights.up,
            weights.down,
        )
        dispatched.handle.exec_y.copy_(grouped_y)
        dispatched.handle.exec_y.mul_(
            dispatched.handle.route_weights.to(torch.bfloat16).unsqueeze(1)
        )
        return dispatched.handle.exec_y
    if backend.balanced:
        exec_x = exec_x[:balanced_execution_rows]
        exec_scales = exec_scales[:balanced_execution_rows]
    dequantized = dequantize_fp8_blocks(
        exec_x, exec_scales
    )
    if backend.balanced:
        if backend.variant == "probeep":
            if materialized.physical_layout is None:
                raise RuntimeError("ProbeEP physical expert layout is missing")
            run_grouped_ffn_stage_out(
                dequantized,
                materialized.physical_layout,
                weights.gate,
                weights.up,
                weights.down,
                dispatched.handle.exec_y[:balanced_execution_rows],
                backend.extension.bf16_grouped_mm_out,
            )
            return dispatched.handle.exec_y
        run_grouped_ffn_stage_out(
            dequantized,
            materialized.layout,
            weights.gate,
            weights.up,
            weights.down,
            dispatched.handle.exec_y[:balanced_execution_rows],
            backend.extension.bf16_grouped_mm_out,
        )
        return dispatched.handle.exec_y

    if materialized.source_rows is None or materialized.route_weights is None:
        raise RuntimeError("official grouped layout has no assignment mapping")
    grouped_x = dequantized.index_select(0, materialized.source_rows)
    grouped_y = run_grouped_ffn_stage(
        grouped_x,
        materialized.layout,
        weights.gate,
        weights.up,
        weights.down,
    )
    grouped_y.mul_(materialized.route_weights.to(torch.bfloat16).unsqueeze(1))
    transport_y = torch.zeros_like(dequantized)
    transport_y.index_add_(0, materialized.source_rows, grouped_y)
    return transport_y


def run_forward(
    backend: RuntimeBackend,
    x_fp8: torch.Tensor,
    x_scales: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    rank: int,
    local_experts: int,
    expert_mode: str,
    grouped_weights: GroupedWeights | None,
    balanced_execution_rows: int,
    balanced_home_execution_rows: int,
    timed: bool,
    profile: bool = False,
    probe_feedback: ProbeFeedbackSet | None = None,
    rdma_path_bandwidth_gbps: float = 200.0,
    controller_alpha: float = 0.90,
) -> ForwardSample:
    events = (
        [torch.cuda.Event(enable_timing=True) for _ in range(7)]
        if timed
        else None
    )
    with nvtx_range(profile, "moe_iteration"):
        if events is not None:
            events[0].record()
        with nvtx_range(profile and not backend.balanced, "route_histogram"):
            official_layout = backend.plan(topk_idx)
        if probe_feedback is not None:
            with nvtx_range(profile, "probeep/feedback_bind"):
                for feedback in probe_feedback.updates:
                    backend.update_probe_feedback(
                        feedback.compute_ns,
                        feedback.network_ns,
                        feedback.dispatch_tx_bytes,
                        feedback.dispatch_rx_bytes,
                        feedback.migration_tx_bytes,
                        feedback.migration_rx_bytes,
                        compute_kind=feedback.compute_kind,
                        rdma_path_bandwidth_gbps=rdma_path_bandwidth_gbps,
                        alpha=controller_alpha,
                    )
        if events is not None:
            events[1].record()
        with nvtx_range(profile, "deepep_dispatch"):
            dispatched = backend.dispatch(
                x_fp8,
                x_scales,
                topk_idx,
                topk_weights,
                official_layout,
                compute_kind=(
                    probe_feedback.dispatch_compute_kind
                    if probe_feedback is not None else 1
                ),
            )
        if events is not None:
            events[2].record()

        materialized = None
        if expert_mode == "grouped":
            with nvtx_range(profile, "layout_materialize"):
                materialized = materialize_grouped_layout(
                    backend,
                    dispatched,
                    rank,
                    local_experts,
                    balanced_execution_rows,
                    balanced_home_execution_rows,
                )
        if events is not None:
            events[3].record()

        if expert_mode == "grouped":
            with nvtx_range(profile, "weight_prefetch"):
                backend.prefetch(dispatched)
        if events is not None:
            events[4].record()

        with nvtx_range(profile, "expert_compute"):
            if expert_mode == "identity":
                exec_y = run_identity_expert(
                    backend, dispatched, balanced_execution_rows
                )
            else:
                if grouped_weights is None or materialized is None:
                    raise RuntimeError(
                        "grouped expert weights/layout were not initialized"
                    )
                exec_y = run_grouped_expert(
                    backend,
                    dispatched,
                    materialized,
                    grouped_weights,
                    local_experts,
                    balanced_execution_rows,
                )
        if events is not None:
            events[5].record()
        with nvtx_range(profile, "deepep_combine"):
            output = backend.combine(dispatched, exec_y)
        if events is not None:
            events[6].record()
            events[6].synchronize()

    if not timed:
        valid_rows = -1
    elif getattr(backend, "probeep_hybrid", False):
        valid_rows = int(dispatched.exec_counts.sum().item())
    elif backend.balanced:
        valid_rows = int(dispatched.exec_counts.sum().item())
    else:
        valid_rows = dispatched.exec_x.size(0)

    local_ms = {name: 0.0 for name in PHASE_NAMES}
    if events is not None:
        # ProbeEP count/plan/materialization remains fused into dispatch_ms.
        # plan_ms records only the previous-observation controller update;
        # first observation and non-ProbeEP balanced paths therefore report 0.
        local_ms["plan_ms"] = (
            events[0].elapsed_time(events[1])
            if backend.variant == "probeep" and probe_feedback is not None
            else (0.0 if backend.balanced else events[0].elapsed_time(events[1]))
        )
        local_ms["layout_materialize_ms"] = (
            events[2].elapsed_time(events[3])
            if expert_mode == "grouped"
            else 0.0
        )
        local_ms["dispatch_ms"] = events[1].elapsed_time(events[2])
        local_ms["weight_prefetch_ms"] = (
            events[3].elapsed_time(events[4])
            if expert_mode == "grouped" and backend.balanced
            else 0.0
        )
        local_ms["expert_compute_ms"] = events[4].elapsed_time(events[5])
        local_ms["combine_ms"] = events[5].elapsed_time(events[6])
        local_ms["e2e_ms"] = events[0].elapsed_time(events[6])
    rank_expert_raw = rank_expert_padded = None
    if materialized is not None:
        rank_expert_raw, rank_expert_padded = local_rank_expert_rows(
            backend,
            dispatched,
            materialized,
            rank=rank,
            local_experts=local_experts,
        )
    return ForwardSample(
        output=output,
        local_ms=local_ms,
        valid_rows=valid_rows,
        handle=dispatched.handle,
        handles=(dispatched.handle,),
        dispatches=(dispatched,),
        local_rank_expert_raw=rank_expert_raw,
        local_rank_expert_padded=rank_expert_padded,
    )


def run_forward_dual_microbatch_ht(
    backend: RuntimeBackend,
    microbatches: tuple[MicrobatchInput, MicrobatchInput],
    *,
    rank: int,
    local_experts: int,
    expert_mode: str,
    grouped_weights: GroupedWeights | None,
    timed: bool,
    profile: bool = False,
    probe_feedback: ProbeFeedbackSet | None = None,
    rdma_path_bandwidth_gbps: float = 200.0,
    controller_alpha: float = 0.90,
    attention_overlap: AttentionProbeState | None = None,
) -> ForwardSample:
    """Run the formal vLLM DBO-style HT benchmark schedule.

    The CUDA-event wavefront is exactly
    ``A0 -> (A1 || W+D0) -> (E0 || W+D1) -> E1``.  In particular, W+D0 is not
    submitted before A0 and W+D1 is not submitted before A1.  The Attention
    and MoE controller observations use the two independent release-to-done
    windows ``A1.start -> W+D0.done`` and ``E0.start -> W+D1.done``.
    """

    if len(microbatches) != 2:
        raise ValueError("dual_microbatch_ht requires exactly two microbatches")

    local_rank_expert_raw = torch.zeros(
        backend.num_experts, dtype=torch.int64, device="cuda"
    )
    local_rank_expert_padded = torch.zeros_like(local_rank_expert_raw)
    local_microbatch_rank_expert_raw = torch.zeros(
        (2, backend.num_experts), dtype=torch.int64, device="cuda"
    )
    local_microbatch_rank_expert_padded = torch.zeros_like(
        local_microbatch_rank_expert_raw
    )

    if timed:
        e2e_start = torch.cuda.Event(enable_timing=True)
        e2e_end = torch.cuda.Event(enable_timing=True)
        combine_start = torch.cuda.Event(enable_timing=True)
        combine_end = torch.cuda.Event(enable_timing=True)
        expert_starts = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        expert_ends = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        network_starts = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        network_ends = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        combine_starts = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        combine_ends = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        attention_starts = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        attention_ends = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    else:
        e2e_start = e2e_end = None
        combine_start = combine_end = None
        expert_starts = expert_ends = []
        network_starts = network_ends = []
        combine_starts = combine_ends = []
        attention_starts = attention_ends = []

    def launch_dispatch(
        index: int,
        microbatch: MicrobatchInput,
        previous_event: object | None,
        compute_kind: int,
    ) -> tuple[DispatchResult, object | None]:
        if timed:
            # Timestamp actual communication-stream service separately from the
            # controller's release-to-done observation window.  Enqueueing the
            # dependency here makes the start event occur only after A0/A1 (and
            # any earlier communication work) is complete.
            with torch.cuda.stream(comm_stream):
                wait_event(previous_event)
                network_starts[index].record(comm_stream)
        with nvtx_range(profile, f"ubatch{index}/ht_dispatch"):
            layout = backend.plan(
                microbatch.topk_idx,
                previous_event=previous_event,
                async_finish=True,
            )
            dispatch_previous = (
                previous_event
                if previous_event is not None
                else (layout.event if layout is not None else None)
            )
            dispatched = backend.dispatch(
                microbatch.x_fp8,
                microbatch.x_scales,
                microbatch.topk_idx,
                microbatch.topk_weights,
                layout,
                compute_kind=compute_kind,
                previous_event=dispatch_previous,
                async_finish=True,
            )
            ready_event = dispatched.event
            if expert_mode == "grouped":
                ready_event = backend.prefetch(
                    dispatched,
                    previous_event=ready_event,
                    async_finish=True,
                )
        return dispatched, ready_event

    def run_expert(
        index: int,
        dispatched: DispatchResult,
        ready_event: object | None,
        microbatch: MicrobatchInput,
        before_compute: Callable[[], None] | None = None,
    ) -> torch.Tensor:
        wait_event(ready_event)
        materialized = None
        if expert_mode == "grouped":
            with nvtx_range(profile, f"ubatch{index}/layout_materialize"):
                materialized = materialize_grouped_layout(
                    backend,
                    dispatched,
                    rank,
                    local_experts,
                    microbatch.balanced_execution_rows,
                    microbatch.balanced_home_execution_rows,
                )
                raw_rows, padded_rows = local_rank_expert_rows(
                    backend,
                    dispatched,
                    materialized,
                    rank=rank,
                    local_experts=local_experts,
                )
                local_rank_expert_raw.add_(raw_rows)
                local_rank_expert_padded.add_(padded_rows)
                local_microbatch_rank_expert_raw[index].copy_(raw_rows)
                local_microbatch_rank_expert_padded[index].copy_(padded_rows)
        if timed:
            expert_starts[index].record()
        if before_compute is not None:
            # The callback only submits communication work on comm_stream.  It
            # runs after the E0 release event is enqueued and before the FFN is
            # enqueued, which makes E0 || W+D1 a real overlap window.
            before_compute()
        with nvtx_range(profile, f"ubatch{index}/expert_mlp"):
            if expert_mode == "identity":
                exec_y = run_identity_expert(
                    backend, dispatched, microbatch.balanced_execution_rows
                )
            else:
                if grouped_weights is None or materialized is None:
                    raise RuntimeError(
                        "grouped expert weights/layout were not initialized"
                    )
                exec_y = run_grouped_expert(
                    backend,
                    dispatched,
                    materialized,
                    grouped_weights,
                    local_experts,
                    microbatch.balanced_execution_rows,
                )
        if timed:
            expert_ends[index].record()
        return exec_y

    comm_stream = backend.buffer.get_comm_stream()
    with nvtx_range(profile, f"{backend.variant}/pipeline_iteration"):
        if timed:
            e2e_start.record()
            # Establish one ordered timestamp origin across both streams.
            # Without this device-side wait the comm stream could record a
            # start event before e2e_start has executed on the compute stream.
            comm_stream.wait_event(e2e_start)

        chain_event = backend.capture_event()
        if probe_feedback is not None:
            chain_identity = tuple(
                (
                    feedback.compute_kind,
                    feedback.dispatch_microbatch,
                    feedback.overlap_microbatch,
                )
                for feedback in probe_feedback.updates
            )
            if chain_identity != ((0, 0, 1), (1, 1, 0)):
                raise RuntimeError(
                    "ProbeEP feedback must preserve independent Attention/MoE chains"
                )
            with nvtx_range(profile, "probeep/feedback_bind"):
                for feedback in probe_feedback.updates:
                    chain_event = backend.update_probe_feedback(
                        feedback.compute_ns,
                        feedback.network_ns,
                        feedback.dispatch_tx_bytes,
                        feedback.dispatch_rx_bytes,
                        feedback.migration_tx_bytes,
                        feedback.migration_rx_bytes,
                        compute_kind=feedback.compute_kind,
                        rdma_path_bandwidth_gbps=rdma_path_bandwidth_gbps,
                        alpha=controller_alpha,
                        previous_event=chain_event,
                        async_finish=True,
                    )

        # A0 produces the routing boundary for microbatch 0.  It is deliberately
        # not a controller sample: the Attention sample paired with W+D0 is A1.
        if timed and attention_overlap is not None:
            attention_starts[0].record()
        attention_output0 = run_attention_overlap_probe(
            attention_overlap,
            profile=profile,
            name="attention_or_gate/ubatch0",
        )
        if timed and attention_overlap is not None:
            attention_ends[0].record()
        attention0_event = backend.capture_event()

        # Enqueue the A1 release before submitting W+D0.  Both streams can then
        # advance as soon as A0 completes, with no host or global barrier.
        if timed and attention_overlap is not None:
            attention_starts[1].record()
        dispatch0, ready0 = launch_dispatch(
            0, microbatches[0], attention0_event, compute_kind=0
        )
        if timed:
            network_ends[0].record(comm_stream)
        attention_output1 = run_attention_overlap_probe(
            attention_overlap,
            profile=profile,
            name="attention_or_gate/ubatch1",
        )
        if timed and attention_overlap is not None:
            attention_ends[1].record()
        attention1_event = backend.capture_event()

        shared_replica_bank = (
            expert_mode == "grouped"
            and backend.variant in {"deepep_moonep_on", "ultraep_hybridep"}
        )
        dispatch1: DispatchResult | None = None
        ready1: object | None = None

        def launch_second_dispatch() -> None:
            nonlocal dispatch1, ready1
            dispatch1, ready1 = launch_dispatch(
                1, microbatches[1], attention1_event, compute_kind=1
            )
            if timed:
                network_ends[1].record(comm_stream)

        exec0 = run_expert(
            0,
            dispatch0,
            ready0,
            microbatches[0],
            before_compute=(None if shared_replica_bank else launch_second_dispatch),
        )
        expert0_event = backend.capture_event()
        if shared_replica_bank:
            # MoonEP and UltraEP expose one physical replica-weight bank.  The
            # next Weight phase must not overwrite it while ubatch0's grouped
            # FFN is still consuming it.  The event dependency is device-side;
            # the host remains asynchronous and the baseline pays its real
            # lack of double-buffered replica storage in the E2E interval.
            dispatch1, ready1 = launch_dispatch(
                1, microbatches[1], expert0_event, compute_kind=1
            )
            if timed:
                network_ends[1].record(comm_stream)
        if dispatch1 is None:
            raise RuntimeError("microbatch 1 dispatch was not submitted")
        if timed:
            combine_start.record(comm_stream)
            combine_starts[0].record(comm_stream)
        with nvtx_range(profile, "ubatch0/ht_combine"):
            output0, combine0 = backend.combine_async(
                dispatch0,
                exec0,
                previous_event=expert0_event,
                async_finish=True,
            )
        if timed:
            combine_ends[0].record(comm_stream)

        exec1 = run_expert(1, dispatch1, ready1, microbatches[1])
        expert1_event = backend.capture_event()
        if timed:
            combine_starts[1].record(comm_stream)
        with nvtx_range(profile, "ubatch1/ht_combine"):
            output1, combine1 = backend.combine_async(
                dispatch1,
                exec1,
                previous_event=expert1_event,
                async_finish=True,
            )
        if timed:
            combine_ends[1].record(comm_stream)
            combine_end.record(comm_stream)

        wait_event(combine0)
        wait_event(combine1)
        output = torch.cat((output0, output1), dim=0)
        # Keep the benchmark-only attention computation alive through the timed
        # interval, so the scheduler cannot discard it in graph-like runtimes.
        _ = (attention_output0, attention_output1)
        if timed:
            e2e_end.record()
            e2e_end.synchronize()

    handles = (dispatch0.handle, dispatch1.handle)
    if not timed:
        valid_rows = -1
    elif getattr(backend, "probeep_hybrid", False):
        valid_rows = int(
            sum(int(item.exec_counts.sum().item()) for item in (dispatch0, dispatch1))
        )
    elif backend.balanced:
        valid_rows = int(
            sum(int(item.exec_counts.sum().item()) for item in (dispatch0, dispatch1))
        )
    else:
        valid_rows = dispatch0.exec_x.size(0) + dispatch1.exec_x.size(0)

    local_ms = {name: 0.0 for name in PHASE_NAMES}
    probe_windows_ms: tuple[tuple[int, float, float], ...] = ()
    timeline_intervals_ms: tuple[
        tuple[str, int, str, float, float], ...
    ] = ()
    if timed:
        local_ms["dispatch_ms"] = (
            attention_starts[1].elapsed_time(network_ends[1])
            if attention_overlap is not None
            else e2e_start.elapsed_time(network_ends[1])
        )
        local_ms["expert_compute_ms"] = sum(
            expert_starts[i].elapsed_time(expert_ends[i]) for i in range(2)
        )
        local_ms["combine_ms"] = combine_start.elapsed_time(combine_end)
        local_ms["e2e_ms"] = e2e_start.elapsed_time(e2e_end)
        # In the DBO-style runner, controller/layout/weight are intentionally
        # fused into the HT dispatch interval.  nsys, not per-phase CUDA events,
        # is the authority for sub-stage overlap.
        local_ms["plan_ms"] = 0.0
        local_ms["layout_materialize_ms"] = 0.0
        local_ms["weight_prefetch_ms"] = 0.0
        moe_compute_ms = expert_starts[0].elapsed_time(expert_ends[0])
        moe_network_ms = max(
            0.0, expert_starts[0].elapsed_time(network_ends[1])
        )
        windows = [(1, moe_compute_ms, moe_network_ms)]
        if attention_overlap is not None:
            windows.insert(
                0,
                (
                    0,
                    attention_starts[1].elapsed_time(attention_ends[1]),
                    max(
                        0.0,
                        attention_starts[1].elapsed_time(network_ends[0]),
                    ),
                ),
            )
        probe_windows_ms = tuple(windows)
        timeline_events: list[
            tuple[str, int, str, torch.cuda.Event, torch.cuda.Event]
        ] = [
            (
                "weight_dispatch",
                0,
                "communication",
                network_starts[0],
                network_ends[0],
            ),
            (
                "weight_dispatch",
                1,
                "communication",
                network_starts[1],
                network_ends[1],
            ),
            ("expert_mlp", 0, "compute", expert_starts[0], expert_ends[0]),
            ("combine", 0, "communication", combine_starts[0], combine_ends[0]),
            ("expert_mlp", 1, "compute", expert_starts[1], expert_ends[1]),
            ("combine", 1, "communication", combine_starts[1], combine_ends[1]),
        ]
        if attention_overlap is not None:
            timeline_events[0:0] = [
                (
                    "attention_or_gate",
                    0,
                    "compute",
                    attention_starts[0],
                    attention_ends[0],
                ),
                (
                    "attention_or_gate",
                    1,
                    "compute",
                    attention_starts[1],
                    attention_ends[1],
                ),
            ]
        timeline_intervals_ms = tuple(
            (
                stage,
                microbatch,
                logical_stream,
                e2e_start.elapsed_time(start_event),
                e2e_start.elapsed_time(end_event),
            )
            for stage, microbatch, logical_stream, start_event, end_event
            in timeline_events
        )

    return ForwardSample(
        output=output,
        local_ms=local_ms,
        valid_rows=valid_rows,
        handle=dispatch1.handle,
        handles=handles,
        dispatches=(dispatch0, dispatch1),
        probe_windows_ms=probe_windows_ms,
        local_rank_expert_raw=local_rank_expert_raw,
        local_rank_expert_padded=local_rank_expert_padded,
        local_microbatch_rank_expert_raw=local_microbatch_rank_expert_raw,
        local_microbatch_rank_expert_padded=local_microbatch_rank_expert_padded,
        timeline_intervals_ms=timeline_intervals_ms,
        timeline_origin_event=e2e_start if timed else None,
    )


def gather_moe_compute_ns(sample: ForwardSample, world_size: int) -> torch.Tensor:
    local = torch.tensor(
        [round(sample.local_ms["expert_compute_ms"] * 1_000_000)],
        dtype=torch.int64,
        device="cuda",
    )
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    return torch.cat(gathered).contiguous()


def gather_probe_window_ns(
    sample: ForwardSample, world_size: int, compute_kind: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather one same-window (compute, W+D) observation across ranks."""

    match = next(
        (entry for entry in sample.probe_windows_ms if entry[0] == compute_kind),
        None,
    )
    if match is None:
        raise RuntimeError(
            f"dual_microbatch_ht did not record compute_kind={compute_kind}"
        )
    local = torch.tensor(
        [round(match[1] * 1_000_000), round(match[2] * 1_000_000)],
        dtype=torch.int64,
        device="cuda",
    )
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    matrix = torch.stack(gathered, dim=0)
    return matrix[:, 0].contiguous(), matrix[:, 1].contiguous()


def build_probe_feedback(
    sample: ForwardSample,
    *,
    iteration: int,
    producer_phase: str,
    runner_mode: str,
    world_size: int,
    rank: int,
    ranks_per_server: int,
    hidden: int,
    topk: int,
    enable_attention_feedback: bool,
    attention_probe: AttentionProbeState | None,
    producer_layer_id: int,
    producer_repeat: int,
) -> ProbeFeedbackSet:
    if runner_mode == "dual_microbatch_ht":
        moe_compute_ns, moe_network_ns = gather_probe_window_ns(
            sample, world_size, 1
        )
    else:
        moe_compute_ns = gather_moe_compute_ns(sample, world_size)
        moe_network_ns = None
    updates = [
        make_probe_feedback_device(
            sample,
            rank=rank,
            world_size=world_size,
            ranks_per_server=ranks_per_server,
            hidden=hidden,
            topk=topk,
            compute_ns=moe_compute_ns,
            compute_kind=1,
            network_ns=moe_network_ns,
        )
    ]
    if enable_attention_feedback and attention_probe is not None:
        if runner_mode == "dual_microbatch_ht":
            attention_compute_ns, attention_network_ns = (
                gather_probe_window_ns(sample, world_size, 0)
            )
        else:
            attention_compute_ns = None
            attention_network_ns = None
        if attention_compute_ns is not None:
            updates.insert(
                0,
                make_probe_feedback_device(
                    sample,
                    rank=rank,
                    world_size=world_size,
                    ranks_per_server=ranks_per_server,
                    hidden=hidden,
                    topk=topk,
                    compute_ns=attention_compute_ns,
                    compute_kind=0,
                    network_ns=attention_network_ns,
                ),
            )
    return ProbeFeedbackSet(
        updates=tuple(updates),
        dispatch_compute_kind=(
            1
            if runner_mode == "dual_microbatch_ht"
            else (
                next_probe_dispatch_compute_kind(iteration + 1)
                if enable_attention_feedback
                else 1
            )
        ),
        producer_phase=producer_phase,
        producer_iteration=iteration,
        producer_layer_id=producer_layer_id,
        producer_repeat=producer_repeat,
    )


def next_probe_dispatch_compute_kind(iteration: int) -> int:
    """Select which feedback chain the next ProbeEP dispatch should consume."""

    pattern = os.getenv("PROBEEP_DISPATCH_COMPUTE_PATTERN", "alternate").lower()
    if pattern in {"attention", "attn", "a", "0"}:
        return 0
    if pattern in {"moe", "expert", "e", "1"}:
        return 1
    if pattern != "alternate":
        raise ValueError(
            "PROBEEP_DISPATCH_COMPUTE_PATTERN must be alternate, attention, or moe"
        )
    # Iteration 0 has no completed feedback and uses the MoE fallback.  Once both
    # chains have a sample, alternate A/M/A/M to exercise the two independent
    # controller rows.
    return 0 if iteration % 2 == 1 else 1


def raw_data1_layer_id(workload_mode: str) -> int:
    prefix = "raw_data1_layer_"
    if not workload_mode.startswith(prefix):
        raise ValueError(
            "formal cross-layer feedback requires raw_data1_layer_<id> workload"
        )
    suffix = workload_mode[len(prefix) :]
    if not suffix.isdigit():
        raise ValueError(f"invalid RawData1 layer workload: {workload_mode}")
    return int(suffix)


def store_probe_feedback(
    bank: dict[tuple[str, int, int], ProbeFeedbackDevice],
    feedback: ProbeFeedbackSet,
) -> None:
    kinds = {update.compute_kind for update in feedback.updates}
    if kinds != {0, 1}:
        raise RuntimeError("dual-microbatch feedback must contain independent A/M chains")
    for update in feedback.updates:
        key = (
            feedback.producer_phase,
            feedback.producer_iteration,
            update.compute_kind,
        )
        if key in bank:
            raise RuntimeError(f"duplicate ProbeEP feedback chain key: {key}")
        bank[key] = update


def load_probe_feedback(
    bank: dict[tuple[str, int, int], ProbeFeedbackDevice],
    *,
    phase: str,
    iteration: int,
    producer_layer_id: int,
    producer_repeat: int,
) -> ProbeFeedbackSet | None:
    updates = tuple(
        bank[key]
        for compute_kind in (0, 1)
        if (key := (phase, iteration, compute_kind)) in bank
    )
    if not updates:
        return None
    if tuple(update.compute_kind for update in updates) != (0, 1):
        raise RuntimeError(
            "previous-layer feedback is incomplete; A/M chains cannot substitute"
        )
    return ProbeFeedbackSet(
        updates=updates,
        dispatch_compute_kind=1,
        producer_phase=phase,
        producer_iteration=iteration,
        producer_layer_id=producer_layer_id,
        producer_repeat=producer_repeat,
    )


def probe_feedback_handles(
    sample: ForwardSample, compute_kind: int
) -> tuple[object, ...]:
    """Return only the dispatch handle paired with one A/M observation."""

    if compute_kind not in (0, 1):
        raise ValueError("ProbeEP compute_kind must be 0=Attention or 1=MoE")
    handles = sample.handles or (sample.handle,)
    if len(handles) == 1:
        return handles
    if len(handles) == 2:
        # dual_microbatch_ht fixes W0+D0 to Attention and W1+D1 to MoE.
        return (handles[compute_kind],)
    raise ValueError("ProbeEP feedback supports one or two dispatch handles")


def dispatch_endpoint_bytes(
    dispatch_matrix: torch.Tensor, ranks_per_server: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map final execution destinations onto DeepEP's same-lane RDMA endpoints.

    ``dispatch_matrix[src_rank, exec_rank]`` records de-duplicated wire bytes
    needed by the final execution rank.  DeepEP does not receive those bytes
    directly on ``exec_rank``: source lane ``i`` sends to lane ``i`` on the
    destination server, which then performs the server-local NVLink forward.
    Controller RX accounting must use that relay endpoint mapping.
    """

    if (
        dispatch_matrix.ndim != 2
        or dispatch_matrix.size(0) != dispatch_matrix.size(1)
        or dispatch_matrix.size(0) % ranks_per_server != 0
    ):
        raise ValueError("dispatch matrix is incompatible with the server topology")
    num_servers = dispatch_matrix.size(0) // ranks_per_server
    directed = dispatch_matrix.view(
        num_servers, ranks_per_server, num_servers, ranks_per_server
    )
    dispatch_tx = directed.sum(dim=(2, 3)).reshape(-1).contiguous()
    # Sum over source server and final destination lane.  The remaining axes
    # are [source_lane, destination_server]; transpose to global rank order.
    dispatch_rx = (
        directed.sum(dim=(0, 3)).transpose(0, 1).reshape(-1).contiguous()
    )
    return dispatch_tx, dispatch_rx


def relay_dispatch_matrix(
    destination_server_bytes: torch.Tensor, ranks_per_server: int
) -> torch.Tensor:
    """Lower per-destination-server wire bytes to DeepEP relay endpoints.

    DeepEP de-duplicates a source token once per destination RDMA rank/server.
    Source lane ``i`` sends that single wire message to lane ``i`` on the
    destination server; the destination then fans out over NVLink.  Expanding
    via final execution ranks would count one physical wire message more than
    once when a token reaches multiple ranks on the same server.
    """

    if destination_server_bytes.ndim != 2:
        raise ValueError("destination_server_bytes must have shape [R,P]")
    world_size, num_servers = destination_server_bytes.shape
    if world_size % ranks_per_server != 0 or (
        world_size // ranks_per_server != num_servers
    ):
        raise ValueError("server byte matrix is incompatible with the topology")
    matrix = torch.zeros(
        (world_size, world_size),
        dtype=destination_server_bytes.dtype,
        device=destination_server_bytes.device,
    )
    sources = torch.arange(world_size, device=destination_server_bytes.device)
    lanes = sources.remainder(ranks_per_server)
    for destination_server in range(num_servers):
        relay = destination_server * ranks_per_server + lanes
        matrix[sources, relay] = destination_server_bytes[:, destination_server]
    return matrix.contiguous()


def make_probe_feedback_device(
    sample: ForwardSample,
    *,
    rank: int,
    world_size: int,
    ranks_per_server: int,
    hidden: int,
    topk: int,
    compute_ns: torch.Tensor,
    compute_kind: int,
    network_ns: torch.Tensor | None = None,
) -> ProbeFeedbackDevice:
    """Publish one measured ProbeEP observation for the next dispatch.

    CUDA events have already completed when this adapter runs.  Python does
    not evaluate the controller or planner: it only publishes measured timing
    scalars and exact plan byte counters as device vectors.
    """

    if compute_kind not in (0, 1):
        raise ValueError("ProbeEP compute_kind must be 0=Attention or 1=MoE")
    if compute_ns.numel() != world_size:
        raise ValueError("ProbeEP compute_ns must have one value per rank")

    if network_ns is None:
        local_network = torch.tensor(
            [
                round(
                    (
                        sample.local_ms["dispatch_ms"]
                        + sample.local_ms["weight_prefetch_ms"]
                    )
                    * 1_000_000
                )
            ],
            dtype=torch.int64,
            device="cuda",
        )
        gathered_network = [
            torch.empty_like(local_network) for _ in range(world_size)
        ]
        dist.all_gather(gathered_network, local_network)
        network_ns = torch.cat(gathered_network).contiguous()
    elif network_ns.numel() != world_size:
        raise ValueError("ProbeEP network_ns must have one value per rank")

    scales = hidden // 128
    source_meta_bytes = 8
    # This is the aligned DeepEP *wire message* from
    # internode::get_num_bytes_per_token(): FP8 activation, block scales,
    # SourceMeta, int32 TopK indices, and fp32 TopK weights.  Do not replace
    # it with the planner's smaller activation+scale admission estimate.
    token_bytes = hidden + scales * 4 + source_meta_bytes + topk * 8
    token_bytes = (token_bytes + 15) // 16 * 16
    local_server = rank // ranks_per_server
    handles = probe_feedback_handles(sample, compute_kind)
    num_servers = world_size // ranks_per_server
    local_destination_server_bytes = torch.zeros(
        num_servers, dtype=torch.int64, device="cuda"
    )
    for handle in handles:
        local_destination_server_bytes.add_(
            handle.num_tokens_per_rdma_rank.to(torch.int64).mul(token_bytes)
        )
    local_destination_server_bytes[local_server] = 0
    gathered_matrix = [
        torch.empty_like(local_destination_server_bytes)
        for _ in range(world_size)
    ]
    dist.all_gather(gathered_matrix, local_destination_server_bytes)
    dispatch_matrix = relay_dispatch_matrix(
        torch.stack(gathered_matrix, dim=0), ranks_per_server
    )
    dispatch_tx, dispatch_rx = dispatch_endpoint_bytes(
        dispatch_matrix, ranks_per_server
    )
    migration_tx = torch.zeros_like(sample.handle.probe_assigned_tx_bytes)
    migration_rx = torch.zeros_like(sample.handle.probe_assigned_rx_bytes)
    for handle in handles:
        # Admission bytes describe the selected plan.  A stable plan/version
        # can hit the per-slot replica cache and issue no Weight RDMA at all.
        # Gate on the device so feedback reflects actual traffic without a
        # host scalar read in the measured path.
        transfer_required = (
            handle.weight_transfer_required.ne(0).to(torch.int64).reshape(())
        )
        migration_tx = (
            migration_tx
            + handle.probe_assigned_tx_bytes * transfer_required
        )
        migration_rx = (
            migration_rx
            + handle.probe_assigned_rx_bytes * transfer_required
        )

    return ProbeFeedbackDevice(
        compute_ns=compute_ns.contiguous(),
        network_ns=network_ns,
        dispatch_tx_bytes=dispatch_tx,
        dispatch_rx_bytes=dispatch_rx,
        migration_tx_bytes=migration_tx.clone(),
        migration_rx_bytes=migration_rx.clone(),
        dispatch_matrix_bytes=dispatch_matrix.clone(),
        compute_kind=compute_kind,
        dispatch_microbatch=compute_kind,
        overlap_microbatch=1 - compute_kind,
    )


def collect_dispatch_rail_telemetry(
    sample: ForwardSample,
    *,
    case: dict[str, object],
    rank: int,
    world_size: int,
    ranks_per_server: int,
) -> list[dict[str, object]]:
    """Collect exact runtime dispatch units after the timed interval.

    The returned rows describe logical DeepEP/HybridEP cross-server paths, not
    mlx5 hardware counters.  Each source rank maps to the same local GPU index
    at the destination server, so every directed rail is present even when it
    carries zero bytes.  That fixed shape lets the report distinguish a quiet
    rail from missing telemetry.
    """

    if not sample.dispatches:
        return []
    num_servers = world_size // ranks_per_server
    _, rails_per_nic, physical_gbps, rail_gbps = probe_rail_topology(
        ranks_per_server
    )
    cache_mode = (
        probe_weight_cache_mode()
        if str(case["system"]) == "probeep"
        else "not_applicable"
    )
    weight_version = (
        int(os.getenv("PROBEEP_WEIGHT_VERSION", "-1"))
        if str(case["system"]) == "probeep"
        else -1
    )
    local_server = rank // ranks_per_server
    gathered_by_microbatch: list[list[torch.Tensor]] = []
    unit_names: list[str] = []
    for dispatch in sample.dispatches:
        if dispatch.wire_routing_map is not None:
            routing = dispatch.wire_routing_map
            if routing.ndim != 2 or routing.size(1) % num_servers != 0:
                raise RuntimeError("HybridEP routing map has an invalid server shape")
            local_units = (
                routing.view(routing.size(0), num_servers, -1)
                .any(dim=2)
                .sum(dim=0, dtype=torch.int64)
            )
            unit_name = "destination_server_token"
        elif dispatch.wire_units is not None:
            units = dispatch.wire_units.to(torch.int64).flatten()
            if dispatch.wire_unit_scope == "rank":
                if units.numel() != world_size:
                    raise RuntimeError("rank-scoped wire units have an invalid shape")
                local_units = units.view(num_servers, ranks_per_server).sum(dim=1)
                unit_name = "route_occurrence"
            elif dispatch.wire_unit_scope == "server":
                if units.numel() != num_servers:
                    raise RuntimeError("server-scoped wire units have an invalid shape")
                local_units = units
                unit_name = "destination_server_token"
            else:
                raise RuntimeError("dispatch wire unit scope is missing")
        else:
            raise RuntimeError("dispatch did not retain post-timing wire telemetry")
        local_units = local_units.clone()
        local_units[local_server] = 0
        gathered = [torch.empty_like(local_units) for _ in range(world_size)]
        dist.all_gather(gathered, local_units)
        gathered_by_microbatch.append(gathered)
        unit_names.append(unit_name)

    if rank != 0:
        return []
    rows: list[dict[str, object]] = []
    for microbatch, (dispatch, gathered, unit_name) in enumerate(
        zip(sample.dispatches, gathered_by_microbatch, unit_names)
    ):
        compute_kind = microbatch if len(sample.dispatches) == 2 else 1
        for source_rank, source_units in enumerate(gathered):
            source_server = source_rank // ranks_per_server
            rail = source_rank % ranks_per_server
            values = source_units.cpu().tolist()
            for destination_server in range(num_servers):
                if source_server == destination_server:
                    continue
                destination_rank = destination_server * ranks_per_server + rail
                units = int(values[destination_server])
                dispatch_bytes = units * int(dispatch.wire_bytes_per_unit)
                path_id = (
                    (source_server * num_servers + destination_server)
                    * ranks_per_server
                    + rail
                )
                rows.append(
                    {
                        "schema_version": case["schema_version"],
                        "run_id": case["run_id"],
                        "slurm_job_id": case["slurm_job_id"],
                        "benchmark_scope": case["benchmark_scope"],
                        "runner_mode": case["runner_mode"],
                        "system": case["system"],
                        "balance": case["balance"],
                        "direction": case["direction"],
                        "workload": case["workload"],
                        "bias_ratio": case["bias_ratio"],
                        "seed": case["seed"],
                        "repeat": case["repeat"],
                        "iteration": case["iteration"],
                        "routing_sha256": case["routing_sha256"],
                        "dispatch_compute_kind": compute_kind,
                        "dispatch_compute_name": (
                            "attention" if compute_kind == 0 else "moe"
                        ),
                        "microbatch": microbatch,
                        "path_id": path_id,
                        "physical_nic": rail // rails_per_nic,
                        "subrail": rail % rails_per_nic,
                        "rail_bandwidth_gbps": rail_gbps,
                        "physical_nic_bandwidth_gbps": physical_gbps,
                        "weight_cache_mode": cache_mode,
                        "expert_weight_version": weight_version,
                        "source_rank": source_rank,
                        "destination_rank": destination_rank,
                        "chunk_count": 0,
                        "dispatch_units": units,
                        "dispatch_unit_name": unit_name,
                        "dispatch_bytes_per_unit": dispatch.wire_bytes_per_unit,
                        "traffic_source": dispatch.wire_traffic_source,
                        "dispatch_bytes": dispatch_bytes,
                        "weight_bytes": 0,
                        "tx_bytes": dispatch_bytes,
                        "rx_bytes": dispatch_bytes,
                    }
                )
    return rows


def probe_observation_rows(
    feedback: ProbeFeedbackSet,
    *,
    case: dict[str, object],
    consumer_iteration: int,
    consumer_layer_id: int,
    consumer_repeat: int,
    ranks_per_server: int,
    controller_alpha: float,
    rdma_path_bandwidth_gbps: float,
) -> list[dict[str, object]]:
    """Serialize the exact A/M observation consumed by this iteration."""

    rows: list[dict[str, object]] = []
    for update in feedback.updates:
        vectors = [
            tensor.cpu().tolist()
            for tensor in (
                update.compute_ns,
                update.network_ns,
                update.dispatch_tx_bytes,
                update.dispatch_rx_bytes,
                update.migration_tx_bytes,
                update.migration_rx_bytes,
            )
        ]
        world_size = len(vectors[0])
        if any(len(values) != world_size for values in vectors):
            raise RuntimeError("ProbeEP observation vectors have inconsistent shapes")
        for global_rank in range(world_size):
            rows.append(
                {
                    **case,
                    "producer_phase": feedback.producer_phase,
                    "producer_iteration": feedback.producer_iteration,
                    "producer_layer_id": feedback.producer_layer_id,
                    "producer_repeat": feedback.producer_repeat,
                    "consumer_iteration": consumer_iteration,
                    "consumer_layer_id": consumer_layer_id,
                    "consumer_repeat": consumer_repeat,
                    "compute_kind": update.compute_kind,
                    "compute_name": (
                        "attention" if update.compute_kind == 0 else "moe"
                    ),
                    "dispatch_microbatch": update.dispatch_microbatch,
                    "overlap_microbatch": update.overlap_microbatch,
                    "global_rank": global_rank,
                    "node_rank": global_rank // ranks_per_server,
                    "local_rank": global_rank % ranks_per_server,
                    "compute_ns": int(vectors[0][global_rank]),
                    "network_ns": int(vectors[1][global_rank]),
                    "dispatch_tx_bytes": int(vectors[2][global_rank]),
                    "dispatch_rx_bytes": int(vectors[3][global_rank]),
                    "migration_tx_bytes": int(vectors[4][global_rank]),
                    "migration_rx_bytes": int(vectors[5][global_rank]),
                    "controller_alpha": controller_alpha,
                    "rdma_path_bandwidth_gbps": rdma_path_bandwidth_gbps,
                }
            )
    return rows


def collect_probe_runtime_telemetry(
    sample: ForwardSample,
    *,
    iteration: int,
    dispatch_compute_kind: int,
    run_id: str,
    scope: str,
    runner_mode: str,
    system: str,
    balance: str,
    workload: str,
    bias_ratio: float,
    seed: int,
    repeat: int,
    routing_sha256: str,
    ranks_per_server: int,
    dispatch_matrix_bytes: torch.Tensor,
    expert_weight_version: int,
    weight_cache_mode: str,
    handle: object | None = None,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Read one completed runtime plan after the timed interval."""

    handle = sample.handle if handle is None else handle
    counts = handle.probe_plan_counts.cpu().tolist()
    admitted_count, chunk_count = int(counts[0]), int(counts[1])
    admitted = handle.probe_admitted_experts[:admitted_count].cpu().tolist()
    deferred_mask = handle.probe_deferred_experts.cpu()
    deferred = torch.nonzero(deferred_mask, as_tuple=False).flatten().tolist()
    chunks = handle.probe_chunk_table[:chunk_count].cpu().tolist()
    assigned_tx = handle.probe_assigned_tx_bytes.cpu().tolist()
    assigned_rx = handle.probe_assigned_rx_bytes.cpu().tolist()

    num_servers = int(handle.probe_server_load_after.numel())
    _, rails_per_nic, physical_gbps, rail_gbps = probe_rail_topology(
        ranks_per_server
    )
    dispatch_matrix = dispatch_matrix_bytes.cpu()
    if tuple(dispatch_matrix.shape) != (
        num_servers * ranks_per_server,
        num_servers * ranks_per_server,
    ):
        raise RuntimeError("ProbeEP dispatch matrix has an invalid topology shape")
    transfer_required = int(
        handle.weight_transfer_required.cpu().item() != 0
    )
    weight_pair_load = handle.probe_pair_load_bytes.cpu().mul(
        transfer_required
    )
    rail_groups: dict[tuple[int, int, int], list[list[int]]] = {}
    for row in chunks:
        key = (int(row[4]), int(row[5]), int(row[10]))
        rail_groups.setdefault(key, []).append(row)
    rdma_path_rows: list[dict[str, object]] = []
    for source_server in range(num_servers):
        for destination_server in range(num_servers):
            if source_server == destination_server:
                continue
            destination_begin = destination_server * ranks_per_server
            destination_end = destination_begin + ranks_per_server
            for rail in range(ranks_per_server):
                source_rank = source_server * ranks_per_server + rail
                destination_rank = destination_server * ranks_per_server + rail
                dispatch_bytes = int(
                    dispatch_matrix[
                        source_rank, destination_begin:destination_end
                    ].sum().item()
                )
                weight_bytes = int(
                    weight_pair_load[source_server, destination_server, rail].item()
                )
                if dispatch_bytes == 0 and weight_bytes == 0:
                    continue
                selected = rail_groups.get(
                    (source_server, destination_server, rail), []
                )
                chunk_bytes = (
                    sum(int(row[9]) for row in selected)
                    * transfer_required
                )
                if chunk_bytes != weight_bytes:
                    raise RuntimeError(
                        "ProbeEP pair rail weight bytes disagree with chunk table"
                    )
                path_id = (
                    (source_server * num_servers + destination_server)
                    * ranks_per_server
                    + rail
                )
                rdma_path_rows.append({
                    "schema_version": SCHEMA_VERSION,
                    "run_id": run_id,
                    "slurm_job_id": os.getenv("SLURM_JOB_ID", ""),
                    "benchmark_scope": scope,
                    "runner_mode": runner_mode,
                    "system": system,
                    "balance": balance,
                    "direction": os.getenv("DIRECTION", "forward"),
                    "workload": workload,
                    "bias_ratio": bias_ratio,
                    "seed": seed,
                    "repeat": repeat,
                    "iteration": iteration,
                    "routing_sha256": routing_sha256,
                    "dispatch_compute_kind": dispatch_compute_kind,
                    "dispatch_compute_name": (
                        "attention" if dispatch_compute_kind == 0 else "moe"
                    ),
                    "microbatch": dispatch_compute_kind,
                    "path_id": path_id,
                    "physical_nic": rail // rails_per_nic,
                    "subrail": rail % rails_per_nic,
                    "rail_bandwidth_gbps": rail_gbps,
                    "physical_nic_bandwidth_gbps": physical_gbps,
                    "weight_cache_mode": weight_cache_mode,
                    "expert_weight_version": expert_weight_version,
                    "source_rank": source_rank,
                    "destination_rank": destination_rank,
                    "chunk_count": len(selected) * transfer_required,
                    "dispatch_units": 0,
                    "dispatch_unit_name": "destination_server_token",
                    "dispatch_bytes_per_unit": 0,
                    "traffic_source": "production_probeep_weight_chunk_table",
                    "dispatch_bytes": dispatch_bytes,
                    "weight_bytes": weight_bytes,
                    "tx_bytes": dispatch_bytes + weight_bytes,
                    "rx_bytes": dispatch_bytes + weight_bytes,
                })

    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "slurm_job_id": os.getenv("SLURM_JOB_ID", ""),
        "benchmark_scope": scope,
        "runner_mode": runner_mode,
        "system": system,
        "balance": balance,
        "direction": os.getenv("DIRECTION", "forward"),
        "workload": workload,
        "bias_ratio": bias_ratio,
        "seed": seed,
        "repeat": repeat,
        "iteration": iteration,
        "routing_sha256": routing_sha256,
        "dispatch_compute_kind": dispatch_compute_kind,
        "dispatch_compute_name": (
            "attention" if dispatch_compute_kind == 0 else "moe"
        ),
        "source": "production_cuda_plan",
        "server_load_before": handle.probe_server_load_before.cpu().tolist(),
        "server_load_after": handle.probe_server_load_after.cpu().tolist(),
        "server_padded_load_before": (
            handle.probe_server_padded_load_before.cpu().tolist()
        ),
        "server_padded_load_after": (
            handle.probe_server_padded_load_after.cpu().tolist()
        ),
        "rank_load_after": handle.slot_count.sum(1).cpu().tolist(),
        "migration_budget_bytes": (
            handle.probe_migration_budget_snapshot.cpu().tolist()
        ),
        "endpoint_total_cap_bytes": (
            handle.probe_endpoint_total_cap_bytes.cpu().tolist()
        ),
        "dispatch_tx_bytes": handle.probe_dispatch_tx_bytes.cpu().tolist(),
        "dispatch_rx_bytes": handle.probe_dispatch_rx_bytes.cpu().tolist(),
        "compute_intents": handle.probe_compute_intents[
            : int(counts[3])
        ].cpu().tolist(),
        "admitted_experts": admitted,
        "admitted_placements": [
            {
                "encoded": int(encoded),
                "expert_id": int(encoded) // PROBEEP_PLAN_SERVER_STRIDE,
                "destination_server": int(encoded) % PROBEEP_PLAN_SERVER_STRIDE,
            }
            for encoded in admitted
        ],
        "deferred_experts": deferred,
        "assigned_tx_bytes": assigned_tx,
        "assigned_rx_bytes": assigned_rx,
        "weight_transfer_required": transfer_required,
        "weight_cache_mode": weight_cache_mode,
        "expert_weight_version": expert_weight_version,
        "plan_counts": counts,
        "invariants": {
            "placement_or_capacity_error": int(counts[2]),
            "max_local_placements": int(counts[5]),
            "negative_count": int(counts[6]),
            "prefix_mismatch_count": int(counts[7]),
            "planning_done": int(counts[8]),
        },
        "planner_phase_cycles": {
            "compute_intent": int(counts[9]),
            "network_admission": int(counts[10]),
            "server_local_packing": int(counts[11]),
            "finalization": int(counts[12]),
        },
        "chunk_table": chunks,
    }
    chunk_rows: list[dict[str, object]] = []
    for chunk_ordinal, row in enumerate(chunks):
        chunk_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "slurm_job_id": os.getenv("SLURM_JOB_ID", ""),
                "benchmark_scope": scope,
                "runner_mode": runner_mode,
                "system": system,
                "balance": balance,
                "direction": os.getenv("DIRECTION", "forward"),
                "workload": workload,
                "bias_ratio": bias_ratio,
                "seed": seed,
                "repeat": repeat,
                "iteration": iteration,
                "routing_sha256": routing_sha256,
                "dispatch_compute_kind": dispatch_compute_kind,
                "dispatch_compute_name": (
                    "attention" if dispatch_compute_kind == 0 else "moe"
                ),
                "chunk_ordinal": chunk_ordinal,
                "expert_id": int(row[0]),
                "replica_id": int(row[1]),
                "seed_rank": int(row[2]),
                "expert_chunk_index": int(row[3]),
                "source_server": int(row[4]),
                "destination_server": int(row[5]),
                "source_rank": int(row[6]),
                "destination_rank": int(row[7]),
                "physical_nic": int(row[10]) // rails_per_nic,
                "subrail": int(row[10]) % rails_per_nic,
                "rail_bandwidth_gbps": rail_gbps,
                "physical_nic_bandwidth_gbps": physical_gbps,
                "weight_cache_mode": weight_cache_mode,
                "expert_weight_version": expert_weight_version,
                "expert_offset_bytes": int(row[8]),
                "chunk_bytes": int(row[9]),
                "rail": int(row[10]),
                "source_path_offset_bytes": int(row[11]),
                "destination_path_offset_bytes": int(row[12]),
                "transfer_required": transfer_required,
            }
        )
    return summary, rdma_path_rows, chunk_rows


def merge_probe_weight_telemetry(
    dispatch_rows: list[dict[str, object]],
    weight_rows: list[dict[str, object]],
) -> None:
    """Overlay ProbeEP cache-miss Weight chunks on fixed dispatch rail rows."""

    index = {
        (
            int(row["iteration"]),
            int(row["dispatch_compute_kind"]),
            int(row["path_id"]),
        ): row
        for row in dispatch_rows
    }
    for weight in weight_rows:
        key = (
            int(weight["iteration"]),
            int(weight["dispatch_compute_kind"]),
            int(weight["path_id"]),
        )
        target = index.get(key)
        if target is None:
            raise RuntimeError(f"ProbeEP weight path has no dispatch rail row: {key}")
        target["chunk_count"] = int(weight["chunk_count"])
        target["weight_bytes"] = int(weight["weight_bytes"])
        target["tx_bytes"] = int(target["dispatch_bytes"]) + int(
            target["weight_bytes"]
        )
        target["rx_bytes"] = target["tx_bytes"]
        target["traffic_source"] = (
            f"{target['traffic_source']}+production_probeep_weight_chunk_table"
        )


def rank_max_and_samples(
    local_ms: dict[str, float], world_size: int
) -> tuple[dict[str, float], list[dict[str, float]]]:
    local = torch.tensor(
        [local_ms[name] for name in PHASE_NAMES],
        dtype=torch.float32,
        device="cuda",
    )
    maximum = local.clone()
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    maxima = dict(zip(PHASE_NAMES, maximum.cpu().tolist()))
    samples = [dict(zip(PHASE_NAMES, item.cpu().tolist())) for item in gathered]
    return maxima, samples


def update_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    routing_mode: str,
    routing_bias_ratio: float,
    routing_seed: int,
    world_size: int,
    num_tokens: int,
    topk: int,
    local_experts: int,
    replica_slots: int,
    token_padding: int,
    execution_rows: int,
    capacity_rows: int,
) -> None:
    if path.exists():
        manifest = json.loads(path.read_text())
    else:
        manifest = {
            "run_id": os.environ["PROBEEP_RUN_ID"],
            "benchmark_scope": "full_moe",
            "slurm": {
                "job_id": os.getenv("SLURM_JOB_ID", ""),
                "nodelist": os.getenv("SLURM_JOB_NODELIST", ""),
            },
            "software": {
                "probeep_commit": git_revision_optional(
                    os.getenv("PROBEEP_ROOT", os.getcwd())
                ),
                "deepep_commit": git_revision_optional(os.getenv("DEEPEP_ROOT")),
                "deepep_moonep_commit": git_revision_optional(
                    os.getenv("DEEPEP_MOONEP_ROOT", "")
                ),
                "deepep_probeep_commit": git_revision_optional(
                    os.getenv("DEEPEP_PROBEEP_ROOT", "")
                ),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
            },
            "config": {
                "world_size": world_size,
                "tokens_per_rank": num_tokens,
                "topk": topk,
                "local_experts": local_experts,
                "replica_slots": replica_slots,
                "token_padding": token_padding,
                "execution_rows": execution_rows,
                "capacity_rows": capacity_rows,
                "hidden": int(os.getenv("HIDDEN", "7168")),
                "ffn_intermediate": int(os.getenv("FFN_INTERMEDIATE", "2048")),
                "dispatch_dtype": "fp8_e4m3fn_block128",
                "combine_dtype": "bf16",
                "weight_dtype": "bf16",
                "weight_cache_mode": probe_weight_cache_mode(),
                "physical_nics_per_server": int(
                    os.getenv("PROBEEP_PHYSICAL_NICS_PER_SERVER", "4")
                ),
                "rails_per_physical_nic": int(
                    os.getenv("PROBEEP_RAILS_PER_PHYSICAL_NIC", "2")
                ),
                "physical_nic_bandwidth_gbps": float(
                    os.getenv("PROBEEP_PHYSICAL_NIC_BANDWIDTH_GBPS", "400")
                ),
                "logical_rail_bandwidth_gbps": float(
                    os.getenv(
                        "PROBEEP_RDMA_PATH_BANDWIDTH_GBPS",
                        os.getenv("RDMA_PATH_BANDWIDTH_GBPS", "200"),
                    )
                ),
            },
            "timing_semantics": {
                "rank_reduction": "maximum",
                "clock": "CUDA events on compute and comm streams",
                "formal_runner": "dual_microbatch_ht",
                "wavefront": "A0 -> (A1 || W+D0) -> (E0 || W+D1) -> E1",
                "attention_observation": "A1.start -> W+D0.done",
                "moe_observation": "E0.start -> W+D1.done",
                "feedback_causality": (
                    "Layer L+1 same round consumes Layer L same compute_kind"
                ),
                "feedback_partition": (
                    "Attention: dispatch MB0/compute MB1; "
                    "MoE: dispatch MB1/compute MB0; no cross-kind fallback"
                ),
                "first_moe_layer": "explicit bootstrap; no fake observation",
                "sync_single": "correctness/smoke/diagnosis only",
                "balanced_dispatch_ms": (
                    "dual HT runner: controller/layout/weight/dispatch on "
                    "the DeepEP communication stream"
                ),
                "balanced_plan_ms": (
                    "0 in dual_microbatch_ht because controller/layout are "
                    "intentionally fused into dispatch_ms"
                ),
            },
            "cases": [],
        }
    case = {
        "variant": args.variant,
        "expert_mode": args.expert_mode,
        "runner_mode": args.runner_mode,
        "workload": routing_mode,
        "bias_ratio": routing_bias_ratio,
        "seed": routing_seed,
    }
    if case not in manifest["cases"]:
        manifest["cases"].append(case)
    if routing_mode.startswith("raw_data1_layer_"):
        workload_root = Path(os.environ["PROBEEP_ROOT"]) / "workload"
        manifest.update(
            raw_source_tree_sha256=runtime_tree_sha256(
                workload_root / "raw_data"
            ),
            raw_data1_tree_sha256=runtime_tree_sha256(
                workload_root / "raw_data1"
            ),
            raw_data_inference=(
                "max-receive primary; redundant physical-slot rows are "
                "redistributed proportional to retained primary load"
            ),
            scaling="largest_remainder",
            experts_per_rank=256 // world_size,
            routes_per_layer=world_size * num_tokens * topk,
        )
    write_manifest(path, manifest)


def initialize_runtime(
    args: argparse.Namespace,
    *,
    rank: int,
    num_experts: int,
    num_tokens: int,
    hidden: int,
    intermediate: int,
    topk: int,
    local_experts: int,
    replica_slots: int,
    token_padding: int,
    ranks_per_server: int,
) -> tuple[RuntimeBackend, GroupedWeights | None]:
    backend_root = root_for_variant(
        args.variant,
        os.getenv("DEEPEP_ROOT", ""),
        os.getenv("DEEPEP_MOONEP_ROOT", ""),
        os.getenv("DEEPEP_PROBEEP_ROOT", ""),
        os.getenv("ULTRAEP_HYBRIDEP_ROOT"),
    )
    backend = RuntimeBackend.load(
        args.variant,
        backend_root,
        dist.group.WORLD,
        num_experts=num_experts,
        num_sms=int(os.getenv("NUM_SMS", "24")),
        num_nvl_bytes=int(os.getenv("NVL_BUFFER_BYTES", "2000000000")),
        num_rdma_bytes=int(os.getenv("RDMA_BUFFER_BYTES", "1000000000")),
        nvl_chunk_size=int(os.getenv("NVL_CHUNK_SIZE", "8")),
        nvl_buffer_size=int(os.getenv("NVL_CHUNK_BUFFER_SIZE", "512")),
        rdma_chunk_size=int(os.getenv("RDMA_CHUNK_SIZE", "16")),
        rdma_buffer_size=int(os.getenv("RDMA_CHUNK_BUFFER_SIZE", "128")),
        hidden=hidden,
        max_num_tokens_per_rank=num_tokens,
        topk=topk,
        local_experts=local_experts,
        replica_slots=replica_slots,
        token_padding=token_padding,
        ranks_per_server=ranks_per_server,
        intermediate=intermediate,
    )
    validate_backend_expert_mode(backend, args.expert_mode)
    grouped_weights = None
    if args.expert_mode == "grouped":
        if getattr(backend, "probeep_hybrid", False):
            masters = make_grouped_weights(
                local_experts,
                hidden,
                intermediate,
                args.seed,
                rank * local_experts,
            )
            backend.register_expert_pools(
                (masters.gate, masters.up, masters.down), local_experts
            )
            grouped_weights = masters
        elif args.variant == "ultraep_hybridep":
            base = make_grouped_weights(
                local_experts,
                hidden,
                intermediate,
                args.seed,
                rank * local_experts,
            )
            physical_gate, physical_up, physical_down = (
                backend.configure_grouped_weights(
                    base.gate, base.up, base.down
                )
            )
            grouped_weights = GroupedWeights(
                gate=physical_gate,
                up=physical_up,
                down=physical_down,
                expert_offset=rank * local_experts,
            )
            del base
        elif backend.balanced:
            views = backend.buffer.get_balanced_expert_pool_views()
            base = make_grouped_weights(
                1,
                hidden,
                intermediate,
                args.seed,
                0,
            )
            for destination, source in zip(
                views[:3], (base.gate, base.up, base.down)
            ):
                destination[:local_experts].copy_(
                    source.expand(local_experts, -1, -1)
                )
                destination[local_experts:].zero_()
            bits = min(EXPERT_FINGERPRINT_BITS, hidden)
            local_ids = torch.arange(
                rank * local_experts,
                (rank + 1) * local_experts,
                dtype=torch.int64,
                device="cuda",
            )
            local_signs = expert_fingerprint_signs(local_ids, bits).to(
                views[2].dtype
            )
            views[2][:local_experts, :, :bits].mul_(
                local_signs.unsqueeze(1)
            )
            for grad in views[3:]:
                grad.zero_()
            backend.register_expert_pools(tuple(views), local_experts)
            grouped_weights = GroupedWeights(
                gate=views[0],
                up=views[1],
                down=views[2],
                expert_offset=rank * local_experts,
            )
            del base
        else:
            grouped_weights = make_grouped_weights(
                local_experts,
                hidden,
                intermediate,
                args.seed,
                rank * local_experts,
            )
    return backend, grouped_weights


def cleanup_persistent_runtime() -> None:
    global _PERSISTENT_BACKEND
    global _PERSISTENT_GROUPED_WEIGHTS
    global _PERSISTENT_VARIANT
    global _PERSISTENT_PROBE_FEEDBACK_BANK
    global _PERSISTENT_PROBE_FEEDBACK_LAYER
    global _PERSISTENT_PROBE_FEEDBACK_REPEAT
    if _PERSISTENT_BACKEND is not None:
        _PERSISTENT_BACKEND.destroy()
    _PERSISTENT_BACKEND = None
    _PERSISTENT_GROUPED_WEIGHTS = None
    _PERSISTENT_VARIANT = None
    _PERSISTENT_PROBE_FEEDBACK_BANK = {}
    _PERSISTENT_PROBE_FEEDBACK_LAYER = None
    _PERSISTENT_PROBE_FEEDBACK_REPEAT = None
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def main(
    argv: list[str] | None = None, *, persistent_runtime: bool = False
) -> None:
    global _PERSISTENT_BACKEND
    global _PERSISTENT_GROUPED_WEIGHTS
    global _PERSISTENT_VARIANT
    global _PERSISTENT_PROBE_FEEDBACK_BANK
    global _PERSISTENT_PROBE_FEEDBACK_LAYER
    global _PERSISTENT_PROBE_FEEDBACK_REPEAT
    args = parse_args(argv)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            "nccl", device_id=torch.device("cuda", local_rank)
        )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    ranks_per_server = int(os.getenv("GPUS_PER_NODE", "8"))
    _, _, _, configured_rail_gbps = probe_rail_topology(ranks_per_server)
    num_tokens = int(os.getenv("NUM_TOKENS_PER_RANK", "4096"))
    hidden = int(os.getenv("HIDDEN", "7168"))
    intermediate = int(os.getenv("FFN_INTERMEDIATE", "2048"))
    topk = int(os.getenv("TOPK", "8"))
    num_experts = int(os.getenv("NUM_EXPERTS", "256"))
    if num_experts % world_size:
        raise ValueError("NUM_EXPERTS must be divisible by distributed world size")
    local_experts = int(
        os.getenv("LOCAL_EXPERTS", str(num_experts // world_size))
    )
    default_replica_slots = 32 if args.variant == "probeep" else 16
    replica_slots = int(os.getenv("REPLICA_SLOTS", str(default_replica_slots)))
    token_padding = int(os.getenv("TOKEN_PADDING", "8"))
    system, balance = variant_identity(args.variant)
    scope = benchmark_scope(args.expert_mode)
    run_dir = Path(os.environ["PROBEEP_RUN_DIR"])
    execution_slots = local_experts + replica_slots
    # This is an allocation upper bound only.  The actual per-rank execution
    # rows below are derived from the selected algorithm's reference plan, so
    # server-imbalanced ProbeEP cases are not silently forced back to 50/50.
    balanced_execution_rows, balanced_capacity_rows = balanced_row_counts(
        num_tokens,
        topk,
        token_padding,
        execution_slots,
        world_size // ranks_per_server,
    )

    workload = make_routing_workload(
        args.workload,
        world_size=world_size,
        num_tokens=num_tokens,
        topk=topk,
        local_experts=local_experts,
        ranks_per_server=ranks_per_server,
        bias_ratio=args.bias_ratio,
        seed=args.seed,
    )
    current_layer_id = (
        raw_data1_layer_id(workload.mode)
        if workload.mode.startswith("raw_data1_layer_")
        else -1
    )
    if args.variant == "probeep":
        reference = plan_probeep(
            workload.topk_experts,
            config=ProbeConfig(
                ranks_per_server=ranks_per_server,
                local_experts=local_experts,
                replica_slots=replica_slots,
                token_padding=token_padding,
                initial_migration_budget_bytes=32 * 1024 * 1024,
            ),
        )
        expert_load = torch.bincount(
            workload.topk_experts.reshape(-1), minlength=num_experts
        )
        destination = torch.arange(world_size).view(1, -1)
        home = (torch.arange(num_experts) // local_experts).view(-1, 1)
        stats = {
            "rank_maxvio_before": max_violation(reference.rank_load_before),
            "rank_maxvio_after": max_violation(reference.rank_load_after),
            "moved_assignments": int(
                (reference.alloc * (destination != home)).sum().item()
            ),
            "replica_count": int((reference.replica_expert >= 0).sum().item()),
        }
    else:
        reference = plan_server_local(
            workload.topk_experts,
            ranks_per_server=ranks_per_server,
            local_experts=local_experts,
            replica_slots=replica_slots,
            token_padding=token_padding,
        )
        expert_load = reference.tokens_per_expert
        stats = plan_statistics(reference)
    reference_slot_count = (
        reference.slot_count
        if args.variant == "probeep"
        else reference.tokens_per_exec_slot
    )
    padded_slot_rows = (
        (reference_slot_count.to(torch.int64) + token_padding - 1)
        // token_padding
        * token_padding
    )
    padded_rows_by_rank = padded_slot_rows.sum(1)
    balanced_execution_rows = int(padded_rows_by_rank.max().item())
    local_balanced_execution_rows = int(padded_rows_by_rank[rank].item())
    if args.variant == "probeep":
        # Production placement is driven by the sampled device controller and
        # may legitimately differ from this offline capacity preview. Keep
        # the fixed runtime extent here; make_grouped_ffn_layout derives the
        # actual grouped-MM end from the completed device slot plan.
        local_balanced_execution_rows = (
            world_size // ranks_per_server * num_tokens * topk
            + (token_padding - 1) * execution_slots
        )
    local_balanced_home_execution_rows = int(
        padded_slot_rows[rank, :local_experts].sum().item()
    )
    local_topk = workload.topk_experts[rank].cuda().contiguous()
    local_weights = workload.topk_weights[rank].cuda().contiguous()
    _, x_fp8, x_scales = make_input(rank, num_tokens, hidden, args.seed)
    if args.runner_mode == "dual_microbatch_ht" and num_tokens < 2:
        raise ValueError("dual_microbatch_ht requires at least two tokens/rank")
    microbatch_ranges = (
        (0, num_tokens // 2),
        (num_tokens // 2, num_tokens),
    )
    microbatch_execution_rows: list[int] = []
    microbatch_home_execution_rows: list[int] = []
    microbatch_references: list[PlanningReference] = []
    for start, end in microbatch_ranges:
        mb_topk = workload.topk_experts[:, start:end].contiguous()
        if args.variant == "probeep":
            mb_reference = plan_probeep(
                mb_topk,
                config=ProbeConfig(
                    ranks_per_server=ranks_per_server,
                    local_experts=local_experts,
                    replica_slots=replica_slots,
                    token_padding=token_padding,
                    initial_migration_budget_bytes=32 * 1024 * 1024,
                ),
            )
        else:
            mb_reference = plan_server_local(
                mb_topk,
                ranks_per_server=ranks_per_server,
                local_experts=local_experts,
                replica_slots=replica_slots,
                token_padding=token_padding,
            )
        microbatch_references.append(mb_reference)
        mb_slot_count = (
            mb_reference.slot_count
            if args.variant == "probeep"
            else mb_reference.tokens_per_exec_slot
        )
        mb_padded_slot_rows = (
            (mb_slot_count.to(torch.int64) + token_padding - 1)
            // token_padding
            * token_padding
        )
        mb_padded_rows = mb_padded_slot_rows.sum(1)
        execution_rows = int(mb_padded_rows[rank].item())
        if args.variant == "probeep":
            # The sampled controller may select a different migration budget
            # after every observation.  Keep the tensor extent at the runtime
            # capacity while make_grouped_ffn_layout supplies a device-derived
            # final offset, so no host read or capacity-tail GEMM is needed.
            execution_rows = (
                world_size // ranks_per_server * (end - start) * topk
                + (token_padding - 1) * execution_slots
            )
        microbatch_execution_rows.append(execution_rows)
        microbatch_home_execution_rows.append(
            int(mb_padded_slot_rows[rank, :local_experts].sum().item())
        )
    microbatches = tuple(
        MicrobatchInput(
            x_fp8=x_fp8[start:end].contiguous(),
            x_scales=x_scales[start:end].contiguous(),
            topk_idx=local_topk[start:end].contiguous(),
            topk_weights=local_weights[start:end].contiguous(),
            balanced_execution_rows=microbatch_execution_rows[index],
            balanced_home_execution_rows=(
                microbatch_home_execution_rows[index]
            ),
        )
        for index, (start, end) in enumerate(microbatch_ranges)
    )
    enable_attention_feedback = (
        args.variant == "probeep"
        and os.getenv("PROBEEP_ENABLE_ATTENTION_FEEDBACK", "1") != "0"
    )
    enable_attention_overlap = (
        args.runner_mode == "dual_microbatch_ht"
        and os.getenv("BENCHMARK_ENABLE_ATTENTION_OVERLAP", "1") != "0"
    )
    attention_probe = (
        make_attention_probe_state(
            microbatch_ranges[0][1] - microbatch_ranges[0][0],
            hidden,
            args.seed,
        )
        if enable_attention_feedback or enable_attention_overlap
        else None
    )

    status_record: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": os.environ["PROBEEP_RUN_ID"],
        "kind": "benchmark_status",
        "benchmark_scope": scope,
        "variant": args.variant,
        "runner_mode": args.runner_mode,
        "expert_mode": args.expert_mode,
        "workload": workload.mode,
        "bias_ratio": workload.bias_ratio,
        "seed": workload.seed,
        "routing_sha256": workload.sha256,
        "weight_cache_mode": (
            probe_weight_cache_mode()
            if args.variant == "probeep"
            else "not_applicable"
        ),
    }

    if rank == 0:
        expert_rows = []
        for expert, receive_rows in enumerate(expert_load.tolist()):
            expert_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": os.environ["PROBEEP_RUN_ID"],
                    "slurm_job_id": os.getenv("SLURM_JOB_ID", ""),
                    "benchmark_scope": scope,
                    "runner_mode": args.runner_mode,
                    "system": system,
                    "balance": balance,
                    "direction": os.getenv("DIRECTION", "forward"),
                    "workload": workload.mode,
                    "bias_ratio": workload.bias_ratio,
                    "seed": workload.seed,
                    "repeat": args.repeat,
                    "iteration": -1,
                    "routing_sha256": workload.sha256,
                    "expert_id": expert,
                    "home_rank": expert // local_experts,
                    "home_server": expert // (ranks_per_server * local_experts),
                    "receive_rows": int(receive_rows),
                }
            )
        append_csv_rows(
            run_dir / "expert_samples.csv",
            EXPERT_SAMPLE_FIELDS,
            expert_rows,
        )

    backend = None
    try:
        if persistent_runtime and _PERSISTENT_BACKEND is not None:
            if _PERSISTENT_VARIANT != args.variant:
                raise RuntimeError(
                    "one persistent worker cannot mix benchmark variants"
                )
            backend = _PERSISTENT_BACKEND
            grouped_weights = _PERSISTENT_GROUPED_WEIGHTS
        else:
            backend, grouped_weights = initialize_runtime(
                args,
                rank=rank,
                num_experts=num_experts,
                num_tokens=num_tokens,
                hidden=hidden,
                intermediate=intermediate,
                topk=topk,
                local_experts=local_experts,
                replica_slots=replica_slots,
                token_padding=token_padding,
                ranks_per_server=ranks_per_server,
            )
            if persistent_runtime:
                _PERSISTENT_BACKEND = backend
                _PERSISTENT_GROUPED_WEIGHTS = grouped_weights
                _PERSISTENT_VARIANT = args.variant

        rdma_path_bandwidth_gbps = configured_rail_gbps
        controller_alpha = float(os.getenv("PROBEEP_ALPHA", "0.90"))
        weight_cache_mode = probe_weight_cache_mode()
        # PROBEEP_WEIGHT_VERSION is the active per-invocation value consumed by
        # the CUDA backend.  It is deliberately mutable.  Never reuse it as the
        # next layer's base in a persistent worker.
        base_weight_version = int(
            os.getenv("PROBEEP_WEIGHT_BASE_VERSION", "1")
        )
        # Every canonical RawData1 sequence starts at Layer 0 with an explicit
        # bootstrap.  The Buffer persists across benchmark repeats, so without
        # this sequence-boundary reset repeat N would silently inherit the A/M
        # budgets and learned cap from repeat N-1's final layer.
        if (
            args.variant == "probeep"
            and persistent_runtime
            and current_layer_id == 0
        ):
            backend.reset_probe_controller(32 * 1024 * 1024)
        previous_layer_feedback: dict[
            tuple[str, int, int], ProbeFeedbackDevice
        ] = {}
        if (
            args.variant == "probeep"
            and persistent_runtime
            and current_layer_id >= 0
            and _PERSISTENT_PROBE_FEEDBACK_LAYER == current_layer_id - 1
            and _PERSISTENT_PROBE_FEEDBACK_REPEAT == args.repeat
        ):
            previous_layer_feedback = dict(
                _PERSISTENT_PROBE_FEEDBACK_BANK
            )
        if (
            args.variant == "probeep"
            and persistent_runtime
            and current_layer_id > 0
        ):
            expected_feedback_keys = {
                *(
                    ("warmup", index, compute_kind)
                    for index in range(args.warmup_iters)
                    for compute_kind in (0, 1)
                ),
                *(
                    ("measured", index, compute_kind)
                    for index in range(args.measure_iters)
                    for compute_kind in (0, 1)
                ),
            }
            missing_feedback = expected_feedback_keys.difference(
                previous_layer_feedback
            )
            if missing_feedback:
                raise RuntimeError(
                    f"Layer {current_layer_id} requires matching feedback from "
                    f"Layer {current_layer_id - 1}; missing {sorted(missing_feedback)}"
                )
        next_layer_feedback: dict[
            tuple[str, int, int], ProbeFeedbackDevice
        ] = {}

        for warmup_iteration in range(args.warmup_iters):
            consumed_warmup_feedback = load_probe_feedback(
                previous_layer_feedback,
                phase="warmup",
                iteration=warmup_iteration,
                producer_layer_id=current_layer_id - 1,
                producer_repeat=args.repeat,
            )
            if args.variant == "probeep":
                set_probe_weight_version(
                    base_weight_version,
                    layer_id=current_layer_id,
                    phase="warmup",
                    iteration=warmup_iteration,
                )
            if args.runner_mode == "dual_microbatch_ht":
                warmup_sample = run_forward_dual_microbatch_ht(
                    backend,
                    microbatches,  # type: ignore[arg-type]
                    rank=rank,
                    local_experts=local_experts,
                    expert_mode=args.expert_mode,
                    grouped_weights=grouped_weights,
                    timed=args.variant == "probeep",
                    profile=False,
                    probe_feedback=consumed_warmup_feedback,
                    rdma_path_bandwidth_gbps=rdma_path_bandwidth_gbps,
                    controller_alpha=controller_alpha,
                    attention_overlap=attention_probe
                    if enable_attention_overlap else None,
                )
            else:
                warmup_sample = run_forward(
                    backend,
                    x_fp8,
                    x_scales,
                    local_topk,
                    local_weights,
                    rank=rank,
                    local_experts=local_experts,
                    expert_mode=args.expert_mode,
                    grouped_weights=grouped_weights,
                    balanced_execution_rows=local_balanced_execution_rows,
                    balanced_home_execution_rows=(
                        local_balanced_home_execution_rows
                    ),
                    timed=args.variant == "probeep",
                    profile=False,
                    probe_feedback=consumed_warmup_feedback,
                    rdma_path_bandwidth_gbps=rdma_path_bandwidth_gbps,
                    controller_alpha=controller_alpha,
                )
            if args.variant == "probeep":
                store_probe_feedback(
                    next_layer_feedback,
                    build_probe_feedback(
                        warmup_sample,
                        iteration=warmup_iteration,
                        producer_phase="warmup",
                        runner_mode=args.runner_mode,
                        world_size=world_size,
                        rank=rank,
                        ranks_per_server=ranks_per_server,
                        hidden=hidden,
                        topk=topk,
                        enable_attention_feedback=enable_attention_feedback,
                        attention_probe=attention_probe,
                        producer_layer_id=current_layer_id,
                        producer_repeat=args.repeat,
                    ),
                )
        torch.cuda.synchronize()

        # Check one dedicated forward after warmup, before creating timing
        # events or profiler ranges. Reference work is local and chunked; the
        # gather makes every rank observe every verdict before measurement.
        correctness_feedback = (
            load_probe_feedback(
                previous_layer_feedback,
                phase="warmup",
                iteration=0,
                producer_layer_id=current_layer_id - 1,
                producer_repeat=args.repeat,
            )
            if args.variant == "probeep" and current_layer_id > 0
            else None
        )
        try:
            if args.variant == "probeep":
                set_probe_weight_version(
                    base_weight_version,
                    layer_id=current_layer_id,
                    phase="correctness",
                    iteration=0,
                )
            if args.runner_mode == "dual_microbatch_ht":
                correctness_sample = run_forward_dual_microbatch_ht(
                    backend,
                    microbatches,  # type: ignore[arg-type]
                    rank=rank,
                    local_experts=local_experts,
                    expert_mode=args.expert_mode,
                    grouped_weights=grouped_weights,
                    timed=False,
                    profile=False,
                    probe_feedback=correctness_feedback,
                    rdma_path_bandwidth_gbps=rdma_path_bandwidth_gbps,
                    controller_alpha=controller_alpha,
                    attention_overlap=(
                        attention_probe if enable_attention_overlap else None
                    ),
                )
            else:
                correctness_sample = run_forward(
                    backend,
                    x_fp8,
                    x_scales,
                    local_topk,
                    local_weights,
                    rank=rank,
                    local_experts=local_experts,
                    expert_mode=args.expert_mode,
                    grouped_weights=grouped_weights,
                    balanced_execution_rows=local_balanced_execution_rows,
                    balanced_home_execution_rows=(
                        local_balanced_home_execution_rows
                    ),
                    timed=False,
                    profile=False,
                    probe_feedback=correctness_feedback,
                    rdma_path_bandwidth_gbps=rdma_path_bandwidth_gbps,
                    controller_alpha=controller_alpha,
                )
            local_correctness = {
                "rank": rank,
                "error": "",
                **check_forward_correctness(
                    correctness_sample.output,
                    x_fp8,
                    x_scales,
                    local_topk,
                    local_weights,
                    expert_mode=args.expert_mode,
                    grouped_weights=grouped_weights,
                ).as_dict(),
            }
        except Exception:
            local_correctness = {
                "rank": rank,
                "passed": False,
                "error": traceback.format_exc(),
            }
        correctness_by_rank: list[dict[str, object] | None] = [None] * world_size
        dist.all_gather_object(correctness_by_rank, local_correctness)
        status_record["forward_correctness"] = local_correctness
        failed_correctness = [
            item
            for item in correctness_by_rank
            if item is None or not bool(item.get("passed", False))
        ]
        if rank == 0:
            append_jsonl(
                run_dir / "correctness.jsonl",
                {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": os.environ["PROBEEP_RUN_ID"],
                    "variant": args.variant,
                    "expert_mode": args.expert_mode,
                    "runner_mode": args.runner_mode,
                    "workload": workload.mode,
                    "repeat": args.repeat,
                    "routing_sha256": workload.sha256,
                    "passed": not failed_correctness,
                    "rank_status": correctness_by_rank,
                },
            )
        if failed_correctness:
            raise AssertionError(
                "untimed forward correctness probe failed: "
                + json.dumps(failed_correctness, sort_keys=True)
            )

        iteration_rows = []
        rank_rows = []
        microbatch_rank_rows: list[dict[str, object]] = []
        microbatch_timeline_rows: list[dict[str, object]] = []
        rank_expert_rows: list[dict[str, object]] = []
        probe_plan_rows: list[dict[str, object]] = []
        runtime_rdma_path_rows: list[dict[str, object]] = []
        probe_observation_sample_rows: list[dict[str, object]] = []
        probe_weight_chunk_rows: list[dict[str, object]] = []
        global_tokens = world_size * num_tokens
        assignments = global_tokens * topk
        after_maxvio = (
            stats["rank_maxvio_after"]
            if backend.balanced
            else stats["rank_maxvio_before"]
        )
        node_load = reference.rank_load_after.view(-1, ranks_per_server).sum(1)
        node_maxvio = max_violation(node_load)
        home_rank = torch.arange(reference.alloc.shape[0]) // local_experts

        for iteration in range(args.measure_iters):
            probe_feedback = load_probe_feedback(
                previous_layer_feedback,
                phase="measured",
                iteration=iteration,
                producer_layer_id=current_layer_id - 1,
                producer_repeat=args.repeat,
            )
            current_weight_version = -1
            if args.variant == "probeep":
                current_weight_version = set_probe_weight_version(
                    base_weight_version,
                    layer_id=current_layer_id,
                    phase="measured",
                    iteration=iteration,
                )
            consumed_probe_feedback = probe_feedback
            current_dispatch_compute_kind = (
                1
                if args.runner_mode == "dual_microbatch_ht"
                else (
                    probe_feedback.dispatch_compute_kind
                    if probe_feedback is not None
                    else 1
                )
            )
            if args.profile and iteration == 0:
                torch.cuda.cudart().cudaProfilerStart()
            if args.profile:
                torch.cuda.nvtx.range_push(
                    f"{args.variant}/measurement_iteration"
                )
            if args.runner_mode == "dual_microbatch_ht":
                sample = run_forward_dual_microbatch_ht(
                    backend,
                    microbatches,  # type: ignore[arg-type]
                    rank=rank,
                    local_experts=local_experts,
                    expert_mode=args.expert_mode,
                    grouped_weights=grouped_weights,
                    timed=True,
                    profile=args.profile,
                    probe_feedback=probe_feedback,
                    rdma_path_bandwidth_gbps=rdma_path_bandwidth_gbps,
                    controller_alpha=controller_alpha,
                    attention_overlap=attention_probe
                    if enable_attention_overlap else None,
                )
            else:
                sample = run_forward(
                    backend,
                    x_fp8,
                    x_scales,
                    local_topk,
                    local_weights,
                    rank=rank,
                    local_experts=local_experts,
                    expert_mode=args.expert_mode,
                    grouped_weights=grouped_weights,
                    balanced_execution_rows=local_balanced_execution_rows,
                    balanced_home_execution_rows=(
                        local_balanced_home_execution_rows
                    ),
                    timed=True,
                    profile=args.profile,
                    probe_feedback=probe_feedback,
                    rdma_path_bandwidth_gbps=rdma_path_bandwidth_gbps,
                    controller_alpha=controller_alpha,
                )
            timeline_intervals = sample.timeline_intervals_ms
            if args.variant == "probeep":
                feedback_start = torch.cuda.Event(enable_timing=True)
                feedback_end = torch.cuda.Event(enable_timing=True)
                if args.profile:
                    torch.cuda.nvtx.range_push("probeep/feedback_prepare")
                feedback_start.record()
                produced_probe_feedback = build_probe_feedback(
                    sample,
                    iteration=iteration,
                    producer_phase="measured",
                    runner_mode=args.runner_mode,
                    world_size=world_size,
                    rank=rank,
                    ranks_per_server=ranks_per_server,
                    hidden=hidden,
                    topk=topk,
                    enable_attention_feedback=enable_attention_feedback,
                    attention_probe=attention_probe,
                    producer_layer_id=current_layer_id,
                    producer_repeat=args.repeat,
                )
                store_probe_feedback(
                    next_layer_feedback,
                    produced_probe_feedback,
                )
                feedback_end.record()
                feedback_end.synchronize()
                if args.profile:
                    torch.cuda.nvtx.range_pop()
                feedback_ms = feedback_start.elapsed_time(feedback_end)
                if (
                    args.runner_mode == "dual_microbatch_ht"
                    and sample.timeline_origin_event is not None
                ):
                    feedback_start_ms = sample.timeline_origin_event.elapsed_time(
                        feedback_start
                    )
                    feedback_end_ms = sample.timeline_origin_event.elapsed_time(
                        feedback_end
                    )
                    timeline_intervals = timeline_intervals + (
                        (
                            "observation_prepare",
                            -1,
                            "compute",
                            feedback_start_ms,
                            feedback_end_ms,
                        ),
                    )
                # Attribute the producer to the invocation whose completed
                # observation it samples.  This keeps steady-state E2E honest:
                # controller/plan are timed in the next fused dispatch, while
                # the device collectives needed to form feedback are paid here.
                sample.local_ms["plan_ms"] += feedback_ms
                sample.local_ms["e2e_ms"] = (
                    sample.timeline_origin_event.elapsed_time(feedback_end)
                    if sample.timeline_origin_event is not None
                    else sample.local_ms["e2e_ms"] + feedback_ms
                )

            if args.profile:
                torch.cuda.nvtx.range_pop()
            maxima, per_rank = rank_max_and_samples(
                sample.local_ms, world_size
            )
            valid_rows = torch.tensor(
                sample.valid_rows, device="cuda", dtype=torch.int64
            )
            all_valid_rows = [
                torch.empty_like(valid_rows) for _ in range(world_size)
            ]
            dist.all_gather(all_valid_rows, valid_rows)
            if (
                sample.local_rank_expert_raw is None
                or sample.local_rank_expert_padded is None
            ):
                raise RuntimeError(
                    "grouped benchmark did not expose actual execution rows"
                )
            local_execution = torch.stack(
                (
                    sample.local_rank_expert_raw.sum(),
                    sample.local_rank_expert_padded.sum(),
                )
            )
            all_execution = [
                torch.empty_like(local_execution) for _ in range(world_size)
            ]
            dist.all_gather(all_execution, local_execution)
            all_microbatch_execution: list[torch.Tensor] = []
            all_timelines: list[torch.Tensor] = []
            if args.runner_mode == "dual_microbatch_ht":
                if (
                    sample.local_microbatch_rank_expert_raw is None
                    or sample.local_microbatch_rank_expert_padded is None
                ):
                    raise RuntimeError(
                        "dual-microbatch benchmark did not expose per-microbatch rows"
                    )
                local_microbatch_execution = torch.stack(
                    (
                        sample.local_microbatch_rank_expert_raw.sum(dim=1),
                        sample.local_microbatch_rank_expert_padded.sum(dim=1),
                    ),
                    dim=1,
                )
                all_microbatch_execution = [
                    torch.empty_like(local_microbatch_execution)
                    for _ in range(world_size)
                ]
                dist.all_gather(
                    all_microbatch_execution, local_microbatch_execution
                )
                timeline_by_key = {
                    (stage, microbatch, logical_stream): (start_ms, end_ms)
                    for stage, microbatch, logical_stream, start_ms, end_ms
                    in timeline_intervals
                }
                local_timeline = torch.full(
                    (len(MICROBATCH_TIMELINE_STAGES), 2),
                    -1.0,
                    dtype=torch.float32,
                    device="cuda",
                )
                for stage_index, key in enumerate(MICROBATCH_TIMELINE_STAGES):
                    interval = timeline_by_key.get(key)
                    if interval is not None:
                        local_timeline[stage_index, 0] = interval[0]
                        local_timeline[stage_index, 1] = interval[1]
                all_timelines = [
                    torch.empty_like(local_timeline) for _ in range(world_size)
                ]
                dist.all_gather(all_timelines, local_timeline)

            case = {
                "schema_version": SCHEMA_VERSION,
                "run_id": os.environ["PROBEEP_RUN_ID"],
                "slurm_job_id": os.getenv("SLURM_JOB_ID", ""),
                "benchmark_scope": scope,
                "runner_mode": args.runner_mode,
                "system": system,
                "balance": balance,
                "direction": os.getenv("DIRECTION", "forward"),
                "workload": workload.mode,
                "bias_ratio": workload.bias_ratio,
                "seed": workload.seed,
                "repeat": args.repeat,
                "iteration": iteration,
                "routing_sha256": workload.sha256,
            }
            iteration_rail_rows = collect_dispatch_rail_telemetry(
                sample,
                case=case,
                rank=rank,
                world_size=world_size,
                ranks_per_server=ranks_per_server,
            )
            if rank == 0:
                runtime_rdma_path_rows.extend(iteration_rail_rows)
                if args.variant == "probeep":
                    if consumed_probe_feedback is None and current_layer_id > 0:
                        raise RuntimeError(
                            "measured ProbeEP layer has no previous-layer A/M observation"
                        )
                    if consumed_probe_feedback is not None:
                        probe_observation_sample_rows.extend(
                            probe_observation_rows(
                                consumed_probe_feedback,
                                case=case,
                                consumer_iteration=iteration,
                                consumer_layer_id=current_layer_id,
                                consumer_repeat=args.repeat,
                                ranks_per_server=ranks_per_server,
                                controller_alpha=controller_alpha,
                                rdma_path_bandwidth_gbps=rdma_path_bandwidth_gbps,
                            )
                        )
            if iteration == args.measure_iters - 1:
                if (
                    sample.local_rank_expert_raw is None
                    or sample.local_rank_expert_padded is None
                ):
                    raise RuntimeError(
                        "formal grouped benchmark did not expose rank-expert rows"
                    )
                local_rank_expert = torch.stack(
                    (
                        sample.local_rank_expert_raw,
                        sample.local_rank_expert_padded,
                    )
                )
                gathered_rank_expert = [
                    torch.empty_like(local_rank_expert) for _ in range(world_size)
                ]
                dist.all_gather(gathered_rank_expert, local_rank_expert)
                if rank == 0:
                    for global_rank, values in enumerate(gathered_rank_expert):
                        raw_values, padded_values = values.cpu().tolist()
                        for expert_id in range(num_experts):
                            rank_expert_rows.append(
                                {
                                    **case,
                                    "global_rank": global_rank,
                                    "node_rank": global_rank // ranks_per_server,
                                    "local_rank": global_rank % ranks_per_server,
                                    "expert_id": expert_id,
                                    "raw_rows": raw_values[expert_id],
                                    "padded_rows": padded_values[expert_id],
                                }
                            )
            if rank == 0:
                runtime_rank_load = torch.tensor(
                    [int(item[0].item()) for item in all_execution],
                    dtype=torch.int64,
                )
                runtime_padded_rows = torch.tensor(
                    [int(item[1].item()) for item in all_execution],
                    dtype=torch.int64,
                )
                runtime_replica_expert = reference.replica_expert
                runtime_slot_expert = None
                runtime_slot_count = reference_slot_count
                runtime_moved_rows = None
                runtime_replica_counts = None
                if args.variant == "probeep":
                    plan_handles = (
                        tuple(enumerate(sample.handles))
                        if args.runner_mode == "dual_microbatch_ht"
                        else ((current_dispatch_compute_kind, sample.handle),)
                    )
                    for plan_kind, plan_handle in plan_handles:
                        plan_row, weight_rows, chunk_rows = collect_probe_runtime_telemetry(
                            sample,
                            iteration=iteration,
                            dispatch_compute_kind=plan_kind,
                            run_id=os.environ["PROBEEP_RUN_ID"],
                            scope=scope,
                            runner_mode=args.runner_mode,
                            system=system,
                            balance=balance,
                            workload=workload.mode,
                            bias_ratio=workload.bias_ratio,
                            seed=workload.seed,
                            repeat=args.repeat,
                            routing_sha256=workload.sha256,
                            ranks_per_server=ranks_per_server,
                            dispatch_matrix_bytes=next(
                                update.dispatch_matrix_bytes
                                for update in produced_probe_feedback.updates
                                if update.compute_kind == plan_kind
                            ),
                            expert_weight_version=current_weight_version,
                            weight_cache_mode=weight_cache_mode,
                            handle=plan_handle,
                        )
                        consumed_update = (
                            next(
                                update
                                for update in consumed_probe_feedback.updates
                                if update.compute_kind == plan_kind
                            )
                            if consumed_probe_feedback is not None
                            else None
                        )
                        plan_row.update(
                            feedback_source=(
                                "previous_layer_observation"
                                if consumed_update is not None
                                else "bootstrap"
                            ),
                            feedback_producer_layer_id=(
                                consumed_probe_feedback.producer_layer_id
                                if consumed_probe_feedback is not None
                                else -1
                            ),
                            feedback_producer_iteration=(
                                consumed_probe_feedback.producer_iteration
                                if consumed_probe_feedback is not None
                                else -1
                            ),
                            feedback_dispatch_microbatch=(
                                consumed_update.dispatch_microbatch
                                if consumed_update is not None
                                else plan_kind
                            ),
                            feedback_overlap_microbatch=(
                                consumed_update.overlap_microbatch
                                if consumed_update is not None
                                else 1 - plan_kind
                            ),
                        )
                        probe_plan_rows.append(plan_row)
                        merge_probe_weight_telemetry(
                            runtime_rdma_path_rows, weight_rows
                        )
                        probe_weight_chunk_rows.extend(chunk_rows)
                    per_plan_slot_count = [
                        item.slot_count.cpu() for _, item in plan_handles
                    ]
                    runtime_moved_rows = torch.zeros_like(runtime_rank_load)
                    runtime_replica_counts = torch.zeros_like(runtime_rank_load)
                    for _, item in plan_handles:
                        counts = item.slot_count.cpu()
                        experts = item.slot_expert.cpu()
                        remote = (experts >= 0) & (
                            experts // local_experts
                            != torch.arange(world_size).view(-1, 1)
                        )
                        runtime_moved_rows += counts.where(remote, 0).sum(1)
                        runtime_replica_counts += (
                            item.replica_expert.cpu() >= 0
                        ).sum(1)
                runtime_rank_maxvio = max_violation(runtime_rank_load)
                runtime_node_load = runtime_rank_load.view(
                    -1, ranks_per_server
                ).sum(1)
                runtime_node_maxvio = max_violation(runtime_node_load)
                seconds = maxima["e2e_ms"] / 1000.0
                iteration_rows.append(
                    {
                        **case,
                        "global_tokens": global_tokens,
                        "global_assignments": assignments,
                        "expert_maxvio": max_violation(expert_load),
                        "rank_maxvio_before": stats["rank_maxvio_before"],
                        "rank_maxvio_after": runtime_rank_maxvio,
                        "node_maxvio": runtime_node_maxvio,
                        "plan_max_ms": maxima["plan_ms"],
                        "count_exchange_max_ms": 0.0,
                        "layout_materialize_max_ms": maxima[
                            "layout_materialize_ms"
                        ],
                        "weight_prefetch_max_ms": maxima[
                            "weight_prefetch_ms"
                        ],
                        "dispatch_max_ms": maxima["dispatch_ms"],
                        "expert_compute_max_ms": maxima["expert_compute_ms"],
                        "combine_max_ms": maxima["combine_ms"],
                        "grad_reduce_max_ms": 0.0,
                        "e2e_max_ms": maxima["e2e_ms"],
                        "tokens_per_second": global_tokens / seconds,
                        "assignments_per_second": assignments / seconds,
                    }
                )
                for global_rank, timings in enumerate(per_rank):
                    balanced_rank = backend.balanced
                    valid = int(all_valid_rows[global_rank].item())
                    if balanced_rank:
                        processed_rows = int(
                            runtime_padded_rows[global_rank].item()
                        )
                        if runtime_moved_rows is not None:
                            moved = int(runtime_moved_rows[global_rank].item())
                        elif runtime_slot_expert is None:
                            remote_mask = home_rank != global_rank
                            moved = int(
                                reference.alloc[remote_mask, global_rank]
                                .sum()
                                .item()
                            )
                        else:
                            experts = runtime_slot_expert[global_rank]
                            counts = runtime_slot_count[global_rank]
                            remote = (experts >= 0) & (
                                experts // local_experts != global_rank
                            )
                            moved = int(counts[remote].sum().item())
                        replica_count = (
                            int(runtime_replica_counts[global_rank].item())
                            if runtime_replica_counts is not None
                            else int(
                                (runtime_replica_expert[global_rank] >= 0)
                                .sum()
                                .item()
                            )
                        )
                        exec_load = int(runtime_rank_load[global_rank].item())
                    else:
                        processed_rows = int(
                            runtime_padded_rows[global_rank].item()
                        )
                        moved = 0
                        replica_count = 0
                        exec_load = int(runtime_rank_load[global_rank].item())
                    rank_rows.append(
                        {
                            **case,
                            "global_rank": global_rank,
                            "node_rank": global_rank // ranks_per_server,
                            "local_rank": global_rank % ranks_per_server,
                            "home_load": int(
                                reference.rank_load_before[global_rank].item()
                            ),
                            "exec_load": exec_load,
                            "replica_count": replica_count,
                            "moved_assignments": moved,
                            "prefetch_bytes": (
                                replica_count * 3 * hidden * intermediate * 2
                                if balanced_rank and args.expert_mode == "grouped"
                                else 0
                            ),
                            "valid_recv_rows": valid,
                            "padding_rows": processed_rows - exec_load,
                            "plan_ms": timings["plan_ms"],
                            "count_exchange_ms": 0.0,
                            "layout_materialize_ms": timings[
                                "layout_materialize_ms"
                            ],
                            "weight_prefetch_ms": timings[
                                "weight_prefetch_ms"
                            ],
                            "dispatch_ms": timings["dispatch_ms"],
                            "expert_compute_ms": timings["expert_compute_ms"],
                            "combine_ms": timings["combine_ms"],
                            "grad_reduce_ms": 0.0,
                            "e2e_ms": timings["e2e_ms"],
                        }
                    )
                    if args.runner_mode == "dual_microbatch_ht":
                        for microbatch in range(2):
                            mb_values = all_microbatch_execution[global_rank]
                            mb_exec_load = int(
                                mb_values[microbatch, 0].item()
                            )
                            mb_padded_rows = int(
                                mb_values[microbatch, 1].item()
                            )
                            microbatch_rank_rows.append(
                                {
                                    **case,
                                    "microbatch": microbatch,
                                    "global_rank": global_rank,
                                    "node_rank": global_rank // ranks_per_server,
                                    "local_rank": global_rank % ranks_per_server,
                                    "home_load": int(
                                        microbatch_references[microbatch]
                                        .rank_load_before[global_rank]
                                        .item()
                                    ),
                                    "exec_load": mb_exec_load,
                                    "padded_rows": mb_padded_rows,
                                }
                            )
                        for stage_index, (
                            stage,
                            microbatch,
                            logical_stream,
                        ) in enumerate(MICROBATCH_TIMELINE_STAGES):
                            start_ms = float(
                                all_timelines[global_rank][stage_index, 0].item()
                            )
                            end_ms = float(
                                all_timelines[global_rank][stage_index, 1].item()
                            )
                            if start_ms < 0.0 or end_ms < start_ms:
                                continue
                            microbatch_timeline_rows.append(
                                {
                                    **case,
                                    "global_rank": global_rank,
                                    "node_rank": global_rank // ranks_per_server,
                                    "local_rank": global_rank % ranks_per_server,
                                    "microbatch": microbatch,
                                    "logical_stream": logical_stream,
                                    "stage": stage,
                                    "start_ms": f"{start_ms:.9f}",
                                    "end_ms": f"{end_ms:.9f}",
                                    "duration_ms": f"{end_ms - start_ms:.9f}",
                                }
                            )
            if args.profile and iteration == min(args.measure_iters, 10) - 1:
                torch.cuda.cudart().cudaProfilerStop()

        if rank == 0:
            append_csv_rows(run_dir / "iterations.csv", ITERATION_FIELDS, iteration_rows)
            append_csv_rows(run_dir / "rank_samples.csv", RANK_SAMPLE_FIELDS, rank_rows)
            if args.runner_mode == "dual_microbatch_ht":
                append_csv_rows(
                    run_dir / "microbatch_rank_samples.csv",
                    MICROBATCH_RANK_SAMPLE_FIELDS,
                    microbatch_rank_rows,
                )
                append_csv_rows(
                    run_dir / "microbatch_timeline.csv",
                    MICROBATCH_TIMELINE_FIELDS,
                    microbatch_timeline_rows,
                )
            append_csv_rows(
                run_dir / "rank_expert_samples.csv",
                RANK_EXPERT_SAMPLE_FIELDS,
                rank_expert_rows,
            )
            append_csv_rows(
                run_dir / "rdma_path_load.csv",
                RDMA_PATH_LOAD_FIELDS,
                runtime_rdma_path_rows,
            )
            if args.variant == "probeep":
                for row in probe_plan_rows:
                    append_jsonl(run_dir / "probeep_plan_summary.jsonl", row)
                append_csv_rows(
                    run_dir / "probeep_observation_samples.csv",
                    PROBEEP_OBSERVATION_SAMPLE_FIELDS,
                    probe_observation_sample_rows,
                )
                append_csv_rows(
                    run_dir / "probeep_weight_chunks.csv",
                    PROBEEP_WEIGHT_CHUNK_FIELDS,
                    probe_weight_chunk_rows,
                )
        if args.variant == "probeep" and persistent_runtime:
            expected_next_count = 2 * (
                args.warmup_iters + args.measure_iters
            )
            if len(next_layer_feedback) != expected_next_count:
                raise RuntimeError(
                    "ProbeEP did not produce a complete per-phase/per-round "
                    "feedback bank for the next layer"
                )
            _PERSISTENT_PROBE_FEEDBACK_BANK = next_layer_feedback
            _PERSISTENT_PROBE_FEEDBACK_LAYER = current_layer_id
            _PERSISTENT_PROBE_FEEDBACK_REPEAT = args.repeat
        status_record.update(status="PASS", error="")
    except BackendUnavailable as error:
        status_record.update(status="BACKEND_UNAVAILABLE", error=str(error))
    except Exception:
        status_record.update(status="FAIL", error=traceback.format_exc())

    statuses: list[dict[str, object] | None] = [None] * world_size
    dist.all_gather_object(statuses, status_record)
    failed = any(item is None or item["status"] != "PASS" for item in statuses)
    if rank == 0:
        append_jsonl(
            run_dir / "benchmark_status.jsonl",
            {
                **status_record,
                "rank_status": [
                    {
                        "rank": index,
                        "status": item["status"],
                        "error": item["error"],
                    }
                    for index, item in enumerate(statuses)
                    if item is not None
                ],
            },
        )
        update_manifest(
            run_dir / "manifest.json",
            args=args,
            routing_mode=workload.mode,
            routing_bias_ratio=workload.bias_ratio,
            routing_seed=workload.seed,
            world_size=world_size,
            num_tokens=num_tokens,
            topk=topk,
            local_experts=local_experts,
            replica_slots=replica_slots,
            token_padding=token_padding,
            execution_rows=balanced_execution_rows,
            capacity_rows=balanced_capacity_rows,
        )
    if backend is not None and not persistent_runtime:
        backend.destroy()
    dist.barrier()
    if not persistent_runtime:
        dist.destroy_process_group()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
