"""Grouped gated-FFN stage shared by correctness and full-MoE runners.

The expert-compute function contains exactly three ``torch._grouped_mm`` calls
and the ``SiLU(gate) * up`` pointwise stage.  The full benchmark records layout
construction in its separate ``layout_materialize`` phase; reference loops are
never part of a benchmark iteration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as functional


GROUPED_MM_BF16_ROW_ALIGNMENT = 8


@dataclass(frozen=True)
class GroupedFFNLayout:
    """Physical row ranges for one rank's 16 execution slots."""

    offsets: torch.Tensor
    slot_begin: torch.Tensor
    slot_count: torch.Tensor
    num_rows: int

    @property
    def num_slots(self) -> int:
        return int(self.offsets.numel())


def pad_grouped_assignment_rows(
    source_rows: torch.Tensor,
    route_weights: torch.Tensor,
    slot_count: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad sorted assignment groups to BF16 grouped-MM row alignment.

    Padding rows repeat the final source row in their group and carry a zero
    route weight.  The expert computation may therefore write those rows,
    while the weighted combine remains identical to the unpadded assignment
    stream.  All groups are expanded together with tensor operations.
    """

    padded_count = (
        (slot_count + GROUPED_MM_BF16_ROW_ALIGNMENT - 1)
        // GROUPED_MM_BF16_ROW_ALIGNMENT
        * GROUPED_MM_BF16_ROW_ALIGNMENT
    )
    padded_cu_seqlens = torch.cat(
        (slot_count.new_zeros(1), padded_count.cumsum(0))
    )
    source_cu_seqlens = torch.cat(
        (slot_count.new_zeros(1), slot_count.cumsum(0))
    )

    slots = torch.repeat_interleave(
        torch.arange(
            slot_count.numel(), device=slot_count.device, dtype=torch.int64
        ),
        padded_count.to(torch.int64),
    )
    physical_row = torch.arange(
        slots.numel(), device=slot_count.device, dtype=torch.int64
    )
    row_in_slot = physical_row - padded_cu_seqlens[:-1][slots]
    real_row = row_in_slot < slot_count[slots]
    source_assignment = source_cu_seqlens[:-1][slots] + torch.minimum(
        row_in_slot, slot_count[slots] - 1
    )

    padded_source_rows = source_rows[source_assignment]
    padded_route_weights = torch.where(
        real_row,
        route_weights[source_assignment],
        route_weights.new_zeros(()),
    )
    return padded_source_rows, padded_route_weights, padded_cu_seqlens


def make_grouped_ffn_layout(
    num_rows: int,
    *,
    cu_seqlens: torch.Tensor | None = None,
    slot_begin: torch.Tensor | None = None,
    slot_count: torch.Tensor | None = None,
    runtime_end: bool = False,
) -> GroupedFFNLayout:
    """Materialize grouped-MM offsets before the expert-compute phase.

    ``cu_seqlens`` is the padded prefix array ``[num_slots + 1]``.  The
    alternative form consumes the planner's local ``slot_begin`` and valid
    ``slot_count`` arrays.  With ``runtime_end=True`` the final offset is
    derived entirely on device from that runtime plan.  The input tensor can
    retain fixed capacity while grouped MM visits only the actual padded slot
    prefix, which is required when ProbeEP feedback changes admission between
    iterations.
    """

    if cu_seqlens is not None:
        begins = cu_seqlens[:-1].to(dtype=torch.int32).contiguous()
        offsets = cu_seqlens[1:].to(dtype=torch.int32).clone()
        if slot_count is None:
            counts = (cu_seqlens[1:] - cu_seqlens[:-1]).to(
                dtype=torch.int32
            ).contiguous()
        else:
            counts = slot_count.to(dtype=torch.int32).contiguous()
    else:
        if slot_begin is None or slot_count is None:
            raise ValueError("provide cu_seqlens or both slot_begin and slot_count")
        begins = slot_begin.to(dtype=torch.int32).contiguous()
        counts = slot_count.to(dtype=torch.int32).contiguous()
        offsets = torch.empty_like(begins)
        offsets[:-1] = begins[1:]

    if runtime_end:
        padded_last = (
            (counts[-1] + GROUPED_MM_BF16_ROW_ALIGNMENT - 1)
            // GROUPED_MM_BF16_ROW_ALIGNMENT
            * GROUPED_MM_BF16_ROW_ALIGNMENT
        )
        offsets[-1] = begins[-1] + padded_last
    else:
        offsets[-1] = num_rows
    return GroupedFFNLayout(
        offsets=offsets,
        slot_begin=begins,
        slot_count=counts,
        num_rows=num_rows,
    )


def grouped_gated_ffn(
    exec_x: torch.Tensor,
    layout: GroupedFFNLayout,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    """Run a 16-slot bias-free gated FFN without a Python expert loop.

    Shapes are ``exec_x[NvS,H]``, gate/up ``[16,H,Hp]``, and down
    ``[16,Hp,H]``.  Inputs and weights are expected to be preallocated,
    contiguous, and on the same device as ``layout.offsets``.
    """

    grouped_mm = torch._grouped_mm
    gate = grouped_mm(exec_x, gate_weight, offs=layout.offsets)
    up = grouped_mm(exec_x, up_weight, offs=layout.offsets)
    functional.silu(gate, inplace=True)
    gate.mul_(up)
    del up
    return grouped_mm(gate, down_weight, offs=layout.offsets)


def grouped_gated_ffn_out(
    exec_x: torch.Tensor,
    layout: GroupedFFNLayout,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    output: torch.Tensor,
    grouped_mm_out: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None
    ],
) -> torch.Tensor:
    """Run the gated FFN with the down projection written into ``output``."""

    grouped_mm = torch._grouped_mm
    gate = grouped_mm(exec_x, gate_weight, offs=layout.offsets)
    up = grouped_mm(exec_x, up_weight, offs=layout.offsets)
    functional.silu(gate, inplace=True)
    gate.mul_(up)
    del up
    grouped_mm_out(gate, down_weight, layout.offsets, output)
    return output


def valid_row_mask(layout: GroupedFFNLayout) -> torch.Tensor:
    """Return the valid, non-padding rows described by the planner counts."""

    rows = torch.arange(
        layout.num_rows, device=layout.slot_begin.device, dtype=torch.int32
    ).unsqueeze(1)
    begins = layout.slot_begin.unsqueeze(0)
    ends = (layout.slot_begin + layout.slot_count).unsqueeze(0)
    return ((rows >= begins) & (rows < ends)).any(dim=1)


def grouped_gated_ffn_reference(
    exec_x: torch.Tensor,
    layout: GroupedFFNLayout,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    """Small-shape correctness oracle; never call this in a timed region."""

    output = torch.zeros_like(exec_x)
    begins = layout.slot_begin.cpu().tolist()
    counts = layout.slot_count.cpu().tolist()
    for slot, (begin, count) in enumerate(zip(begins, counts)):
        if count == 0:
            continue
        x = exec_x[begin : begin + count]
        gate = x @ gate_weight[slot]
        up = x @ up_weight[slot]
        output[begin : begin + count] = (
            functional.silu(gate) * up
        ) @ down_weight[slot]
    return output


def identity_exec_reference(
    exec_x: torch.Tensor, layout: GroupedFFNLayout
) -> torch.Tensor:
    """Identity expert oracle with padding rows forced to zero."""

    return exec_x * valid_row_mask(layout).unsqueeze(1)


def affine_exec_reference(
    exec_x: torch.Tensor,
    layout: GroupedFFNLayout,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Per-slot affine expert oracle used before real FFN correctness tests."""

    output = torch.zeros_like(exec_x)
    begins = layout.slot_begin.cpu().tolist()
    counts = layout.slot_count.cpu().tolist()
    for slot, (begin, count) in enumerate(zip(begins, counts)):
        if count:
            x = exec_x[begin : begin + count]
            output[begin : begin + count] = x @ weight[slot] + bias[slot]
    return output


def run_grouped_ffn_stage(
    exec_x: torch.Tensor,
    layout: GroupedFFNLayout,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    """Timed adapter called by the full-MoE benchmark runtime."""

    return grouped_gated_ffn(
        exec_x, layout, gate_weight, up_weight, down_weight
    )


def run_grouped_ffn_stage_out(
    exec_x: torch.Tensor,
    layout: GroupedFFNLayout,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    output: torch.Tensor,
    grouped_mm_out: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None
    ],
) -> torch.Tensor:
    """Timed adapter for the balanced runtime's preallocated execution output."""

    return grouped_gated_ffn_out(
        exec_x,
        layout,
        gate_weight,
        up_weight,
        down_weight,
        output,
        grouped_mm_out,
    )
