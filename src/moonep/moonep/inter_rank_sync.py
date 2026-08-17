"""
MoonEP inter-rank sync - a minimal CuTe DSL cross-rank barrier kernel.

This is used to align EP ranks immediately before launching the planning
kernel, so planning/communication timings are less affected by CPU-side or
upstream stream skew.
"""

import functools

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass.cute.runtime import make_ptr

from moonep._common import cross_rank_barrier


class InterRankSyncKernel:
    """Single-CTA inter-rank barrier over the shared meta_buf barrier slots."""

    def __init__(self, R: int, meta_stride: int, num_threads: int):
        self.R = R
        self.meta_stride = meta_stride
        self.num_threads = num_threads

    @cute.jit
    def __call__(
        self,
        meta_ptr: cute.Pointer,      # int32 [R * meta_stride]
        bar_ptr: cute.Pointer,       # int32 [1] grid barrier counter
        rank: Int32,
        barrier_off: Int32,
        stream: cuda.CUstream,
    ):
        R = cutlass.const_expr(self.R)
        meta_stride = cutlass.const_expr(self.meta_stride)

        meta_tensor = cute.make_tensor(
            meta_ptr,
            cute.make_layout((R * meta_stride,)),
        )
        bar_tensor = cute.make_tensor(bar_ptr, cute.make_layout((1,)))

        self.kernel(
            meta_tensor,
            bar_tensor,
            rank,
            barrier_off,
        ).launch(
            grid=(1, 1, 1),
            block=(self.num_threads, 1, 1),
            smem=0,
            stream=stream,
            cooperative=True,
        )

    @cute.kernel
    def kernel(
        self,
        meta_tensor: cute.Tensor,
        bar_tensor: cute.Tensor,
        rank: Int32,
        barrier_off: Int32,
    ):
        meta_stride = cutlass.const_expr(self.meta_stride)
        num_ranks = cutlass.const_expr(self.R)
        tid = cute.arch.thread_idx()[0]

        cross_rank_barrier(
            meta_tensor,
            meta_stride,
            barrier_off,
            rank,
            num_ranks,
            bar_tensor.iterator,
            Int32(1),
            cutlass.const_expr(self.num_threads),
            tid,
        )


def _num_threads_for_ranks(R: int) -> int:
    assert R > 0, f"R must be positive, got {R}"
    assert R <= 1024, f"inter_rank_sync supports at most 1024 ranks, got {R}"
    return max(32, ((R + 31) // 32) * 32)


@functools.lru_cache(maxsize=None)
def _get_compiled(
    R: int,
    meta_stride: int,
    num_threads: int,
    device_index: int,
):
    _ = device_index
    kernel = InterRankSyncKernel(
        R=R,
        meta_stride=meta_stride,
        num_threads=num_threads,
    )

    i32_ptr = make_ptr(Int32, 0, cute.AddressSpace.gmem, assumed_align=16)
    stream_arg = cuda.CUstream(0)

    return cute.compile(
        kernel,
        i32_ptr,       # meta_buf
        i32_ptr,       # bar (grid barrier counter)
        Int32(0),      # rank
        Int32(0),      # barrier_off
        stream_arg,
    )


def launch_inter_rank_sync(ctx: dict) -> None:
    """Launch the pre-planning inter-rank sync kernel on the current stream."""
    assert ctx['meta_buf'].dtype == torch.int32 and ctx['meta_buf'].is_contiguous()

    R = int(ctx['R'])
    meta_stride = int(ctx['meta_chunk_padded'])
    num_threads = _num_threads_for_ranks(R)
    device_index = ctx['meta_buf'].device.index

    compiled = _get_compiled(
        R,
        meta_stride,
        num_threads,
        device_index,
    )

    meta_ptr = make_ptr(
        Int32,
        ctx['meta_buf'].data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    bar_ptr = make_ptr(
        Int32,
        ctx['grid_sync_bar'].data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compiled(
        meta_ptr,
        bar_ptr,
        Int32(int(ctx['rank'])),
        Int32(int(ctx['BARRIER_OFF'])),
        stream,
    )
