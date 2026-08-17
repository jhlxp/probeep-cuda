#include "probeep_histogram.cuh"

#include "../probeep_topology.hpp"

namespace deep_ep::probeep {
namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kHistogramMaxBlocks = 128;
constexpr int kHistogramSegmentRoutes = 256;
constexpr int kPackedOrdinalBits = 16;
constexpr int kPackedOrdinalMask = (1 << kPackedOrdinalBits) - 1;

__global__ void serial_histogram_and_ordinal(
        const std::int64_t* topk_idx,
        int* tokens_per_expert,
        int* local_ordinal,
        int routes_per_rank) {
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    __shared__ int running[kNumExperts];
    __shared__ int warp_counts[kWarps][kNumExperts];

    running[tid] = 0;
    __syncthreads();
    for (int chunk = 0; chunk < routes_per_rank; chunk += kThreads) {
        for (int index = tid; index < kWarps * kNumExperts;
             index += kThreads)
            reinterpret_cast<int*>(warp_counts)[index] = 0;
        __syncthreads();

        const int route = chunk + tid;
        const unsigned active = __ballot_sync(
                0xffffffffu, route < routes_per_rank);
        int expert = -1;
        int within_warp = 0;
        if (route < routes_per_rank) {
            expert = static_cast<int>(topk_idx[route]);
            const unsigned peers = __match_any_sync(active, expert);
            const unsigned lanes_before = lane == 0
                    ? 0u : ((1u << lane) - 1u);
            within_warp = __popc(peers & lanes_before);
            if (within_warp == 0)
                atomicAdd(&warp_counts[warp][expert], __popc(peers));
        }
        __syncthreads();
        if (route < routes_per_rank) {
            int ordinal = running[expert] + within_warp;
#pragma unroll
            for (int source_warp = 0; source_warp < kWarps; ++source_warp)
                ordinal += source_warp < warp
                        ? warp_counts[source_warp][expert] : 0;
            local_ordinal[route] = ordinal;
        }
        __syncthreads();
        int count = 0;
#pragma unroll
        for (int source_warp = 0; source_warp < kWarps; ++source_warp)
            count += warp_counts[source_warp][tid];
        running[tid] += count;
        __syncthreads();
    }
    tokens_per_expert[tid] = running[tid];
}

__global__ void segmented_histogram_and_ordinal(
        const std::int64_t* topk_idx,
        int* local_ordinal,
        int routes_per_rank,
        int histogram_blocks) {
    const int segment = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int segment_begin = segment * kHistogramSegmentRoutes;
    const int segment_end = segment + 1 == histogram_blocks
            ? routes_per_rank
            : segment_begin + kHistogramSegmentRoutes;
    __shared__ int running[kNumExperts];
    __shared__ int warp_counts[kWarps][kNumExperts];

    running[tid] = 0;
    __syncthreads();
    for (int chunk = segment_begin; chunk < segment_end; chunk += kThreads) {
        for (int index = tid; index < kWarps * kNumExperts;
             index += kThreads)
            reinterpret_cast<int*>(warp_counts)[index] = 0;
        __syncthreads();

        const int route = chunk + tid;
        const unsigned active = __ballot_sync(
                0xffffffffu, route < segment_end);
        int expert = -1;
        int within_warp = 0;
        if (route < segment_end) {
            expert = static_cast<int>(topk_idx[route]);
            const unsigned peers = __match_any_sync(active, expert);
            const unsigned lanes_before = lane == 0
                    ? 0u : ((1u << lane) - 1u);
            within_warp = __popc(peers & lanes_before);
            if (within_warp == 0)
                atomicAdd(&warp_counts[warp][expert], __popc(peers));
        }
        __syncthreads();
        if (route < segment_end) {
            int ordinal = running[expert] + within_warp;
#pragma unroll
            for (int source_warp = 0; source_warp < kWarps; ++source_warp)
                ordinal += source_warp < warp
                        ? warp_counts[source_warp][expert] : 0;
            local_ordinal[route] = ordinal;
        }
        __syncthreads();
        int count = 0;
#pragma unroll
        for (int source_warp = 0; source_warp < kWarps; ++source_warp)
            count += warp_counts[source_warp][tid];
        running[tid] += count;
        __syncthreads();
    }

    const int scratch_route = segment_begin + tid;
    local_ordinal[scratch_route] =
            (running[tid] << kPackedOrdinalBits) |
            (local_ordinal[scratch_route] & kPackedOrdinalMask);
}

__global__ void prefix_segment_histograms(
        int* tokens_per_expert,
        int* local_ordinal,
        int histogram_blocks) {
    const int expert = static_cast<int>(threadIdx.x);
    int prefix = 0;
    for (int segment = 0; segment < histogram_blocks; ++segment) {
        const int route = segment * kHistogramSegmentRoutes + expert;
        const int packed = local_ordinal[route];
        local_ordinal[route] =
                (prefix << kPackedOrdinalBits) |
                (packed & kPackedOrdinalMask);
        prefix += packed >> kPackedOrdinalBits;
    }
    tokens_per_expert[expert] = prefix;
}

}  // namespace

void launch_local_histogram_and_ordinal(
        const std::int64_t* local_topk_idx,
        int num_tokens,
        int* local_histogram,
        int* local_ordinal,
        cudaStream_t stream) {
    const int routes = num_tokens * kTopK;
    const int available_blocks = routes / kHistogramSegmentRoutes;
    const int histogram_blocks = available_blocks < kHistogramMaxBlocks
            ? available_blocks : kHistogramMaxBlocks;
    if (histogram_blocks <= 1) {
        serial_histogram_and_ordinal<<<1, kThreads, 0, stream>>>(
                local_topk_idx, local_histogram, local_ordinal, routes);
        return;
    }
    segmented_histogram_and_ordinal
            <<<histogram_blocks, kThreads, 0, stream>>>(
                    local_topk_idx, local_ordinal, routes, histogram_blocks);
    prefix_segment_histograms<<<1, kThreads, 0, stream>>>(
            local_histogram, local_ordinal, histogram_blocks);
}

}  // namespace deep_ep::probeep
