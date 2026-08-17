#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "moonep_plan.cuh"

namespace deep_ep::moonep {

namespace {

constexpr int kNumRanks = 16;
constexpr int kRanksPerServer = 8;
constexpr int kNumServers = 2;
constexpr int kNumExperts = 256;
constexpr int kLocalExperts = 16;
constexpr int kReplicaSlots = 16;
constexpr int kNumSlots = kLocalExperts + kReplicaSlots;
constexpr int kTopK = 8;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kHistogramMaxBlocks = 128;
constexpr int kHistogramSegmentRoutes = 256;
constexpr int kPackedOrdinalBits = 16;
constexpr int kPackedOrdinalMask = (1 << kPackedOrdinalBits) - 1;

struct alignas(256) IpcPlanReserveView {
    int source_counts[kNumServers][kNumExperts];
};

__global__ void serial_histogram_and_ordinal(const int64_t* topk_idx,
                                             int* tokens_per_expert,
                                             int* local_ordinal,
                                             int routes_per_rank,
                                             int input_source_stride) {
    const int input_src = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;

    __shared__ int running[kNumExperts];
    __shared__ int warp_counts[kWarps][kNumExperts];

    for (int e = tid; e < kNumExperts; e += kThreads)
        running[e] = 0;
    __syncthreads();

    const auto* src_topk = topk_idx + static_cast<int64_t>(input_src) * input_source_stride;
    auto* src_ordinal = local_ordinal + static_cast<int64_t>(input_src) * routes_per_rank;

    for (int chunk = 0; chunk < routes_per_rank; chunk += kThreads) {
        for (int i = tid; i < kWarps * kNumExperts; i += kThreads)
            reinterpret_cast<int*>(warp_counts)[i] = 0;
        __syncthreads();

        const int route_idx = chunk + tid;
        const unsigned active = __ballot_sync(0xffffffffu, route_idx < routes_per_rank);
        int expert = -1;
        unsigned peers = 0;
        int within_warp = 0;
        if (route_idx < routes_per_rank) {
            expert = static_cast<int>(src_topk[route_idx]);
            peers = __match_any_sync(active, expert);
            const unsigned lanes_before = lane == 0 ? 0u : ((1u << lane) - 1u);
            within_warp = __popc(peers & lanes_before);
            if (within_warp == 0)
                atomicAdd(&warp_counts[warp][expert], __popc(peers));
        }
        __syncthreads();

        if (route_idx < routes_per_rank) {
            int ordinal = running[expert] + within_warp;
            #pragma unroll
            for (int w = 0; w < kWarps; ++w)
                ordinal += w < warp ? warp_counts[w][expert] : 0;
            src_ordinal[route_idx] = ordinal;
        }
        __syncthreads();

        if (tid < kNumExperts) {
            int count = 0;
            #pragma unroll
            for (int w = 0; w < kWarps; ++w)
                count += warp_counts[w][tid];
            running[tid] += count;
        }
        __syncthreads();
    }

    for (int e = tid; e < kNumExperts; e += kThreads)
        tokens_per_expert[input_src * kNumExperts + e] = running[e];
}

// Each source is split into contiguous route intervals.  The first 128
// ordinal words of an interval temporarily carry both that interval's count
// and the original within-interval ordinal.  The interval owns those words,
// so this needs neither atomics nor a separate O(blocks * experts) workspace.
__global__ void segmented_histogram_and_ordinal(const int64_t* topk_idx,
                                                int* local_ordinal,
                                                int routes_per_rank,
                                                int input_source_stride,
                                                int histogram_blocks) {
    const int segment = static_cast<int>(blockIdx.x);
    const int input_src = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int segment_begin = segment * kHistogramSegmentRoutes;
    const int segment_end = segment + 1 == histogram_blocks ?
            routes_per_rank : segment_begin + kHistogramSegmentRoutes;

    __shared__ int running[kNumExperts];
    __shared__ int warp_counts[kWarps][kNumExperts];

    if (tid < kNumExperts)
        running[tid] = 0;
    __syncthreads();

    const auto* src_topk = topk_idx +
            static_cast<int64_t>(input_src) * input_source_stride;
    auto* src_ordinal = local_ordinal +
            static_cast<int64_t>(input_src) * routes_per_rank;

    for (int chunk = segment_begin; chunk < segment_end; chunk += kThreads) {
        for (int i = tid; i < kWarps * kNumExperts; i += kThreads)
            reinterpret_cast<int*>(warp_counts)[i] = 0;
        __syncthreads();

        const int route_idx = chunk + tid;
        const unsigned active = __ballot_sync(
                0xffffffffu, route_idx < segment_end);
        int expert = -1;
        unsigned peers = 0;
        int within_warp = 0;
        if (route_idx < segment_end) {
            expert = static_cast<int>(src_topk[route_idx]);
            peers = __match_any_sync(active, expert);
            const unsigned lanes_before = lane == 0 ? 0u : ((1u << lane) - 1u);
            within_warp = __popc(peers & lanes_before);
            if (within_warp == 0)
                atomicAdd(&warp_counts[warp][expert], __popc(peers));
        }
        __syncthreads();

        if (route_idx < segment_end) {
            int ordinal = running[expert] + within_warp;
            #pragma unroll
            for (int w = 0; w < kWarps; ++w)
                ordinal += w < warp ? warp_counts[w][expert] : 0;
            src_ordinal[route_idx] = ordinal;
        }
        __syncthreads();

        if (tid < kNumExperts) {
            int count = 0;
            #pragma unroll
            for (int w = 0; w < kWarps; ++w)
                count += warp_counts[w][tid];
            running[tid] += count;
        }
        __syncthreads();
    }

    if (tid < kNumExperts) {
        const int scratch_route = segment_begin + tid;
        src_ordinal[scratch_route] =
                (running[tid] << kPackedOrdinalBits) |
                (src_ordinal[scratch_route] & kPackedOrdinalMask);
    }
}

__global__ void prefix_segment_histograms(int* tokens_per_expert,
                                          int* local_ordinal,
                                          int routes_per_rank,
                                          int histogram_blocks) {
    const int input_src = static_cast<int>(blockIdx.x);
    const int expert = static_cast<int>(threadIdx.x);
    auto* src_ordinal = local_ordinal +
            static_cast<int64_t>(input_src) * routes_per_rank;

    int prefix = 0;
    for (int segment = 0; segment < histogram_blocks; ++segment) {
        const int segment_begin = segment * kHistogramSegmentRoutes;
        const int packed = src_ordinal[segment_begin + expert];
        src_ordinal[segment_begin + expert] =
                (prefix << kPackedOrdinalBits) |
                (packed & kPackedOrdinalMask);
        prefix += packed >> kPackedOrdinalBits;
    }
    tokens_per_expert[input_src * kNumExperts + expert] = prefix;
}

void launch_histogram_and_ordinal(const int64_t* topk_idx,
                                  int num_sources,
                                  int* tokens_per_expert,
                                  int* local_ordinal,
                                  int routes_per_rank,
                                  int input_source_stride,
                                  cudaStream_t stream) {
    const int available_blocks = routes_per_rank / kHistogramSegmentRoutes;
    const int histogram_blocks = available_blocks < kHistogramMaxBlocks ?
            available_blocks : kHistogramMaxBlocks;
    if (histogram_blocks <= 1) {
        serial_histogram_and_ordinal<<<num_sources, kThreads, 0, stream>>>(
                topk_idx, tokens_per_expert, local_ordinal,
                routes_per_rank, input_source_stride);
        return;
    }

    const dim3 segmented_grid(histogram_blocks, num_sources);
    segmented_histogram_and_ordinal<<<segmented_grid, kThreads, 0, stream>>>(
            topk_idx, local_ordinal, routes_per_rank,
            input_source_stride, histogram_blocks);
    prefix_segment_histograms<<<num_sources, kNumExperts, 0, stream>>>(
            tokens_per_expert, local_ordinal,
            routes_per_rank, histogram_blocks);
}

template <bool kCountsFromIpc>
__global__ void build_server_local_plan(const int* tokens_per_expert,
                                        void** count_buffer_ptrs,
                                        int64_t plan_reserve_offset,
                                        int* tokens_per_expert_prefix,
                                        int* alloc_prefix,
                                        int* expert_slot,
                                        int* slot_count,
                                        int* slot_begin,
                                        int* replica_expert,
                                        int* slot_expert,
                                        int* num_tokens_per_rank,
                                        int* num_tokens_per_rdma_rank,
                                        int* num_tokens_per_exec_expert,
                                        int num_sources,
                                        int source_rank_base,
                                        int token_padding) {
    const int tid = static_cast<int>(threadIdx.x);
    __shared__ int expert_total[kNumExperts];
    __shared__ int group_tokens[kNumRanks];
    __shared__ int balance[kNumRanks];
    __shared__ int migration[kNumRanks][kNumRanks];
    __shared__ int quotas[kRanksPerServer];
    __shared__ int remaining[kLocalExperts];

    if (tid < kNumExperts) {
        int prefix = 0;
        for (int src = 0; src < kNumRanks; ++src) {
            int count;
            if constexpr (kCountsFromIpc) {
                const int source_nvl_rank = src % kRanksPerServer;
                const int source_server = src / kRanksPerServer;
                const auto* reserve = reinterpret_cast<const IpcPlanReserveView*>(
                    static_cast<const uint8_t*>(
                        count_buffer_ptrs[source_nvl_rank]) +
                    plan_reserve_offset);
                count = reserve->source_counts[source_server][tid];
            } else {
                count = tokens_per_expert[src * kNumExperts + tid];
            }
            prefix += count;
            tokens_per_expert_prefix[src * kNumExperts + tid] = prefix;
        }
        expert_total[tid] = prefix;
    }

    for (int i = tid; i < kNumRanks * kNumRanks; i += blockDim.x)
        reinterpret_cast<int*>(migration)[i] = 0;
    for (int i = tid; i < kNumExperts * kNumRanks; i += blockDim.x)
        alloc_prefix[i] = 0;
    for (int i = tid; i < kNumRanks * kNumExperts; i += blockDim.x)
        expert_slot[i] = -1;
    for (int i = tid; i < kNumRanks * kNumSlots; i += blockDim.x)
        slot_expert[i] = -1;
    for (int i = tid; i < kNumRanks * kReplicaSlots; i += blockDim.x)
        replica_expert[i] = -1;
    for (int i = tid; i < num_sources * kNumRanks; i += blockDim.x)
        num_tokens_per_rank[i] = 0;
    for (int i = tid; i < num_sources * kNumServers; i += blockDim.x)
        num_tokens_per_rdma_rank[i] = 0;
    for (int i = tid; i < num_sources * kNumRanks * kNumSlots; i += blockDim.x)
        num_tokens_per_exec_expert[i] = 0;
    __syncthreads();

    if (tid < kNumRanks) {
        const int rank = tid;
        int load = 0;
        for (int le = 0; le < kLocalExperts; ++le) {
            const int expert = rank * kLocalExperts + le;
            load += expert_total[expert];
            alloc_prefix[expert * kNumRanks + rank] = expert_total[expert];
            expert_slot[rank * kNumExperts + expert] = le;
            slot_expert[rank * kNumSlots + le] = expert;
        }
        group_tokens[rank] = load;
    }
    __syncthreads();

    if (tid == 0) {
        for (int server = 0; server < kNumServers; ++server) {
            const int rank_begin = server * kRanksPerServer;
            int server_tokens = 0;
            for (int i = 0; i < kRanksPerServer; ++i)
                server_tokens += group_tokens[rank_begin + i];
            const int capacity = server_tokens / kRanksPerServer;
            const int remainder = server_tokens - capacity * kRanksPerServer;
            for (int i = 0; i < kRanksPerServer; ++i) {
                const int rank = rank_begin + i;
                balance[rank] = group_tokens[rank] -
                        capacity - (i < remainder ? 1 : 0);
            }

            while (true) {
                int surplus_rank = -1;
                int surplus = 0;
                int deficit_rank = -1;
                int deficit = 0;
                for (int i = 0; i < kRanksPerServer; ++i) {
                    const int rank = rank_begin + i;
                    if (balance[rank] > surplus ||
                        (balance[rank] == surplus && balance[rank] > 0 && rank < surplus_rank)) {
                        surplus = balance[rank];
                        surplus_rank = rank;
                    }
                    if (balance[rank] < deficit ||
                        (balance[rank] == deficit && balance[rank] < 0 && rank < deficit_rank)) {
                        deficit = balance[rank];
                        deficit_rank = rank;
                    }
                }
                if (surplus <= 0)
                    break;
                const int move = -deficit;
                migration[surplus_rank][deficit_rank] += move;
                balance[surplus_rank] -= move;
                balance[deficit_rank] = 0;
            }

            for (int oi = 0; oi < kRanksPerServer; ++oi) {
                const int owner = rank_begin + oi;
                for (int di = 0; di < kRanksPerServer; ++di)
                    quotas[di] = migration[owner][rank_begin + di];
                for (int le = 0; le < kLocalExperts; ++le)
                    remaining[le] = expert_total[owner * kLocalExperts + le];

                while (true) {
                    int target_local = -1;
                    int max_quota = 0;
                    for (int di = 0; di < kRanksPerServer; ++di) {
                        if (quotas[di] > max_quota ||
                            (quotas[di] == max_quota && quotas[di] > 0 && di < target_local)) {
                            max_quota = quotas[di];
                            target_local = di;
                        }
                    }
                    if (max_quota <= 0)
                        break;

                    int selected_local_expert = -1;
                    int max_remaining = 0;
                    for (int le = 0; le < kLocalExperts; ++le) {
                        if (remaining[le] > max_remaining ||
                            (remaining[le] == max_remaining && remaining[le] > 0 && le < selected_local_expert)) {
                            max_remaining = remaining[le];
                            selected_local_expert = le;
                        }
                    }

                    const int take = max_remaining < max_quota ? max_remaining : max_quota;
                    const int expert = owner * kLocalExperts + selected_local_expert;
                    const int target = rank_begin + target_local;
                    alloc_prefix[expert * kNumRanks + target] += take;
                    alloc_prefix[expert * kNumRanks + owner] -= take;
                    remaining[selected_local_expert] -= take;
                    quotas[target_local] -= take;
                }
            }
        }
    }
    __syncthreads();

    if (tid < kNumRanks) {
        const int dest = tid;
        const int local_begin = dest * kLocalExperts;
        const int local_end = local_begin + kLocalExperts;
        for (int b = 0; b < kReplicaSlots; ++b) {
            int best_expert = -1;
            int best_count = 0;
            for (int expert = 0; expert < kNumExperts; ++expert) {
                const int count = alloc_prefix[expert * kNumRanks + dest];
                const bool is_local = expert >= local_begin && expert < local_end;
                const bool already_selected = expert_slot[dest * kNumExperts + expert] >= 0;
                if (!is_local && !already_selected &&
                    (count > best_count || (count == best_count && count > 0 && expert > best_expert))) {
                    best_count = count;
                    best_expert = expert;
                }
            }
            if (best_count > 0) {
                const int slot = kLocalExperts + b;
                replica_expert[dest * kReplicaSlots + b] = best_expert;
                slot_expert[dest * kNumSlots + slot] = best_expert;
                expert_slot[dest * kNumExperts + best_expert] = slot;
            }
        }

        int begin = 0;
        for (int slot = 0; slot < kNumSlots; ++slot) {
            const int expert = slot_expert[dest * kNumSlots + slot];
            const int count = expert >= 0 ? alloc_prefix[expert * kNumRanks + dest] : 0;
            slot_begin[dest * kNumSlots + slot] = begin;
            slot_count[dest * kNumSlots + slot] = count;
            if (count > 0)
                begin += ((count + token_padding - 1) / token_padding) * token_padding;
        }
    }
    __syncthreads();

    if (tid < kNumExperts) {
        for (int input_src = 0; input_src < num_sources; ++input_src) {
            const int src = source_rank_base < 0 ? input_src : source_rank_base;
            const int source_begin = src == 0 ? 0 :
                tokens_per_expert_prefix[(src - 1) * kNumExperts + tid];
            const int source_end =
                tokens_per_expert_prefix[src * kNumExperts + tid];
            int destination_begin = 0;
            for (int dest = 0; dest < kNumRanks; ++dest) {
                const int destination_end = destination_begin +
                    alloc_prefix[tid * kNumRanks + dest];
                const int overlap_begin = source_begin > destination_begin ?
                    source_begin : destination_begin;
                const int overlap_end = source_end < destination_end ?
                    source_end : destination_end;
                const int count = overlap_end > overlap_begin ?
                    overlap_end - overlap_begin : 0;
                if (count > 0) {
                    const int slot = expert_slot[dest * kNumExperts + tid];
                    num_tokens_per_exec_expert[
                        input_src * (kNumRanks * kNumSlots) +
                        dest * kNumSlots + slot] = count;
                }
                destination_begin = destination_end;
            }
        }

        int prefix = 0;
        for (int dest = 0; dest < kNumRanks; ++dest) {
            prefix += alloc_prefix[tid * kNumRanks + dest];
            alloc_prefix[tid * kNumRanks + dest] = prefix;
        }
    }
}

template <bool kClearExecWeights, bool kSingleSource>
__global__ void materialize_routes_and_token_layout(
                                   const int64_t* topk_idx,
                                   const int* local_ordinal,
                                   const int* tokens_per_expert_prefix,
                                   const int* alloc_prefix,
                                   const int* expert_slot,
                                   const int* slot_begin,
                                   int* route_dst,
                                   int* exec_rank,
                                   int* exec_slot,
                                   bool* is_token_in_rank,
                                   int* num_tokens_per_rank,
                                   int* num_tokens_per_rdma_rank,
                                   float* exec_route_weight,
                                   int total_routes,
                                   int routes_per_rank,
                                   int num_tokens,
                                   int source_rank_base,
                                   int nvs) {
    __shared__ int block_rank_counts[kNumRanks];
    __shared__ int block_server_counts[kNumServers];
    if constexpr (kSingleSource) {
        if (threadIdx.x < kNumRanks)
            block_rank_counts[threadIdx.x] = 0;
        if (threadIdx.x < kNumServers)
            block_server_counts[threadIdx.x] = 0;
        __syncthreads();
    }

    if constexpr (kClearExecWeights) {
        for (int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
             idx < nvs;
             idx += static_cast<int>(blockDim.x * gridDim.x))
            exec_route_weight[idx] = 0.0f;
    }

    for (int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
         idx < total_routes;
         idx += static_cast<int>(blockDim.x * gridDim.x)) {
        const int token_idx = idx / kTopK;
        const int input_src = token_idx / num_tokens;
        const int src = source_rank_base < 0 ? input_src : source_rank_base;
        const int expert = static_cast<int>(topk_idx[idx]);
        const int previous_sources = src == 0 ? 0 : tokens_per_expert_prefix[(src - 1) * kNumExperts + expert];
        const int local_route = idx - input_src * routes_per_rank;
        int ordinal = local_ordinal[idx];
        const int available_blocks = routes_per_rank / kHistogramSegmentRoutes;
        const int histogram_blocks = available_blocks < kHistogramMaxBlocks ?
                available_blocks : kHistogramMaxBlocks;
        if (histogram_blocks > 1) {
            int segment = local_route / kHistogramSegmentRoutes;
            segment = segment < histogram_blocks ? segment : histogram_blocks - 1;
            const int segment_begin = segment * kHistogramSegmentRoutes;
            const int packed_prefix = local_ordinal[
                    input_src * routes_per_rank + segment_begin + expert];
            if (local_route - segment_begin < kNumExperts)
                ordinal &= kPackedOrdinalMask;
            ordinal += packed_prefix >> kPackedOrdinalBits;
        }
        const int global_ordinal = previous_sources + ordinal;

        int lo = 0;
        int hi = kNumRanks;
        #pragma unroll
        for (int step = 0; step < 5; ++step) {
            const int mid = (lo + hi) >> 1;
            if (alloc_prefix[expert * kNumRanks + mid] <= global_ordinal)
                lo = mid + 1;
            else
                hi = mid;
        }
        const int dest = lo;
        const int previous_dest = dest == 0 ? 0 : alloc_prefix[expert * kNumRanks + dest - 1];
        const int slot = expert_slot[dest * kNumExperts + expert];
        const int local_offset = slot_begin[dest * kNumSlots + slot] + global_ordinal - previous_dest;
        route_dst[idx] = dest * nvs + local_offset;
        exec_rank[idx] = dest;
        exec_slot[idx] = slot;

        const unsigned active = __activemask();
        unsigned rank_mask = 1u << dest;
        rank_mask |= __shfl_down_sync(active, rank_mask, 4, kTopK);
        rank_mask |= __shfl_down_sync(active, rank_mask, 2, kTopK);
        rank_mask |= __shfl_down_sync(active, rank_mask, 1, kTopK);
        rank_mask = __shfl_sync(active, rank_mask, 0, kTopK);

        const int group_lane = threadIdx.x & (kTopK - 1);
        const int rank_lo = group_lane;
        const int rank_hi = group_lane + kTopK;
        const bool selected_lo = (rank_mask & (1u << rank_lo)) != 0;
        const bool selected_hi = (rank_mask & (1u << rank_hi)) != 0;
        auto* token_layout = is_token_in_rank +
                static_cast<int64_t>(token_idx) * kNumRanks;
        token_layout[rank_lo] = selected_lo;
        token_layout[rank_hi] = selected_hi;
        if (selected_lo) {
            if constexpr (kSingleSource)
                atomicAdd(block_rank_counts + rank_lo, 1);
            else
                atomicAdd(num_tokens_per_rank +
                          input_src * kNumRanks + rank_lo, 1);
        }
        if (selected_hi) {
            if constexpr (kSingleSource)
                atomicAdd(block_rank_counts + rank_hi, 1);
            else
                atomicAdd(num_tokens_per_rank +
                          input_src * kNumRanks + rank_hi, 1);
        }

        if (group_lane < kNumServers) {
            const unsigned server_mask = 0xffu << (group_lane * kRanksPerServer);
            if (rank_mask & server_mask) {
                if constexpr (kSingleSource)
                    atomicAdd(block_server_counts + group_lane, 1);
                else
                    atomicAdd(num_tokens_per_rdma_rank +
                              input_src * kNumServers + group_lane, 1);
            }
        }
    }

    if constexpr (kSingleSource) {
        __syncthreads();
        if (threadIdx.x < kNumRanks and block_rank_counts[threadIdx.x] != 0)
            atomicAdd(num_tokens_per_rank + threadIdx.x,
                      block_rank_counts[threadIdx.x]);
        if (threadIdx.x < kNumServers and block_server_counts[threadIdx.x] != 0)
            atomicAdd(num_tokens_per_rdma_rank + threadIdx.x,
                      block_server_counts[threadIdx.x]);
    }
}

} // namespace

void launch_local_histogram_and_ordinal(const int64_t* local_topk_idx,
                                        int num_tokens,
                                        int* local_histogram,
                                        int* local_ordinal,
                                        cudaStream_t stream) {
    const int routes_per_rank = num_tokens * kTopK;
    launch_histogram_and_ordinal(
            local_topk_idx, 1, local_histogram, local_ordinal,
            routes_per_rank, routes_per_rank, stream);
}

void launch_server_local_plan_from_counts(const int64_t* local_topk_idx,
                                          const int* global_counts,
                                          int source_rank,
                                          int num_tokens,
                                          int token_padding,
                                          const ServerLocalPlanWorkspace& workspace,
                                          int* route_dst,
                                          int* exec_rank,
                                          int* exec_slot,
                                          bool* is_token_in_rank,
                                          int* slot_count,
                                          int* slot_begin,
                                          int* replica_expert,
                                          int* slot_expert,
                                          int* num_tokens_per_rank,
                                          int* num_tokens_per_rdma_rank,
                                          int* num_tokens_per_exec_expert,
                                          cudaStream_t stream) {
    const int routes_per_rank = num_tokens * kTopK;
    // All routes in the EP16 world may legally select experts on the same
    // server.  After balancing across its eight ranks, one execution rank can
    // therefore own two source-ranks' worth of assignments, plus padding for
    // every local/replica slot.
    const int nvs = kNumServers * routes_per_rank +
                    (token_padding - 1) * kNumSlots;

    build_server_local_plan<false><<<1, kNumExperts, 0, stream>>>(
        global_counts,
        nullptr,
        0,
        workspace.tokens_per_expert_prefix,
        workspace.alloc_prefix,
        workspace.expert_slot,
        slot_count,
        slot_begin,
        replica_expert,
        slot_expert,
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_exec_expert,
        1,
        source_rank,
        token_padding);

    const int route_blocks_unbounded =
            (routes_per_rank + kThreads - 1) / kThreads;
    const int route_blocks = route_blocks_unbounded < 256 ?
            route_blocks_unbounded : 256;
    materialize_routes_and_token_layout<false, true><<<route_blocks, kThreads, 0, stream>>>(
        local_topk_idx,
        workspace.local_ordinal,
        workspace.tokens_per_expert_prefix,
        workspace.alloc_prefix,
        workspace.expert_slot,
        slot_begin,
        route_dst,
        exec_rank,
        exec_slot,
        is_token_in_rank,
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        nullptr,
        routes_per_rank,
        routes_per_rank,
        num_tokens,
        source_rank,
        nvs);
}

void launch_server_local_plan_from_ipc_counts(
                                          const int64_t* local_topk_idx,
                                          void** buffer_ptrs,
                                          int64_t plan_reserve_offset,
                                          int source_rank,
                                          int num_tokens,
                                          int token_padding,
                                          const ServerLocalPlanWorkspace& workspace,
                                          int* route_dst,
                                          int* exec_rank,
                                          int* exec_slot,
                                          bool* is_token_in_rank,
                                          int* slot_count,
                                          int* slot_begin,
                                          int* replica_expert,
                                          int* slot_expert,
                                          int* num_tokens_per_rank,
                                          int* num_tokens_per_rdma_rank,
                                          int* num_tokens_per_exec_expert,
                                          float* exec_route_weight,
                                          cudaStream_t stream) {
    const int routes_per_rank = num_tokens * kTopK;
    const int nvs = kNumServers * routes_per_rank +
                    (token_padding - 1) * kNumSlots;

    build_server_local_plan<true><<<1, kNumExperts, 0, stream>>>(
        nullptr,
        buffer_ptrs,
        plan_reserve_offset,
        workspace.tokens_per_expert_prefix,
        workspace.alloc_prefix,
        workspace.expert_slot,
        slot_count,
        slot_begin,
        replica_expert,
        slot_expert,
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_exec_expert,
        1,
        source_rank,
        token_padding);

    const int route_blocks_unbounded =
            (routes_per_rank + kThreads - 1) / kThreads;
    const int route_blocks = route_blocks_unbounded < 256 ?
            route_blocks_unbounded : 256;
    materialize_routes_and_token_layout<true, true><<<route_blocks, kThreads, 0, stream>>>(
        local_topk_idx,
        workspace.local_ordinal,
        workspace.tokens_per_expert_prefix,
        workspace.alloc_prefix,
        workspace.expert_slot,
        slot_begin,
        route_dst,
        exec_rank,
        exec_slot,
        is_token_in_rank,
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        exec_route_weight,
        routes_per_rank,
        routes_per_rank,
        num_tokens,
        source_rank,
        nvs);
}

void launch_server_local_plan(const int64_t* local_topk_idx,
                              const int* global_counts,
                              int source_rank,
                              int num_tokens,
                              int token_padding,
                              const ServerLocalPlanWorkspace& workspace,
                              int* route_dst,
                              int* exec_rank,
                              int* exec_slot,
                              bool* is_token_in_rank,
                              int* slot_count,
                              int* slot_begin,
                              int* replica_expert,
                              int* slot_expert,
                              int* num_tokens_per_rank,
                              int* num_tokens_per_rdma_rank,
                              int* num_tokens_per_exec_expert,
                              cudaStream_t stream) {
    launch_local_histogram_and_ordinal(
        local_topk_idx,
        num_tokens,
        workspace.local_histogram,
        workspace.local_ordinal,
        stream);
    launch_server_local_plan_from_counts(
        local_topk_idx,
        global_counts,
        source_rank,
        num_tokens,
        token_padding,
        workspace,
        route_dst,
        exec_rank,
        exec_slot,
        is_token_in_rank,
        slot_count,
        slot_begin,
        replica_expert,
        slot_expert,
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_exec_expert,
        stream);
}

ServerLocalPlan plan_server_local_cuda(const torch::Tensor& topk_idx,
                                       int ranks_per_server,
                                       int local_experts,
                                       int replica_slots,
                                       int token_padding) {
    (void)ranks_per_server;
    (void)local_experts;
    (void)replica_slots;

    const int num_tokens = static_cast<int>(topk_idx.size(1));
    const int routes_per_rank = num_tokens * kTopK;
    const int total_routes = kNumRanks * routes_per_rank;
    const int nvs = kNumServers * routes_per_rank +
                    (token_padding - 1) * kNumSlots;
    const auto int_options = topk_idx.options().dtype(torch::kInt32);
    const auto bool_options = topk_idx.options().dtype(torch::kBool);

    auto tokens_per_home_expert = torch::empty({kNumRanks, kNumExperts}, int_options);
    auto local_ordinal = torch::empty(topk_idx.sizes(), int_options);
    auto tokens_per_expert_prefix = torch::empty({kNumRanks, kNumExperts}, int_options);
    auto alloc_prefix = torch::empty({kNumExperts, kNumRanks}, int_options);
    auto expert_slot = torch::empty({kNumRanks, kNumExperts}, int_options);

    auto route_dst = torch::empty(topk_idx.sizes(), int_options);
    auto exec_rank = torch::empty(topk_idx.sizes(), int_options);
    auto exec_slot = torch::empty(topk_idx.sizes(), int_options);
    auto is_token_in_rank = torch::empty({kNumRanks, num_tokens, kNumRanks}, bool_options);
    auto slot_count = torch::empty({kNumRanks, kNumSlots}, int_options);
    auto slot_begin = torch::empty({kNumRanks, kNumSlots}, int_options);
    auto replica_expert = torch::empty({kNumRanks, kReplicaSlots}, int_options);
    auto slot_expert = torch::empty({kNumRanks, kNumSlots}, int_options);
    auto num_tokens_per_rank = torch::empty({kNumRanks, kNumRanks}, int_options);
    auto num_tokens_per_rdma_rank = torch::empty({kNumRanks, kNumServers}, int_options);
    auto num_tokens_per_exec_expert = torch::empty({kNumRanks, kNumRanks * kNumSlots}, int_options);

    const auto stream = at::cuda::getCurrentCUDAStream();
    launch_histogram_and_ordinal(
        topk_idx.data_ptr<int64_t>(), kNumRanks,
        tokens_per_home_expert.data_ptr<int>(),
        local_ordinal.data_ptr<int>(), routes_per_rank,
        routes_per_rank, stream);
    build_server_local_plan<false><<<1, kNumExperts, 0, stream>>>(
        tokens_per_home_expert.data_ptr<int>(),
        nullptr,
        0,
        tokens_per_expert_prefix.data_ptr<int>(),
        alloc_prefix.data_ptr<int>(),
        expert_slot.data_ptr<int>(),
        slot_count.data_ptr<int>(),
        slot_begin.data_ptr<int>(),
        replica_expert.data_ptr<int>(),
        slot_expert.data_ptr<int>(),
        num_tokens_per_rank.data_ptr<int>(),
        num_tokens_per_rdma_rank.data_ptr<int>(),
        num_tokens_per_exec_expert.data_ptr<int>(),
        kNumRanks,
        -1,
        token_padding);

    constexpr int kRouteBlocks = 256;
    materialize_routes_and_token_layout<false, false><<<kRouteBlocks, kThreads, 0, stream>>>(
        topk_idx.data_ptr<int64_t>(),
        local_ordinal.data_ptr<int>(),
        tokens_per_expert_prefix.data_ptr<int>(),
        alloc_prefix.data_ptr<int>(),
        expert_slot.data_ptr<int>(),
        slot_begin.data_ptr<int>(),
        route_dst.data_ptr<int>(),
        exec_rank.data_ptr<int>(),
        exec_slot.data_ptr<int>(),
        is_token_in_rank.data_ptr<bool>(),
        num_tokens_per_rank.data_ptr<int>(),
        num_tokens_per_rdma_rank.data_ptr<int>(),
        nullptr,
        total_routes,
        routes_per_rank,
        num_tokens,
        -1,
        nvs);

    return {
        route_dst,
        exec_rank,
        exec_slot,
        is_token_in_rank,
        slot_count,
        slot_begin,
        replica_expert,
        slot_expert,
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_exec_expert,
        nvs,
    };
}

} // namespace deep_ep::moonep
