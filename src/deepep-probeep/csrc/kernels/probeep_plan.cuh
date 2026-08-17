#pragma once

#include <cuda_runtime.h>
#include <torch/types.h>

#include "../probeep_topology.hpp"

namespace deep_ep::probeep {

inline constexpr int kProbeChunkFields = 13;
// [0:9] are correctness/lifecycle counters; [9:13] are device clock-cycle
// phase costs for init+intent, admission, local packing and finalization.
// Keeping the phase telemetry in the persistent plan avoids profiler-only
// blind spots and adds no host synchronization to the hot path.
inline constexpr int kProbePlanCounterFields = 13;
inline constexpr int kProbeMaxComputeIntents =
        2 * kNumExperts + 2 * kMaxServers + 1;
inline constexpr int kProbeMaxRemoteReplicas =
        kNumExperts * (kMaxServers - 1);
inline constexpr int kProbeMaxChunksPerExpert = 64;
// Each endpoint can inject at most 64 MiB in one probe window. With 4-MiB
// chunks the global table cannot contain more than R*16 committed chunks.
inline constexpr int kProbeMaxChunks = kMaxWorldSize * 16;

struct ProbePlanCuda {
    torch::Tensor route_dst;
    torch::Tensor exec_rank;
    torch::Tensor exec_slot;
    torch::Tensor is_token_in_rank;
    torch::Tensor slot_count;
    torch::Tensor slot_begin;
    torch::Tensor replica_expert;
    torch::Tensor slot_expert;
    torch::Tensor alloc;
    torch::Tensor num_tokens_per_rank;
    torch::Tensor num_tokens_per_rdma_rank;
    torch::Tensor num_tokens_per_exec_expert;
    torch::Tensor server_load_before;
    torch::Tensor server_load_after;
    torch::Tensor server_padded_load_before;
    torch::Tensor server_padded_load_after;
    torch::Tensor assigned_tx_bytes;
    torch::Tensor assigned_rx_bytes;
    torch::Tensor dispatch_tx_bytes;
    torch::Tensor dispatch_rx_bytes;
    torch::Tensor pair_load_bytes;
    torch::Tensor compute_intents;
    torch::Tensor migration_budget_snapshot;
    torch::Tensor endpoint_total_cap_bytes;
    torch::Tensor admitted_experts;
    torch::Tensor deferred_experts;
    torch::Tensor chunk_table;
    torch::Tensor plan_counts;
    int nvs;
};

// Persistent pointers owned by one BalancedRuntime ring slot.  The hot path
// only launches CUDA kernels into these buffers; it performs no iteration
// allocation and never asks the host to inspect the plan.
struct ProbePlanWorkspace {
    int* global_counts;
    int* local_ordinal;
    int* count_prefix;
    int* alloc_prefix;
    int* alloc;
    int* expert_slot;
    int* route_dst;
    int* exec_rank;
    int* exec_slot;
    bool* is_token_in_rank;
    int* slot_count;
    int* slot_begin;
    int* replica_expert;
    int* slot_expert;
    int* num_tokens_per_rank;
    int* num_tokens_per_rdma_rank;
    int* num_tokens_per_exec_expert;
    int* server_load_before;
    int* server_load_after;
    int* server_padded_load_before;
    int* server_padded_load_after;
    std::int64_t* assigned_tx_bytes;
    std::int64_t* assigned_rx_bytes;
    // Absolute Token Dispatch footprint for the current admitted placement.
    // It is conservative at expert-occurrence granularity; actual DeepEP
    // token de-duplication is fed back by the measured controller sample.
    std::int64_t* dispatch_tx_bytes;
    std::int64_t* dispatch_rx_bytes;
    std::int64_t* pair_load_bytes;
    int* server_expert_rows;
    int* compute_intents;  // [kProbeMaxComputeIntents,6]
    std::int64_t* migration_budget_snapshot;
    std::int64_t* endpoint_total_cap_bytes;
    int* admitted_experts;
    bool* deferred_experts;
    std::int64_t* chunk_table;
    int* plan_counts;
};

// Build the distributed ProbeEP plan after publish_server_histograms() made
// every source histogram visible in the local CUDA-IPC reserve.  topk_idx is
// only this rank's [S,8] routing; all global decisions use the device counts.
void launch_probeep_plan_from_ipc_counts(
        const std::int64_t* local_topk_idx,
        void** buffer_ptrs,
        std::int64_t plan_reserve_offset,
        int source_rank,
        int world_size,
        int num_tokens,
        int token_padding,
        const std::int64_t* migration_budget_bytes,
        const std::int64_t* controller_summary,
        std::int64_t dispatch_bytes_per_route,
        std::int64_t expert_weight_bytes,
        std::int64_t weight_chunk_bytes,
        const ProbePlanWorkspace& workspace,
        cudaStream_t stream);

ProbePlanCuda plan_probeep_cuda(
        const torch::Tensor& topk_idx,
        const torch::Tensor& migration_budget_bytes,
        std::int64_t expert_weight_bytes,
        std::int64_t weight_chunk_bytes,
        int ranks_per_server,
        int local_experts,
        int replica_slots,
        int token_padding,
        std::int64_t learned_total_bytes);

} // namespace deep_ep::probeep
