#pragma once

#include <cuda_runtime.h>
#include <torch/types.h>

namespace deep_ep::moonep {

struct ServerLocalPlan {
    torch::Tensor route_dst;
    torch::Tensor exec_rank;
    torch::Tensor exec_slot;
    torch::Tensor is_token_in_rank;
    torch::Tensor slot_count;
    torch::Tensor slot_begin;
    torch::Tensor replica_expert;
    torch::Tensor slot_expert;
    torch::Tensor num_tokens_per_rank;
    torch::Tensor num_tokens_per_rdma_rank;
    torch::Tensor num_tokens_per_exec_expert;
    int nvs;
};

struct ServerLocalPlanWorkspace {
    int* local_histogram;
    int* local_ordinal;
    int* tokens_per_expert_prefix;
    int* alloc_prefix;
    int* expert_slot;
};

void launch_local_histogram_and_ordinal(const int64_t* local_topk_idx,
                                        int num_tokens,
                                        int* local_histogram,
                                        int* local_ordinal,
                                        cudaStream_t stream);

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
                                          cudaStream_t stream);

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
                                          cudaStream_t stream);

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
                              cudaStream_t stream);

ServerLocalPlan plan_server_local_cuda(const torch::Tensor& topk_idx,
                                       int ranks_per_server,
                                       int local_experts,
                                       int replica_slots,
                                       int token_padding);

} // namespace deep_ep::moonep
