#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace deep_ep::internode {

// Fixed EP16 balanced transport.  All shapes are capacities resident in
// persistent CUDA buffers; recv_count and recv_rdma_count are device scalars.
// In-place combine-backward post-processing.  balanced_dispatch can move a
// BF16 token gradient with num_scales == 0 directly into grouped execution
// rows; this kernel then applies the saved forward gate for each row.
void launch_bf16_route_weight_scale(void* exec_rows,
                                    const float* exec_route_weight,
                                    int num_rows,
                                    int hidden,
                                    cudaStream_t stream);

// Convert expert-grouped output into one contiguous row per received token.
// A wide persistent grid performs the local K-way reduction before the
// 12-SM DeepEP sender, leaving the sender as a pure TMA/network stage.
void launch_precombine_bf16(void* transport_rows,
                            const void* exec_rows,
                            const float* exec_route_weight,
                            const int* recv_route_rows,
                            const int* recv_count,
                            int hidden,
                            cudaStream_t stream);

// Exchange only rank counts, build the same four prefix objects consumed by
// the normal DeepEP queue engine, and publish receive totals on the device.
void balanced_notify_dispatch(const int* num_tokens_per_rank,
                              const int* num_tokens_per_rdma_rank,
                              int* recv_count,
                              int* recv_rdma_count,
                              const bool* is_token_in_rank,
                              int num_tokens,
                              int num_channels,
                              int hidden_int4,
                              int num_scales,
                              int* rdma_channel_prefix_matrix,
                              int* recv_rdma_rank_prefix_sum,
                              int* gbl_channel_prefix_matrix,
                              int* recv_gbl_rank_prefix_sum,
                              void* rdma_buffer_ptr,
                              int num_max_rdma_chunked_recv_tokens,
                              void** buffer_ptrs,
                              int num_max_nvl_chunked_recv_tokens,
                              int** barrier_signal_ptrs,
                              int rank,
                              cudaStream_t stream,
                              std::int64_t num_rdma_bytes,
                              std::int64_t num_nvl_bytes);

// route_dst is source-token-major int32 [S, 8], encoded as
// exec_rank * nvs + exec_row.  One transport row is sent per destination rank;
// the receiver fans it directly into grouped execution rows without an
// intermediate receive tensor or permutation.  x and exec_x are opaque byte
// rows sized by hidden_int4; with num_scales == 0 the two scale pointers may be
// null, which is the BF16 combine-backward path.
void balanced_dispatch(void* exec_x,
                       float* exec_x_scales,
                       float* exec_route_weight,
                       int* recv_route_rows,
                       void* recv_src_meta,
                       const void* x,
                       const float* x_scales,
                       const int* route_dst,
                       const float* route_weight,
                       int* send_rdma_head,
                       int* send_nvl_head,
                       int* recv_rdma_channel_prefix_matrix,
                       int* recv_gbl_channel_prefix_matrix,
                       const int* rdma_channel_prefix_matrix,
                       const int* recv_rdma_rank_prefix_sum,
                       const int* gbl_channel_prefix_matrix,
                       const int* recv_gbl_rank_prefix_sum,
                       const bool* is_token_in_rank,
                       int num_tokens,
                       int hidden_int4,
                       int num_scales,
                       int scale_token_stride,
                       int scale_hidden_stride,
                       int nvs,
                       void* rdma_buffer_ptr,
                       int num_max_rdma_chunked_send_tokens,
                       int num_max_rdma_chunked_recv_tokens,
                       void** buffer_ptrs,
                       int num_max_nvl_chunked_send_tokens,
                       int num_max_nvl_chunked_recv_tokens,
                       int rank,
                       cudaStream_t stream,
                       int num_channels);

// Replay combine backward through the saved forward plan.  This performs the
// cached-dispatch barrier/queue cleanup and then moves BF16 grad_out rows
// directly into their original grouped execution rows.  Saved execution-row
// weights remain untouched for launch_bf16_route_weight_scale.
void balanced_cached_dispatch(
        void* exec_x,
        const void* grad_out,
        const int* route_dst,
        const int* rdma_channel_prefix_matrix,
        const int* recv_rdma_rank_prefix_sum,
        const int* gbl_channel_prefix_matrix,
        const int* recv_gbl_rank_prefix_sum,
        const bool* is_token_in_rank,
        int num_tokens,
        int hidden_int4,
        int nvs,
        void* rdma_buffer_ptr,
        int num_max_rdma_chunked_send_tokens,
        int num_max_rdma_chunked_recv_tokens,
        void** buffer_ptrs,
        int num_max_nvl_chunked_send_tokens,
        int num_max_nvl_chunked_recv_tokens,
        int** barrier_signal_ptrs,
        int rank,
        cudaStream_t stream,
        int num_channels,
        std::int64_t num_rdma_bytes,
        std::int64_t num_nvl_bytes);

// recv_route_rows maps each received transport row to its local grouped rows.
// The NVL sender performs the BF16 local reduction in place, after which the
// original DeepEP NVL/RDMA reverse path produces [S, H].  Passing a null
// exec_route_weight selects an unweighted reduction (dispatch backward).
void balanced_combine(void* combined_x,
                      const bool* is_combined_token_in_rank,
                      const void* exec_y,
                      const float* exec_route_weight,
                      const int* recv_route_rows,
                      const int* recv_count,
                      const int* combined_rdma_head,
                      const int* combined_nvl_head,
                      const void* src_meta,
                      const int* rdma_channel_prefix_matrix,
                      const int* rdma_rank_prefix_sum,
                      const int* gbl_channel_prefix_matrix,
                      int num_combined_tokens,
                      int hidden,
                      int nvs,
                      void* rdma_buffer_ptr,
                      int num_max_rdma_chunked_send_tokens,
                      int num_max_rdma_chunked_recv_tokens,
                      void** buffer_ptrs,
                      int num_max_nvl_chunked_send_tokens,
                      int num_max_nvl_chunked_recv_tokens,
                      int rank,
                      cudaStream_t stream,
                      int num_channels);

} // namespace deep_ep::internode
