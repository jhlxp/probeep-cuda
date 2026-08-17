#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/types.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace deep_ep::moonep {

// Phase-one ProbeEP is intentionally specialized for the production EP16
// topology.  Keeping the topology in one place also makes the CUDA workspace
// layout identical in the planner, transport and expert-I/O kernels.
constexpr int kWorldSize = 16;
constexpr int kRanksPerServer = 8;
constexpr int kNumServers = 2;
constexpr int kNumExperts = 256;
constexpr int kLocalExperts = 16;
constexpr int kReplicaSlots = 16;
constexpr int kExecutionSlots = kLocalExperts + kReplicaSlots;
constexpr int kTopK = 8;
constexpr int kMaxTokensPerRank = 4096;
// BF16 torch._grouped_mm on Hopper requires each expert group to end on an
// eight-row boundary.  The old 128-row padding wasted up to 1,920 execution
// rows per rank without buying any additional kernel compatibility.
constexpr int kTokenPadding = 8;
constexpr int kMaxAssignmentsPerRank = kMaxTokensPerRank * kTopK;
// All routes may legally target one home server.  Server-local balancing then
// gives each of its eight execution ranks R*S*K/G assignments.
constexpr int kMaxAssignmentsPerExecutionRank =
        kWorldSize * kMaxAssignmentsPerRank / kRanksPerServer;
constexpr int kMaxExecutionRows =
        kMaxAssignmentsPerExecutionRank +
        (kTokenPadding - 1) * kExecutionSlots;
// Hidden payloads are deduplicated per destination rank, not per assignment.
// In the worst case every source token selects this rank, hence R*S rows.
constexpr int kMaxTransportRows = kWorldSize * kMaxTokensPerRank;
constexpr int kPlanRingSlots = 3;
constexpr int kHidden = 7168;
constexpr int kFp8ScaleBlock = 128;
constexpr int kFp8Scales = kHidden / kFp8ScaleBlock;
constexpr int kNumSms = 24;
constexpr int kNumChannels = kNumSms / 2;

// Each CUDA-IPC allocation carries this small control tail.  The normal
// DeepEP kernels still see the original num_nvl_bytes and therefore cannot
// overlap it.  A rank publishes the two source-count rows belonging to its
// NVLink lane here after the paired NVSHMEM exchange.
struct alignas(256) IpcPlanReserve {
    int source_counts[kNumServers][kNumExperts];
};

constexpr std::size_t kIpcPlanReserveBytes =
        ((sizeof(IpcPlanReserve) + 255) / 256) * 256;
constexpr std::size_t kSymmetricPlanBytes =
        kNumServers * kNumExperts * sizeof(int);

static_assert(kWorldSize == kRanksPerServer * kNumServers);
static_assert(kNumExperts == kWorldSize * kLocalExperts);
static_assert(kMaxExecutionRows == 65760);
static_assert(kMaxTransportRows == 65536);
static_assert(kFp8Scales == 56);

// All tensors in this structure are allocated once by configure_balanced().
// A handle only copies Tensor references; it never allocates iteration-sized
// device storage.  Receive tensors expose their full fixed capacity and
// recv_count is the device-resident valid prefix consumed by combine.
struct BalancedSlotStorage {
    // Planner scratch and materialized plan.
    torch::Tensor local_histogram;                 // [E], int32
    torch::Tensor local_ordinal;                   // [S*K], int32
    torch::Tensor tokens_per_expert_prefix;        // [R,E], int32
    torch::Tensor alloc_prefix;                    // [E,R], int32
    torch::Tensor expert_slot;                     // [R,E], int32
    torch::Tensor route_dst;                       // [S,K], int32
    torch::Tensor exec_rank;                       // [S,K], int32
    torch::Tensor exec_slot;                       // [S,K], int32
    torch::Tensor is_token_in_rank;                // [S,R], bool
    torch::Tensor slot_count;                      // [R,L+B], int32
    torch::Tensor slot_begin;                      // [R,L+B], int32
    torch::Tensor replica_expert;                  // [R,B], int32
    torch::Tensor slot_expert;                     // [R,L+B], int32
    torch::Tensor num_tokens_per_rank;             // [R], int32
    torch::Tensor num_tokens_per_rdma_rank;        // [N], int32
    torch::Tensor num_tokens_per_exec_slot;        // [R*(L+B)], int32

    // Fixed-capacity grouped execution input and combine output.
    torch::Tensor exec_x;                          // [max_NvS,H], fp8 e4m3
    torch::Tensor exec_x_scales;                   // [max_NvS,H/128], fp32
    torch::Tensor exec_weights;                    // [max_NvS], fp32
    torch::Tensor exec_y;                          // [max_NvS,H], bf16
    torch::Tensor combined_x;                      // [S,H], bf16
    torch::Tensor combined_topk_weights;           // [S,K], fp32

    // Persistent normal-path receive metadata.  A transport row can fan out
    // to up to K execution rows without another full-size pack/unpack pass.
    torch::Tensor recv_route_rows;                 // [R*S,K], int32
    torch::Tensor recv_src_meta;                   // [R*S,source_meta_bytes], uint8
    torch::Tensor rdma_channel_prefix_matrix;      // [N,C], int32
    torch::Tensor recv_rdma_channel_prefix_matrix; // [N,C], int32
    torch::Tensor recv_rdma_rank_prefix_sum;       // [N], int32
    torch::Tensor gbl_channel_prefix_matrix;       // [R,C], int32
    torch::Tensor recv_gbl_channel_prefix_matrix;  // [R,C], int32
    torch::Tensor recv_gbl_rank_prefix_sum;        // [R], int32
    torch::Tensor send_rdma_head;                  // [S,N], int32
    torch::Tensor send_nvl_head;                   // [R*S,G], int32
    torch::Tensor recv_count;                      // [1], int32
    torch::Tensor recv_rdma_count;                 // [1], int32
};

class BalancedRuntime;

class BalancedHandle {
public:
    BalancedHandle() = default;

    int slot() const { return slot_; }
    std::uint64_t generation() const { return generation_; }
    int num_tokens() const { return num_tokens_; }

    const torch::Tensor& route_dst() const { return storage_->route_dst; }
    const torch::Tensor& exec_rank() const { return storage_->exec_rank; }
    const torch::Tensor& exec_slot() const { return storage_->exec_slot; }
    const torch::Tensor& is_token_in_rank() const { return storage_->is_token_in_rank; }
    const torch::Tensor& slot_count() const { return storage_->slot_count; }
    const torch::Tensor& slot_begin() const { return storage_->slot_begin; }
    const torch::Tensor& replica_expert() const { return storage_->replica_expert; }
    const torch::Tensor& slot_expert() const { return storage_->slot_expert; }
    const torch::Tensor& num_tokens_per_rank() const { return storage_->num_tokens_per_rank; }
    const torch::Tensor& num_tokens_per_rdma_rank() const { return storage_->num_tokens_per_rdma_rank; }
    const torch::Tensor& num_tokens_per_exec_slot() const { return storage_->num_tokens_per_exec_slot; }
    const torch::Tensor& exec_x() const { return storage_->exec_x; }
    const torch::Tensor& exec_x_scales() const { return storage_->exec_x_scales; }
    const torch::Tensor& exec_weights() const { return storage_->exec_weights; }
    const torch::Tensor& exec_y() const { return storage_->exec_y; }
    const torch::Tensor& combined_x() const { return storage_->combined_x; }
    const torch::Tensor& recv_count() const { return storage_->recv_count; }

    // Marks whether the plan must remain live through backward.  Buffer's
    // combine path calls this exactly once: forward-only releases immediately,
    // training records a forward-complete event and grad_reduce releases it.
    void finish(cudaStream_t completion_stream, bool backward_follows) const;

private:
    friend class BalancedRuntime;

    BalancedHandle(int slot, std::uint64_t generation, int num_tokens,
                   std::shared_ptr<BalancedSlotStorage> storage,
                   BalancedRuntime* runtime):
            slot_(slot), generation_(generation), num_tokens_(num_tokens),
            storage_(std::move(storage)), runtime_(runtime) {}

    int slot_ = -1;
    std::uint64_t generation_ = 0;
    int num_tokens_ = 0;
    std::shared_ptr<BalancedSlotStorage> storage_;
    BalancedRuntime* runtime_ = nullptr;
};

// Owns the persistent EP16 execution ring.  acquire() and release() only
// enqueue CUDA event dependencies; neither method queries or synchronizes the
// device.  The release event is recorded by combine for forward-only use and
// by gradient reduction when backward is enabled.
class BalancedRuntime {
public:
    BalancedRuntime() = default;
    ~BalancedRuntime();

    BalancedRuntime(const BalancedRuntime&) = delete;
    BalancedRuntime& operator=(const BalancedRuntime&) = delete;

    void configure(int rank, int device_id, int source_meta_bytes,
                   int64_t transport_nvl_bytes, void* local_nvl_base,
                   void** peer_nvl_bases, int** barrier_signal_ptrs,
                   void* symmetric_plan_base,
                   const at::cuda::CUDAStream& comm_stream);

    bool is_configured() const { return configured_; }
    int64_t ipc_plan_reserve_offset() const { return transport_nvl_bytes_; }
    int* symmetric_counts() const {
        return static_cast<int*>(symmetric_plan_base_);
    }
    torch::Tensor& transport_y() { return transport_y_; }

    BalancedHandle acquire(int num_tokens);
    BalancedSlotStorage& storage(const BalancedHandle& handle);
    const BalancedSlotStorage& storage(const BalancedHandle& handle) const;
    void release(const BalancedHandle& handle, cudaStream_t completion_stream);
    void defer_release(const BalancedHandle& handle,
                       cudaStream_t completion_stream);

    void register_expert_pools(
            const std::vector<torch::Tensor>& home_weight_shards,
            const std::vector<torch::Tensor>& replica_weight_shards,
            const std::vector<torch::Tensor>& home_grad_shards,
            const std::vector<torch::Tensor>& replica_grad_shards);
    void launch_weight_sync(const BalancedHandle& handle, cudaStream_t stream);
    void launch_grad_reduce(const BalancedHandle& handle, cudaStream_t stream);
    void complete_backward(const BalancedHandle& handle, cudaStream_t stream);

private:
    struct RingState {
        std::shared_ptr<BalancedSlotStorage> storage;
        cudaEvent_t reusable_event = nullptr;
        std::uint64_t generation = 0;
        bool release_enqueued = true;
        bool backward_pending = false;
    };

    void validate(const BalancedHandle& handle) const;

    std::array<RingState, kPlanRingSlots> ring_;
    int next_slot_ = 0;
    int rank_ = -1;
    int device_id_ = -1;
    int source_meta_bytes_ = 0;
    int64_t transport_nvl_bytes_ = 0;
    void* local_nvl_base_ = nullptr;
    // These pointer tables are CUDA-resident copies built by Buffer::sync.
    void** peer_nvl_bases_ = nullptr;
    int** barrier_signal_ptrs_ = nullptr;
    void* symmetric_plan_base_ = nullptr;
    cudaStream_t comm_stream_ = nullptr;
    bool configured_ = false;

    // Registration-time tensors retain the framework-owned expert pools and
    // contain device pointer arrays/descriptors consumed by the batched copy
    // and deterministic reduction launchers.
    std::vector<torch::Tensor> home_weight_shards_;
    std::vector<torch::Tensor> replica_weight_shards_;
    std::vector<torch::Tensor> home_grad_shards_;
    std::vector<torch::Tensor> replica_grad_shards_;
    std::vector<torch::Tensor> home_weight_pointer_tables_;
    std::vector<torch::Tensor> home_grad_pointer_tables_;
    torch::Tensor weight_descriptors_;
    torch::Tensor grad_descriptors_;
    // Combine launches are serialized on comm_stream, so all ring slots share
    // one full-card precombine target instead of carrying three copies.
    torch::Tensor transport_y_;
};

} // namespace deep_ep::moonep
