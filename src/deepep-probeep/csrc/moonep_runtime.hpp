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

#include "probeep_topology.hpp"

namespace deep_ep::moonep {

// Each CUDA-IPC allocation carries this small control tail.  The normal
// DeepEP kernels still see the original num_nvl_bytes and therefore cannot
// overlap it. A rank publishes one E256 row per server for its NVLink lane
// after the compact NVSHMEM exchange.
struct alignas(256) IpcPlanReserve {
    int source_counts[probeep::kMaxServers][probeep::kNumExperts];
    int probe_replica_chunk_count[probeep::kPlanRingSlots]
                                     [probeep::kReplicaSlots];
    int probe_replica_ready[probeep::kPlanRingSlots]
                            [probeep::kReplicaSlots];
    int probe_owner_grad_chunk_count[probeep::kPlanRingSlots]
                                      [probeep::kMaxLocalExperts];
    int probe_owner_grad_ready[probeep::kPlanRingSlots]
                               [probeep::kMaxLocalExperts];
    // Weight progress streams must not reuse DeepEP's transport barrier
    // signals.  Monotonic tickets make both local phases reusable across ring
    // generations without a reset race.
    int probe_weight_barrier_ticket[probeep::kPlanRingSlots][2];
};

constexpr std::size_t kIpcPlanReserveBytes =
        ((sizeof(IpcPlanReserve) + 255) / 256) * 256;
constexpr std::size_t kSymmetricPlanBytes =
        probeep::kMaxServers * probeep::kNumExperts * sizeof(int);

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

    // ProbeEP server-first state.  These buffers are intentionally colocated
    // with the MoonEP ring slot so dispatch, combine and backward retain one
    // immutable plan generation without allocating in the hot path.
    torch::Tensor probe_global_counts;              // [R,E], int32
    torch::Tensor probe_alloc;                      // [E,R], int32
    torch::Tensor probe_server_load_before;         // [P], int32
    torch::Tensor probe_server_load_after;          // [P], int32
    torch::Tensor probe_server_padded_load_before;  // [P], int32
    torch::Tensor probe_server_padded_load_after;   // [P], int32
    torch::Tensor probe_assigned_tx_bytes;           // weight [R], int64
    torch::Tensor probe_assigned_rx_bytes;           // weight [R], int64
    torch::Tensor probe_dispatch_tx_bytes;            // absolute footprint [R]
    torch::Tensor probe_dispatch_rx_bytes;            // absolute footprint [R]
    torch::Tensor probe_pair_load_bytes;             // [P,P,8], int64
    torch::Tensor probe_server_expert_rows;           // [E,P], int32
    torch::Tensor probe_compute_intents;               // [2E+2Pmax+1,6], int32
    torch::Tensor probe_migration_budget_snapshot;   // [R], int64
    torch::Tensor probe_endpoint_total_cap_bytes;     // Dispatch+Weight [R]
    torch::Tensor probe_admitted_experts;            // [E*(P-1)], int32
    torch::Tensor probe_deferred_experts;            // [E], bool
    torch::Tensor probe_chunk_table;                 // [max_chunks,13], int64
    torch::Tensor probe_plan_counts;                 // [13], int32
    // The physical replica bank belongs to this ring slot.  Its layout cache
    // therefore has the same lifetime and can bypass the complete weight-sync
    // sequence when the next generation selects the same experts.
    torch::Tensor cached_replica_expert;             // [R,B], int32
    torch::Tensor weight_transfer_required;          // [1], int32

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
    const torch::Tensor& probe_server_load_before() const { return storage_->probe_server_load_before; }
    const torch::Tensor& probe_server_load_after() const { return storage_->probe_server_load_after; }
    const torch::Tensor& probe_server_padded_load_before() const { return storage_->probe_server_padded_load_before; }
    const torch::Tensor& probe_server_padded_load_after() const { return storage_->probe_server_padded_load_after; }
    const torch::Tensor& probe_assigned_tx_bytes() const { return storage_->probe_assigned_tx_bytes; }
    const torch::Tensor& probe_assigned_rx_bytes() const { return storage_->probe_assigned_rx_bytes; }
    const torch::Tensor& probe_dispatch_tx_bytes() const { return storage_->probe_dispatch_tx_bytes; }
    const torch::Tensor& probe_dispatch_rx_bytes() const { return storage_->probe_dispatch_rx_bytes; }
    const torch::Tensor& probe_pair_load_bytes() const { return storage_->probe_pair_load_bytes; }
    const torch::Tensor& probe_compute_intents() const { return storage_->probe_compute_intents; }
    const torch::Tensor& probe_migration_budget_snapshot() const { return storage_->probe_migration_budget_snapshot; }
    const torch::Tensor& probe_endpoint_total_cap_bytes() const { return storage_->probe_endpoint_total_cap_bytes; }
    const torch::Tensor& probe_admitted_experts() const { return storage_->probe_admitted_experts; }
    const torch::Tensor& probe_deferred_experts() const { return storage_->probe_deferred_experts; }
    const torch::Tensor& probe_chunk_table() const { return storage_->probe_chunk_table; }
    const torch::Tensor& probe_plan_counts() const { return storage_->probe_plan_counts; }
    const torch::Tensor& weight_transfer_required() const { return storage_->weight_transfer_required; }
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

// Owns the persistent runtime-sized E256 execution ring. acquire/release only
// enqueue CUDA event dependencies; neither method queries or synchronizes the
// device.  The release event is recorded by combine for forward-only use and
// by gradient reduction when backward is enabled.
class BalancedRuntime {
public:
    BalancedRuntime() = default;
    ~BalancedRuntime();

    BalancedRuntime(const BalancedRuntime&) = delete;
    BalancedRuntime& operator=(const BalancedRuntime&) = delete;

    void configure(int rank, int world_size, int device_id,
                   int source_meta_bytes,
                   int64_t transport_nvl_bytes, void* local_nvl_base,
                   void** peer_nvl_bases, int** barrier_signal_ptrs,
                   void* symmetric_plan_base,
                   const at::cuda::CUDAStream& comm_stream);

    bool is_configured() const { return configured_; }
    bool expert_pools_registered() const {
        return !home_weight_shards_.empty();
    }
    std::int64_t expert_weight_bytes() const {
        return expert_weight_bytes_;
    }
    int grad_weight_byte_ratio() const {
        return grad_weight_byte_ratio_;
    }
    int64_t ipc_plan_reserve_offset() const { return transport_nvl_bytes_; }
    const probeep::Topology& topology() const { return topology_; }
    int* symmetric_counts() const {
        return static_cast<int*>(symmetric_plan_base_);
    }
    torch::Tensor& transport_y() { return transport_y_; }
    torch::Tensor probe_migration_budget(int compute_kind) const;
    torch::Tensor probe_controller_summary(int compute_kind) const;
    void reset_probe_controller(std::int64_t fallback_budget_bytes,
                                cudaStream_t stream);

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
    void set_weight_version(const BalancedHandle& handle,
                            std::int64_t weight_version);
    void launch_weight_sync(const BalancedHandle& handle, cudaStream_t stream);
    void launch_weight_send(const BalancedHandle& handle, cudaStream_t stream);
    void join_weight_sync(const BalancedHandle& handle, cudaStream_t stream);
    void launch_grad_reduce(const BalancedHandle& handle, cudaStream_t stream);
    void complete_backward(const BalancedHandle& handle, cudaStream_t stream);
    void update_probe_controller(
            const torch::Tensor& compute_ns,
            const torch::Tensor& network_ns,
            const torch::Tensor& dispatch_tx_bytes,
            const torch::Tensor& dispatch_rx_bytes,
            const torch::Tensor& migration_tx_bytes,
            const torch::Tensor& migration_rx_bytes,
            int compute_kind,
            double rdma_path_bandwidth_gbps, double alpha,
            std::int64_t fallback_budget_bytes, bool valid,
            cudaStream_t stream);

private:
    struct RingState {
        std::shared_ptr<BalancedSlotStorage> storage;
        cudaEvent_t reusable_event = nullptr;
        cudaEvent_t weight_plan_ready_event = nullptr;
        cudaEvent_t weight_consumer_ready_event = nullptr;
        cudaStream_t weight_progress_stream = nullptr;
        std::uint64_t generation = 0;
        bool release_enqueued = true;
        bool backward_pending = false;
        bool weight_sync_enqueued = false;
        bool weight_send_enqueued = false;
        // Negative requested versions are volatile: training integrations that
        // do not provide an optimizer version conservatively refresh every
        // invocation. Non-negative versions permit cache hits within a plan
        // bank until the caller advances the globally synchronized version.
        std::int64_t requested_weight_version = -1;
        std::int64_t cached_weight_version = INT64_MIN;
    };

    void validate(const BalancedHandle& handle) const;

    std::array<RingState, probeep::kPlanRingSlots> ring_;
    int rank_ = -1;
    probeep::Topology topology_{0, 0, 0, 0};
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
    // one full-card precombine target.
    torch::Tensor transport_y_;
    // Derived from the registered model shards; never inferred from a fixed
    // DSV3 example inside the planner or transport completion logic.
    std::int64_t expert_weight_bytes_ = 0;
    int grad_weight_byte_ratio_ = 0;
    // Row 0 is Attention-overlap state and row 1 is MoE-overlap state.  A
    // dispatch reads exactly one row; updating either chain cannot overwrite
    // the other chain's most recent admissible window.
    torch::Tensor probe_migration_budget_;   // [2,R], int64
    torch::Tensor probe_controller_summary_; // [2,6], int64
};

} // namespace deep_ep::moonep
