#include <ATen/cuda/CUDADataType.h>
#include <ATen/native/cuda/GroupMM.h>
#include <cuda_runtime.h>

#include "deep_ep.hpp"
#include "moonep_binding.hpp"
#include "moonep_expert_pool.hpp"

#include <cstring>

#include <pybind11/stl.h>
#include <torch/extension.h>

#include "kernels/api.cuh"
#include "kernels/exception.cuh"
#include "kernels/moonep_expert_io.cuh"
#include "kernels/probeep_controller.cuh"
#include "kernels/probeep_plan.cuh"
#include "kernels/probeep_weight_transport.cuh"

namespace deep_ep::moonep {
namespace {

torch::Tensor copy_bytes_to_device(const void* source, std::size_t bytes,
                                   int device_id) {
    auto host = torch::empty(
            {static_cast<int64_t>(bytes)},
            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
    std::memcpy(host.data_ptr(), source, bytes);
    return host.to(torch::Device(torch::kCUDA, device_id));
}

void bf16_grouped_mm_out(torch::Tensor mat_a, torch::Tensor mat_b,
                         torch::Tensor offsets, torch::Tensor output) {
    at::cuda::detail::bf16bf16_grouped_mm(
            std::move(mat_a), std::move(mat_b), std::move(offsets),
            std::nullopt, output);
}

std::shared_ptr<BalancedSlotStorage> allocate_slot(
        int device_id, int source_meta_bytes,
        const probeep::Topology& topology) {
    auto slot = std::make_shared<BalancedSlotStorage>();
    const auto device = torch::Device(torch::kCUDA, device_id);
    const auto ints = torch::TensorOptions().dtype(torch::kInt32).device(device);
    const auto bools = torch::TensorOptions().dtype(torch::kBool).device(device);
    const auto bytes = torch::TensorOptions().dtype(torch::kUInt8).device(device);
    const auto floats = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    const auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(device);
    const auto fp8 = torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(device);

    const int world = topology.world_size;
    const int servers = topology.num_servers;
    const int slots = topology.execution_slots;
    slot->local_histogram = torch::empty({probeep::kNumExperts}, ints);
    slot->local_ordinal = torch::empty(
            {probeep::kMaxTokensPerRank * probeep::kTopK}, ints);
    slot->tokens_per_expert_prefix =
            torch::empty({world, probeep::kNumExperts}, ints);
    slot->alloc_prefix =
            torch::empty({probeep::kNumExperts, world}, ints);
    slot->expert_slot =
            torch::empty({world, probeep::kNumExperts}, ints);
    slot->route_dst = torch::empty(
            {probeep::kMaxTokensPerRank, probeep::kTopK}, ints);
    slot->exec_rank = torch::empty(
            {probeep::kMaxTokensPerRank, probeep::kTopK}, ints);
    slot->exec_slot = torch::empty(
            {probeep::kMaxTokensPerRank, probeep::kTopK}, ints);
    slot->is_token_in_rank = torch::empty(
            {probeep::kMaxTokensPerRank, world}, bools);
    slot->slot_count = torch::empty({world, slots}, ints);
    slot->slot_begin = torch::empty({world, slots}, ints);
    slot->replica_expert =
            torch::empty({world, probeep::kReplicaSlots}, ints);
    slot->slot_expert = torch::empty({world, slots}, ints);
    slot->num_tokens_per_rank = torch::empty({world}, ints);
    slot->num_tokens_per_rdma_rank = torch::empty({servers}, ints);
    slot->num_tokens_per_exec_slot =
            torch::empty({world * slots}, ints);
    slot->probe_global_counts =
            torch::empty({world, probeep::kNumExperts}, ints);
    slot->probe_alloc =
            torch::empty({probeep::kNumExperts, world}, ints);
    slot->probe_server_load_before = torch::empty({servers}, ints);
    slot->probe_server_load_after = torch::empty({servers}, ints);
    slot->probe_server_padded_load_before = torch::empty({servers}, ints);
    slot->probe_server_padded_load_after = torch::empty({servers}, ints);
    slot->probe_assigned_tx_bytes = torch::empty(
            {world}, torch::TensorOptions().dtype(torch::kInt64).device(device));
    slot->probe_assigned_rx_bytes = torch::empty(
            {world}, torch::TensorOptions().dtype(torch::kInt64).device(device));
    slot->probe_dispatch_tx_bytes = torch::empty(
            {world}, torch::TensorOptions().dtype(torch::kInt64).device(device));
    slot->probe_dispatch_rx_bytes = torch::empty(
            {world}, torch::TensorOptions().dtype(torch::kInt64).device(device));
    slot->probe_pair_load_bytes = torch::empty(
            {servers, servers, probeep::kRanksPerServer},
            torch::TensorOptions().dtype(torch::kInt64).device(device));
    slot->probe_server_expert_rows = torch::empty(
            {probeep::kNumExperts, servers}, ints);
    slot->probe_compute_intents = torch::empty(
            {probeep::kProbeMaxComputeIntents, 6}, ints);
    slot->probe_migration_budget_snapshot = torch::empty(
            {world}, torch::TensorOptions().dtype(torch::kInt64).device(device));
    slot->probe_endpoint_total_cap_bytes = torch::empty(
            {world}, torch::TensorOptions().dtype(torch::kInt64).device(device));
    slot->probe_admitted_experts = torch::empty(
            {probeep::kProbeMaxRemoteReplicas}, ints);
    slot->probe_deferred_experts =
            torch::empty({probeep::kNumExperts}, bools);
    slot->probe_chunk_table = torch::empty(
            {probeep::kProbeMaxChunks, probeep::kProbeChunkFields},
            torch::TensorOptions().dtype(torch::kInt64).device(device));
    slot->probe_plan_counts = torch::empty(
            {probeep::kProbePlanCounterFields}, ints);
    slot->cached_replica_expert = torch::empty(
            {world, probeep::kReplicaSlots}, ints);
    slot->weight_transfer_required = torch::empty({1}, ints);

    const int max_execution_rows = probeep::max_execution_rows(topology);
    const int max_transport_rows = probeep::max_transport_rows(topology);
    slot->exec_x = torch::empty(
            {max_execution_rows, probeep::kHidden}, fp8);
    slot->exec_x_scales = torch::empty(
            {max_execution_rows, probeep::kFp8Scales}, floats);
    slot->exec_weights = torch::empty({max_execution_rows}, floats);
    slot->exec_y = torch::empty(
            {max_execution_rows, probeep::kHidden}, bf16);
    slot->combined_x = torch::empty(
            {probeep::kMaxTokensPerRank, probeep::kHidden}, bf16);
    slot->combined_topk_weights =
            torch::empty(
                    {probeep::kMaxTokensPerRank, probeep::kTopK}, floats);

    slot->recv_route_rows = torch::empty(
            {max_transport_rows, probeep::kTopK}, ints);
    slot->recv_src_meta =
            torch::empty({max_transport_rows, source_meta_bytes}, bytes);
    slot->rdma_channel_prefix_matrix =
            torch::empty({servers, probeep::kNumChannels}, ints);
    slot->recv_rdma_channel_prefix_matrix =
            torch::empty({servers, probeep::kNumChannels}, ints);
    slot->recv_rdma_rank_prefix_sum = torch::empty({servers}, ints);
    slot->gbl_channel_prefix_matrix =
            torch::empty({world, probeep::kNumChannels}, ints);
    slot->recv_gbl_channel_prefix_matrix =
            torch::empty({world, probeep::kNumChannels}, ints);
    slot->recv_gbl_rank_prefix_sum = torch::empty({world}, ints);
    slot->send_rdma_head =
            torch::empty({probeep::kMaxTokensPerRank, servers}, ints);
    slot->send_nvl_head =
            torch::empty(
                    {max_transport_rows, probeep::kRanksPerServer}, ints);
    slot->recv_count = torch::empty({1}, ints);
    slot->recv_rdma_count = torch::empty({1}, ints);
    return slot;
}

} // namespace

BalancedRuntime::~BalancedRuntime() {
    for (auto& state : ring_) {
        if (state.reusable_event != nullptr)
            cudaEventDestroy(state.reusable_event);
        if (state.weight_plan_ready_event != nullptr)
            cudaEventDestroy(state.weight_plan_ready_event);
        if (state.weight_consumer_ready_event != nullptr)
            cudaEventDestroy(state.weight_consumer_ready_event);
        if (state.weight_progress_stream != nullptr)
            cudaStreamDestroy(state.weight_progress_stream);
    }
}

void BalancedRuntime::configure(int rank, int world_size, int device_id,
                                int source_meta_bytes,
                                int64_t transport_nvl_bytes,
                                void* local_nvl_base, void** peer_nvl_bases,
                                int** barrier_signal_ptrs,
                                void* symmetric_plan_base,
                                const at::cuda::CUDAStream& comm_stream) {
    EP_HOST_ASSERT(!configured_);
    EP_HOST_ASSERT(probeep::supported_world_size(world_size));
    EP_HOST_ASSERT(rank >= 0 && rank < world_size);
    EP_HOST_ASSERT(source_meta_bytes > 0);

    rank_ = rank;
    topology_ = probeep::make_topology(world_size);
    device_id_ = device_id;
    source_meta_bytes_ = source_meta_bytes;
    transport_nvl_bytes_ = transport_nvl_bytes;
    local_nvl_base_ = local_nvl_base;
    peer_nvl_bases_ = peer_nvl_bases;
    barrier_signal_ptrs_ = barrier_signal_ptrs;
    symmetric_plan_base_ = symmetric_plan_base;
    comm_stream_ = comm_stream.stream();

    const auto previous_stream = at::cuda::getCurrentCUDAStream(device_id_);
    at::cuda::setCurrentCUDAStream(comm_stream);
    probe_migration_budget_ = torch::full(
            {2, topology_.world_size}, 32LL * 1024 * 1024,
            torch::TensorOptions().dtype(torch::kInt64).device(
                    torch::Device(torch::kCUDA, device_id_)));
    probe_controller_summary_ = torch::zeros(
            {2, 6}, torch::TensorOptions().dtype(torch::kInt64).device(
                    torch::Device(torch::kCUDA, device_id_)));
    transport_y_ = torch::empty(
            {probeep::max_transport_rows(topology_), probeep::kHidden},
            torch::TensorOptions().dtype(torch::kBFloat16).device(
                    torch::Device(torch::kCUDA, device_id_)));
    for (auto& state : ring_) {
        state.storage = allocate_slot(
                device_id_, source_meta_bytes_, topology_);
        CUDA_CHECK(cudaMemsetAsync(
                state.storage->cached_replica_expert.data_ptr(), 0xff,
                state.storage->cached_replica_expert.nbytes(), comm_stream_));
        CUDA_CHECK(cudaEventCreateWithFlags(
                &state.reusable_event, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(
                &state.weight_plan_ready_event, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(
                &state.weight_consumer_ready_event, cudaEventDisableTiming));
        int least_priority = 0;
        int greatest_priority = 0;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(
                &least_priority, &greatest_priority));
        CUDA_CHECK(cudaStreamCreateWithPriority(
                &state.weight_progress_stream, cudaStreamNonBlocking,
                greatest_priority));
        CUDA_CHECK(cudaEventRecord(state.reusable_event, comm_stream_));
        CUDA_CHECK(cudaEventRecord(
                state.weight_plan_ready_event, comm_stream_));
        CUDA_CHECK(cudaEventRecord(
                state.weight_consumer_ready_event, comm_stream_));
    }
    at::cuda::setCurrentCUDAStream(previous_stream);
    configured_ = true;
}

BalancedHandle BalancedRuntime::acquire(int num_tokens) {
    EP_HOST_ASSERT(configured_);
    EP_HOST_ASSERT(num_tokens > 0 &&
                   num_tokens <= probeep::kMaxTokensPerRank);

    int slot = -1;
    for (int candidate = 0; candidate < probeep::kPlanRingSlots; ++candidate) {
        if (ring_[candidate].release_enqueued) {
            slot = candidate;
            break;
        }
    }
    EP_HOST_ASSERT(slot >= 0);
    auto& state = ring_[slot];
    CUDA_CHECK(cudaStreamWaitEvent(comm_stream_, state.reusable_event, 0));
    state.release_enqueued = false;
    state.weight_sync_enqueued = false;
    state.weight_send_enqueued = false;
    state.requested_weight_version = -1;
    ++state.generation;
    CUDA_CHECK(cudaMemsetAsync(
            state.storage->recv_count.data_ptr(), 0, sizeof(int), comm_stream_));
    CUDA_CHECK(cudaMemsetAsync(
            state.storage->recv_rdma_count.data_ptr(), 0, sizeof(int), comm_stream_));
    return BalancedHandle(slot, state.generation, num_tokens, state.storage,
                          this);
}

void BalancedHandle::finish(cudaStream_t completion_stream,
                            bool backward_follows) const {
    if (backward_follows)
        runtime_->defer_release(*this, completion_stream);
    else
        runtime_->release(*this, completion_stream);
}

void BalancedRuntime::release(const BalancedHandle& handle,
                              cudaStream_t completion_stream) {
    validate(handle);
    auto& state = ring_[handle.slot_];
    EP_HOST_ASSERT(!state.release_enqueued);
    EP_HOST_ASSERT(!state.backward_pending);
    CUDA_CHECK(cudaEventRecord(state.reusable_event, completion_stream));
    state.release_enqueued = true;
}

void BalancedRuntime::defer_release(const BalancedHandle& handle,
                                    cudaStream_t completion_stream) {
    validate(handle);
    auto& state = ring_[handle.slot_];
    EP_HOST_ASSERT(!state.release_enqueued && !state.backward_pending);
    // This event expresses completion of the forward consumer.  Gradient
    // reduction waits for it before touching replica slots, but it does not
    // make the ring slot reusable yet.
    CUDA_CHECK(cudaEventRecord(state.reusable_event, completion_stream));
    state.backward_pending = true;
}

BalancedSlotStorage& BalancedRuntime::storage(const BalancedHandle& handle) {
    validate(handle);
    return *ring_[handle.slot_].storage;
}

const BalancedSlotStorage& BalancedRuntime::storage(
        const BalancedHandle& handle) const {
    validate(handle);
    return *ring_[handle.slot_].storage;
}

void BalancedRuntime::validate(const BalancedHandle& handle) const {
    EP_HOST_ASSERT(configured_);
    EP_HOST_ASSERT(handle.slot_ >= 0 &&
                   handle.slot_ < probeep::kPlanRingSlots);
    EP_HOST_ASSERT(handle.runtime_ == this);
    const auto& state = ring_[handle.slot_];
    EP_HOST_ASSERT(handle.generation_ == state.generation);
    EP_HOST_ASSERT(handle.storage_.get() == state.storage.get());
}

void BalancedRuntime::register_expert_pools(
        const std::vector<torch::Tensor>& home_weight_shards,
        const std::vector<torch::Tensor>& replica_weight_shards,
        const std::vector<torch::Tensor>& home_grad_shards,
        const std::vector<torch::Tensor>& replica_grad_shards) {
    EP_HOST_ASSERT(configured_);
    EP_HOST_ASSERT(home_weight_shards.size() == replica_weight_shards.size());
    EP_HOST_ASSERT(home_grad_shards.size() == replica_grad_shards.size());
    EP_HOST_ASSERT(!home_weight_shards.empty());
    EP_HOST_ASSERT(!home_grad_shards.empty());
    EP_HOST_ASSERT(home_weight_shards.size() == home_grad_shards.size());

    home_weight_shards_ = home_weight_shards;
    replica_weight_shards_ = replica_weight_shards;
    home_grad_shards_ = home_grad_shards;
    replica_grad_shards_ = replica_grad_shards;
    home_weight_pointer_tables_.clear();
    home_grad_pointer_tables_.clear();
    home_weight_pointer_tables_.reserve(home_weight_shards.size());
    home_grad_pointer_tables_.reserve(home_grad_shards.size());

    std::vector<WeightShardDescriptor> weight_descriptors(
            home_weight_shards_.size());
    std::int64_t registered_expert_weight_bytes = 0;
    std::uint32_t registered_weight_element_bytes = 0;
    for (std::size_t shard = 0; shard < weight_descriptors.size(); ++shard) {
        const auto& home = home_weight_shards_[shard];
        const auto& replica = replica_weight_shards_[shard];
        EP_HOST_ASSERT(home.is_cuda() && replica.is_cuda());
        EP_HOST_ASSERT(home.device().index() == device_id_ &&
                       replica.device().index() == device_id_);
        EP_HOST_ASSERT(home.dim() >= 1 &&
                       home.size(0) == topology_.local_experts);
        EP_HOST_ASSERT(replica.dim() >= 1 &&
                       replica.size(0) == kExpertPoolWeightSlots);
        EP_HOST_ASSERT(home.is_contiguous() && replica.is_contiguous());
        EP_HOST_ASSERT(home.scalar_type() == replica.scalar_type());
        EP_HOST_ASSERT(home.numel() / topology_.local_experts ==
                       replica.numel() / kExpertPoolWeightSlots);
        auto* home_begin = static_cast<std::uint8_t*>(home.data_ptr());
        auto* replica_storage_begin =
                static_cast<std::uint8_t*>(replica.data_ptr());
        const auto replica_stride_bytes =
                replica.stride(0) * replica.element_size();
        auto* replica_begin = replica_storage_begin +
                kExpertPoolHomeSlots * replica_stride_bytes;
        auto* nvl_begin = static_cast<std::uint8_t*>(local_nvl_base_);
        const auto home_address = reinterpret_cast<std::uintptr_t>(home_begin);
        const auto replica_storage_address =
                reinterpret_cast<std::uintptr_t>(replica_storage_begin);
        const auto replica_address =
                reinterpret_cast<std::uintptr_t>(replica_begin);
        const auto nvl_address = reinterpret_cast<std::uintptr_t>(nvl_begin);
        EP_HOST_ASSERT(home_address >= nvl_address &&
                       home_address + home.nbytes() <=
                               nvl_address + transport_nvl_bytes_ +
                                       kIpcPlanReserveBytes + kExpertPoolBytes);
        EP_HOST_ASSERT(replica_storage_address >= nvl_address &&
                       replica_storage_address + replica.nbytes() <=
                               nvl_address + transport_nvl_bytes_ +
                                       kIpcPlanReserveBytes + kExpertPoolBytes);

        std::vector<std::uint64_t> master_pointers(topology_.local_experts);
        const auto home_stride_bytes = home.stride(0) * home.element_size();
        auto* home_base = static_cast<std::uint8_t*>(home.data_ptr());
        for (int expert = 0; expert < topology_.local_experts; ++expert)
            master_pointers[expert] = reinterpret_cast<std::uint64_t>(
                    home_base + expert * home_stride_bytes);
        auto master_pointer_table = copy_bytes_to_device(
                master_pointers.data(),
                master_pointers.size() * sizeof(std::uint64_t), device_id_);
        // The device descriptor retains a raw address into this table.  Keep
        // it alive next to the registered home tensors.
        home_weight_pointer_tables_.push_back(master_pointer_table);

        auto& descriptor = weight_descriptors[shard];
        descriptor.master_expert_ptrs =
                reinterpret_cast<const std::uint64_t*>(
                        master_pointer_table.data_ptr());
        descriptor.home_buffer_offset_bytes =
                static_cast<std::uint64_t>(home_address - nvl_address);
        descriptor.home_slot_stride_bytes =
                static_cast<std::uint64_t>(home_stride_bytes);
        descriptor.replica_buffer_offset_bytes =
                static_cast<std::uint64_t>(replica_address - nvl_address);
        descriptor.replica_slot_stride_bytes =
                static_cast<std::uint64_t>(replica_stride_bytes);
        descriptor.replica_plan_stride_bytes =
                static_cast<std::uint64_t>(
                        kExpertPoolWeightSlotsPerPlan * replica_stride_bytes);
        EP_HOST_ASSERT(descriptor.replica_slot_stride_bytes % 16 == 0);
        EP_HOST_ASSERT(descriptor.replica_plan_stride_bytes % 16 == 0);
        EP_HOST_ASSERT((home.stride(0) * home.element_size()) % 16 == 0);
        descriptor.num_elements =
                static_cast<std::uint64_t>(
                        home.numel() / topology_.local_experts);
        descriptor.element_bytes =
                static_cast<std::uint32_t>(home.element_size());
        if (registered_weight_element_bytes == 0)
            registered_weight_element_bytes = descriptor.element_bytes;
        EP_HOST_ASSERT(descriptor.element_bytes ==
                       registered_weight_element_bytes);
        EP_HOST_ASSERT((descriptor.num_elements *
                        descriptor.element_bytes) % 16 == 0);
        registered_expert_weight_bytes += static_cast<std::int64_t>(
                descriptor.num_elements * descriptor.element_bytes);
        EP_HOST_ASSERT(reinterpret_cast<std::uintptr_t>(home.data_ptr()) % 16 == 0);
        EP_HOST_ASSERT(replica_address % 16 == 0);
    }
    weight_descriptors_ = copy_bytes_to_device(
            weight_descriptors.data(),
            weight_descriptors.size() * sizeof(WeightShardDescriptor),
            device_id_);
    EP_HOST_ASSERT(registered_expert_weight_bytes > 0);
    EP_HOST_ASSERT((registered_expert_weight_bytes +
                    probeep::kProbeWeightChunkBytes - 1) /
                           probeep::kProbeWeightChunkBytes <=
                   probeep::kProbeMaxChunksPerExpert);
    expert_weight_bytes_ = registered_expert_weight_bytes;

    std::vector<Fp32GradShardDescriptor> grad_descriptors(
            home_grad_shards_.size());
    for (std::size_t shard = 0; shard < grad_descriptors.size(); ++shard) {
        const auto& home = home_grad_shards_[shard];
        const auto& replica = replica_grad_shards_[shard];
        EP_HOST_ASSERT(home.is_cuda() && replica.is_cuda());
        EP_HOST_ASSERT(home.device().index() == device_id_ &&
                       replica.device().index() == device_id_);
        EP_HOST_ASSERT(home.scalar_type() == torch::kFloat32 &&
                       replica.scalar_type() == torch::kFloat32);
        EP_HOST_ASSERT(home.dim() >= 1 &&
                       home.size(0) == topology_.local_experts);
        EP_HOST_ASSERT(replica.dim() >= 1 &&
                       replica.size(0) == kExpertPoolReplicaSlots);
        EP_HOST_ASSERT(home.is_contiguous() && replica.is_contiguous());
        EP_HOST_ASSERT(home.numel() / topology_.local_experts ==
                       replica.numel() / kExpertPoolReplicaSlots);
        auto* replica_begin = static_cast<std::uint8_t*>(replica.data_ptr());
        auto* home_begin = static_cast<std::uint8_t*>(home.data_ptr());
        auto* nvl_begin = static_cast<std::uint8_t*>(local_nvl_base_);
        const auto home_address = reinterpret_cast<std::uintptr_t>(home_begin);
        const auto replica_address = reinterpret_cast<std::uintptr_t>(replica_begin);
        const auto nvl_address = reinterpret_cast<std::uintptr_t>(nvl_begin);
        EP_HOST_ASSERT(home_address >= nvl_address &&
                       home_address + home.nbytes() <=
                               nvl_address + transport_nvl_bytes_ +
                                       kIpcPlanReserveBytes + kExpertPoolBytes);
        EP_HOST_ASSERT(replica_address >= nvl_address &&
                       replica_address + replica.nbytes() <=
                               nvl_address + transport_nvl_bytes_ +
                                       kIpcPlanReserveBytes + kExpertPoolBytes);

        std::vector<std::uint64_t> master_pointers(topology_.local_experts);
        const auto home_stride_bytes = home.stride(0) * home.element_size();
        auto* home_base = static_cast<std::uint8_t*>(home.data_ptr());
        for (int expert = 0; expert < topology_.local_experts; ++expert)
            master_pointers[expert] = reinterpret_cast<std::uint64_t>(
                    home_base + expert * home_stride_bytes);
        auto master_pointer_table = copy_bytes_to_device(
                master_pointers.data(),
                master_pointers.size() * sizeof(std::uint64_t), device_id_);
        home_grad_pointer_tables_.push_back(master_pointer_table);

        auto& descriptor = grad_descriptors[shard];
        descriptor.master_expert_ptrs =
                reinterpret_cast<const std::uint64_t*>(
                        master_pointer_table.data_ptr());
        descriptor.home_buffer_offset_bytes =
                static_cast<std::uint64_t>(home_address - nvl_address);
        descriptor.home_slot_stride_bytes =
                static_cast<std::uint64_t>(home_stride_bytes);
        descriptor.replica_buffer_offset_bytes =
                static_cast<std::uint64_t>(replica_address - nvl_address);
        descriptor.replica_slot_stride_bytes =
                static_cast<std::uint64_t>(
                        replica.stride(0) * replica.element_size());
        EP_HOST_ASSERT(descriptor.replica_slot_stride_bytes % 16 == 0);
        EP_HOST_ASSERT((home.stride(0) * home.element_size()) % 16 == 0);
        descriptor.num_elements =
                static_cast<std::uint64_t>(
                        replica.numel() / kExpertPoolReplicaSlots);
        EP_HOST_ASSERT(descriptor.num_elements ==
                       weight_descriptors[shard].num_elements);
        EP_HOST_ASSERT(reinterpret_cast<std::uintptr_t>(home.data_ptr()) % 16 == 0);
        EP_HOST_ASSERT(replica_address % 16 == 0);
    }
    grad_descriptors_ = copy_bytes_to_device(
            grad_descriptors.data(),
            grad_descriptors.size() * sizeof(Fp32GradShardDescriptor),
            device_id_);
    EP_HOST_ASSERT(registered_weight_element_bytes > 0 &&
                   sizeof(float) % registered_weight_element_bytes == 0);
    grad_weight_byte_ratio_ = static_cast<int>(
            sizeof(float) / registered_weight_element_bytes);
    EP_HOST_ASSERT(grad_weight_byte_ratio_ > 0 &&
                   grad_weight_byte_ratio_ <=
                           probeep::kProbeMaxGradWeightByteRatio);
    for (auto& state : ring_) {
        CUDA_CHECK(cudaMemsetAsync(
                state.storage->cached_replica_expert.data_ptr(), 0xff,
                state.storage->cached_replica_expert.nbytes(), comm_stream_));
        state.cached_weight_version = INT64_MIN;
    }
}

void BalancedRuntime::set_weight_version(
        const BalancedHandle& handle, std::int64_t weight_version) {
    validate(handle);
    EP_HOST_ASSERT(weight_version >= -1);
    auto& state = ring_[handle.slot_];
    if (state.weight_sync_enqueued)
        return;
    state.requested_weight_version = weight_version;
}

void BalancedRuntime::launch_weight_sync(const BalancedHandle& handle,
                                         cudaStream_t stream) {
    validate(handle);
    EP_HOST_ASSERT(expert_pools_registered());
    auto& state = ring_[handle.slot_];
    if (state.weight_sync_enqueued)
        return;
    state.weight_sync_enqueued = true;
    const auto& slot = storage(handle);
    const int server = topology_.server(rank_);
    const auto* domain_replica_expert =
            slot.replica_expert.data_ptr<int>() +
            server * probeep::kRanksPerServer * probeep::kReplicaSlots;
    probeep::launch_prepare_weight_transfer(
            slot.replica_expert.data_ptr<int>(),
            slot.cached_replica_expert.data_ptr<int>(),
            slot.weight_transfer_required.data_ptr<int>(),
            topology_.world_size,
            state.requested_weight_version < 0 ||
                    state.requested_weight_version !=
                            state.cached_weight_version,
            stream);
    CUDA_CHECK(cudaEventRecord(state.weight_plan_ready_event, stream));
    CUDA_CHECK(cudaStreamWaitEvent(
            state.weight_progress_stream,
            state.weight_plan_ready_event, 0));
    launch_direct_replica_weight_copy(
            domain_replica_expert, peer_nvl_bases_,
            reinterpret_cast<const WeightShardDescriptor*>(
                    weight_descriptors_.data_ptr()),
            static_cast<int>(home_weight_shards_.size()), rank_,
            topology_.local_experts, probeep::kReplicaSlots,
            handle.slot() * probeep::kReplicaSlots,
            slot.weight_transfer_required.data_ptr<int>(),
            state.weight_progress_stream);
    probeep::launch_probeep_weight_receive(
            slot.probe_chunk_table.data_ptr<std::int64_t>(),
            slot.probe_plan_counts.data_ptr<int>(),
            slot.replica_expert.data_ptr<int>(), peer_nvl_bases_,
            slot.cached_replica_expert.data_ptr<int>(),
            slot.weight_transfer_required.data_ptr<int>(),
            reinterpret_cast<const WeightShardDescriptor*>(
                    weight_descriptors_.data_ptr()),
            static_cast<int>(home_weight_shards_.size()),
            transport_nvl_bytes_ + kIpcPlanReserveBytes,
            symmetric_plan_base_, handle.slot(), rank_,
            topology_.world_size, topology_.local_experts,
            expert_weight_bytes_,
            state.weight_progress_stream);
    CUDA_CHECK(cudaEventRecord(
            state.weight_consumer_ready_event,
            state.weight_progress_stream));
    state.cached_weight_version = state.requested_weight_version >= 0
            ? state.requested_weight_version : INT64_MIN;
}

void BalancedRuntime::launch_weight_send(const BalancedHandle& handle,
                                         cudaStream_t stream) {
    validate(handle);
    auto& state = ring_[handle.slot_];
    EP_HOST_ASSERT(state.weight_sync_enqueued);
    if (state.weight_send_enqueued)
        return;
    state.weight_send_enqueued = true;
    const auto& slot = storage(handle);
    probeep::launch_probeep_weight_send(
            slot.probe_chunk_table.data_ptr<std::int64_t>(),
            slot.probe_plan_counts.data_ptr<int>(),
            slot.weight_transfer_required.data_ptr<int>(), peer_nvl_bases_,
            reinterpret_cast<const WeightShardDescriptor*>(
                    weight_descriptors_.data_ptr()),
            static_cast<int>(home_weight_shards_.size()),
            transport_nvl_bytes_ + kIpcPlanReserveBytes,
            symmetric_plan_base_, handle.slot(), rank_,
            topology_.local_experts, stream);
}

void BalancedRuntime::join_weight_sync(const BalancedHandle& handle,
                                       cudaStream_t stream) {
    validate(handle);
    const auto& state = ring_[handle.slot_];
    EP_HOST_ASSERT(state.weight_sync_enqueued &&
                   state.weight_send_enqueued);
    // This join is deliberately after DeepEP Dispatch.  It is the Expert FFN
    // consumer dependency, not a Weight-before-Dispatch barrier.
    CUDA_CHECK(cudaStreamWaitEvent(
            stream, state.weight_consumer_ready_event, 0));
}

void BalancedRuntime::launch_grad_reduce(const BalancedHandle& handle,
                                         cudaStream_t stream) {
    const auto& slot = storage(handle);
    auto& state = ring_[handle.slot_];
    EP_HOST_ASSERT(state.backward_pending && !state.release_enqueued);
    CUDA_CHECK(cudaStreamWaitEvent(stream, state.reusable_event, 0));
    const int server = topology_.server(rank_);
    const auto* domain_replica_expert =
            slot.replica_expert.data_ptr<int>() +
            server * probeep::kRanksPerServer * probeep::kReplicaSlots;
    intranode::barrier(barrier_signal_ptrs_, topology_.lane(rank_),
                       probeep::kRanksPerServer, stream);
    launch_deterministic_fp32_replica_grad_reduce(
            domain_replica_expert, peer_nvl_bases_,
            reinterpret_cast<const Fp32GradShardDescriptor*>(
                    grad_descriptors_.data_ptr()),
            static_cast<int>(home_grad_shards_.size()), rank_,
            topology_.local_experts, probeep::kReplicaSlots,
            handle.slot() * probeep::kReplicaSlots, stream);
    intranode::barrier(barrier_signal_ptrs_, topology_.lane(rank_),
                       probeep::kRanksPerServer, stream);
    probeep::launch_probeep_grad_transport(
            slot.probe_chunk_table.data_ptr<std::int64_t>(),
            slot.probe_plan_counts.data_ptr<int>(),
            slot.probe_admitted_experts.data_ptr<int>(),
            slot.replica_expert.data_ptr<int>(), peer_nvl_bases_,
            barrier_signal_ptrs_,
            reinterpret_cast<const Fp32GradShardDescriptor*>(
                    grad_descriptors_.data_ptr()),
            static_cast<int>(home_grad_shards_.size()),
            transport_nvl_bytes_ + kIpcPlanReserveBytes,
            symmetric_plan_base_, handle.slot(), rank_,
            topology_.world_size, topology_.local_experts,
            expert_weight_bytes_, grad_weight_byte_ratio_, stream);
    CUDA_CHECK(cudaMemsetAsync(
            slot.cached_replica_expert.data_ptr(), 0xff,
            slot.cached_replica_expert.nbytes(), stream));
    state.backward_pending = false;
    release(handle, stream);
}

void BalancedRuntime::complete_backward(const BalancedHandle& handle,
                                        cudaStream_t stream) {
    validate(handle);
    auto& state = ring_[handle.slot_];
    EP_HOST_ASSERT(state.backward_pending && !state.release_enqueued);
    CUDA_CHECK(cudaStreamWaitEvent(stream, state.reusable_event, 0));
    CUDA_CHECK(cudaMemsetAsync(
            state.storage->cached_replica_expert.data_ptr(), 0xff,
            state.storage->cached_replica_expert.nbytes(), stream));
    state.backward_pending = false;
    release(handle, stream);
}

void BalancedRuntime::update_probe_controller(
        const torch::Tensor& compute_ns,
        const torch::Tensor& network_ns,
        const torch::Tensor& dispatch_tx_bytes,
        const torch::Tensor& dispatch_rx_bytes,
        const torch::Tensor& migration_tx_bytes,
        const torch::Tensor& migration_rx_bytes,
        int compute_kind,
        double rdma_path_bandwidth_gbps, double alpha,
        std::int64_t fallback_budget_bytes, bool valid,
        cudaStream_t stream) {
    EP_HOST_ASSERT(configured_);
    EP_HOST_ASSERT(compute_kind == 0 || compute_kind == 1);
    // Write directly into persistent runtime state: no temporary tensors, no
    // D2D copies and no host inspection in the sampled update path.
    auto budget = probe_migration_budget(compute_kind);
    auto summary = probe_controller_summary(compute_kind);
    probeep::update_controller_cuda(
            compute_ns, network_ns, dispatch_tx_bytes, dispatch_rx_bytes,
            migration_tx_bytes, migration_rx_bytes,
            budget, summary,
            rdma_path_bandwidth_gbps, alpha, fallback_budget_bytes, valid,
            true, stream);
}

torch::Tensor BalancedRuntime::probe_migration_budget(int compute_kind) const {
    EP_HOST_ASSERT(compute_kind == 0 || compute_kind == 1);
    return probe_migration_budget_.select(0, compute_kind);
}

torch::Tensor BalancedRuntime::probe_controller_summary(int compute_kind) const {
    EP_HOST_ASSERT(compute_kind == 0 || compute_kind == 1);
    return probe_controller_summary_.select(0, compute_kind);
}

void BalancedRuntime::reset_probe_controller(
        std::int64_t fallback_budget_bytes, cudaStream_t stream) {
    EP_HOST_ASSERT(configured_);
    EP_HOST_ASSERT(fallback_budget_bytes >= 0 &&
                   fallback_budget_bytes <=
                           probeep::kProbeMaxMigrationBytesPerEndpoint);
    const auto previous_stream = at::cuda::getCurrentCUDAStream(device_id_);
    at::cuda::setCurrentCUDAStream(
            at::cuda::getStreamFromExternal(stream, device_id_));
    probe_migration_budget_.fill_(fallback_budget_bytes);
    probe_controller_summary_.zero_();
    at::cuda::setCurrentCUDAStream(previous_stream);
}

} // namespace deep_ep::moonep

namespace deep_ep {

void Buffer::register_balanced_expert_pools(
        const std::vector<torch::Tensor>& home_weight_shards,
        const std::vector<torch::Tensor>& replica_weight_shards,
        const std::vector<torch::Tensor>& home_grad_shards,
        const std::vector<torch::Tensor>& replica_grad_shards) {
    balanced_runtime->register_expert_pools(
            home_weight_shards, replica_weight_shards,
            home_grad_shards, replica_grad_shards);
}

std::optional<EventHandle> Buffer::balanced_weight_sync(
        const moonep::BalancedHandle& handle,
        std::optional<EventHandle>& previous_event, bool async) {
    const auto compute_stream = at::cuda::getCurrentCUDAStream();
    if (previous_event.has_value())
        stream_wait(comm_stream, previous_event.value());
    else
        stream_wait(comm_stream, compute_stream);

    balanced_runtime->set_weight_version(handle, -1);
    balanced_runtime->launch_weight_sync(handle, comm_stream);
    balanced_runtime->launch_weight_send(handle, comm_stream);
    balanced_runtime->join_weight_sync(handle, comm_stream);
    if (async)
        return EventHandle(comm_stream);

    stream_wait(compute_stream, comm_stream);
    return std::nullopt;
}

std::optional<EventHandle> Buffer::balanced_grad_reduce(
        const moonep::BalancedHandle& handle,
        std::optional<EventHandle>& previous_event, bool async) {
    const auto compute_stream = at::cuda::getCurrentCUDAStream();
    if (previous_event.has_value())
        stream_wait(comm_stream, previous_event.value());
    else
        stream_wait(comm_stream, compute_stream);

    balanced_runtime->launch_grad_reduce(handle, comm_stream);
    if (async)
        return EventHandle(comm_stream);

    stream_wait(compute_stream, comm_stream);
    return std::nullopt;
}

std::optional<EventHandle> Buffer::balanced_finish_backward(
        const moonep::BalancedHandle& handle,
        std::optional<EventHandle>& previous_event, bool async) {
    const auto compute_stream = at::cuda::getCurrentCUDAStream();
    if (previous_event.has_value())
        stream_wait(comm_stream, previous_event.value());
    else
        stream_wait(comm_stream, compute_stream);

    balanced_runtime->complete_backward(handle, comm_stream);
    if (async)
        return EventHandle(comm_stream);

    stream_wait(compute_stream, comm_stream);
    return std::nullopt;
}

} // namespace deep_ep

namespace deep_ep::moonep {

void bind_balanced_handle(pybind11::module_& module) {
    module.def("bf16_grouped_mm_out", &bf16_grouped_mm_out,
               pybind11::arg("mat_a"), pybind11::arg("mat_b"),
               pybind11::arg("offsets"), pybind11::arg("output"));
    pybind11::class_<BalancedHandle>(module, "BalancedHandle")
        .def_property_readonly("slot", &BalancedHandle::slot)
        .def_property_readonly("generation", &BalancedHandle::generation)
        .def_property_readonly("num_tokens", &BalancedHandle::num_tokens)
        .def_property_readonly("route_dst", &BalancedHandle::route_dst)
        .def_property_readonly("exec_rank", &BalancedHandle::exec_rank)
        .def_property_readonly("exec_slot", &BalancedHandle::exec_slot)
        .def_property_readonly("is_token_in_rank", &BalancedHandle::is_token_in_rank)
        .def_property_readonly("slot_count", &BalancedHandle::slot_count)
        .def_property_readonly("slot_begin", &BalancedHandle::slot_begin)
        .def_property_readonly("replica_expert", &BalancedHandle::replica_expert)
        .def_property_readonly("slot_expert", &BalancedHandle::slot_expert)
        .def_property_readonly("num_tokens_per_rank", &BalancedHandle::num_tokens_per_rank)
        .def_property_readonly("num_tokens_per_rdma_rank", &BalancedHandle::num_tokens_per_rdma_rank)
        .def_property_readonly("num_tokens_per_exec_slot", &BalancedHandle::num_tokens_per_exec_slot)
        .def_property_readonly("probe_server_load_before", &BalancedHandle::probe_server_load_before)
        .def_property_readonly("probe_server_load_after", &BalancedHandle::probe_server_load_after)
        .def_property_readonly("probe_server_padded_load_before", &BalancedHandle::probe_server_padded_load_before)
        .def_property_readonly("probe_server_padded_load_after", &BalancedHandle::probe_server_padded_load_after)
        .def_property_readonly("probe_assigned_tx_bytes", &BalancedHandle::probe_assigned_tx_bytes)
        .def_property_readonly("probe_assigned_rx_bytes", &BalancedHandle::probe_assigned_rx_bytes)
        .def_property_readonly("probe_dispatch_tx_bytes", &BalancedHandle::probe_dispatch_tx_bytes)
        .def_property_readonly("probe_dispatch_rx_bytes", &BalancedHandle::probe_dispatch_rx_bytes)
        .def_property_readonly("probe_pair_load_bytes", &BalancedHandle::probe_pair_load_bytes)
        .def_property_readonly("probe_compute_intents", &BalancedHandle::probe_compute_intents)
        .def_property_readonly("probe_migration_budget_snapshot", &BalancedHandle::probe_migration_budget_snapshot)
        .def_property_readonly("probe_endpoint_total_cap_bytes", &BalancedHandle::probe_endpoint_total_cap_bytes)
        .def_property_readonly("probe_admitted_experts", &BalancedHandle::probe_admitted_experts)
        .def_property_readonly("probe_deferred_experts", &BalancedHandle::probe_deferred_experts)
        .def_property_readonly("probe_chunk_table", &BalancedHandle::probe_chunk_table)
        .def_property_readonly("probe_plan_counts", &BalancedHandle::probe_plan_counts)
        .def_property_readonly("weight_transfer_required", &BalancedHandle::weight_transfer_required)
        .def_property_readonly("exec_x", &BalancedHandle::exec_x)
        .def_property_readonly("exec_x_scales", &BalancedHandle::exec_x_scales)
        .def_property_readonly("exec_weights", &BalancedHandle::exec_weights)
        .def_property_readonly("exec_y", &BalancedHandle::exec_y)
        .def_property_readonly("combined_x", &BalancedHandle::combined_x)
        .def_property_readonly("recv_count", &BalancedHandle::recv_count);
}

void bind_balanced_buffer(pybind11::class_<Buffer>& buffer_class) {
    namespace py = pybind11;
    buffer_class
        .def("configure_balanced", &Buffer::configure_balanced)
        .def("reset_balanced_probe_controller",
             &Buffer::reset_balanced_probe_controller,
             py::arg("fallback_budget_bytes") = 32LL * 1024 * 1024)
        .def("get_balanced_expert_pool_views",
             &Buffer::get_balanced_expert_pool_views)
        .def("balanced_dispatch", &Buffer::balanced_dispatch,
             py::arg("x"), py::arg("x_scales"), py::arg("topk_idx"),
             py::arg("topk_weights"), py::arg("config"),
             py::arg("compute_kind"),
             py::arg("previous_event"), py::arg("async_finish"),
             py::arg("compute_ns") = std::nullopt,
             py::arg("network_ns") = std::nullopt,
             py::arg("dispatch_tx_bytes") = std::nullopt,
             py::arg("dispatch_rx_bytes") = std::nullopt,
             py::arg("migration_tx_bytes") = std::nullopt,
             py::arg("migration_rx_bytes") = std::nullopt,
             py::arg("feedback_valid") = false,
             py::arg("rdma_path_bandwidth_gbps") = 200.0,
             py::arg("controller_alpha") = 0.90,
             py::arg("fallback_budget_bytes") = 0,
             py::arg("expert_weight_version") = -1)
        .def("balanced_combine", &Buffer::balanced_combine,
             py::arg("x"), py::arg("handle"), py::arg("config"),
             py::arg("previous_event"), py::arg("async_finish"),
             py::arg("release_after_combine"))
        .def("balanced_dispatch_backward",
             &Buffer::balanced_dispatch_backward,
             py::arg("grad_out"), py::arg("handle"), py::arg("config"),
             py::arg("previous_event"), py::arg("async_finish"))
        .def("balanced_combine_backward",
             &Buffer::balanced_combine_backward,
             py::arg("exec_grad_x"), py::arg("handle"),
             py::arg("config"), py::arg("previous_event"),
             py::arg("async_finish"))
        .def("register_balanced_expert_pools",
             &Buffer::register_balanced_expert_pools,
             py::arg("home_weight_shards"), py::arg("replica_weight_shards"),
             py::arg("home_grad_shards"), py::arg("replica_grad_shards"))
        .def("balanced_weight_sync", &Buffer::balanced_weight_sync,
             py::arg("handle"), py::arg("previous_event"),
             py::arg("async_finish"))
        .def("balanced_grad_reduce", &Buffer::balanced_grad_reduce,
             py::arg("handle"), py::arg("previous_event"),
             py::arg("async_finish"))
        .def("balanced_finish_backward", &Buffer::balanced_finish_backward,
             py::arg("handle"), py::arg("previous_event"),
             py::arg("async_finish"));
}

} // namespace deep_ep::moonep
