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

std::shared_ptr<BalancedSlotStorage> allocate_slot(int device_id,
                                                   int source_meta_bytes) {
    auto slot = std::make_shared<BalancedSlotStorage>();
    const auto device = torch::Device(torch::kCUDA, device_id);
    const auto ints = torch::TensorOptions().dtype(torch::kInt32).device(device);
    const auto bools = torch::TensorOptions().dtype(torch::kBool).device(device);
    const auto bytes = torch::TensorOptions().dtype(torch::kUInt8).device(device);
    const auto floats = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    const auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(device);
    const auto fp8 = torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(device);

    slot->local_histogram = torch::empty({kNumExperts}, ints);
    slot->local_ordinal = torch::empty({kMaxAssignmentsPerRank}, ints);
    slot->tokens_per_expert_prefix = torch::empty({kWorldSize, kNumExperts}, ints);
    slot->alloc_prefix = torch::empty({kNumExperts, kWorldSize}, ints);
    slot->expert_slot = torch::empty({kWorldSize, kNumExperts}, ints);
    slot->route_dst = torch::empty({kMaxTokensPerRank, kTopK}, ints);
    slot->exec_rank = torch::empty({kMaxTokensPerRank, kTopK}, ints);
    slot->exec_slot = torch::empty({kMaxTokensPerRank, kTopK}, ints);
    slot->is_token_in_rank = torch::empty({kMaxTokensPerRank, kWorldSize}, bools);
    slot->slot_count = torch::empty({kWorldSize, kExecutionSlots}, ints);
    slot->slot_begin = torch::empty({kWorldSize, kExecutionSlots}, ints);
    slot->replica_expert = torch::empty({kWorldSize, kReplicaSlots}, ints);
    slot->slot_expert = torch::empty({kWorldSize, kExecutionSlots}, ints);
    slot->num_tokens_per_rank = torch::empty({kWorldSize}, ints);
    slot->num_tokens_per_rdma_rank = torch::empty({kNumServers}, ints);
    slot->num_tokens_per_exec_slot =
            torch::empty({kWorldSize * kExecutionSlots}, ints);

    slot->exec_x = torch::empty({kMaxExecutionRows, kHidden}, fp8);
    slot->exec_x_scales = torch::empty({kMaxExecutionRows, kFp8Scales}, floats);
    slot->exec_weights = torch::empty({kMaxExecutionRows}, floats);
    slot->exec_y = torch::empty({kMaxExecutionRows, kHidden}, bf16);
    slot->combined_x = torch::empty({kMaxTokensPerRank, kHidden}, bf16);
    slot->combined_topk_weights =
            torch::empty({kMaxTokensPerRank, kTopK}, floats);

    slot->recv_route_rows = torch::empty({kMaxTransportRows, kTopK}, ints);
    slot->recv_src_meta =
            torch::empty({kMaxTransportRows, source_meta_bytes}, bytes);
    slot->rdma_channel_prefix_matrix =
            torch::empty({kNumServers, kNumChannels}, ints);
    slot->recv_rdma_channel_prefix_matrix =
            torch::empty({kNumServers, kNumChannels}, ints);
    slot->recv_rdma_rank_prefix_sum = torch::empty({kNumServers}, ints);
    slot->gbl_channel_prefix_matrix =
            torch::empty({kWorldSize, kNumChannels}, ints);
    slot->recv_gbl_channel_prefix_matrix =
            torch::empty({kWorldSize, kNumChannels}, ints);
    slot->recv_gbl_rank_prefix_sum = torch::empty({kWorldSize}, ints);
    slot->send_rdma_head =
            torch::empty({kMaxTokensPerRank, kNumServers}, ints);
    slot->send_nvl_head =
            torch::empty({kMaxTransportRows, kRanksPerServer}, ints);
    slot->recv_count = torch::empty({1}, ints);
    slot->recv_rdma_count = torch::empty({1}, ints);
    return slot;
}

} // namespace

BalancedRuntime::~BalancedRuntime() {
    for (auto& state : ring_) {
        if (state.reusable_event != nullptr)
            cudaEventDestroy(state.reusable_event);
    }
}

void BalancedRuntime::configure(int rank, int device_id, int source_meta_bytes,
                                int64_t transport_nvl_bytes,
                                void* local_nvl_base, void** peer_nvl_bases,
                                int** barrier_signal_ptrs,
                                void* symmetric_plan_base,
                                const at::cuda::CUDAStream& comm_stream) {
    EP_HOST_ASSERT(!configured_);
    EP_HOST_ASSERT(rank >= 0 && rank < kWorldSize);
    EP_HOST_ASSERT(source_meta_bytes > 0);

    rank_ = rank;
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
    transport_y_ = torch::empty(
            {kMaxTransportRows, kHidden},
            torch::TensorOptions().dtype(torch::kBFloat16).device(
                    torch::Device(torch::kCUDA, device_id_)));
    for (auto& state : ring_) {
        state.storage = allocate_slot(device_id_, source_meta_bytes_);
        CUDA_CHECK(cudaEventCreateWithFlags(
                &state.reusable_event, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(state.reusable_event, comm_stream_));
    }
    at::cuda::setCurrentCUDAStream(previous_stream);
    configured_ = true;
}

BalancedHandle BalancedRuntime::acquire(int num_tokens) {
    EP_HOST_ASSERT(configured_);
    EP_HOST_ASSERT(num_tokens > 0 && num_tokens <= kMaxTokensPerRank);

    const int slot = next_slot_;
    auto& state = ring_[slot];
    EP_HOST_ASSERT(state.release_enqueued);
    CUDA_CHECK(cudaStreamWaitEvent(comm_stream_, state.reusable_event, 0));
    state.release_enqueued = false;
    ++state.generation;
    next_slot_ = (next_slot_ + 1) % kPlanRingSlots;

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
    EP_HOST_ASSERT(handle.slot_ >= 0 && handle.slot_ < kPlanRingSlots);
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
    for (std::size_t shard = 0; shard < weight_descriptors.size(); ++shard) {
        const auto& home = home_weight_shards_[shard];
        const auto& replica = replica_weight_shards_[shard];
        EP_HOST_ASSERT(home.is_cuda() && replica.is_cuda());
        EP_HOST_ASSERT(home.device().index() == device_id_ &&
                       replica.device().index() == device_id_);
        EP_HOST_ASSERT(home.dim() >= 1 && home.size(0) == kLocalExperts);
        EP_HOST_ASSERT(replica.dim() >= 1 && replica.size(0) == kReplicaSlots);
        EP_HOST_ASSERT(home.is_contiguous() && replica.is_contiguous());
        EP_HOST_ASSERT(home.scalar_type() == replica.scalar_type());
        EP_HOST_ASSERT(home.numel() / kLocalExperts ==
                       replica.numel() / kReplicaSlots);
        auto* replica_begin = static_cast<std::uint8_t*>(replica.data_ptr());
        auto* nvl_begin = static_cast<std::uint8_t*>(local_nvl_base_);
        const auto replica_address = reinterpret_cast<std::uintptr_t>(replica_begin);
        const auto nvl_address = reinterpret_cast<std::uintptr_t>(nvl_begin);
        EP_HOST_ASSERT(replica_address >= nvl_address &&
                       replica_address + replica.nbytes() <=
                               nvl_address + transport_nvl_bytes_ +
                                       kIpcPlanReserveBytes + kExpertPoolBytes);

        std::vector<std::uint64_t> master_pointers(kLocalExperts);
        const auto home_stride_bytes = home.stride(0) * home.element_size();
        auto* home_base = static_cast<std::uint8_t*>(home.data_ptr());
        for (int expert = 0; expert < kLocalExperts; ++expert)
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
        descriptor.replica_buffer_offset_bytes =
                static_cast<std::uint64_t>(replica_address - nvl_address);
        descriptor.replica_slot_stride_bytes =
                static_cast<std::uint64_t>(
                        replica.stride(0) * replica.element_size());
        EP_HOST_ASSERT(descriptor.replica_slot_stride_bytes % 16 == 0);
        EP_HOST_ASSERT((home.stride(0) * home.element_size()) % 16 == 0);
        descriptor.num_elements =
                static_cast<std::uint64_t>(replica.numel() / kReplicaSlots);
        descriptor.element_bytes =
                static_cast<std::uint32_t>(home.element_size());
        EP_HOST_ASSERT(reinterpret_cast<std::uintptr_t>(home.data_ptr()) % 16 == 0);
        EP_HOST_ASSERT(replica_address % 16 == 0);
    }
    weight_descriptors_ = copy_bytes_to_device(
            weight_descriptors.data(),
            weight_descriptors.size() * sizeof(WeightShardDescriptor),
            device_id_);

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
        EP_HOST_ASSERT(home.dim() >= 1 && home.size(0) == kLocalExperts);
        EP_HOST_ASSERT(replica.dim() >= 1 && replica.size(0) == kReplicaSlots);
        EP_HOST_ASSERT(home.is_contiguous() && replica.is_contiguous());
        EP_HOST_ASSERT(home.numel() / kLocalExperts ==
                       replica.numel() / kReplicaSlots);
        auto* replica_begin = static_cast<std::uint8_t*>(replica.data_ptr());
        auto* nvl_begin = static_cast<std::uint8_t*>(local_nvl_base_);
        const auto replica_address = reinterpret_cast<std::uintptr_t>(replica_begin);
        const auto nvl_address = reinterpret_cast<std::uintptr_t>(nvl_begin);
        EP_HOST_ASSERT(replica_address >= nvl_address &&
                       replica_address + replica.nbytes() <=
                               nvl_address + transport_nvl_bytes_ +
                                       kIpcPlanReserveBytes + kExpertPoolBytes);

        std::vector<std::uint64_t> master_pointers(kLocalExperts);
        const auto home_stride_bytes = home.stride(0) * home.element_size();
        auto* home_base = static_cast<std::uint8_t*>(home.data_ptr());
        for (int expert = 0; expert < kLocalExperts; ++expert)
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
        descriptor.replica_buffer_offset_bytes =
                static_cast<std::uint64_t>(replica_address - nvl_address);
        descriptor.replica_slot_stride_bytes =
                static_cast<std::uint64_t>(
                        replica.stride(0) * replica.element_size());
        EP_HOST_ASSERT(descriptor.replica_slot_stride_bytes % 16 == 0);
        EP_HOST_ASSERT((home.stride(0) * home.element_size()) % 16 == 0);
        descriptor.num_elements =
                static_cast<std::uint64_t>(replica.numel() / kReplicaSlots);
        EP_HOST_ASSERT(reinterpret_cast<std::uintptr_t>(home.data_ptr()) % 16 == 0);
        EP_HOST_ASSERT(replica_address % 16 == 0);
    }
    grad_descriptors_ = copy_bytes_to_device(
            grad_descriptors.data(),
            grad_descriptors.size() * sizeof(Fp32GradShardDescriptor),
            device_id_);
}

void BalancedRuntime::launch_weight_sync(const BalancedHandle& handle,
                                         cudaStream_t stream) {
    const auto& slot = storage(handle);
    const int server = rank_ / kRanksPerServer;
    const auto* domain_replica_expert =
            slot.replica_expert.data_ptr<int>() +
            server * kRanksPerServer * kReplicaSlots;
    launch_direct_replica_weight_copy(
            domain_replica_expert, peer_nvl_bases_,
            reinterpret_cast<const WeightShardDescriptor*>(
                    weight_descriptors_.data_ptr()),
            static_cast<int>(home_weight_shards_.size()), rank_,
            kLocalExperts, kReplicaSlots, stream);
    intranode::barrier(barrier_signal_ptrs_, rank_ % kRanksPerServer,
                       kRanksPerServer, stream);
}

void BalancedRuntime::launch_grad_reduce(const BalancedHandle& handle,
                                         cudaStream_t stream) {
    const auto& slot = storage(handle);
    auto& state = ring_[handle.slot_];
    EP_HOST_ASSERT(state.backward_pending && !state.release_enqueued);
    CUDA_CHECK(cudaStreamWaitEvent(stream, state.reusable_event, 0));
    const int server = rank_ / kRanksPerServer;
    const auto* domain_replica_expert =
            slot.replica_expert.data_ptr<int>() +
            server * kRanksPerServer * kReplicaSlots;
    intranode::barrier(barrier_signal_ptrs_, rank_ % kRanksPerServer,
                       kRanksPerServer, stream);
    launch_deterministic_fp32_replica_grad_reduce(
            domain_replica_expert, peer_nvl_bases_,
            reinterpret_cast<const Fp32GradShardDescriptor*>(
                    grad_descriptors_.data_ptr()),
            static_cast<int>(home_grad_shards_.size()), rank_,
            kLocalExperts, kReplicaSlots, stream);
    intranode::barrier(barrier_signal_ptrs_, rank_ % kRanksPerServer,
                       kRanksPerServer, stream);
    state.backward_pending = false;
    release(handle, stream);
}

void BalancedRuntime::complete_backward(const BalancedHandle& handle,
                                        cudaStream_t stream) {
    validate(handle);
    auto& state = ring_[handle.slot_];
    EP_HOST_ASSERT(state.backward_pending && !state.release_enqueued);
    CUDA_CHECK(cudaStreamWaitEvent(stream, state.reusable_event, 0));
    state.backward_pending = false;
    release(handle, stream);
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

    balanced_runtime->launch_weight_sync(handle, comm_stream);
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
        .def("get_balanced_expert_pool_views",
             &Buffer::get_balanced_expert_pool_views)
        .def("balanced_dispatch", &Buffer::balanced_dispatch,
             py::arg("x"), py::arg("x_scales"), py::arg("topk_idx"),
             py::arg("topk_weights"), py::arg("config"),
             py::arg("previous_event"), py::arg("async_finish"))
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
