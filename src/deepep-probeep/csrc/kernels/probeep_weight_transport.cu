#include "configs.cuh"
#include "probeep_weight_transport.cuh"

#include "../moonep_expert_pool.hpp"
#include "../moonep_runtime.hpp"
#include "api.cuh"
#include "ibgda_device.cuh"
#include "utils.cuh"

#include <cstdint>

namespace deep_ep::probeep {
namespace {

constexpr int kThreads = 256;
// Each process owns one rail/QP.  A single batched CTA serializes that QP's
// put/quiet/AMO completion state while still processing the full device-side
// chunk table without per-chunk host launches.
constexpr int kTransportCtas = 1;
constexpr int kReceiveCtas = 1;

__global__ void prepare_weight_transfer_kernel(
        const int* replica_expert,
        const int* cached_replica_expert,
        int* transfer_required,
        int world_size,
        bool force_transfer) {
    if (force_transfer) {
        if (threadIdx.x == 0)
            *transfer_required = 1;
        return;
    }
    for (int index = static_cast<int>(threadIdx.x);
         index < world_size * kReplicaSlots; index += blockDim.x)
        if (replica_expert[index] != cached_replica_expert[index])
            atomicExch(transfer_required, 1);
}

__global__ void publish_weight_layout_kernel(
        const int* replica_expert,
        int* cached_replica_expert,
        int world_size) {
    for (int index = static_cast<int>(threadIdx.x);
         index < world_size * kReplicaSlots; index += blockDim.x)
        cached_replica_expert[index] = replica_expert[index];
}

__global__ void conditional_weight_barrier_kernel(
        void** peer_nvl_bases,
        std::int64_t plan_reserve_offset,
        const int* transfer_required,
        int plan_slot,
        int phase) {
    if (*transfer_required == 0)
        return;
    auto* reserve = reinterpret_cast<moonep::IpcPlanReserve*>(
            static_cast<std::uint8_t*>(peer_nvl_bases[0]) +
            plan_reserve_offset);
    if (threadIdx.x == 0) {
        int* ticket = &reserve->probe_weight_barrier_ticket[plan_slot][phase];
        const int arrival = atomicAdd_system(ticket, 1);
        const int target = (arrival / kRanksPerServer + 1) * kRanksPerServer;
        while (ld_volatile_global(ticket) < target) {}
    }
}

__device__ __forceinline__ void copy_registered_weight_range(
        void* rank_base,
        const moonep::WeightShardDescriptor* shards,
        int num_shards,
        int local_slot,
        std::int64_t logical_offset,
        std::int64_t bytes,
        std::uint8_t* staging,
        bool from_home) {
    std::int64_t shard_begin = 0;
    const std::int64_t logical_end = logical_offset + bytes;
    for (int shard_index = 0; shard_index < num_shards; ++shard_index) {
        const auto shard = shards[shard_index];
        const std::int64_t shard_bytes = static_cast<std::int64_t>(
                shard.num_elements * shard.element_bytes);
        const std::int64_t shard_end = shard_begin + shard_bytes;
        const std::int64_t begin = max(logical_offset, shard_begin);
        const std::int64_t end = min(logical_end, shard_end);
        if (begin < end) {
            const std::int64_t within_shard = begin - shard_begin;
            const std::int64_t within_staging = begin - logical_offset;
            const std::int64_t range_bytes = end - begin;
            const auto base_offset = from_home
                    ? shard.home_buffer_offset_bytes
                    : shard.replica_buffer_offset_bytes;
            const auto slot_stride = from_home
                    ? shard.home_slot_stride_bytes
                    : shard.replica_slot_stride_bytes;
            const auto plan_offset = from_home
                    ? 0
                    : static_cast<std::int64_t>(
                              local_slot / kReplicaSlots) *
                              shard.replica_plan_stride_bytes;
            const auto slot_in_plan = from_home
                    ? local_slot
                    : local_slot % kReplicaSlots;
            auto* registered = static_cast<std::uint8_t*>(rank_base) +
                    base_offset + plan_offset +
                    static_cast<std::int64_t>(slot_in_plan) * slot_stride +
                    within_shard;
            auto* stage = staging + within_staging;
            const std::int64_t vector_bytes =
                    range_bytes / sizeof(uint4) * sizeof(uint4);
            for (std::int64_t index = threadIdx.x * sizeof(uint4);
                 index < vector_bytes;
                 index += blockDim.x * sizeof(uint4)) {
                if (from_home)
                    *reinterpret_cast<uint4*>(stage + index) =
                            *reinterpret_cast<const uint4*>(registered + index);
                else
                    *reinterpret_cast<uint4*>(registered + index) =
                            *reinterpret_cast<const uint4*>(stage + index);
            }
            for (std::int64_t index = vector_bytes + threadIdx.x;
                 index < range_bytes; index += blockDim.x) {
                if (from_home)
                    stage[index] = registered[index];
                else
                    registered[index] = stage[index];
            }
        }
        __syncthreads();
        shard_begin = shard_end;
    }
}

__global__ void reset_probe_weight_state(
        void** peer_nvl_bases,
        std::int64_t plan_reserve_offset,
        const int* transfer_required,
        int plan_slot,
        int nvl_rank) {
    if (transfer_required != nullptr && *transfer_required == 0)
        return;
    auto* reserve = reinterpret_cast<moonep::IpcPlanReserve*>(
            static_cast<std::uint8_t*>(peer_nvl_bases[nvl_rank]) +
            plan_reserve_offset);
    for (int slot = threadIdx.x; slot < kReplicaSlots; slot += blockDim.x) {
        reserve->probe_replica_chunk_count[plan_slot][slot] = 0;
        reserve->probe_replica_ready[plan_slot][slot] = 0;
    }
    for (int local = threadIdx.x; local < kMaxLocalExperts;
         local += blockDim.x) {
        reserve->probe_owner_grad_chunk_count[plan_slot][local] = 0;
        reserve->probe_owner_grad_ready[plan_slot][local] = 0;
    }
}

template <bool kSender>
__device__ __forceinline__ void probe_weight_chunk(
        const std::int64_t* chunk_table,
        const int* plan_counts,
        const int* replica_expert,
        void** peer_nvl_bases,
        const moonep::WeightShardDescriptor* weight_shards,
        int num_weight_shards,
        std::uint8_t* tx_staging,
        std::uint8_t* rx_staging,
        std::uint64_t* transfer_signal,
        int* signal_expected,
        std::int64_t plan_reserve_offset,
        const int* transfer_required,
        int plan_slot,
        int global_rank,
        int local_experts,
        int chunks_per_expert,
        int chunk) {
    if (*transfer_required == 0)
        return;
    const auto* row = chunk_table +
            static_cast<std::int64_t>(chunk) * kProbeChunkFields;
    const int expert = static_cast<int>(row[0]);
    const int source_rank = static_cast<int>(row[6]);
    const int destination_rank = static_cast<int>(row[7]);
    const std::int64_t expert_offset = row[8];
    const std::int64_t bytes = row[9];
    const std::int64_t source_path_offset = row[11];
    const std::int64_t destination_path_offset = row[12];
    if (source_path_offset < 0 || destination_path_offset < 0 ||
        source_path_offset + bytes > kProbeRdmaPathStagingBytes ||
        destination_path_offset + bytes > kProbeRdmaPathStagingBytes)
        return;

    const int destination_server = static_cast<int>(row[5]);
    const auto plan_staging = static_cast<std::int64_t>(plan_slot) *
                              kProbeRdmaPathStagingBytes;
    std::uint64_t* local_signal = transfer_signal +
            plan_slot * kProbeMaxTransfers + chunk;
    int* local_signal_expected = kSender ? nullptr :
            signal_expected + plan_slot * kProbeMaxTransfers + chunk;

    if constexpr (kSender) {
        if (global_rank != source_rank)
            return;
        auto* local_staging = tx_staging + plan_staging + source_path_offset;
        auto* remote_staging = rx_staging + plan_staging +
                               destination_path_offset;
        const int owner = expert / local_experts;
        const int owner_lane = owner % kRanksPerServer;
        const int local_expert = expert % local_experts;
        copy_registered_weight_range(
                peer_nvl_bases[owner_lane], weight_shards,
                num_weight_shards, local_expert, expert_offset, bytes,
                local_staging, true);
        nvshmemx_putmem_signal_nbi_block(
                remote_staging, local_staging,
                static_cast<std::size_t>(bytes), local_signal, 1,
                NVSHMEM_SIGNAL_ADD, destination_server);
        return;
    }

    if (global_rank != destination_rank)
        return;
    auto* local_staging = rx_staging + plan_staging + destination_path_offset;
    if (threadIdx.x == 0) {
        const int expected = atomicAdd(local_signal_expected, 1) + 1;
        while (ld_acquire_sys_global(local_signal) <
               static_cast<std::uint64_t>(expected)) {}
    }
    __syncthreads();

    const int local_destination_server = global_rank / kRanksPerServer;
    for (int local = 0; local < kRanksPerServer; ++local) {
        const int target_rank =
                local_destination_server * kRanksPerServer + local;
        for (int replica_slot = 0; replica_slot < kReplicaSlots;
             ++replica_slot) {
            if (replica_expert[target_rank * kReplicaSlots + replica_slot] !=
                expert)
                continue;
            copy_registered_weight_range(
                    peer_nvl_bases[local], weight_shards,
                    num_weight_shards,
                    plan_slot * kReplicaSlots + replica_slot,
                    expert_offset, bytes, local_staging, false);
            if (threadIdx.x == 0) {
                auto* target_reserve =
                        reinterpret_cast<moonep::IpcPlanReserve*>(
                            static_cast<std::uint8_t*>(peer_nvl_bases[local]) +
                            plan_reserve_offset);
                const int completed = atomicAdd_system(
                        &target_reserve->probe_replica_chunk_count
                                [plan_slot][replica_slot], 1) + 1;
                if (completed == chunks_per_expert) {
                    __threadfence_system();
                    atomicExch_system(
                            &target_reserve->probe_replica_ready
                                    [plan_slot][replica_slot], 1);
                }
            }
            __syncthreads();
        }
    }
}

template <bool kSender>
__global__ void probe_weight_chunks(
        const std::int64_t* chunk_table,
        const int* plan_counts,
        const int* replica_expert,
        void** peer_nvl_bases,
        const moonep::WeightShardDescriptor* weight_shards,
        int num_weight_shards,
        std::uint8_t* tx_staging,
        std::uint8_t* rx_staging,
        std::uint64_t* transfer_signal,
        int* signal_expected,
        std::int64_t plan_reserve_offset,
        const int* transfer_required,
        int plan_slot,
        int global_rank,
        int local_experts,
        int chunks_per_expert) {
    for (int chunk = static_cast<int>(blockIdx.x);
         chunk < plan_counts[1]; chunk += static_cast<int>(gridDim.x))
        probe_weight_chunk<kSender>(
                chunk_table, plan_counts, replica_expert, peer_nvl_bases,
                weight_shards, num_weight_shards, tx_staging, rx_staging,
                transfer_signal, signal_expected, plan_reserve_offset,
                transfer_required, plan_slot, global_rank, local_experts,
                chunks_per_expert, chunk);
}

__device__ __forceinline__ void pack_registered_grad_range(
        const moonep::Fp32GradShardDescriptor* shards,
        int num_shards,
        const int* replica_expert,
        void** peer_nvl_bases,
        int destination_server,
        int expert,
        int plan_slot,
        std::int64_t logical_offset,
        std::int64_t bytes,
        std::uint8_t* staging) {
    std::int64_t shard_begin = 0;
    const std::int64_t logical_end = logical_offset + bytes;
    for (int shard_index = 0; shard_index < num_shards; ++shard_index) {
        const auto shard = shards[shard_index];
        const std::int64_t shard_bytes = static_cast<std::int64_t>(
                shard.num_elements * sizeof(float));
        const std::int64_t shard_end = shard_begin + shard_bytes;
        const std::int64_t begin = max(logical_offset, shard_begin);
        const std::int64_t end = min(logical_end, shard_end);
        const std::int64_t range_bytes = end > begin ? end - begin : 0;
        const std::int64_t elements = range_bytes / sizeof(float);
        const std::int64_t within_shard = begin - shard_begin;
        const std::int64_t within_staging = begin - logical_offset;
        auto* packed = reinterpret_cast<float*>(staging + within_staging);
        for (std::int64_t index = threadIdx.x; index < elements;
             index += blockDim.x) {
            float sum = 0.0f;
            for (int local = 0; local < kRanksPerServer; ++local) {
                const int target_rank =
                        destination_server * kRanksPerServer + local;
                for (int replica_slot = 0; replica_slot < kReplicaSlots;
                     ++replica_slot) {
                    if (replica_expert[target_rank * kReplicaSlots +
                                       replica_slot] != expert)
                        continue;
                    auto* replica = reinterpret_cast<float*>(
                            static_cast<std::uint8_t*>(peer_nvl_bases[local]) +
                            shard.replica_buffer_offset_bytes +
                            static_cast<std::int64_t>(
                                    plan_slot * kReplicaSlots + replica_slot) *
                                    shard.replica_slot_stride_bytes +
                            within_shard);
                    const int value_bits = ld_acquire_sys_global(
                            reinterpret_cast<const int*>(replica + index));
                    const float value = __int_as_float(value_bits);
                    sum = __fadd_rn(sum, value);
                    replica[index] = 0.0f;
                }
            }
            packed[index] = sum;
        }
        __syncthreads();
        shard_begin = shard_end;
    }
}

__device__ __forceinline__ void accumulate_registered_owner_grad_range(
        void* owner_rank_base,
        const moonep::Fp32GradShardDescriptor* shards,
        int num_shards,
        int local_expert,
        std::int64_t logical_offset,
        std::int64_t bytes,
        const std::uint8_t* staging) {
    std::int64_t shard_begin = 0;
    const std::int64_t logical_end = logical_offset + bytes;
    for (int shard_index = 0; shard_index < num_shards; ++shard_index) {
        const auto shard = shards[shard_index];
        const std::int64_t shard_bytes = static_cast<std::int64_t>(
                shard.num_elements * sizeof(float));
        const std::int64_t shard_end = shard_begin + shard_bytes;
        const std::int64_t begin = max(logical_offset, shard_begin);
        const std::int64_t end = min(logical_end, shard_end);
        const std::int64_t range_bytes = end > begin ? end - begin : 0;
        const std::int64_t elements = range_bytes / sizeof(float);
        const std::int64_t within_shard = begin - shard_begin;
        const std::int64_t within_staging = begin - logical_offset;
        auto* master = reinterpret_cast<float*>(
                static_cast<std::uint8_t*>(owner_rank_base) +
                shard.home_buffer_offset_bytes +
                static_cast<std::int64_t>(local_expert) *
                        shard.home_slot_stride_bytes +
                within_shard);
        const auto* packed = reinterpret_cast<const float*>(
                staging + within_staging);
        for (std::int64_t index = threadIdx.x; index < elements;
             index += blockDim.x)
            atomicAdd_system(master + index, ld_nc_global(packed + index));
        __syncthreads();
        shard_begin = shard_end;
    }
}

__device__ __forceinline__ void probe_grad_chunk(
        const std::int64_t* chunk_table,
        const int* plan_counts,
        const int* replica_expert,
        void** peer_nvl_bases,
        const moonep::Fp32GradShardDescriptor* grad_shards,
        int num_grad_shards,
        std::uint8_t* tx_staging,
        std::uint8_t* rx_staging,
        std::uint64_t* transfer_signal,
        std::uint64_t* transfer_ack,
        int* signal_expected,
        int* ack_expected,
        std::int64_t plan_reserve_offset,
        int plan_slot,
        int global_rank,
        int local_experts,
        int grad_weight_byte_ratio,
        int transfer) {
    const int chunk = transfer / grad_weight_byte_ratio;
    const int subchunk = transfer % grad_weight_byte_ratio;
    if (chunk >= plan_counts[1])
        return;
    const auto* row = chunk_table +
            static_cast<std::int64_t>(chunk) * kProbeChunkFields;
    const int expert = static_cast<int>(row[0]);
    const std::int64_t grad_offset = row[8] * grad_weight_byte_ratio +
            static_cast<std::int64_t>(subchunk) * kProbeWeightChunkBytes;
    const std::int64_t grad_total = row[9] * grad_weight_byte_ratio;
    const std::int64_t bytes_left = grad_total -
            static_cast<std::int64_t>(subchunk) * kProbeWeightChunkBytes;
    const std::int64_t bytes = bytes_left < kProbeWeightChunkBytes
            ? bytes_left : kProbeWeightChunkBytes;
    if (bytes <= 0)
        return;
    const std::int64_t source_path_offset =
            row[11] * grad_weight_byte_ratio +
            static_cast<std::int64_t>(subchunk) * kProbeWeightChunkBytes;
    const std::int64_t destination_path_offset =
            row[12] * grad_weight_byte_ratio +
            static_cast<std::int64_t>(subchunk) * kProbeWeightChunkBytes;
    if (source_path_offset < 0 || destination_path_offset < 0 ||
        source_path_offset + bytes > kProbeRdmaPathStagingBytes ||
        destination_path_offset + bytes > kProbeRdmaPathStagingBytes)
        return;

    const int weight_source_rank = static_cast<int>(row[6]);
    const int weight_destination_rank = static_cast<int>(row[7]);
    const int source_server = static_cast<int>(row[4]);
    const int destination_server = static_cast<int>(row[5]);
    const auto plan_staging = static_cast<std::int64_t>(plan_slot) *
                              kProbeRdmaPathStagingBytes;
    std::uint64_t* local_signal = transfer_signal +
            plan_slot * kProbeMaxTransfers + transfer;
    std::uint64_t* local_ack = transfer_ack +
            plan_slot * kProbeMaxTransfers + transfer;
    int* local_signal_expected = signal_expected +
            plan_slot * kProbeMaxTransfers + transfer;
    int* local_ack_expected = ack_expected +
            plan_slot * kProbeMaxTransfers + transfer;

    // Gradients originate at the server that received the weight replica.
    if (global_rank == weight_destination_rank) {
        auto* local_staging = tx_staging + plan_staging +
                              destination_path_offset;
        auto* remote_staging = rx_staging + plan_staging + source_path_offset;
        const int local_destination_server =
                global_rank / kRanksPerServer;
        pack_registered_grad_range(
                grad_shards, num_grad_shards, replica_expert, peer_nvl_bases,
                local_destination_server, expert, plan_slot, grad_offset,
                bytes, local_staging);
        nvshmemx_putmem_signal_nbi_block(
                remote_staging, local_staging,
                static_cast<std::size_t>(bytes), local_signal, 1,
                NVSHMEM_SIGNAL_ADD, source_server);
        if (threadIdx.x == 0) {
            const int expected = atomicAdd(local_ack_expected, 1) + 1;
            while (ld_acquire_sys_global(local_ack) <
                   static_cast<std::uint64_t>(expected)) {}
        }
        return;
    }

    if (global_rank != weight_source_rank)
        return;
    auto* local_staging = rx_staging + plan_staging + source_path_offset;
    if (threadIdx.x == 0) {
        const int expected = atomicAdd(local_signal_expected, 1) + 1;
        while (ld_acquire_sys_global(local_signal) <
               static_cast<std::uint64_t>(expected)) {}
    }
    __syncthreads();
    const int owner = expert / local_experts;
    const int owner_lane = owner % kRanksPerServer;
    const int local_expert = expert % local_experts;
    accumulate_registered_owner_grad_range(
            peer_nvl_bases[owner_lane], grad_shards, num_grad_shards,
            local_expert, grad_offset, bytes, local_staging);
    if (threadIdx.x == 0) {
        auto* owner_reserve = reinterpret_cast<moonep::IpcPlanReserve*>(
                static_cast<std::uint8_t*>(peer_nvl_bases[owner_lane]) +
                plan_reserve_offset);
        const int completed = atomicAdd_system(
                &owner_reserve->probe_owner_grad_chunk_count
                        [plan_slot][local_expert], 1) + 1;
        int expected_chunks = 0;
        for (int scheduled = 0; scheduled < plan_counts[1]; ++scheduled) {
            const auto* scheduled_row = chunk_table +
                    static_cast<std::int64_t>(scheduled) * kProbeChunkFields;
            if (static_cast<int>(scheduled_row[0]) == expert)
                expected_chunks += static_cast<int>(
                        (scheduled_row[9] * grad_weight_byte_ratio +
                         kProbeWeightChunkBytes - 1) /
                        kProbeWeightChunkBytes);
        }
        if (completed == expected_chunks) {
            __threadfence_system();
            atomicExch_system(
                    &owner_reserve->probe_owner_grad_ready
                            [plan_slot][local_expert], 1);
        }
        __threadfence_system();
        nvshmemx_signal_op(
                local_ack, 1, NVSHMEM_SIGNAL_ADD, destination_server);
    }
}

__global__ void probe_grad_chunks(
        const std::int64_t* chunk_table,
        const int* plan_counts,
        const int* replica_expert,
        void** peer_nvl_bases,
        const moonep::Fp32GradShardDescriptor* grad_shards,
        int num_grad_shards,
        std::uint8_t* tx_staging,
        std::uint8_t* rx_staging,
        std::uint64_t* transfer_signal,
        std::uint64_t* transfer_ack,
        int* signal_expected,
        int* ack_expected,
        std::int64_t plan_reserve_offset,
        int plan_slot,
        int global_rank,
        int local_experts,
        int grad_weight_byte_ratio) {
    const int transfers = grad_weight_byte_ratio * plan_counts[1];
    for (int transfer = static_cast<int>(blockIdx.x);
         transfer < transfers; transfer += static_cast<int>(gridDim.x))
        probe_grad_chunk(
                chunk_table, plan_counts, replica_expert, peer_nvl_bases,
                grad_shards, num_grad_shards, tx_staging, rx_staging,
                transfer_signal, transfer_ack,
                signal_expected, ack_expected, plan_reserve_offset,
                plan_slot, global_rank, local_experts,
                grad_weight_byte_ratio, transfer);
}

__global__ void wait_local_owner_grads(
        const int* admitted_experts,
        const int* plan_counts,
        void** peer_nvl_bases,
        std::int64_t plan_reserve_offset,
        int plan_slot,
        int global_rank,
        int local_experts) {
    const int nvl_rank = global_rank % kRanksPerServer;
    auto* reserve = reinterpret_cast<moonep::IpcPlanReserve*>(
            static_cast<std::uint8_t*>(peer_nvl_bases[nvl_rank]) +
            plan_reserve_offset);
    for (int replica = threadIdx.x; replica < plan_counts[0];
         replica += blockDim.x) {
        const int expert = admitted_experts[replica] / kMaxServers;
        if (expert / local_experts != global_rank)
            continue;
        const int local_expert = expert % local_experts;
        while (ld_acquire_sys_global(
                &reserve->probe_owner_grad_ready
                        [plan_slot][local_expert]) == 0) {}
    }
}

} // namespace

void launch_prepare_weight_transfer(
        const int* replica_expert,
        const int* cached_replica_expert,
        int* transfer_required,
        int world_size,
        bool force_transfer,
        cudaStream_t stream) {
    cudaMemsetAsync(transfer_required, 0, sizeof(int), stream);
    const int threads = min(1024, world_size * kReplicaSlots);
    prepare_weight_transfer_kernel<<<1, threads, 0, stream>>>(
            replica_expert, cached_replica_expert, transfer_required,
            world_size, force_transfer);
}

void launch_probeep_weight_send(
        const std::int64_t* chunk_table,
        const int* plan_counts,
        const int* transfer_required,
        void** peer_nvl_bases,
        const moonep::WeightShardDescriptor* weight_shards,
        int num_weight_shards,
        std::int64_t expert_pool_offset,
        void* symmetric_tail_base,
        int plan_slot,
        int global_rank,
        int local_experts,
        cudaStream_t stream) {
    auto* symmetric = static_cast<std::uint8_t*>(symmetric_tail_base);
    auto* tx_staging = symmetric + kProbeTxStagingOffset;
    auto* rx_staging = symmetric + kProbeRxStagingOffset;
    auto* signal = reinterpret_cast<std::uint64_t*>(
            symmetric + kProbeWeightSignalOffset);
    // replica_expert is never inspected by the source specialization.
    probe_weight_chunks<true><<<kTransportCtas, kThreads, 0, stream>>>(
            chunk_table, plan_counts, nullptr, peer_nvl_bases,
            weight_shards, num_weight_shards, tx_staging, rx_staging,
            signal, nullptr,
            expert_pool_offset - moonep::kIpcPlanReserveBytes,
            transfer_required, plan_slot, global_rank, local_experts, 0);
    // putmem_signal_nbi only enqueues the transfer.  DeepEP Dispatch uses its
    // own IBGDA QPs, so CUDA kernel order alone is not a transport completion
    // dependency.  Quiet this PE's stream before balanced_dispatch is launched:
    // this is a per-source-rail edge, not a peer/world Weight barrier.
    nvshmemx_quiet_on_stream(stream);
}

void launch_probeep_weight_receive(
        const std::int64_t* chunk_table,
        const int* plan_counts,
        const int* replica_expert,
        void** peer_nvl_bases,
        int* cached_replica_expert,
        const int* transfer_required,
        const moonep::WeightShardDescriptor* weight_shards,
        int num_weight_shards,
        std::int64_t expert_pool_offset,
        void* symmetric_tail_base,
        int plan_slot,
        int global_rank,
        int world_size,
        int local_experts,
        std::int64_t expert_weight_bytes,
        cudaStream_t stream) {
    auto* symmetric = static_cast<std::uint8_t*>(symmetric_tail_base);
    auto* tx_staging = symmetric + kProbeTxStagingOffset;
    auto* rx_staging = symmetric + kProbeRxStagingOffset;
    auto* signal = reinterpret_cast<std::uint64_t*>(
            symmetric + kProbeWeightSignalOffset);
    auto* signal_expected = reinterpret_cast<int*>(
            symmetric + kProbeWeightSignalExpectedOffset);
    const auto reserve_offset = expert_pool_offset -
            moonep::kIpcPlanReserveBytes;
    const int nvl_rank = global_rank % kRanksPerServer;
    reset_probe_weight_state<<<1, kThreads, 0, stream>>>(
            peer_nvl_bases, reserve_offset, transfer_required,
            plan_slot, nvl_rank);
    // The first local barrier orders readiness reset and same-server direct
    // copies. Dispatch is already free to progress on the public comm stream.
    conditional_weight_barrier_kernel<<<1, 32, 0, stream>>>(
            peer_nvl_bases, reserve_offset, transfer_required, plan_slot, 0);
    const int chunks_per_expert = static_cast<int>(
            (expert_weight_bytes + kProbeWeightChunkBytes - 1) /
            kProbeWeightChunkBytes);
    probe_weight_chunks<false><<<kReceiveCtas, kThreads, 0, stream>>>(
            chunk_table, plan_counts, replica_expert, peer_nvl_bases,
            weight_shards, num_weight_shards, tx_staging, rx_staging,
            signal, signal_expected, reserve_offset, transfer_required,
            plan_slot, global_rank, local_experts, chunks_per_expert);
    // Every rank may consume replicas populated by several destination rails.
    // Join them only here, on the private consumer-readiness stream.
    conditional_weight_barrier_kernel<<<1, 32, 0, stream>>>(
            peer_nvl_bases, reserve_offset, transfer_required, plan_slot, 1);
    const int threads = min(1024, world_size * kReplicaSlots);
    publish_weight_layout_kernel<<<1, threads, 0, stream>>>(
            replica_expert, cached_replica_expert, world_size);
}

void launch_probeep_grad_transport(
        const std::int64_t* chunk_table,
        const int* plan_counts,
        const int* admitted_experts,
        const int* replica_expert,
        void** peer_nvl_bases,
        int** barrier_signal_ptrs,
        const moonep::Fp32GradShardDescriptor* grad_shards,
        int num_grad_shards,
        std::int64_t expert_pool_offset,
        void* symmetric_tail_base,
        int plan_slot,
        int global_rank,
        int world_size,
        int local_experts,
        std::int64_t expert_weight_bytes,
        int grad_weight_byte_ratio,
        cudaStream_t stream) {
    auto* symmetric = static_cast<std::uint8_t*>(symmetric_tail_base);
    auto* tx_staging = symmetric + kProbeTxStagingOffset;
    auto* rx_staging = symmetric + kProbeRxStagingOffset;
    auto* signal = reinterpret_cast<std::uint64_t*>(
            symmetric + kProbeSignalOffset);
    auto* ack = reinterpret_cast<std::uint64_t*>(
            symmetric + kProbeAckOffset);
    auto* signal_expected = reinterpret_cast<int*>(
            symmetric + kProbeSignalExpectedOffset);
    auto* ack_expected = reinterpret_cast<int*>(
            symmetric + kProbeAckExpectedOffset);
    const auto reserve_offset = expert_pool_offset -
            moonep::kIpcPlanReserveBytes;
    const int nvl_rank = global_rank % kRanksPerServer;
    reset_probe_weight_state<<<1, kThreads, 0, stream>>>(
            peer_nvl_bases, reserve_offset, nullptr,
            plan_slot, nvl_rank);
    intranode::barrier(
            barrier_signal_ptrs, nvl_rank, kRanksPerServer, stream);
    (void)expert_weight_bytes;
    probe_grad_chunks<<<kTransportCtas, kThreads, 0, stream>>>(
            chunk_table, plan_counts, replica_expert, peer_nvl_bases,
            grad_shards, num_grad_shards, tx_staging, rx_staging, signal, ack,
            signal_expected, ack_expected, reserve_offset,
            plan_slot, global_rank, local_experts,
            grad_weight_byte_ratio);
    // A local replica can be consumed in disjoint ranges by every rail rank on
    // its destination server.  Do not publish this rank's grad completion
    // until all local rail kernels have finished reading and clearing those
    // ranges.
    intranode::barrier(
            barrier_signal_ptrs, nvl_rank, kRanksPerServer, stream);
    wait_local_owner_grads<<<1, 32, 0, stream>>>(
            admitted_experts, plan_counts, peer_nvl_bases, reserve_offset,
            plan_slot, global_rank, local_experts);
}

} // namespace deep_ep::probeep
