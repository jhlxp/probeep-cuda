// Copyright (c) 2026 PKU-DASYS and Dots-Infra
//
// Direct NVL weight synchronization and deterministic FP32 gradient reduction
// adapted from UltraEP commit 94cab099b44fffa99a82fea99e7c12d89cf65e4f.
// The UltraEP runtime, NVSHMEM, relay path, and VMM are intentionally absent.
// See NOTICE for the vendored-source license.
// SPDX-License-Identifier: MIT

#include "moonep_expert_io.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace deep_ep::moonep {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kCtasPerExpertShard = 4;

__device__ __forceinline__ void store_streaming(
    void* destination,
    std::uint32_t x,
    std::uint32_t y,
    std::uint32_t z,
    std::uint32_t w) {
    asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};"
                 :
                 : "l"(destination), "r"(x), "r"(y), "r"(z), "r"(w)
                 : "memory");
}

__global__ __launch_bounds__(kThreadsPerBlock) void direct_replica_weight_copy_kernel(
    const std::int32_t* __restrict__ domain_replica_expert,
    void* const* __restrict__ peer_ipc_base_ptrs,
    const WeightShardDescriptor* __restrict__ shard_descriptors,
    int global_rank,
    int num_local_experts,
    int num_replica_slots,
    int replica_slot_base,
    const std::int32_t* __restrict__ transfer_required) {
    if (*transfer_required == 0)
        return;
    extern __shared__ std::uint64_t replica_shard_addresses[];
    __shared__ int num_replicas;

    const int local_expert = static_cast<int>(blockIdx.z);
    const int global_expert = global_rank * num_local_experts + local_expert;
    const WeightShardDescriptor shard = shard_descriptors[blockIdx.y];

    if (threadIdx.x == 0) {
        int replica_count = 0;
        for (int peer_rank = 0; peer_rank < kNvlDomainRanks; ++peer_rank) {
            auto* peer_base = static_cast<std::uint8_t*>(peer_ipc_base_ptrs[peer_rank]);
            for (int replica_slot = 0; replica_slot < num_replica_slots; ++replica_slot) {
                const int map_offset = peer_rank * num_replica_slots + replica_slot;
                if (domain_replica_expert[map_offset] == global_expert) {
                    const int physical_slot = replica_slot_base + replica_slot;
                    const int plan_slot = physical_slot / num_replica_slots;
                    const int replica_in_plan = physical_slot % num_replica_slots;
                    replica_shard_addresses[replica_count++] = reinterpret_cast<std::uint64_t>(
                        peer_base + shard.replica_buffer_offset_bytes +
                        static_cast<std::uint64_t>(plan_slot) *
                            shard.replica_plan_stride_bytes +
                        static_cast<std::uint64_t>(replica_in_plan) *
                            shard.replica_slot_stride_bytes);
                }
            }
        }
        num_replicas = replica_count;
    }
    __syncthreads();

    if (num_replicas == 0) {
        return;
    }

    const auto* source = reinterpret_cast<const std::uint8_t*>(
        shard.master_expert_ptrs[local_expert]);
    const std::uint64_t num_bytes = shard.num_elements * shard.element_bytes;
    const std::uint64_t num_vectors = num_bytes / sizeof(uint4);
    const auto* source_vectors = reinterpret_cast<const uint4*>(source);

    for (std::uint64_t vector_idx =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         vector_idx < num_vectors;
         vector_idx += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const uint4 value = source_vectors[vector_idx];
        for (int replica_idx = 0; replica_idx < num_replicas; ++replica_idx) {
            auto* destination = reinterpret_cast<uint4*>(replica_shard_addresses[replica_idx]);
            store_streaming(destination + vector_idx, value.x, value.y, value.z, value.w);
        }
    }

    if (blockIdx.x == 0) {
        const std::uint64_t tail_begin = num_vectors * sizeof(uint4);
        for (std::uint64_t byte_idx = tail_begin + threadIdx.x;
             byte_idx < num_bytes;
             byte_idx += blockDim.x) {
            const std::uint8_t value = source[byte_idx];
            for (int replica_idx = 0; replica_idx < num_replicas; ++replica_idx) {
                auto* destination = reinterpret_cast<std::uint8_t*>(
                    replica_shard_addresses[replica_idx]);
                destination[byte_idx] = value;
            }
        }
    }
}

__global__ __launch_bounds__(kThreadsPerBlock) void deterministic_fp32_replica_grad_reduce_kernel(
    const std::int32_t* __restrict__ domain_replica_expert,
    void* const* __restrict__ peer_ipc_base_ptrs,
    const Fp32GradShardDescriptor* __restrict__ shard_descriptors,
    int global_rank,
    int num_local_experts,
    int num_replica_slots,
    int replica_slot_base) {
    extern __shared__ std::uint64_t replica_shard_addresses[];
    __shared__ int num_replicas;

    const int local_expert = static_cast<int>(blockIdx.z);
    const int global_expert = global_rank * num_local_experts + local_expert;
    const Fp32GradShardDescriptor shard = shard_descriptors[blockIdx.y];

    if (threadIdx.x == 0) {
        int replica_count = 0;
        for (int peer_rank = 0; peer_rank < kNvlDomainRanks; ++peer_rank) {
            auto* peer_base = static_cast<std::uint8_t*>(peer_ipc_base_ptrs[peer_rank]);
            for (int replica_slot = 0; replica_slot < num_replica_slots; ++replica_slot) {
                const int map_offset = peer_rank * num_replica_slots + replica_slot;
                if (domain_replica_expert[map_offset] == global_expert) {
                    replica_shard_addresses[replica_count++] = reinterpret_cast<std::uint64_t>(
                        peer_base + shard.replica_buffer_offset_bytes +
                        static_cast<std::uint64_t>(replica_slot_base + replica_slot) *
                            shard.replica_slot_stride_bytes);
                }
            }
        }
        num_replicas = replica_count;
    }
    __syncthreads();

    if (num_replicas == 0) {
        return;
    }

    auto* master = reinterpret_cast<float*>(shard.master_expert_ptrs[local_expert]);
    auto* master_vectors = reinterpret_cast<float4*>(master);
    const std::uint64_t num_vectors = shard.num_elements / 4;

    for (std::uint64_t vector_idx =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         vector_idx < num_vectors;
         vector_idx += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        float4 accumulator = master_vectors[vector_idx];
        for (int replica_idx = 0; replica_idx < num_replicas; ++replica_idx) {
            auto* replica = reinterpret_cast<float4*>(replica_shard_addresses[replica_idx]);
            const float4 value = replica[vector_idx];
            accumulator.x = __fadd_rn(accumulator.x, value.x);
            accumulator.y = __fadd_rn(accumulator.y, value.y);
            accumulator.z = __fadd_rn(accumulator.z, value.z);
            accumulator.w = __fadd_rn(accumulator.w, value.w);
            store_streaming(replica + vector_idx, 0, 0, 0, 0);
        }
        master_vectors[vector_idx] = accumulator;
    }

    if (blockIdx.x == 0) {
        const std::uint64_t tail_begin = num_vectors * 4;
        for (std::uint64_t element_idx = tail_begin + threadIdx.x;
             element_idx < shard.num_elements;
             element_idx += blockDim.x) {
            float accumulator = master[element_idx];
            for (int replica_idx = 0; replica_idx < num_replicas; ++replica_idx) {
                auto* replica = reinterpret_cast<float*>(replica_shard_addresses[replica_idx]);
                accumulator = __fadd_rn(accumulator, replica[element_idx]);
                replica[element_idx] = 0.0f;
            }
            master[element_idx] = accumulator;
        }
    }
}

}  // namespace

void launch_direct_replica_weight_copy(
    const std::int32_t* domain_replica_expert,
    void* const* peer_ipc_base_ptrs,
    const WeightShardDescriptor* shard_descriptors,
    int num_shards,
    int global_rank,
    int num_local_experts,
    int num_replica_slots,
    int replica_slot_base,
    const std::int32_t* transfer_required,
    cudaStream_t stream) {
    const dim3 grid(kCtasPerExpertShard, num_shards, num_local_experts);
    const std::size_t shared_bytes =
        static_cast<std::size_t>(kNvlDomainRanks) * num_replica_slots * sizeof(std::uint64_t);
    direct_replica_weight_copy_kernel<<<grid, kThreadsPerBlock, shared_bytes, stream>>>(
        domain_replica_expert,
        peer_ipc_base_ptrs,
        shard_descriptors,
        global_rank,
        num_local_experts,
        num_replica_slots,
        replica_slot_base,
        transfer_required);
}

void launch_deterministic_fp32_replica_grad_reduce(
    const std::int32_t* domain_replica_expert,
    void* const* peer_ipc_base_ptrs,
    const Fp32GradShardDescriptor* shard_descriptors,
    int num_shards,
    int global_rank,
    int num_local_experts,
    int num_replica_slots,
    int replica_slot_base,
    cudaStream_t stream) {
    const dim3 grid(kCtasPerExpertShard, num_shards, num_local_experts);
    const std::size_t shared_bytes =
        static_cast<std::size_t>(kNvlDomainRanks) * num_replica_slots * sizeof(std::uint64_t);
    deterministic_fp32_replica_grad_reduce_kernel<<<grid, kThreadsPerBlock, shared_bytes, stream>>>(
        domain_replica_expert,
        peer_ipc_base_ptrs,
        shard_descriptors,
        global_rank,
        num_local_experts,
        num_replica_slots,
        replica_slot_base);
}

}  // namespace deep_ep::moonep
