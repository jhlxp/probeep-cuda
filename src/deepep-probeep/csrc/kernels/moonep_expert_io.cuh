// Copyright (c) 2026 PKU-DASYS and Dots-Infra
//
// This file contains an adaptation of the direct weight synchronization and
// deterministic FP32 gradient reduction contracts from UltraEP.  See NOTICE.
// SPDX-License-Identifier: MIT

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace deep_ep::moonep {

inline constexpr int kNvlDomainRanks = 8;

// Device-resident description of one weight shard (for example gate, up,
// down, or a quantization-scale shard).  master_expert_ptrs contains one
// device address per home expert on this rank.  The replica fields describe
// this shard's location inside every rank's symmetric DeepEP CUDA-IPC buffer.
struct WeightShardDescriptor {
    const std::uint64_t* master_expert_ptrs;
    std::uint64_t home_buffer_offset_bytes;
    std::uint64_t home_slot_stride_bytes;
    std::uint64_t replica_buffer_offset_bytes;
    std::uint64_t replica_slot_stride_bytes;
    std::uint64_t replica_plan_stride_bytes;
    std::uint64_t num_elements;
    std::uint32_t element_bytes;
};

// Device-resident description of one FP32 gradient shard.  Master gradients
// may be ordinary framework allocations; replica gradients live in the
// symmetric DeepEP CUDA-IPC buffer described by offset and slot stride.
struct Fp32GradShardDescriptor {
    const std::uint64_t* master_expert_ptrs;
    std::uint64_t home_buffer_offset_bytes;
    std::uint64_t home_slot_stride_bytes;
    std::uint64_t replica_buffer_offset_bytes;
    std::uint64_t replica_slot_stride_bytes;
    std::uint64_t num_elements;
};

// Push every local home expert's weight shards directly to its server-local
// replicas.  All pointer arguments and descriptors are device-resident:
//
//   domain_replica_expert  [8, num_replica_slots], row-major global expert ids
//   peer_ipc_base_ptrs     [8], DeepEP cudaIpcOpenMemHandle pointer table
//   shard_descriptors      [num_shards]
//
// Global expert ownership is the standard contiguous DeepEP layout:
// global_expert = global_rank * num_local_experts + local_expert.
// Master pointers, IPC offsets, and slot strides are 16-byte aligned; arbitrary
// byte tails are supported through num_elements * element_bytes.
//
// Completion of this rank's launch covers its outgoing writes only.  Every
// rank in the NVL8 domain must enqueue deep_ep::intranode::barrier on the same
// communication stream before any rank consumes an incoming replica weight.
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
    cudaStream_t stream);

// Pull every replica gradient belonging to this rank's home experts, add it to
// the local FP32 master gradient in deterministic peer-rank/slot order, and
// clear the consumed remote replica storage in place.
//
// Before launch, all eight ranks must finish replica Wgrad production and pass
// an intranode barrier on this stream.  A second intranode barrier is required
// after launch before any rank reuses replica-gradient slots.  The local master
// gradients are ready for same-rank consumers when this launch completes.
void launch_deterministic_fp32_replica_grad_reduce(
    const std::int32_t* domain_replica_expert,
    void* const* peer_ipc_base_ptrs,
    const Fp32GradShardDescriptor* shard_descriptors,
    int num_shards,
    int global_rank,
    int num_local_experts,
    int num_replica_slots,
    int replica_slot_base,
    cudaStream_t stream);

// Clear exactly the populated replica slots after local and remote gradient
// accumulation. Empty slots are left untouched so their lifecycle sentinel
// remains observable by correctness checks.
}  // namespace deep_ep::moonep
