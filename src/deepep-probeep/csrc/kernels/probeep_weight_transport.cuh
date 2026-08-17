#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "../moonep_expert_pool.hpp"
#include "moonep_expert_io.cuh"
#include "probeep_plan.cuh"

namespace deep_ep::probeep {

inline constexpr std::int64_t kProbeWeightChunkBytes = 4LL * 1024 * 1024;
// Standalone planner/debug default only.  The production runtime derives the
// complete expert size from all registered model shards.
inline constexpr std::int64_t kDefaultDsv3ExpertWeightBytes =
        3LL * moonep::kExpertPoolElementsPerSlot * sizeof(std::uint16_t);
// FP32 gradient return is 2x BF16 or 4x FP8 weight traffic. One staging slot
// reserves the largest supported ratio of the clamped weight window.
inline constexpr int kProbeMaxGradWeightByteRatio = 4;
inline constexpr std::int64_t kProbeRdmaPathStagingBytes =
        kProbeMaxGradWeightByteRatio * 64LL * 1024 * 1024;
inline constexpr std::int64_t kProbeMaxMigrationBytesPerEndpoint =
        64LL * 1024 * 1024;
// One direction has eight source endpoints. Complete-expert admission cannot
// schedule more chunks than their aggregate endpoint budgets can hold. Keep
// the planner table large, but do not launch thousands of empty CTAs.
inline constexpr int kProbeMaxScheduledWeightChunks = kProbeMaxChunks;
inline constexpr int kProbePlanRingSlots = 3;
inline constexpr int kProbeMaxTransfers =
        kProbeMaxGradWeightByteRatio * kProbeMaxChunks;

constexpr std::size_t align_probe_tail(std::size_t bytes) {
    return (bytes + 255) / 256 * 256;
}

inline constexpr std::size_t kProbeTxStagingOffset =
        align_probe_tail(
                kMaxServers * kNumExperts * sizeof(int));
// TX and RX offsets are allocated independently by the planner and both start
// at zero.  They must therefore address disjoint symmetric banks: a rail can
// inject one expert while another server writes an incoming expert on the
// private progress stream.  Sharing one bank would corrupt full-duplex
// Weight/Grad traffic even though each direction is individually in bounds.
inline constexpr std::size_t kProbeRxStagingOffset =
        kProbeTxStagingOffset +
        kProbePlanRingSlots * kProbeRdmaPathStagingBytes;
inline constexpr std::size_t kProbeWeightSignalOffset =
        kProbeRxStagingOffset +
        kProbePlanRingSlots * kProbeRdmaPathStagingBytes;
// Weight and gradient return use independent NVSHMEM signal counters.  Their
// expected-value state must be independent as well: a completed forward weight
// signal must not advance the expected value of the first backward grad signal.
inline constexpr std::size_t kProbeWeightSignalExpectedOffset =
        kProbeWeightSignalOffset +
        kProbePlanRingSlots * kProbeMaxTransfers * sizeof(std::uint64_t);
inline constexpr std::size_t kProbeSignalOffset =
        kProbeWeightSignalExpectedOffset +
        kProbePlanRingSlots * kProbeMaxTransfers * sizeof(int);
inline constexpr std::size_t kProbeAckOffset =
        kProbeSignalOffset +
        kProbePlanRingSlots * kProbeMaxTransfers * sizeof(std::uint64_t);
inline constexpr std::size_t kProbeSignalExpectedOffset =
        kProbeAckOffset +
        kProbePlanRingSlots * kProbeMaxTransfers * sizeof(std::uint64_t);
inline constexpr std::size_t kProbeAckExpectedOffset =
        kProbeSignalExpectedOffset +
        kProbePlanRingSlots * kProbeMaxTransfers * sizeof(int);
inline constexpr std::size_t kProbeSymmetricBytes = align_probe_tail(
        kProbeAckExpectedOffset +
        kProbePlanRingSlots * kProbeMaxTransfers * sizeof(int));

static_assert(kDefaultDsv3ExpertWeightBytes == 84LL * 1024 * 1024);
static_assert(kProbeWeightChunkBytes == 4LL * 1024 * 1024);
static_assert(kProbeMaxScheduledWeightChunks == 2048);

void launch_prepare_weight_transfer(
        const int* replica_expert,
        const int* cached_replica_expert,
        int* transfer_required,
        int world_size,
        bool force_transfer,
        cudaStream_t stream);

// Source-only half of the admitted chunk table.  It returns after the local
// rail has issued and quieted its own Weight puts and completion signals.  It
// never waits for remote scatter or another rail, so the caller can launch
// that rail's DeepEP Dispatch immediately afterward.
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
        cudaStream_t stream);

// Destination-only half of the admitted chunk table.  This runs on a private
// progress stream, scatters complete experts to the final server-local slots,
// and joins only at the Expert consumer boundary.  The two local barriers
// order peer resets/copies; neither is a Weight-before-Dispatch barrier.
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
        cudaStream_t stream);

// Reverse of the weight path.  Destination-server replica FP32 gradients are
// reduced while packing, returned over the same eight RDMA paths, accumulated
// into the home expert, and cleared in-place.  Each owner waits only for its
// own migrated experts.
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
        cudaStream_t stream);

} // namespace deep_ep::probeep
