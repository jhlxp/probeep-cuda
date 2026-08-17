#pragma once

#include <torch/types.h>

#include <cstdint>
#include <tuple>

namespace deep_ep::moonep {

// Phase-one expert storage is part of every rank's CUDA-IPC allocation.  The
// first sixteen slots are the maximum home-expert extent.  The remaining
// slots are three independent thirty-two-slot replica banks, one per plan
// ring slot.  Route metadata and expert weights therefore have the same
// lifetime: a later microbatch cannot overwrite an earlier handle's weights.
inline constexpr std::int64_t kExpertPoolHomeSlots = 16;
inline constexpr std::int64_t kExpertPoolReplicaSlotsPerPlan = 32;
inline constexpr std::int64_t kExpertPoolPlanSlots = 3;
inline constexpr std::int64_t kExpertPoolReplicaSlots =
        kExpertPoolReplicaSlotsPerPlan * kExpertPoolPlanSlots;
inline constexpr std::int64_t kExpertPoolSlots =
        kExpertPoolHomeSlots + kExpertPoolReplicaSlots;
inline constexpr std::int64_t kExpertPoolHidden = 7168;
inline constexpr std::int64_t kExpertPoolIntermediate = 2048;
inline constexpr std::int64_t kExpertPoolAlignment = 256;

inline constexpr std::int64_t kExpertPoolElementsPerSlot =
        kExpertPoolHidden * kExpertPoolIntermediate;
inline constexpr std::int64_t kExpertPoolWeightShardBytes =
        kExpertPoolSlots * kExpertPoolElementsPerSlot * 2;
inline constexpr std::int64_t kExpertPoolGradShardBytes =
        kExpertPoolSlots * kExpertPoolElementsPerSlot * 4;

inline constexpr std::int64_t kExpertPoolGateWeightOffset = 0;
inline constexpr std::int64_t kExpertPoolUpWeightOffset =
        kExpertPoolGateWeightOffset + kExpertPoolWeightShardBytes;
inline constexpr std::int64_t kExpertPoolDownWeightOffset =
        kExpertPoolUpWeightOffset + kExpertPoolWeightShardBytes;
inline constexpr std::int64_t kExpertPoolGateGradOffset =
        kExpertPoolDownWeightOffset + kExpertPoolWeightShardBytes;
inline constexpr std::int64_t kExpertPoolUpGradOffset =
        kExpertPoolGateGradOffset + kExpertPoolGradShardBytes;
inline constexpr std::int64_t kExpertPoolDownGradOffset =
        kExpertPoolUpGradOffset + kExpertPoolGradShardBytes;
inline constexpr std::int64_t kExpertPoolBytes =
        kExpertPoolDownGradOffset + kExpertPoolGradShardBytes;

static_assert(kExpertPoolWeightShardBytes == 3288334336);
static_assert(kExpertPoolGradShardBytes == 6576668672);
static_assert(kExpertPoolBytes == 29595009024);
static_assert(kExpertPoolBytes % kExpertPoolAlignment == 0);

// The returned views do not own memory.  local_base is the beginning of the
// rank's CUDA-IPC allocation and pool_offset is the byte offset immediately
// after the normal NVL transport extent and IpcPlanReserve.
//
// Shapes, in tuple order:
//   gate_weight [112, 7168, 2048]  BF16
//   up_weight   [112, 7168, 2048]  BF16
//   down_weight [112, 2048, 7168]  BF16
//   gate_grad   [112, 7168, 2048]  FP32
//   up_grad     [112, 7168, 2048]  FP32
//   down_grad   [112, 2048, 7168]  FP32
using ExpertPoolViews = std::tuple<
        torch::Tensor, torch::Tensor, torch::Tensor,
        torch::Tensor, torch::Tensor, torch::Tensor>;

ExpertPoolViews make_expert_pool_views(
        void* local_base, std::int64_t pool_offset, int device_id);

}  // namespace deep_ep::moonep
