#pragma once

#include <torch/types.h>

#include <cstdint>
#include <tuple>

namespace deep_ep::moonep {

// Phase-one expert storage is part of every rank's CUDA-IPC allocation.  The
// Each plan-ring weight bank is one contiguous [16 home + 16 replica] block.
// Duplicating the small home prefix lets grouped GEMM consume the selected
// 48-slot bank directly instead of presenting 112 groups with 64 empty groups.
// Replica gradients retain one shared home prefix plus three replica banks.
inline constexpr std::int64_t kExpertPoolHomeSlots = 16;
inline constexpr std::int64_t kExpertPoolReplicaSlotsPerPlan = 16;
inline constexpr std::int64_t kExpertPoolPlanSlots = 3;
inline constexpr std::int64_t kExpertPoolReplicaSlots =
        kExpertPoolReplicaSlotsPerPlan * kExpertPoolPlanSlots;
inline constexpr std::int64_t kExpertPoolWeightSlotsPerPlan =
        kExpertPoolHomeSlots + kExpertPoolReplicaSlotsPerPlan;
inline constexpr std::int64_t kExpertPoolWeightSlots =
        kExpertPoolWeightSlotsPerPlan * kExpertPoolPlanSlots;
inline constexpr std::int64_t kExpertPoolGradSlots =
        kExpertPoolHomeSlots + kExpertPoolReplicaSlots;
inline constexpr std::int64_t kExpertPoolHidden = 7168;
inline constexpr std::int64_t kExpertPoolIntermediate = 2048;
inline constexpr std::int64_t kExpertPoolAlignment = 256;

inline constexpr std::int64_t kExpertPoolElementsPerSlot =
        kExpertPoolHidden * kExpertPoolIntermediate;
inline constexpr std::int64_t kExpertPoolWeightShardBytes =
        kExpertPoolWeightSlots * kExpertPoolElementsPerSlot * 2;
inline constexpr std::int64_t kExpertPoolGradShardBytes =
        kExpertPoolGradSlots * kExpertPoolElementsPerSlot * 4;

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

static_assert(kExpertPoolWeightShardBytes == 2818572288);
static_assert(kExpertPoolGradShardBytes == 3758096384);
static_assert(kExpertPoolBytes == 19730006016);
static_assert(kExpertPoolBytes % kExpertPoolAlignment == 0);

// The returned views do not own memory.  local_base is the beginning of the
// rank's CUDA-IPC allocation and pool_offset is the byte offset immediately
// after the normal NVL transport extent and IpcPlanReserve.
//
// Shapes, in tuple order:
//   gate_weight [96, 7168, 2048]  BF16
//   up_weight   [96, 7168, 2048]  BF16
//   down_weight [96, 2048, 7168]  BF16
//   gate_grad   [64, 7168, 2048]  FP32
//   up_grad     [64, 7168, 2048]  FP32
//   down_grad   [64, 2048, 7168]  FP32
using ExpertPoolViews = std::tuple<
        torch::Tensor, torch::Tensor, torch::Tensor,
        torch::Tensor, torch::Tensor, torch::Tensor>;

ExpertPoolViews make_expert_pool_views(
        void* local_base, std::int64_t pool_offset, int device_id);

}  // namespace deep_ep::moonep
