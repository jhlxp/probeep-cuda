#include "moonep_expert_pool.hpp"

#include <torch/extension.h>

#include <cstdint>

namespace deep_ep::moonep {

ExpertPoolViews make_expert_pool_views(
        void* local_base, std::int64_t pool_offset, int device_id) {
    auto* pool = static_cast<std::uint8_t*>(local_base) + pool_offset;
    const auto cuda_device = torch::Device(torch::kCUDA, device_id);
    const auto bf16 = torch::TensorOptions()
                              .dtype(torch::kBFloat16)
                              .device(cuda_device);
    const auto fp32 = torch::TensorOptions()
                              .dtype(torch::kFloat32)
                              .device(cuda_device);

    auto gate_weight = torch::from_blob(
            pool + kExpertPoolGateWeightOffset,
            {kExpertPoolWeightSlots, kExpertPoolHidden, kExpertPoolIntermediate},
            bf16);
    auto up_weight = torch::from_blob(
            pool + kExpertPoolUpWeightOffset,
            {kExpertPoolWeightSlots, kExpertPoolHidden, kExpertPoolIntermediate},
            bf16);
    auto down_weight = torch::from_blob(
            pool + kExpertPoolDownWeightOffset,
            {kExpertPoolWeightSlots, kExpertPoolIntermediate, kExpertPoolHidden},
            bf16);
    auto gate_grad = torch::from_blob(
            pool + kExpertPoolGateGradOffset,
            {kExpertPoolGradSlots, kExpertPoolHidden, kExpertPoolIntermediate},
            fp32);
    auto up_grad = torch::from_blob(
            pool + kExpertPoolUpGradOffset,
            {kExpertPoolGradSlots, kExpertPoolHidden, kExpertPoolIntermediate},
            fp32);
    auto down_grad = torch::from_blob(
            pool + kExpertPoolDownGradOffset,
            {kExpertPoolGradSlots, kExpertPoolIntermediate, kExpertPoolHidden},
            fp32);

    return {gate_weight, up_weight, down_weight,
            gate_grad, up_grad, down_grad};
}

}  // namespace deep_ep::moonep
