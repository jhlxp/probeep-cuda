#pragma once

#include <torch/types.h>

namespace deep_ep::probeep {

struct ControllerOutput {
    torch::Tensor migration_budget_bytes;  // [R], int64
    torch::Tensor summary;                 // [6], int64
};

void update_controller_cuda(
        const torch::Tensor& compute_ns,
        const torch::Tensor& network_ns,
        const torch::Tensor& dispatch_tx_bytes,
        const torch::Tensor& dispatch_rx_bytes,
        const torch::Tensor& migration_tx_bytes,
        const torch::Tensor& migration_rx_bytes,
        torch::Tensor& migration_budget_bytes,
        torch::Tensor& summary,
        double rdma_path_bandwidth_gbps,
        double alpha,
        std::int64_t fallback_budget_bytes,
        bool valid,
        bool hold_budget_on_invalid,
        cudaStream_t stream);

// Standalone CUDA entry used by the CPU/CUDA contract test.  The production
// planner consumes the same device formula from a fixed runtime workspace.
ControllerOutput run_controller_cuda(
        const torch::Tensor& compute_ns,
        const torch::Tensor& network_ns,
        const torch::Tensor& dispatch_tx_bytes,
        const torch::Tensor& dispatch_rx_bytes,
        const torch::Tensor& migration_tx_bytes,
        const torch::Tensor& migration_rx_bytes,
        double rdma_path_bandwidth_gbps,
        double alpha,
        std::int64_t fallback_budget_bytes,
        bool valid);

} // namespace deep_ep::probeep
