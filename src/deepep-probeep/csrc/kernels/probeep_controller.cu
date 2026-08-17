#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "probeep_controller.cuh"
#include "probeep_weight_transport.cuh"

namespace deep_ep::probeep {
namespace {

constexpr std::int64_t kMaxRdmaPathStagingBytes =
        kProbeMaxMigrationBytesPerEndpoint;

__global__ void controller_kernel(
        const std::int64_t* compute_ns,
        const std::int64_t* network_ns,
        const std::int64_t* dispatch_tx_bytes,
        const std::int64_t* dispatch_rx_bytes,
        const std::int64_t* migration_tx_bytes,
        const std::int64_t* migration_rx_bytes,
        std::int64_t* budgets,
        std::int64_t* summary,
        double rdma_path_bytes_per_ns,
        double alpha,
        std::int64_t fallback_budget_bytes,
        bool valid,
        bool hold_budget_on_invalid,
        int num_ranks) {
    if (threadIdx.x != 0)
        return;

    std::int64_t compute_max = 0;
    std::int64_t network_max = 0;
    for (int rank = 0; rank < num_ranks; ++rank) {
        compute_max = max(compute_max, compute_ns[rank]);
        network_max = max(network_max, network_ns[rank]);
    }

    if (!valid || compute_max <= 0 || network_max <= 0) {
        // Runtime feedback updates must hold the complete independent A/M
        // state, not only its per-endpoint budget.  Replacing learned_total
        // while retaining budgets changes the next planner cap and therefore
        // is not a hold at all.
        if (hold_budget_on_invalid)
            return;
        for (int rank = 0; rank < num_ranks; ++rank)
            budgets[rank] = min(
                    fallback_budget_bytes,
                    kMaxRdmaPathStagingBytes);
        summary[0] = compute_max;
        summary[1] = network_max;
        summary[2] = 0;
        summary[3] = 0;
        summary[4] = 0;
        summary[5] = fallback_budget_bytes;
        return;
    }

    std::int64_t sampled_total = 0;
    for (int rank = 0; rank < num_ranks; ++rank) {
        if (network_ns[rank] != network_max)
            continue;
        const auto tx = dispatch_tx_bytes[rank] + migration_tx_bytes[rank];
        const auto rx = dispatch_rx_bytes[rank] + migration_rx_bytes[rank];
        sampled_total = max(sampled_total, max(tx, rx));
    }
    // A positive event duration without any observed wire byte is not a
    // usable bandwidth sample (for example an all-local route or an empty
    // cache-hit window). Treat it exactly like the other invalid samples:
    // the persistent runtime keeps the previous A/M row, while the standalone
    // diagnostic entry initializes from its explicit fallback. Collapsing
    // the budget to zero here would let one quiet window permanently disable
    // later expert migration on that independent controller chain.
    if (sampled_total <= 0) {
        if (hold_budget_on_invalid)
            return;
        for (int rank = 0; rank < num_ranks; ++rank)
            budgets[rank] = min(
                    fallback_budget_bytes,
                    kMaxRdmaPathStagingBytes);
        summary[0] = compute_max;
        summary[1] = network_max;
        summary[2] = 0;
        summary[3] = 0;
        summary[4] = 0;
        summary[5] = fallback_budget_bytes;
        return;
    }
    const auto probe_total = static_cast<std::int64_t>(
            alpha * static_cast<double>(compute_max) /
            static_cast<double>(network_max) *
            static_cast<double>(sampled_total));
    const auto theory_total = static_cast<std::int64_t>(
            rdma_path_bytes_per_ns * static_cast<double>(compute_max));
    const auto target_total = min(probe_total, theory_total);

    for (int rank = 0; rank < num_ranks; ++rank) {
        const auto dispatch_tx = dispatch_tx_bytes[rank];
        const auto dispatch_rx = dispatch_rx_bytes[rank];
        const auto target_remaining = target_total - max(dispatch_tx, dispatch_rx);
        const auto hard_tx = theory_total - dispatch_tx;
        const auto hard_rx = theory_total - dispatch_rx;
        const auto bounded = min(target_remaining, min(hard_tx, hard_rx));
        budgets[rank] = bounded <= 0 ? 0 :
                min(bounded, kMaxRdmaPathStagingBytes);
    }
    summary[0] = compute_max;
    summary[1] = network_max;
    summary[2] = sampled_total;
    summary[3] = probe_total;
    summary[4] = theory_total;
    summary[5] = target_total;
}

int validate_vector(const torch::Tensor& tensor, int expected = -1) {
    TORCH_CHECK(tensor.is_cuda(), "controller input must be CUDA");
    TORCH_CHECK(tensor.is_contiguous(), "controller input must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == torch::kInt64,
                "controller input must be int64");
    TORCH_CHECK(tensor.dim() == 1 && tensor.numel() > 0,
                "controller input must have shape [R]");
    if (expected >= 0)
        TORCH_CHECK(tensor.numel() == expected,
                    "all controller vectors must have the same R");
    return static_cast<int>(tensor.numel());
}

} // namespace

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
        cudaStream_t stream) {
    const int num_ranks = validate_vector(compute_ns);
    for (const auto* tensor : {
             &compute_ns, &network_ns, &dispatch_tx_bytes, &dispatch_rx_bytes,
             &migration_tx_bytes, &migration_rx_bytes})
        validate_vector(*tensor, num_ranks);
    validate_vector(migration_budget_bytes, num_ranks);
    TORCH_CHECK(summary.is_cuda() && summary.is_contiguous() &&
                summary.scalar_type() == torch::kInt64 && summary.numel() == 6,
                "controller summary must be contiguous CUDA int64 [6]");
    TORCH_CHECK(
            rdma_path_bandwidth_gbps > 0.0,
            "RDMA path bandwidth must be positive");
    TORCH_CHECK(alpha > 0.0 && alpha <= 1.0, "alpha must be in (0, 1]");
    TORCH_CHECK(fallback_budget_bytes >= 0,
                "fallback budget must be non-negative");
    controller_kernel<<<1, 32, 0, stream>>>(
            compute_ns.data_ptr<std::int64_t>(),
            network_ns.data_ptr<std::int64_t>(),
            dispatch_tx_bytes.data_ptr<std::int64_t>(),
            dispatch_rx_bytes.data_ptr<std::int64_t>(),
            migration_tx_bytes.data_ptr<std::int64_t>(),
            migration_rx_bytes.data_ptr<std::int64_t>(),
            migration_budget_bytes.data_ptr<std::int64_t>(),
            summary.data_ptr<std::int64_t>(), rdma_path_bandwidth_gbps / 8.0,
            alpha, fallback_budget_bytes, valid, hold_budget_on_invalid,
            num_ranks);
}

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
        bool valid) {
    auto budgets = torch::empty({compute_ns.numel()}, compute_ns.options());
    auto summary = torch::empty({6}, compute_ns.options());
    const auto stream = at::cuda::getCurrentCUDAStream();
    update_controller_cuda(
            compute_ns, network_ns, dispatch_tx_bytes, dispatch_rx_bytes,
            migration_tx_bytes, migration_rx_bytes, budgets, summary,
            rdma_path_bandwidth_gbps, alpha, fallback_budget_bytes, valid,
            false, stream);
    return {budgets, summary};
}

} // namespace deep_ep::probeep
