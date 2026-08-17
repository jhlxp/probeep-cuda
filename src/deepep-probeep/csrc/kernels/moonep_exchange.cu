#include "configs.cuh"
#include "../moonep_runtime.hpp"
#include "exception.cuh"
#include "ibgda_device.cuh"
#include "launch.cuh"
#include "moonep_exchange.cuh"
#include "utils.cuh"

namespace deep_ep::moonep {

__global__ void publish_server_histograms_kernel(const int* local_counts,
                                             int* symmetric_counts,
                                             void** buffer_ptrs,
                                             int** barrier_signal_ptrs,
                                             int64_t plan_reserve_offset,
                                             int rank,
                                             int num_servers) {
    const int rdma_rank = rank / probeep::kRanksPerServer;
    const int nvl_rank = rank % probeep::kRanksPerServer;
    int* own_row = symmetric_counts + rdma_rank * probeep::kNumExperts;

    if (threadIdx.x < probeep::kNumExperts / 4)
        reinterpret_cast<int4*>(own_row)[threadIdx.x] =
                reinterpret_cast<const int4*>(local_counts)[threadIdx.x];
    __syncthreads();

    const int lane_id = threadIdx.x & 31;
    for (int remote_pe = 0; remote_pe < num_servers; ++remote_pe) {
        if (remote_pe == rdma_rank)
            continue;
        if (threadIdx.x < 32)
            nvshmemi_ibgda_put_nbi_warp<true>(
                    reinterpret_cast<uint64_t>(own_row),
                    reinterpret_cast<uint64_t>(own_row),
                    probeep::kNumExperts * sizeof(int), remote_pe, 0,
                    lane_id, remote_pe);
        if (threadIdx.x == 0)
            nvshmemi_ibgda_quiet(remote_pe, 0);
        __syncthreads();
    }
    __syncthreads();
    if (threadIdx.x == 0)
        nvshmem_sync_all();
    __syncthreads();

    auto* reserve = reinterpret_cast<IpcPlanReserve*>(
            static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + plan_reserve_offset);
    if (threadIdx.x < num_servers * probeep::kNumExperts / 4)
        reinterpret_cast<int4*>(reserve->source_counts)[threadIdx.x] =
                reinterpret_cast<const int4*>(symmetric_counts)[threadIdx.x];

    // The reserve is consumed directly by the planner on every NVLink peer.
    // Folding the publication barrier into this kernel removes a standalone
    // 32-thread launch from every dispatch.
    barrier_block<probeep::kRanksPerServer>(barrier_signal_ptrs, nvl_rank);
}

void publish_server_histograms(const int* local_counts,
                           int* symmetric_counts,
                           void** buffer_ptrs,
                           int** barrier_signal_ptrs,
                           int64_t plan_reserve_offset,
                           int rank,
                           int num_servers,
                           cudaStream_t stream) {
    constexpr int kThreads = 256;
    SETUP_LAUNCH_CONFIG(1, kThreads, stream);
    LAUNCH_KERNEL(&cfg, publish_server_histograms_kernel,
                  local_counts, symmetric_counts, buffer_ptrs,
                  barrier_signal_ptrs,
                  plan_reserve_offset, rank, num_servers);
}

} // namespace deep_ep::moonep
