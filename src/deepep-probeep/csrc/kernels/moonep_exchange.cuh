#pragma once

#include <cstdint>

namespace deep_ep::moonep {

// Exchange one rank's compact [256] histogram with the same NVLink lane on
// every server, publish P rows through CUDA IPC, and synchronize visibility
// to the eight local planner kernels. Expanded TopK routes never leave rank.
void publish_server_histograms(const int* local_counts,
                           int* symmetric_counts,
                           void** buffer_ptrs,
                           int** barrier_signal_ptrs,
                           int64_t plan_reserve_offset,
                           int rank,
                           int num_servers,
                           cudaStream_t stream);

} // namespace deep_ep::moonep
