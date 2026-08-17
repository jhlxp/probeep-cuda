#pragma once

#include <cstdint>

namespace deep_ep::moonep {

// Exchange one rank's [128] histogram with the same NVLink lane on the other
// server, publish both rows through CUDA IPC and synchronize their visibility
// to the eight local planner kernels.
void publish_paired_counts(const int* local_counts,
                           int* symmetric_counts,
                           void** buffer_ptrs,
                           int** barrier_signal_ptrs,
                           int64_t plan_reserve_offset,
                           int rank,
                           cudaStream_t stream);

} // namespace deep_ep::moonep
