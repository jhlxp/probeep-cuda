#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace deep_ep::probeep {

// Build this rank's compact E256 histogram and stable per-expert ordinal.
// The segmented path uses the ordinal output itself as prefix scratch, so the
// production hot path needs no temporary allocation.
void launch_local_histogram_and_ordinal(
        const std::int64_t* local_topk_idx,
        int num_tokens,
        int* local_histogram,
        int* local_ordinal,
        cudaStream_t stream);

}  // namespace deep_ep::probeep
