#include "moonep_transport.cuh"

#include <cuda_bf16.h>

#include <cstddef>

namespace deep_ep::internode {

namespace {

__global__ void scale_bf16_rows(__nv_bfloat162* rows,
                                const float* row_weights,
                                int hidden_pairs) {
    const int row = static_cast<int>(blockIdx.x);
    const float weight = row_weights[row];
    auto* row_data = rows + static_cast<std::size_t>(row) * hidden_pairs;
    for (int column = static_cast<int>(threadIdx.x);
         column < hidden_pairs;
         column += static_cast<int>(blockDim.x)) {
        if (weight == 0.0f) {
            row_data[column] = __float2bfloat162_rn(0.0f);
            continue;
        }
        const float2 value = __bfloat1622float2(row_data[column]);
        row_data[column] = __floats2bfloat162_rn(value.x * weight,
                                                value.y * weight);
    }
}

} // namespace

void launch_bf16_route_weight_scale(void* exec_rows,
                                    const float* exec_route_weight,
                                    int num_rows,
                                    int hidden,
                                    cudaStream_t stream) {
    constexpr int kNumThreads = 256;
    scale_bf16_rows<<<num_rows, kNumThreads, 0, stream>>>(
        static_cast<__nv_bfloat162*>(exec_rows), exec_route_weight,
        hidden / 2);
}

} // namespace deep_ep::internode
