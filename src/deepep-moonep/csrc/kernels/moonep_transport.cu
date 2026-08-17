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

template <bool kWeighted>
__global__ void precombine_bf16_rows(
        int4* transport_rows,
        const int4* exec_rows,
        const float* exec_route_weight,
        const int* recv_route_rows,
        const int* recv_count,
        int hidden_int4) {
    __shared__ int route_rows[8];
    __shared__ float route_weights[8];

    const int num_rows = *recv_count;
    for (int row = static_cast<int>(blockIdx.x); row < num_rows;
         row += static_cast<int>(gridDim.x)) {
        if (threadIdx.x < 8) {
            const int exec_row = recv_route_rows[row * 8 + threadIdx.x];
            route_rows[threadIdx.x] = exec_row;
            if constexpr (kWeighted)
                route_weights[threadIdx.x] = exec_row < 0
                    ? 0.0f : exec_route_weight[exec_row];
        }
        __syncthreads();

        unsigned route_mask = 0;
        #pragma unroll
        for (int route = 0; route < 8; ++route)
            route_mask |= static_cast<unsigned>(route_rows[route] >= 0)
                          << route;
        const int active_routes = __popc(route_mask);
        const int single_route = __ffs(route_mask) - 1;

        auto* output = transport_rows +
                       static_cast<std::size_t>(row) * hidden_int4;
        if (active_routes == 1) {
            const int exec_row = route_rows[single_route];
            float weight = 1.0f;
            if constexpr (kWeighted)
                weight = route_weights[single_route];
            for (int column = static_cast<int>(threadIdx.x);
                 column < hidden_int4;
                 column += static_cast<int>(blockDim.x)) {
                const int4 packed = exec_rows[
                    static_cast<std::size_t>(exec_row) * hidden_int4 +
                    column];
                if constexpr (kWeighted) {
                    int4 packed_output;
                    const auto* values =
                        reinterpret_cast<const __nv_bfloat162*>(&packed);
                    auto* results =
                        reinterpret_cast<__nv_bfloat162*>(&packed_output);
                    #pragma unroll
                    for (int pair = 0; pair < 4; ++pair) {
                        const float2 value = __bfloat1622float2(values[pair]);
                        results[pair] = __floats2bfloat162_rn(
                            value.x * weight, value.y * weight);
                    }
                    output[column] = packed_output;
                } else {
                    output[column] = packed;
                }
            }
        } else {
            for (int column = static_cast<int>(threadIdx.x);
                 column < hidden_int4;
                 column += static_cast<int>(blockDim.x)) {
                float2 sums[4];
                #pragma unroll
                for (int pair = 0; pair < 4; ++pair)
                    sums[pair] = make_float2(0.0f, 0.0f);
                unsigned remaining = route_mask;
                while (remaining != 0) {
                    const int route = __ffs(remaining) - 1;
                    const int exec_row = route_rows[route];
                    const int4 packed = exec_rows[
                        static_cast<std::size_t>(exec_row) * hidden_int4 +
                        column];
                    const auto* values =
                        reinterpret_cast<const __nv_bfloat162*>(&packed);
                    #pragma unroll
                    for (int pair = 0; pair < 4; ++pair) {
                        const float2 value =
                            __bfloat1622float2(values[pair]);
                        if constexpr (kWeighted) {
                            const float route_weight = route_weights[route];
                            sums[pair].x = fmaf(
                                value.x, route_weight, sums[pair].x);
                            sums[pair].y = fmaf(
                                value.y, route_weight, sums[pair].y);
                        } else {
                            sums[pair].x += value.x;
                            sums[pair].y += value.y;
                        }
                    }
                    remaining &= remaining - 1;
                }
                int4 packed_output;
                auto* results =
                    reinterpret_cast<__nv_bfloat162*>(&packed_output);
                #pragma unroll
                for (int pair = 0; pair < 4; ++pair)
                    results[pair] = __floats2bfloat162_rn(
                        sums[pair].x, sums[pair].y);
                output[column] = packed_output;
            }
        }
        __syncthreads();
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

void launch_precombine_bf16(void* transport_rows,
                            const void* exec_rows,
                            const float* exec_route_weight,
                            const int* recv_route_rows,
                            const int* recv_count,
                            int hidden,
                            cudaStream_t stream) {
    constexpr int kNumBlocks = 512;
    constexpr int kNumThreads = 256;
    if (exec_route_weight == nullptr) {
        precombine_bf16_rows<false><<<kNumBlocks, kNumThreads, 0, stream>>>(
            static_cast<int4*>(transport_rows),
            static_cast<const int4*>(exec_rows), nullptr,
            recv_route_rows, recv_count, hidden / 8);
    } else {
        precombine_bf16_rows<true><<<kNumBlocks, kNumThreads, 0, stream>>>(
            static_cast<int4*>(transport_rows),
            static_cast<const int4*>(exec_rows),
            exec_route_weight, recv_route_rows, recv_count, hidden / 8);
    }
}

} // namespace deep_ep::internode
