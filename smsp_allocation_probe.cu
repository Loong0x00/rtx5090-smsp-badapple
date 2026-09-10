#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _err = (expr);                                              \
        if (_err != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,       \
                         __LINE__, cudaGetErrorString(_err));                    \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

__global__ void ilp_ffma_probe(float* sink, int iterations, uint32_t warp_mask) {
    extern __shared__ unsigned occupancy_limiter[];
    if (threadIdx.x == 0) {
        occupancy_limiter[0] = blockIdx.x;
    }
    __syncthreads();

    const unsigned logical_warp = threadIdx.x >> 5;
    if ((warp_mask & (1u << logical_warp)) == 0) {
        return;
    }

    float a0 = 1.0001f + threadIdx.x * 1e-7f;
    float a1 = 1.0002f + threadIdx.x * 1e-7f;
    float a2 = 1.0003f + threadIdx.x * 1e-7f;
    float a3 = 1.0004f + threadIdx.x * 1e-7f;
    float a4 = 1.0005f + threadIdx.x * 1e-7f;
    float a5 = 1.0006f + threadIdx.x * 1e-7f;
    float a6 = 1.0007f + threadIdx.x * 1e-7f;
    float a7 = 1.0008f + threadIdx.x * 1e-7f;
    const float mul = 0.99999994f;
    const float add = 0.00000006f;

    for (int i = 0; i < iterations; ++i) {
        // Eight independent accumulators let one warp issue near one FFMA per
        // cycle instead of exposing a dependency-latency bottleneck.
        asm volatile("fma.rn.f32 %0, %0, %8, %9;\n\t"
                     "fma.rn.f32 %1, %1, %8, %9;\n\t"
                     "fma.rn.f32 %2, %2, %8, %9;\n\t"
                     "fma.rn.f32 %3, %3, %8, %9;\n\t"
                     "fma.rn.f32 %4, %4, %8, %9;\n\t"
                     "fma.rn.f32 %5, %5, %8, %9;\n\t"
                     "fma.rn.f32 %6, %6, %8, %9;\n\t"
                     "fma.rn.f32 %7, %7, %8, %9;"
                     : "+f"(a0), "+f"(a1), "+f"(a2), "+f"(a3),
                       "+f"(a4), "+f"(a5), "+f"(a6), "+f"(a7)
                     : "f"(mul), "f"(add));
    }
    sink[blockIdx.x * blockDim.x + threadIdx.x] =
        a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

static float measure(int sms,
                     int threads,
                     uint32_t warp_mask,
                     int iterations,
                     int repeats,
                     size_t dynamic_shared,
                     float* sink) {
    void* args[] = {&sink, &iterations, &warp_mask};
    // Warm up the exact launch configuration.
    CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(ilp_ffma_probe), dim3(sms), dim3(threads),
        args, dynamic_shared, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) {
        CUDA_CHECK(cudaLaunchCooperativeKernel(
            reinterpret_cast<void*>(ilp_ffma_probe), dim3(sms), dim3(threads),
            args, dynamic_shared, nullptr));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));
    return elapsed_ms / static_cast<float>(repeats);
}

static int parse_bounded_int(const char* name,
                             const char* text,
                             int minimum,
                             int maximum) {
    errno = 0;
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (text == end || end == nullptr || *end != '\0' || errno == ERANGE ||
        value < minimum || value > maximum) {
        std::fprintf(stderr, "%s must be an integer in [%d, %d], got '%s'\n",
                     name, minimum, maximum, text);
        std::exit(2);
    }
    return static_cast<int>(value);
}

int main(int argc, char** argv) {
    if (argc > 3) {
        std::fprintf(stderr, "Usage: %s [iterations=500000] [repeats=7]\n", argv[0]);
        return 2;
    }
    const int iterations =
        argc > 1 ? parse_bounded_int("iterations", argv[1], 1, 5'000'000)
                 : 500000;
    const int repeats =
        argc > 2 ? parse_bounded_int("repeats", argv[2], 1, 100) : 7;

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int max_shared = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_shared,
                                      cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    CUDA_CHECK(cudaFuncSetAttribute(ilp_ffma_probe,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    max_shared));

    int active128 = 0;
    int active512 = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active128, ilp_ffma_probe, 128, max_shared));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active512, ilp_ffma_probe, 512, max_shared));
    if (active128 != 1 || active512 != 1) {
        std::fprintf(stderr, "Expected one active CTA/SM, got %d and %d.\n",
                     active128, active512);
        return 1;
    }

    float* sink = nullptr;
    CUDA_CHECK(cudaMalloc(&sink,
                          static_cast<size_t>(prop.multiProcessorCount) * 512 *
                              sizeof(float)));

    std::array<float, 4> singles{};
    for (int warp = 0; warp < 4; ++warp) {
        singles[warp] = measure(prop.multiProcessorCount, 128, 1u << warp,
                                iterations, repeats, max_shared, sink);
    }
    const float single_avg =
        (singles[0] + singles[1] + singles[2] + singles[3]) * 0.25f;

    const std::array<uint32_t, 6> pair_masks = {
        0b0011u, 0b0101u, 0b1001u, 0b0110u, 0b1010u, 0b1100u};
    std::array<float, 6> pairs{};
    for (size_t i = 0; i < pair_masks.size(); ++i) {
        pairs[i] = measure(prop.multiProcessorCount, 128, pair_masks[i],
                           iterations, repeats, max_shared, sink);
    }
    const float t4 = measure(prop.multiProcessorCount, 128, 0x0fu,
                             iterations, repeats, max_shared, sink);
    const float t16 = measure(prop.multiProcessorCount, 512, 0xffffu,
                              iterations, repeats, max_shared, sink);
    const std::array<uint32_t, 4> same_low_bits_pair_masks = {
        0x0011u, 0x0022u, 0x0044u, 0x0088u};
    std::array<float, 4> same_low_bits_pairs{};
    for (size_t i = 0; i < same_low_bits_pair_masks.size(); ++i) {
        same_low_bits_pairs[i] = measure(
            prop.multiProcessorCount, 512, same_low_bits_pair_masks[i],
            iterations, repeats, max_shared, sink);
    }
    const std::array<uint32_t, 4> stride4_masks = {
        0x1111u, 0x2222u, 0x4444u, 0x8888u};
    std::array<float, 4> stride4{};
    for (size_t i = 0; i < stride4_masks.size(); ++i) {
        stride4[i] = measure(prop.multiProcessorCount, 512, stride4_masks[i],
                             iterations, repeats, max_shared, sink);
    }

    float max_pair_ratio = 0.0f;
    for (float pair : pairs) {
        max_pair_ratio = std::max(max_pair_ratio, pair / single_avg);
    }
    const float full_ratio = t4 / single_avg;
    const float scaling_ratio = t16 / t4;
    float min_same_low_pair_ratio = same_low_bits_pairs[0] / single_avg;
    for (float value : same_low_bits_pairs) {
        min_same_low_pair_ratio =
            std::min(min_same_low_pair_ratio, value / single_avg);
    }
    float min_stride4_ratio = stride4[0] / single_avg;
    for (float value : stride4) {
        min_stride4_ratio = std::min(min_stride4_ratio, value / single_avg);
    }

    std::printf("GPU: %s, SMs=%d\n", prop.name, prop.multiProcessorCount);
    std::printf("One cooperative CTA/SM, 8 independent FFMA instructions/iteration/warp\n");
    std::printf("Single logical warp times [0,1,2,3]: %.3f %.3f %.3f %.3f ms\n",
                singles[0], singles[1], singles[2], singles[3]);
    std::printf("Pair times [01,02,03,12,13,23]:      %.3f %.3f %.3f %.3f %.3f %.3f ms\n",
                pairs[0], pairs[1], pairs[2], pairs[3], pairs[4], pairs[5]);
    std::printf("All first 4 warps: %.3f ms (%.3fx single average)\n",
                t4, full_ratio);
    std::printf("All first 16 warps: %.3f ms (%.3fx four-warp time)\n",
                t16, scaling_ratio);
    std::printf("Same-low-bit pairs [04,15,26,37]:    %.3f %.3f %.3f %.3f ms\n",
                same_low_bits_pairs[0], same_low_bits_pairs[1],
                same_low_bits_pairs[2], same_low_bits_pairs[3]);
    std::printf("Stride-4 groups [048C,159D,26AE,37BF]: %.3f %.3f %.3f %.3f ms\n",
                stride4[0], stride4[1], stride4[2], stride4[3]);
    std::printf("Interpretation: warps 0..3 do not contend, while physical warp-slot\n"
                "groups with the same low two bits do contend. On this GB202 that\n"
                "supports SMSP = physical_warpid & 3. This is empirical, not a PTX ABI.\n");

    CUDA_CHECK(cudaFree(sink));
    const bool four_distinct = max_pair_ratio < 1.6f && full_ratio < 1.8f &&
                               min_same_low_pair_ratio > 1.4f &&
                               min_stride4_ratio > 2.0f;
    return four_distinct ? 0 : 3;
}
