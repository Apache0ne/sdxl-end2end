#include "fp8_internal.hpp"
#include "runtime_internal.hpp"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <mma.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <type_traits>

namespace sdxl::cuda {
namespace {

__global__ void fp16_amax_kernel(const __half* source,
                                 std::size_t count,
                                 unsigned int* maximum_bits) {
    float local = 0.0F;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        local = fmaxf(local, fabsf(__half2float(source[index])));
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        local = fmaxf(local, __shfl_down_sync(0xffffffffU, local, offset));
    }
    if ((threadIdx.x & 31) == 0) atomicMax(maximum_bits, __float_as_uint(local));
}

__global__ void fp8_scale_kernel(const unsigned int* maximum_bits,
                                 float* dequant_scale,
                                 float maximum_fp8) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const float maximum = __uint_as_float(*maximum_bits);
        *dequant_scale = maximum > 0.0F ? fmaxf(maximum / maximum_fp8, 1.0e-12F) : 1.0F;
    }
}

template <typename FP8>
__global__ void fp16_to_fp8_kernel(const __half* source,
                                   FP8* destination,
                                   std::size_t count,
                                   const float* dequant_scale) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float inverse_scale = 1.0F / *dequant_scale;
    destination[index] = FP8(__half2float(source[index]) * inverse_scale);
}

template <typename FP8, int WarpsPerBlock, bool KMajor>
__global__ void fp8_weight_only_wmma_kernel(const __half* input,
                                            const FP8* weight,
                                            const float* weight_scale,
                                            std::size_t weight_scale_count,
                                            __half* output,
                                            int m,
                                            int n,
                                            int k) {
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    constexpr int Tile = 16;
    const int warp = static_cast<int>(threadIdx.x) / warpSize;
    const int lane = static_cast<int>(threadIdx.x) & (warpSize - 1);
    if (warp >= WarpsPerBlock) return;

    const int tile_m = static_cast<int>(blockIdx.y);
    const int tile_n = static_cast<int>(blockIdx.x) * WarpsPerBlock + warp;
    const int row_base = tile_m * Tile;
    const int column_base = tile_n * Tile;

    // All warps in the CTA share the same MxK activation tile and process
    // adjacent N tiles. The previous kernel loaded A independently in every
    // warp, multiplying global activation traffic by WarpsPerBlock.
    __shared__ __align__(16) __half shared_a[Tile * Tile];
    __shared__ __align__(16) __half shared_b[WarpsPerBlock][Tile * Tile];
    __shared__ __align__(16) __half shared_c[WarpsPerBlock][Tile * Tile];

    // Strict non-VAE mode keeps the custom Ampere path in FP16 accumulation.
    // Native FP8 cuBLASLt is different: NVIDIA requires CUBLAS_COMPUTE_32F.
    wmma::fragment<wmma::accumulator, Tile, Tile, Tile, __half> accumulator;
    wmma::fill_fragment(accumulator, __float2half(0.0F));

    for (int k_base = 0; k_base < k; k_base += Tile) {
        // The full block cooperatively loads A exactly once.
        for (int linear = static_cast<int>(threadIdx.x);
             linear < Tile * Tile; linear += static_cast<int>(blockDim.x)) {
            const int row = linear / Tile;
            const int column = linear - row * Tile;
            const int global_row = row_base + row;
            const int global_k = k_base + column;
            shared_a[linear] = (global_row < m && global_k < k)
                ? input[static_cast<std::size_t>(global_row) * k + global_k]
                : __float2half(0.0F);
        }

        // Each warp decodes one adjacent FP8 KxN tile. No complete FP16
        // weight matrix is ever materialized.
        for (int linear = lane; linear < Tile * Tile; linear += warpSize) {
            const int row = linear / Tile;
            const int column = linear - row * Tile;
            const int global_n = column_base + column;
            const int weight_k = k_base + row;
            if (global_n < n && weight_k < k) {
                const std::size_t scale_index = weight_scale_count == 1
                    ? 0 : static_cast<std::size_t>(global_n);
                const float scale = weight_scale == nullptr ? 1.0F : weight_scale[scale_index];
                const std::size_t weight_index = KMajor
                    ? static_cast<std::size_t>(weight_k) * n + global_n
                    : static_cast<std::size_t>(global_n) * k + weight_k;
                shared_b[warp][linear] = __float2half(
                    static_cast<float>(weight[weight_index]) * scale);
            } else {
                shared_b[warp][linear] = __float2half(0.0F);
            }
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, Tile, Tile, Tile, __half, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, Tile, Tile, Tile, __half, wmma::row_major> b;
        wmma::load_matrix_sync(a, shared_a, Tile);
        wmma::load_matrix_sync(b, shared_b[warp], Tile);
        wmma::mma_sync(accumulator, a, b, accumulator);
        __syncthreads();
    }

    wmma::store_matrix_sync(shared_c[warp], accumulator, Tile, wmma::mem_row_major);
    __syncwarp();
    for (int linear = lane; linear < Tile * Tile; linear += warpSize) {
        const int row = linear / Tile;
        const int column = linear - row * Tile;
        const int global_row = row_base + row;
        const int global_column = column_base + column;
        if (global_row < m && global_column < n) {
            output[static_cast<std::size_t>(global_row) * n + global_column] =
                shared_c[warp][linear];
        }
    }
#else
    (void)input;
    (void)weight;
    (void)weight_scale;
    (void)weight_scale_count;
    (void)output;
    (void)m;
    (void)n;
    (void)k;
#endif
}

[[nodiscard]] unsigned blocks_for(std::size_t count, unsigned threads = 256) {
    return static_cast<unsigned>((count + threads - 1) / threads);
}

} // namespace

Tensor quantize_tensor_fp8(const Runtime& runtime,
                           const Tensor& source_f16,
                           ScalarType destination_type) {
    if (source_f16.type() != ScalarType::Float16 || !is_fp8(destination_type)) {
        throw CudaError("FP8 activation quantization requires float16 input and an FP8 destination");
    }
    Tensor destination = Tensor::allocate(runtime, source_f16.shape(), destination_type,
                                          TensorRole::Model);
    Tensor scale = Tensor::zeros(runtime, {1}, ScalarType::Float32,
                                 TensorRole::FP8ScaleMetadata);
    constexpr unsigned threads = 256;
    const unsigned reduction_blocks = static_cast<unsigned>(std::max<std::size_t>(
        1, std::min<std::size_t>(4096, (source_f16.elements() + threads - 1) / threads)));
    fp16_amax_kernel<<<reduction_blocks, threads, 0, runtime.stream()>>>(
        source_f16.half_data(), source_f16.elements(),
        reinterpret_cast<unsigned int*>(scale.float_data()));
    SDXL_CUDA_CHECK(cudaGetLastError());
    const float maximum = destination_type == ScalarType::Float8E4M3 ? 448.0F : 57344.0F;
    fp8_scale_kernel<<<1, 1, 0, runtime.stream()>>>(
        reinterpret_cast<const unsigned int*>(scale.float_data()), scale.float_data(), maximum);
    SDXL_CUDA_CHECK(cudaGetLastError());
    if (destination_type == ScalarType::Float8E4M3) {
        fp16_to_fp8_kernel<<<blocks_for(source_f16.elements()), threads, 0, runtime.stream()>>>(
            source_f16.half_data(), reinterpret_cast<__nv_fp8_e4m3*>(destination.data()),
            source_f16.elements(), scale.float_data());
    } else {
        fp16_to_fp8_kernel<<<blocks_for(source_f16.elements()), threads, 0, runtime.stream()>>>(
            source_f16.half_data(), reinterpret_cast<__nv_fp8_e5m2*>(destination.data()),
            source_f16.elements(), scale.float_data());
    }
    SDXL_CUDA_CHECK(cudaGetLastError());
    destination.attach_dequant_scale(scale, FP8ScaleMode::TensorWide);
    return destination;
}

Tensor fp8_weight_only_linear(const Runtime& runtime,
                                     const Tensor& input_f16,
                                     const Tensor& weight_fp8) {
    if (input_f16.type() != ScalarType::Float16 || !is_fp8(weight_fp8.type()) ||
        input_f16.rank() < 2 || weight_fp8.rank() != 2) {
        throw CudaError("FP8 weight-only linear received invalid tensor types or ranks");
    }
    const std::size_t k = input_f16.shape().back();
    const std::size_t n = weight_fp8.size(0);
    const std::size_t m = input_f16.elements() / k;
    if (weight_fp8.size(1) != k) throw CudaError("FP8 linear feature mismatch");
    if (m > static_cast<std::size_t>(INT_MAX) || n > static_cast<std::size_t>(INT_MAX) ||
        k > static_cast<std::size_t>(INT_MAX)) {
        throw CudaError("FP8 linear dimensions exceed kernel integer range");
    }
    if (!weight_fp8.has_dequant_scale() ||
        (weight_fp8.dequant_scale_count() != 1 && weight_fp8.dequant_scale_count() != n)) {
        throw CudaError("FP8 weight has invalid scaling metadata");
    }
    std::vector<std::size_t> output_shape = input_f16.shape();
    output_shape.back() = n;
    Tensor output = Tensor::allocate(runtime, output_shape, ScalarType::Float16,
                                     TensorRole::Model);
    constexpr int warps = 4;
    const dim3 block(warps * 32U, 1U, 1U);
    const dim3 grid(static_cast<unsigned>((n + warps * 16 - 1) / (warps * 16)),
                    static_cast<unsigned>((m + 15) / 16), 1U);
    const bool k_major = weight_fp8.fp8_storage_layout() == FP8StorageLayout::KMajorKN;
    const auto launch_e4m3 = [&](auto layout_tag) {
        constexpr bool packed_k_major = decltype(layout_tag)::value;
        fp8_weight_only_wmma_kernel<__nv_fp8_e4m3, warps, packed_k_major>
            <<<grid, block, 0, runtime.stream()>>>(
                input_f16.half_data(), reinterpret_cast<const __nv_fp8_e4m3*>(weight_fp8.data()),
                weight_fp8.dequant_scale_data(), weight_fp8.dequant_scale_count(),
                output.half_data(), static_cast<int>(m), static_cast<int>(n), static_cast<int>(k));
    };
    const auto launch_e5m2 = [&](auto layout_tag) {
        constexpr bool packed_k_major = decltype(layout_tag)::value;
        fp8_weight_only_wmma_kernel<__nv_fp8_e5m2, warps, packed_k_major>
            <<<grid, block, 0, runtime.stream()>>>(
                input_f16.half_data(), reinterpret_cast<const __nv_fp8_e5m2*>(weight_fp8.data()),
                weight_fp8.dequant_scale_data(), weight_fp8.dequant_scale_count(),
                output.half_data(), static_cast<int>(m), static_cast<int>(n), static_cast<int>(k));
    };
    if (weight_fp8.type() == ScalarType::Float8E4M3) {
        if (k_major) launch_e4m3(std::true_type{});
        else launch_e4m3(std::false_type{});
    } else {
        if (k_major) launch_e5m2(std::true_type{});
        else launch_e5m2(std::false_type{});
    }
    SDXL_CUDA_CHECK(cudaGetLastError());
    return output;
}

} // namespace sdxl::cuda
