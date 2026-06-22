#include "flash_attention_sm80.hpp"

#include "sdxl/cuda/runtime.hpp"

#include <cuda_fp16.h>
#include <mma.h>
#include <math_constants.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace sdxl::cuda {
namespace {

namespace wmma = nvcuda::wmma;

constexpr int kHeadDimension = 64;
constexpr int kQueryTile = 32;
constexpr int kKeyTile = 16;
constexpr int kThreads = 256;
constexpr unsigned kFullMask = 0xffffffffU;
constexpr float kAttentionScale = 0.125F; // 1 / sqrt(64)

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(kFullMask, value, offset);
    }
    return __shfl_sync(kFullMask, value, 0);
}

__device__ __forceinline__ float warp_max(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(kFullMask, value, offset));
    }
    return __shfl_sync(kFullMask, value, 0);
}

// A compact FlashAttention-style forward kernel specialized for SDXL's
// head-dimension 64. Two warps compute two 16x16 QK score tiles with WMMA while
// all eight warps update four query rows each. K and V tiles are loaded once per
// CTA and reused by 32 query rows. The online softmax keeps only O(Q*D) output
// state and never creates a global score tensor.
template <int FixedKeySequence, bool Causal>
__global__ void flash_attention_hdim64_kernel(const __half* __restrict__ query,
                                               const __half* __restrict__ key,
                                               const __half* __restrict__ value,
                                               const std::int32_t* __restrict__ key_mask,
                                               __half* __restrict__ output,
                                               std::size_t batch,
                                               std::size_t query_sequence,
                                               std::size_t key_sequence,
                                               std::size_t heads) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __shared__ __align__(16) __half query_tile[kQueryTile * kHeadDimension];
    __shared__ __align__(16) __half key_tile[kKeyTile * kHeadDimension];
    __shared__ __align__(16) __half value_tile[kKeyTile * kHeadDimension];
    __shared__ __align__(16) float scores[kQueryTile * kKeyTile];
    __shared__ __align__(16) __half probabilities[kQueryTile * kKeyTile];
    __shared__ __align__(16) float probability_value[kQueryTile * kHeadDimension];
    __shared__ float running_max[kQueryTile];
    __shared__ float running_sum[kQueryTile];
    __shared__ float row_alpha[kQueryTile];

    const int thread = static_cast<int>(threadIdx.x);
    const int lane = thread & 31;
    const int warp = thread >> 5;
    const std::size_t key_count = FixedKeySequence == 0
        ? key_sequence : static_cast<std::size_t>(FixedKeySequence);
    const std::size_t query_tile_begin = static_cast<std::size_t>(blockIdx.x) * kQueryTile;
    const std::size_t head = static_cast<std::size_t>(blockIdx.y);
    const std::size_t batch_index = static_cast<std::size_t>(blockIdx.z);
    if (batch_index >= batch || head >= heads) return;

    for (int linear = thread; linear < kQueryTile * kHeadDimension; linear += kThreads) {
        const int local_query = linear / kHeadDimension;
        const int channel = linear % kHeadDimension;
        const std::size_t global_query = query_tile_begin + static_cast<std::size_t>(local_query);
        __half loaded = __float2half(0.0F);
        if (global_query < query_sequence) {
            const std::size_t offset =
                ((batch_index * query_sequence + global_query) * heads + head) * kHeadDimension +
                static_cast<std::size_t>(channel);
            loaded = query[offset];
        }
        query_tile[linear] = loaded;
    }
    if (thread < kQueryTile) {
        running_max[thread] = -CUDART_INF_F;
        running_sum[thread] = 0.0F;
        row_alpha[thread] = 0.0F;
    }
    __syncthreads();

    // Each thread owns eight [query,channel] output elements. QK^T and P*V
    // both use Tensor Core WMMA; only the online-softmax state and final output
    // remain in FP32 registers/shared memory.
    float output_accumulator[8] = {0.0F, 0.0F, 0.0F, 0.0F,
                                   0.0F, 0.0F, 0.0F, 0.0F};

    for (std::size_t key_begin = 0; key_begin < key_count; key_begin += kKeyTile) {
        for (int linear = thread; linear < kKeyTile * kHeadDimension; linear += kThreads) {
            const int local_key = linear / kHeadDimension;
            const int channel = linear % kHeadDimension;
            const std::size_t global_key = key_begin + static_cast<std::size_t>(local_key);
            __half loaded_key = __float2half(0.0F);
            __half loaded_value = __float2half(0.0F);
            if (global_key < key_count) {
                const std::size_t offset =
                    ((batch_index * key_count + global_key) * heads + head) * kHeadDimension +
                    static_cast<std::size_t>(channel);
                loaded_key = key[offset];
                loaded_value = value[offset];
            }
            key_tile[linear] = loaded_key;
            value_tile[linear] = loaded_value;
        }
        __syncthreads();

        // Two warps calculate QK^T for 32 queries x 16 keys. The row-major
        // [16,64] K tile is loaded as column-major [64,16], so no transpose
        // buffer or global score matrix is needed.
        if (warp < 2) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
            wmma::fill_fragment(c, 0.0F);
            const __half* q_base = query_tile + warp * 16 * kHeadDimension;
            for (int k = 0; k < kHeadDimension; k += 16) {
                wmma::load_matrix_sync(a, q_base + k, kHeadDimension);
                wmma::load_matrix_sync(b, key_tile + k, kHeadDimension);
                wmma::mma_sync(c, a, b, c);
            }
            wmma::store_matrix_sync(scores + warp * 16 * kKeyTile, c,
                                    kKeyTile, wmma::mem_row_major);
        }
        __syncthreads();

        // Eight warps normalize four rows each. P is stored as FP16 because it
        // immediately becomes the Tensor Core A operand for P*V.
        #pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int local_query = warp * 4 + item;
            const std::size_t global_query = query_tile_begin + static_cast<std::size_t>(local_query);
            const bool query_valid = global_query < query_sequence;
            float score = -CUDART_INF_F;
            if (lane < kKeyTile && query_valid) {
                const std::size_t global_key = key_begin + static_cast<std::size_t>(lane);
                const bool key_valid = global_key < key_count;
                const bool mask_valid = key_mask == nullptr || !key_valid ||
                    key_mask[batch_index * key_count + global_key] != 0;
                const bool causal_valid = !Causal || global_key <= global_query;
                if (key_valid && mask_valid && causal_valid) {
                    score = scores[local_query * kKeyTile + lane] * kAttentionScale;
                }
            }

            const float tile_maximum = warp_max(score);
            const float previous_maximum = running_max[local_query];
            const float next_maximum = fmaxf(previous_maximum, tile_maximum);
            const float alpha = isfinite(previous_maximum)
                ? __expf(previous_maximum - next_maximum) : 0.0F;
            const float beta = isfinite(score) ? __expf(score - next_maximum) : 0.0F;
            if (lane < kKeyTile) {
                probabilities[local_query * kKeyTile + lane] = __float2half_rn(beta);
            }
            const float tile_sum = warp_sum(beta);
            if (lane == 0) {
                row_alpha[local_query] = alpha;
                running_sum[local_query] = running_sum[local_query] * alpha + tile_sum;
                running_max[local_query] = next_maximum;
            }
        }
        __syncthreads();

        // All eight warps now compute P[16,16] * V[16,16]. Warp groups 0-3
        // cover query rows 0-15, groups 4-7 cover rows 16-31, and each warp
        // owns one 16-channel output tile.
        {
            const int query_block = warp >> 2;
            const int channel_block = warp & 3;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> p_fragment;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> v_fragment;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> pv_fragment;
            wmma::fill_fragment(pv_fragment, 0.0F);
            wmma::load_matrix_sync(
                p_fragment, probabilities + query_block * 16 * kKeyTile, kKeyTile);
            wmma::load_matrix_sync(
                v_fragment, value_tile + channel_block * 16, kHeadDimension);
            wmma::mma_sync(pv_fragment, p_fragment, v_fragment, pv_fragment);
            wmma::store_matrix_sync(
                probability_value + query_block * 16 * kHeadDimension + channel_block * 16,
                pv_fragment, kHeadDimension, wmma::mem_row_major);
        }
        __syncthreads();

        #pragma unroll
        for (int item = 0; item < 8; ++item) {
            const int linear = thread + item * kThreads;
            const int local_query = linear / kHeadDimension;
            output_accumulator[item] = output_accumulator[item] * row_alpha[local_query] +
                                       probability_value[linear];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int item = 0; item < 8; ++item) {
        const int linear = thread + item * kThreads;
        const int local_query = linear / kHeadDimension;
        const int channel = linear % kHeadDimension;
        const std::size_t global_query = query_tile_begin + static_cast<std::size_t>(local_query);
        if (global_query >= query_sequence) continue;
        const float inverse = running_sum[local_query] > 0.0F
            ? 1.0F / running_sum[local_query] : 0.0F;
        const std::size_t output_offset =
            ((batch_index * query_sequence + global_query) * heads + head) * kHeadDimension +
            static_cast<std::size_t>(channel);
        output[output_offset] = __float2half_rn(output_accumulator[item] * inverse);
    }
#else
    (void)query; (void)key; (void)value; (void)key_mask; (void)output;
    (void)batch; (void)query_sequence; (void)key_sequence; (void)heads;
#endif
}

} // namespace

void launch_flash_attention_sm80_hdim64(const Runtime& runtime,
                                        const Tensor& query,
                                        const Tensor& key,
                                        const Tensor& value,
                                        const Tensor* key_mask,
                                        Tensor& output,
                                        std::size_t heads,
                                        bool causal) {
    const dim3 grid(
        static_cast<unsigned>((query.size(1) + kQueryTile - 1) / kQueryTile),
        static_cast<unsigned>(heads),
        static_cast<unsigned>(query.size(0)));
    const auto launch = [&](auto fixed_keys, auto causal_tag) {
        constexpr int fixed = decltype(fixed_keys)::value;
        constexpr bool is_causal = decltype(causal_tag)::value;
        flash_attention_hdim64_kernel<fixed, is_causal>
            <<<grid, kThreads, 0, runtime.stream()>>>(
                query.half_data(), key.half_data(), value.half_data(),
                key_mask == nullptr ? nullptr : key_mask->int32_data(),
                output.half_data(), query.size(0), query.size(1),
                key.size(1), heads);
    };
    const auto dispatch_keys = [&](auto causal_tag) {
        switch (key.size(1)) {
        case 77: launch(std::integral_constant<int, 77>{}, causal_tag); break;
        case 256: launch(std::integral_constant<int, 256>{}, causal_tag); break;
        case 1024: launch(std::integral_constant<int, 1024>{}, causal_tag); break;
        case 4096: launch(std::integral_constant<int, 4096>{}, causal_tag); break;
        case 16384: launch(std::integral_constant<int, 16384>{}, causal_tag); break;
        default: launch(std::integral_constant<int, 0>{}, causal_tag); break;
        }
    };
    if (causal) dispatch_keys(std::true_type{});
    else dispatch_keys(std::false_type{});
    SDXL_CUDA_CHECK(cudaGetLastError());
}

} // namespace sdxl::cuda
