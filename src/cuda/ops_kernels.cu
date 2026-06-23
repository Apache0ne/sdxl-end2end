#include "sdxl/cuda/ops.hpp"
#include "sdxl/cuda/profiler.hpp"
#include "runtime_internal.hpp"
#include "flash_attention_sm80.hpp"
#include "cudnn_sdpa.hpp"

#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <type_traits>
#include <mutex>
#include <string>

namespace sdxl::cuda {
namespace {

constexpr unsigned kThreads = 256;

[[nodiscard]] unsigned blocks_for(std::size_t count, unsigned threads = kThreads) {
    return static_cast<unsigned>((count + threads - 1) / threads);
}

[[nodiscard]] unsigned attention_threads(std::size_t head_dimension) {
    unsigned threads = 32;
    while (threads < head_dimension && threads < 1024U) threads <<= 1U;
    return threads;
}

__device__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return value;
}

__device__ float block_sum(float value, float* shared) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_sum(value);
    if (lane == 0) shared[warp] = value;
    __syncthreads();
    value = threadIdx.x < (blockDim.x + 31) / 32 ? shared[lane] : 0.0F;
    if (warp == 0) value = warp_sum(value);
    if (threadIdx.x == 0) shared[0] = value;
    __syncthreads();
    return shared[0];
}

__global__ void add_kernel(const __half* first, const __half* second,
                           __half* output, std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = __hadd(first[index], second[index]);
}

__global__ void add_in_place_kernel(__half* destination, const __half* source,
                                    std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) destination[index] = __hadd(destination[index], source[index]);
}

__global__ void add_silu_kernel(const __half* first, const __half* second,
                                __half* output, std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const float x = __half2float(first[index]) + __half2float(second[index]);
        output[index] = __float2half_rn(x / (1.0F + expf(-x)));
    }
}


__global__ void add_last_dim_bias_kernel(__half* destination, const __half* bias,
                                         std::size_t count, std::size_t width) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        destination[index] = __hadd(destination[index], bias[index % width]);
    }
}

__global__ void scale_kernel(const __half* input, __half* output,
                                     std::size_t count, float scale) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = __float2half_rn(__half2float(input[index]) * scale);
}

__global__ void scale_in_place_kernel(__half* values, std::size_t count, float scale) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) values[index] = __float2half_rn(__half2float(values[index]) * scale);
}

__global__ void silu_kernel(const __half* input, __half* output, std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const float x = __half2float(input[index]);
        output[index] = __float2half_rn(x / (1.0F + expf(-x)));
    }
}

__global__ void silu_in_place_kernel(__half* values, std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const float x = __half2float(values[index]);
        values[index] = __float2half_rn(x / (1.0F + expf(-x)));
    }
}

__global__ void quick_gelu_kernel(const __half* input, __half* output, std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const float x = __half2float(input[index]);
        output[index] = __float2half_rn(x / (1.0F + expf(-1.702F * x)));
    }
}


__global__ void gelu_kernel(const __half* input, __half* output, std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const float x = __half2float(input[index]);
        output[index] = __float2half_rn(0.5F * x * (1.0F + erff(x * 0.7071067811865475F)));
    }
}

__global__ void geglu_kernel(const __half* input, __half* output,
                             std::size_t rows, std::size_t width) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = rows * width;
    if (index < count) {
        const std::size_t row = index / width;
        const std::size_t column = index % width;
        const float value = __half2float(input[row * width * 2 + column]);
        const float gate = __half2float(input[row * width * 2 + width + column]);
        const float gelu = 0.5F * gate * (1.0F + erff(gate * 0.7071067811865475F));
        output[index] = __float2half_rn(value * gelu);
    }
}

__device__ __forceinline__ float apply_linear_activation(float x, int activation) {
    if (activation == static_cast<int>(LinearActivation::SiLU)) {
        return x / (1.0F + expf(-x));
    }
    if (activation == static_cast<int>(LinearActivation::GELU)) {
        return 0.5F * x * (1.0F + erff(x * 0.7071067811865475F));
    }
    if (activation == static_cast<int>(LinearActivation::QuickGELU)) {
        return x / (1.0F + expf(-1.702F * x));
    }
    return x;
}

__global__ void bias_activation_in_place_kernel(__half* values,
                                                 const __half* bias,
                                                 std::size_t count,
                                                 std::size_t width,
                                                 int activation) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const float x = __half2float(values[index]) + __half2float(bias[index % width]);
        values[index] = __float2half_rn(apply_linear_activation(x, activation));
    }
}

__global__ void bias_geglu_kernel(const __half* projected,
                                  const __half* bias,
                                  __half* output,
                                  std::size_t rows,
                                  std::size_t width) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < rows * width) {
        const std::size_t row = index / width;
        const std::size_t column = index % width;
        const std::size_t base = row * width * 2;
        const float value = __half2float(projected[base + column]) + __half2float(bias[column]);
        const float gate = __half2float(projected[base + width + column]) +
                           __half2float(bias[width + column]);
        const float gelu = 0.5F * gate * (1.0F + erff(gate * 0.7071067811865475F));
        output[index] = __float2half_rn(value * gelu);
    }
}

__global__ void layer_norm_kernel(const __half* input,
                                  const __half* weight,
                                  const __half* bias,
                                  __half* output,
                                  std::size_t rows,
                                  std::size_t width,
                                  float epsilon) {
    extern __shared__ float shared[];
    const std::size_t row = blockIdx.x;
    if (row >= rows) return;

    float sum = 0.0F;
    for (std::size_t column = threadIdx.x; column < width; column += blockDim.x) {
        sum += __half2float(input[row * width + column]);
    }
    const float total = block_sum(sum, shared);
    const float mean = total / static_cast<float>(width);

    // Stable two-pass variance. E[x^2] - E[x]^2 loses too much precision for
    // large FP16 transformer activations and was the source of late-UNet NaNs.
    float squared_centered = 0.0F;
    for (std::size_t column = threadIdx.x; column < width; column += blockDim.x) {
        const float centered = __half2float(input[row * width + column]) - mean;
        squared_centered += centered * centered;
    }
    const float total_squared_centered = block_sum(squared_centered, shared);
    const float variance = fmaxf(
        0.0F, total_squared_centered / static_cast<float>(width));
    const float inverse = rsqrtf(variance + epsilon);

    for (std::size_t column = threadIdx.x; column < width; column += blockDim.x) {
        const float value = __half2float(input[row * width + column]);
        const float gamma = __half2float(weight[column]);
        const float beta = __half2float(bias[column]);
        output[row * width + column] =
            __float2half_rn((value - mean) * inverse * gamma + beta);
    }
}

template <std::size_t Width>
__global__ void layer_norm_fixed_kernel(const __half* input,
                                        const __half* weight,
                                        const __half* bias,
                                        __half* output,
                                        std::size_t rows,
                                        float epsilon) {
    extern __shared__ float shared[];
    const std::size_t row = blockIdx.x;
    if (row >= rows) return;

    float sum = 0.0F;
    for (std::size_t column = threadIdx.x; column < Width; column += blockDim.x) {
        sum += __half2float(input[row * Width + column]);
    }
    const float mean = block_sum(sum, shared) / static_cast<float>(Width);

    float squared_centered = 0.0F;
    for (std::size_t column = threadIdx.x; column < Width; column += blockDim.x) {
        const float centered = __half2float(input[row * Width + column]) - mean;
        squared_centered += centered * centered;
    }
    const float variance = fmaxf(
        0.0F, block_sum(squared_centered, shared) / static_cast<float>(Width));
    const float inverse = rsqrtf(variance + epsilon);

    for (std::size_t column = threadIdx.x; column < Width; column += blockDim.x) {
        const float value = __half2float(input[row * Width + column]);
        output[row * Width + column] = __float2half_rn(
            (value - mean) * inverse * __half2float(weight[column]) +
            __half2float(bias[column]));
    }
}

__global__ void group_norm_kernel(const __half* input,
                                  const __half* weight,
                                  const __half* bias,
                                  __half* output,
                                  std::size_t batch,
                                  std::size_t channels,
                                  std::size_t height,
                                  std::size_t width,
                                  std::size_t groups,
                                  float epsilon,
                                  bool apply_silu) {
    extern __shared__ float shared[];
    const std::size_t group_index = blockIdx.x;
    if (group_index >= batch * groups) return;
    const std::size_t batch_index = group_index / groups;
    const std::size_t group = group_index % groups;
    const std::size_t channels_per_group = channels / groups;
    const std::size_t spatial = height * width;
    const std::size_t count = channels_per_group * spatial;
    const std::size_t channel_begin = group * channels_per_group;

    float sum = 0.0F;
    for (std::size_t local = threadIdx.x; local < count; local += blockDim.x) {
        const std::size_t channel = channel_begin + local / spatial;
        const std::size_t position = local % spatial;
        sum += __half2float(input[(batch_index * channels + channel) * spatial + position]);
    }
    const float mean = block_sum(sum, shared) / static_cast<float>(count);

    float squared_centered = 0.0F;
    for (std::size_t local = threadIdx.x; local < count; local += blockDim.x) {
        const std::size_t channel = channel_begin + local / spatial;
        const std::size_t position = local % spatial;
        const float centered =
            __half2float(input[(batch_index * channels + channel) * spatial + position]) - mean;
        squared_centered += centered * centered;
    }
    const float variance = fmaxf(
        0.0F, block_sum(squared_centered, shared) / static_cast<float>(count));
    const float inverse = rsqrtf(variance + epsilon);

    for (std::size_t local = threadIdx.x; local < count; local += blockDim.x) {
        const std::size_t channel = channel_begin + local / spatial;
        const std::size_t position = local % spatial;
        const std::size_t index = (batch_index * channels + channel) * spatial + position;
        const float value = __half2float(input[index]);
        float normalized =
            (value - mean) * inverse * __half2float(weight[channel]) +
            __half2float(bias[channel]);
        if (apply_silu) normalized = normalized / (1.0F + expf(-normalized));
        output[index] = __float2half_rn(normalized);
    }
}

__global__ void add_spatial_bias_kernel(const __half* input,
                                        const __half* bias,
                                        __half* output,
                                        std::size_t batch,
                                        std::size_t channels,
                                        std::size_t spatial) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = batch * channels * spatial;
    if (index < count) {
        const std::size_t channel = (index / spatial) % channels;
        const std::size_t batch_index = index / (channels * spatial);
        output[index] = __float2half_rn(__half2float(input[index]) +
                                        __half2float(bias[batch_index * channels + channel]));
    }
}

__global__ void flatten_spatial_kernel(const __half* input,
                                       __half* output,
                                       std::size_t batch,
                                       std::size_t channels,
                                       std::size_t height,
                                       std::size_t width) {
    const std::size_t count = batch * channels * height * width;
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const std::size_t x = index % width;
        const std::size_t y = (index / width) % height;
        const std::size_t channel = (index / (width * height)) % channels;
        const std::size_t batch_index = index / (width * height * channels);
        const std::size_t sequence = y * width + x;
        output[(batch_index * height * width + sequence) * channels + channel] = input[index];
    }
}

__global__ void unflatten_spatial_kernel(const __half* input,
                                         __half* output,
                                         std::size_t batch,
                                         std::size_t channels,
                                         std::size_t height,
                                         std::size_t width) {
    const std::size_t count = batch * channels * height * width;
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const std::size_t x = index % width;
        const std::size_t y = (index / width) % height;
        const std::size_t channel = (index / (width * height)) % channels;
        const std::size_t batch_index = index / (width * height * channels);
        const std::size_t sequence = y * width + x;
        output[index] = input[(batch_index * height * width + sequence) * channels + channel];
    }
}

__global__ void concat_channels_kernel(const __half* first,
                                       const __half* second,
                                       __half* output,
                                       std::size_t batch,
                                       std::size_t first_channels,
                                       std::size_t second_channels,
                                       std::size_t spatial) {
    const std::size_t total_channels = first_channels + second_channels;
    const std::size_t count = batch * total_channels * spatial;
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const std::size_t position = index % spatial;
        const std::size_t channel = (index / spatial) % total_channels;
        const std::size_t batch_index = index / (spatial * total_channels);
        if (channel < first_channels) {
            output[index] = first[(batch_index * first_channels + channel) * spatial + position];
        } else {
            output[index] = second[(batch_index * second_channels + channel - first_channels) * spatial + position];
        }
    }
}

__global__ void concat_last_kernel(const __half* first,
                                   const __half* second,
                                   __half* output,
                                   std::size_t rows,
                                   std::size_t first_width,
                                   std::size_t second_width) {
    const std::size_t total = first_width + second_width;
    const std::size_t count = rows * total;
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const std::size_t row = index / total;
        const std::size_t column = index % total;
        output[index] = column < first_width
                            ? first[row * first_width + column]
                            : second[row * second_width + column - first_width];
    }
}

__global__ void repeat_batch_kernel(const __half* input, __half* output,
                                    std::size_t per_batch,
                                    std::size_t batch,
                                    std::size_t repeats) {
    const std::size_t count = per_batch * batch * repeats;
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const std::size_t source_batch = (index / per_batch) % batch;
        output[index] = input[source_batch * per_batch + index % per_batch];
    }
}

__global__ void nearest_upsample_kernel(const __half* input,
                                        __half* output,
                                        std::size_t batch,
                                        std::size_t channels,
                                        std::size_t input_height,
                                        std::size_t input_width,
                                        std::size_t output_height,
                                        std::size_t output_width) {
    const std::size_t count = batch * channels * output_height * output_width;
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const std::size_t x = index % output_width;
        const std::size_t y = (index / output_width) % output_height;
        const std::size_t channel = (index / (output_width * output_height)) % channels;
        const std::size_t batch_index = index / (output_width * output_height * channels);
        const std::size_t mapped_y = y * input_height / output_height;
        const std::size_t source_y = mapped_y < input_height ? mapped_y : input_height - 1;
        const std::size_t mapped_x = x * input_width / output_width;
        const std::size_t source_x = mapped_x < input_width ? mapped_x : input_width - 1;
        output[index] = input[((batch_index * channels + channel) * input_height + source_y) * input_width + source_x];
    }
}

__global__ void embedding_kernel(const std::int32_t* token_ids,
                                 const __half* token_embedding,
                                 const __half* position_embedding,
                                 __half* output,
                                 std::size_t batch,
                                 std::size_t sequence,
                                 std::size_t hidden,
                                 std::size_t vocabulary) {
    const std::size_t count = batch * sequence * hidden;
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const std::size_t hidden_index = index % hidden;
        const std::size_t position = (index / hidden) % sequence;
        const std::size_t batch_index = index / (hidden * sequence);
        const int token = token_ids[batch_index * sequence + position];
        if (token >= 0 && static_cast<std::size_t>(token) < vocabulary) {
            output[index] = __hadd(token_embedding[static_cast<std::size_t>(token) * hidden + hidden_index],
                                   position_embedding[position * hidden + hidden_index]);
        } else {
            output[index] = __float2half(CUDART_NAN_F);
        }
    }
}

__global__ void pool_eos_kernel(const __half* hidden,
                                const std::int32_t* token_ids,
                                __half* output,
                                std::size_t batch,
                                std::size_t sequence,
                                std::size_t width) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = batch * width;
    if (index < count) {
        const std::size_t batch_index = index / width;
        const std::size_t channel = index % width;
        std::size_t selected = 0;
        int maximum = token_ids[batch_index * sequence];
        for (std::size_t position = 1; position < sequence; ++position) {
            const int value = token_ids[batch_index * sequence + position];
            if (value > maximum) {
                maximum = value;
                selected = position;
            }
        }
        output[index] = hidden[(batch_index * sequence + selected) * width + channel];
    }
}

// Warp-specialized online-softmax attention for SDXL's fixed 64-wide heads.
// One warp computes one (batch, head, query) row; each lane owns two channels.
// This removes the shared-memory reductions and block-wide barriers used by the
// original reference kernel while retaining O(Q*D) output storage.
template <int FixedKeySequence>
__global__ void online_attention_kernel(const __half* query,
                                        const __half* key,
                                        const __half* value,
                                        const std::int32_t* key_mask,
                                        __half* output,
                                        std::size_t batch,
                                        std::size_t query_sequence,
                                        std::size_t key_sequence,
                                        std::size_t heads,
                                        std::size_t head_dimension,
                                        bool causal) {
    const std::size_t block = blockIdx.x;
    const std::size_t query_position = block % query_sequence;
    const std::size_t head = (block / query_sequence) % heads;
    const std::size_t batch_index = block / (query_sequence * heads);
    if (batch_index >= batch || head_dimension != 64 || blockDim.x != 32) return;

    const unsigned lane = threadIdx.x;
    const std::size_t effective_key_sequence = FixedKeySequence > 0
        ? static_cast<std::size_t>(FixedKeySequence) : key_sequence;
    const std::size_t query_base =
        (batch_index * query_sequence + query_position) * heads * 64 + head * 64;
    const float q0 = __half2float(query[query_base + lane]);
    const float q1 = __half2float(query[query_base + lane + 32]);
    float acc0 = 0.0F;
    float acc1 = 0.0F;
    float running_max = -CUDART_INF_F;
    float running_sum = 0.0F;
    const std::size_t visible_keys = causal
        ? (query_position + 1 < effective_key_sequence
            ? query_position + 1 : effective_key_sequence)
        : effective_key_sequence;
    constexpr float scale = 0.125F; // 1/sqrt(64)

    for (std::size_t key_position = 0; key_position < visible_keys; ++key_position) {
        const bool visible = key_mask == nullptr ||
            key_mask[batch_index * effective_key_sequence + key_position] != 0;
        const std::size_t key_base =
            (batch_index * effective_key_sequence + key_position) * heads * 64 + head * 64;
        float dot = visible
            ? q0 * __half2float(key[key_base + lane]) +
              q1 * __half2float(key[key_base + lane + 32])
            : 0.0F;
        dot = warp_sum(dot);

        float alpha = 0.0F;
        float beta = 0.0F;
        if (lane == 0) {
            const float score = visible ? dot * scale : -CUDART_INF_F;
            const float next_max = fmaxf(running_max, score);
            alpha = isfinite(running_max) ? expf(running_max - next_max) : 0.0F;
            beta = isfinite(score) ? expf(score - next_max) : 0.0F;
            running_sum = running_sum * alpha + beta;
            running_max = next_max;
        }
        alpha = __shfl_sync(0xffffffffU, alpha, 0);
        beta = __shfl_sync(0xffffffffU, beta, 0);
        acc0 = acc0 * alpha + beta * __half2float(value[key_base + lane]);
        acc1 = acc1 * alpha + beta * __half2float(value[key_base + lane + 32]);
    }
    running_sum = __shfl_sync(0xffffffffU, running_sum, 0);
    const float inverse = 1.0F / fmaxf(running_sum, 1.0e-20F);
    output[query_base + lane] = __float2half_rn(acc0 * inverse);
    output[query_base + lane + 32] = __float2half_rn(acc1 * inverse);
}

// Generic online-softmax attention used by the VAE mid-block (one 512-wide head).
// It keeps O(B*H*Q*D) storage and never materializes the QxK score matrix.
__global__ void generic_online_attention_kernel(const __half* query,
                                                const __half* key,
                                                const __half* value,
                                                const std::int32_t* key_mask,
                                                __half* output,
                                                std::size_t batch,
                                                std::size_t query_sequence,
                                                std::size_t key_sequence,
                                                std::size_t heads,
                                                std::size_t head_dimension,
                                                bool causal) {
    extern __shared__ float shared[];
    const unsigned warp_count = (blockDim.x + 31U) / 32U;
    float* warp_sums = shared;
    float* scalar = shared + warp_count;

    const std::size_t block = blockIdx.x;
    const std::size_t query_position = block % query_sequence;
    const std::size_t head = (block / query_sequence) % heads;
    const std::size_t batch_index = block / (query_sequence * heads);
    if (batch_index >= batch) return;

    const std::size_t dimension = threadIdx.x;
    const std::size_t query_base =
        (batch_index * query_sequence + query_position) * heads * head_dimension +
        head * head_dimension;
    const float query_value = dimension < head_dimension
        ? __half2float(query[query_base + dimension]) : 0.0F;
    float accumulator = 0.0F;
    float running_max = -CUDART_INF_F;
    float running_sum = 0.0F;
    const std::size_t causal_limit = query_position + 1 < key_sequence
        ? query_position + 1 : key_sequence;
    const std::size_t visible_keys = causal ? causal_limit : key_sequence;
    const float scale = rsqrtf(static_cast<float>(head_dimension));

    for (std::size_t key_position = 0; key_position < visible_keys; ++key_position) {
        const bool visible = key_mask == nullptr ||
            key_mask[batch_index * key_sequence + key_position] != 0;
        const std::size_t key_base =
            (batch_index * key_sequence + key_position) * heads * head_dimension +
            head * head_dimension;
        float product = 0.0F;
        if (visible && dimension < head_dimension) {
            product = query_value * __half2float(key[key_base + dimension]);
        }
        const float dot = block_sum(product, warp_sums);
        if (threadIdx.x == 0) {
            const float score = visible ? dot * scale : -CUDART_INF_F;
            const float next_max = fmaxf(running_max, score);
            const float alpha = isfinite(running_max) ? expf(running_max - next_max) : 0.0F;
            const float beta = isfinite(score) ? expf(score - next_max) : 0.0F;
            running_sum = running_sum * alpha + beta;
            running_max = next_max;
            scalar[0] = alpha;
            scalar[1] = beta;
        }
        __syncthreads();
        if (dimension < head_dimension) {
            accumulator = accumulator * scalar[0] +
                scalar[1] * __half2float(value[key_base + dimension]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) scalar[0] = running_sum;
    __syncthreads();
    if (dimension < head_dimension) {
        output[query_base + dimension] =
            __float2half_rn(accumulator / fmaxf(scalar[0], 1.0e-20F));
    }
}

__global__ void timestep_scalar_kernel(float timestep,
                                       __half* output,
                                       std::size_t batch,
                                       std::size_t dimension) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = batch * dimension;
    if (index < count) {
        const std::size_t column = index % dimension;
        const std::size_t half = dimension / 2;
        const std::size_t frequency = column < half ? column : column - half;
        const float exponent = -logf(10000.0F) * static_cast<float>(frequency) /
                               static_cast<float>(half == 0 ? 1 : half);
        const float angle = timestep * expf(exponent);
        output[index] = __float2half_rn(column < half ? cosf(angle) : sinf(angle));
    }
}

__global__ void timestep_values_kernel(const void* values,
                                       int value_type,
                                       __half* output,
                                       std::size_t rows,
                                       std::size_t dimension) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = rows * dimension;
    if (index < count) {
        const std::size_t row = index / dimension;
        const std::size_t column = index % dimension;
        const std::size_t half = dimension / 2;
        const std::size_t frequency = column < half ? column : column - half;
        const float scalar = value_type == 0
                                 ? __half2float(static_cast<const __half*>(values)[row])
                                 : static_cast<const float*>(values)[row];
        const float exponent = -logf(10000.0F) * static_cast<float>(frequency) /
                               static_cast<float>(half == 0 ? 1 : half);
        const float angle = scalar * expf(exponent);
        output[index] = __float2half_rn(column < half ? cosf(angle) : sinf(angle));
    }
}

__global__ void cfg_kernel(const __half* model_output,
                           __half* guided,
                           std::size_t batch,
                           std::size_t per_batch,
                           float guidance_scale) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = batch * per_batch;
    if (index < count) {
        const std::size_t batch_index = index / per_batch;
        const std::size_t local = index % per_batch;
        const float unconditional = __half2float(model_output[batch_index * per_batch + local]);
        const float conditional = __half2float(model_output[(batch + batch_index) * per_batch + local]);
        guided[index] = __float2half_rn(unconditional + guidance_scale * (conditional - unconditional));
    }
}

__global__ void guidance_stats_kernel(const __half* guided,
                                      const __half* model_output,
                                      __half* ratios,
                                      std::size_t batch,
                                      std::size_t per_batch) {
    extern __shared__ float shared[];
    float* first = shared;
    float* second = shared + 8;
    const std::size_t batch_index = blockIdx.x;
    if (batch_index >= batch) return;
    float guided_sum = 0.0F;
    float conditional_sum = 0.0F;
    for (std::size_t index = threadIdx.x; index < per_batch; index += blockDim.x) {
        guided_sum += __half2float(guided[batch_index * per_batch + index]);
        conditional_sum += __half2float(
            model_output[(batch + batch_index) * per_batch + index]);
    }
    const float guided_mean =
        block_sum(guided_sum, first) / static_cast<float>(per_batch);
    const float conditional_mean =
        block_sum(conditional_sum, second) / static_cast<float>(per_batch);

    // Guidance rescale uses the same stable centered two-pass variance policy
    // as LayerNorm. This avoids cancellation when predictions have a large
    // common offset but relatively small variation.
    float guided_squared_centered = 0.0F;
    float conditional_squared_centered = 0.0F;
    for (std::size_t index = threadIdx.x; index < per_batch; index += blockDim.x) {
        const float g = __half2float(guided[batch_index * per_batch + index]) - guided_mean;
        const float c = __half2float(
            model_output[(batch + batch_index) * per_batch + index]) - conditional_mean;
        guided_squared_centered += g * g;
        conditional_squared_centered += c * c;
    }
    const float denominator = static_cast<float>(per_batch > 1 ? per_batch - 1 : 1);
    const float guided_variance = fmaxf(
        0.0F, block_sum(guided_squared_centered, first) / denominator);
    const float conditional_variance = fmaxf(
        0.0F, block_sum(conditional_squared_centered, second) / denominator);
    if (threadIdx.x == 0) {
        ratios[batch_index] = __float2half_rn(
            guided_variance > 0.0F
                ? sqrtf(conditional_variance / guided_variance)
                : 1.0F);
    }
}

__global__ void guidance_rescale_kernel(__half* guided,
                                        const __half* ratios,
                                        std::size_t batch,
                                        std::size_t per_batch,
                                        float amount) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = batch * per_batch;
    if (index < count) {
        const std::size_t batch_index = index / per_batch;
        const float value = __half2float(guided[index]);
        const float rescaled = value * __half2float(ratios[batch_index]);
        guided[index] = __float2half_rn(amount * rescaled + (1.0F - amount) * value);
    }
}

__global__ void random_normal_half_kernel(__half* output,
                                         std::size_t count,
                                         unsigned long long seed,
                                         float scale) {
    const std::size_t group = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t base = group * 4;
    if (base >= count) return;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, static_cast<unsigned long long>(group), 0ULL, &state);
    const float4 values = curand_normal4(&state);
    const float samples[4] = {values.x, values.y, values.z, values.w};
    #pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
        const std::size_t index = base + static_cast<std::size_t>(lane);
        if (index < count) output[index] = __float2half_rn(samples[lane] * scale);
    }
}

__global__ void random_normal_float_kernel(float* output,
                                          std::size_t count,
                                          unsigned long long seed,
                                          float scale) {
    const std::size_t group = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t base = group * 4;
    if (base >= count) return;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, static_cast<unsigned long long>(group), 0ULL, &state);
    const float4 values = curand_normal4(&state);
    const float samples[4] = {values.x, values.y, values.z, values.w};
    #pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
        const std::size_t index = base + static_cast<std::size_t>(lane);
        if (index < count) output[index] = samples[lane] * scale;
    }
}

__global__ void random_normal_batch_half_kernel(__half* output,
                                               std::size_t per_batch,
                                               std::size_t batch,
                                               const unsigned long long* seeds,
                                               float scale) {
    const std::size_t group = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t groups_per_batch = (per_batch + 3U) / 4U;
    const std::size_t total_groups = groups_per_batch * batch;
    if (group >= total_groups) return;
    const std::size_t batch_index = group / groups_per_batch;
    const std::size_t local_group = group % groups_per_batch;
    const std::size_t base = batch_index * per_batch + local_group * 4U;
    curandStatePhilox4_32_10_t state;
    curand_init(seeds[batch_index], static_cast<unsigned long long>(local_group), 0ULL, &state);
    const float4 values = curand_normal4(&state);
    const float samples[4] = {values.x, values.y, values.z, values.w};
    #pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
        const std::size_t local = local_group * 4U + static_cast<std::size_t>(lane);
        if (local < per_batch) output[base + static_cast<std::size_t>(lane)] =
            __float2half_rn(samples[lane] * scale);
    }
}

__global__ void random_normal_batch_float_kernel(float* output,
                                                std::size_t per_batch,
                                                std::size_t batch,
                                                const unsigned long long* seeds,
                                                float scale) {
    const std::size_t group = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t groups_per_batch = (per_batch + 3U) / 4U;
    const std::size_t total_groups = groups_per_batch * batch;
    if (group >= total_groups) return;
    const std::size_t batch_index = group / groups_per_batch;
    const std::size_t local_group = group % groups_per_batch;
    const std::size_t base = batch_index * per_batch + local_group * 4U;
    curandStatePhilox4_32_10_t state;
    curand_init(seeds[batch_index], static_cast<unsigned long long>(local_group), 0ULL, &state);
    const float4 values = curand_normal4(&state);
    const float samples[4] = {values.x, values.y, values.z, values.w};
    #pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
        const std::size_t local = local_group * 4U + static_cast<std::size_t>(lane);
        if (local < per_batch) output[base + static_cast<std::size_t>(lane)] =
            samples[lane] * scale;
    }
}

__global__ void euler_scale_kernel(const __half* sample, __half* output,
                                   std::size_t count, float inverse_scale) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = __float2half_rn(__half2float(sample[index]) * inverse_scale);
}

__global__ void euler_scale_repeat_kernel(const __half* sample,
                                           __half* output,
                                           std::size_t input_count,
                                           std::size_t repeats,
                                           float inverse_scale) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = input_count * repeats;
    if (index < count) {
        output[index] = __float2half_rn(
            __half2float(sample[index % input_count]) * inverse_scale);
    }
}

__global__ void euler_scale_f32_to_half_kernel(const float* sample, __half* output,
                                               std::size_t count, float inverse_scale) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = __float2half_rn(sample[index] * inverse_scale);
}

__global__ void euler_scale_repeat_f32_to_half_kernel(const float* sample,
                                                      __half* output,
                                                      std::size_t input_count,
                                                      std::size_t repeats,
                                                      float inverse_scale) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = input_count * repeats;
    if (index < count) {
        output[index] = __float2half_rn(sample[index % input_count] * inverse_scale);
    }
}

__device__ float cfg_value(const __half* model_output,
                           std::size_t index,
                           std::size_t batch,
                           std::size_t per_batch,
                           float guidance_scale) {
    const std::size_t batch_index = index / per_batch;
    const std::size_t local = index % per_batch;
    const float unconditional =
        __half2float(model_output[batch_index * per_batch + local]);
    const float conditional =
        __half2float(model_output[(batch + batch_index) * per_batch + local]);
    return unconditional + guidance_scale * (conditional - unconditional);
}

__global__ void predicted_original_kernel(const __half* model_output,
                                          const __half* sample,
                                          __half* output,
                                          std::size_t count,
                                          float sigma,
                                          int prediction_type) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const float x = __half2float(sample[index]);
        const float prediction = __half2float(model_output[index]);
        float original = 0.0F;
        if (prediction_type == 0) original = x - sigma * prediction;
        else if (prediction_type == 1) original = prediction;
        else original = prediction * (-sigma / sqrtf(sigma * sigma + 1.0F)) +
                        x / (sigma * sigma + 1.0F);
        output[index] = __float2half_rn(original);
    }
}

__global__ void predicted_original_f16_f32_kernel(const __half* model_output,
                                                  const float* sample,
                                                  float* output,
                                                  std::size_t count,
                                                  float sigma,
                                                  int prediction_type) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const float x = sample[index];
        const float prediction = __half2float(model_output[index]);
        float original = 0.0F;
        if (prediction_type == 0) original = x - sigma * prediction;
        else if (prediction_type == 1) original = prediction;
        else original = prediction * (-sigma / sqrtf(sigma * sigma + 1.0F)) +
                        x / (sigma * sigma + 1.0F);
        output[index] = original;
    }
}

__global__ void combine_half_kernel(const __half* a, const __half* b,
                                    const __half* c, const __half* noise,
                                    __half* output, std::size_t count,
                                    float ca, float cb, float cc, float cn) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        float value = ca * __half2float(a[index]);
        if (b != nullptr) value += cb * __half2float(b[index]);
        if (c != nullptr) value += cc * __half2float(c[index]);
        if (noise != nullptr) value += cn * __half2float(noise[index]);
        output[index] = __float2half_rn(value);
    }
}

__global__ void combine_float_kernel(const float* a, const float* b,
                                     const float* c, const float* noise,
                                     float* output, std::size_t count,
                                     float ca, float cb, float cc, float cn) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        float value = ca * a[index];
        if (b != nullptr) value += cb * b[index];
        if (c != nullptr) value += cc * c[index];
        if (noise != nullptr) value += cn * noise[index];
        output[index] = value;
    }
}

__global__ void dpmpp_2m_step_kernel(const __half* denoised,
                                     const __half* sample,
                                     const __half* old_denoised,
                                     __half* output,
                                     std::size_t count,
                                     float sample_coefficient,
                                     float denoised_coefficient,
                                     float old_coefficient) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const float old_value = old_denoised == nullptr ? 0.0F :
                                __half2float(old_denoised[index]);
        const float result = sample_coefficient * __half2float(sample[index]) +
                             denoised_coefficient * __half2float(denoised[index]) +
                             old_coefficient * old_value;
        output[index] = __float2half_rn(result);
    }
}

__global__ void dpmpp_2m_step_float_kernel(const float* denoised,
                                           const float* sample,
                                           const float* old_denoised,
                                           float* output,
                                           std::size_t count,
                                           float sample_coefficient,
                                           float denoised_coefficient,
                                           float old_coefficient) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const float old_value = old_denoised == nullptr ? 0.0F : old_denoised[index];
        output[index] = sample_coefficient * sample[index] +
                        denoised_coefficient * denoised[index] +
                        old_coefficient * old_value;
    }
}

__global__ void euler_cfg_step_kernel(const __half* model_output,
                                      const __half* sample,
                                      __half* output,
                                      std::size_t batch,
                                      std::size_t per_batch,
                                      float guidance_scale,
                                      float sigma,
                                      float delta_sigma,
                                      int prediction_type) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = batch * per_batch;
    if (index < count) {
        const float x = __half2float(sample[index]);
        const float prediction = cfg_value(
            model_output, index, batch, per_batch, guidance_scale);
        float original = 0.0F;
        if (prediction_type == 0) original = x - sigma * prediction;
        else if (prediction_type == 1) original = prediction;
        else original = prediction * (-sigma / sqrtf(sigma * sigma + 1.0F)) +
                        x / (sigma * sigma + 1.0F);
        const float derivative = sigma == 0.0F ? 0.0F : (x - original) / sigma;
        output[index] = __float2half_rn(x + derivative * delta_sigma);
    }
}

__global__ void euler_step_kernel(const __half* model_output,
                                  const __half* sample,
                                  __half* output,
                                  std::size_t count,
                                  float sigma,
                                  float delta_sigma,
                                  int prediction_type) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const float x = __half2float(sample[index]);
        const float prediction = __half2float(model_output[index]);
        float original = 0.0F;
        if (prediction_type == 0) original = x - sigma * prediction;
        else if (prediction_type == 1) original = prediction;
        else original = prediction * (-sigma / sqrtf(sigma * sigma + 1.0F)) + x / (sigma * sigma + 1.0F);
        const float derivative = sigma == 0.0F ? 0.0F : (x - original) / sigma;
        output[index] = __float2half_rn(x + derivative * delta_sigma);
    }
}

__global__ void euler_step_f16_f32_kernel(const __half* model_output,
                                            const float* sample,
                                            float* output,
                                            std::size_t count,
                                            float sigma,
                                            float delta_sigma,
                                            int prediction_type) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const float x = sample[index];
        const float prediction = __half2float(model_output[index]);
        float original = 0.0F;
        if (prediction_type == 0) original = x - sigma * prediction;
        else if (prediction_type == 1) original = prediction;
        else original = prediction * (-sigma / sqrtf(sigma * sigma + 1.0F)) +
                        x / (sigma * sigma + 1.0F);
        const float derivative = sigma == 0.0F ? 0.0F : (x - original) / sigma;
        output[index] = x + derivative * delta_sigma;
    }
}

__global__ void ddim_step_kernel(const __half* model_output,
                                 const __half* sample,
                                 const __half* noise,
                                 __half* output,
                                 std::size_t count,
                                 float alpha_t,
                                 float alpha_prev,
                                 float eta,
                                 int prediction_type) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const float x = __half2float(sample[index]);
        const float prediction = __half2float(model_output[index]);
        const float beta_t = 1.0F - alpha_t;
        float predicted_original = 0.0F;
        float predicted_epsilon = 0.0F;
        if (prediction_type == 0) {
            predicted_original = (x - sqrtf(beta_t) * prediction) / sqrtf(alpha_t);
            predicted_epsilon = prediction;
        } else if (prediction_type == 1) {
            predicted_original = prediction;
            predicted_epsilon = (x - sqrtf(alpha_t) * predicted_original) / sqrtf(beta_t);
        } else {
            predicted_original = sqrtf(alpha_t) * x - sqrtf(beta_t) * prediction;
            predicted_epsilon = sqrtf(alpha_t) * prediction + sqrtf(beta_t) * x;
        }
        const float variance = (1.0F - alpha_prev) / (1.0F - alpha_t) *
                               (1.0F - alpha_t / alpha_prev);
        const float standard_deviation = eta * sqrtf(fmaxf(0.0F, variance));
        const float direction = sqrtf(fmaxf(0.0F, 1.0F - alpha_prev - standard_deviation * standard_deviation)) *
                                predicted_epsilon;
        float result = sqrtf(alpha_prev) * predicted_original + direction;
        if (noise != nullptr && standard_deviation != 0.0F) {
            result += standard_deviation * __half2float(noise[index]);
        }
        output[index] = __float2half_rn(result);
    }
}

__global__ void ddim_step_f16_f32_kernel(const __half* model_output,
                                           const float* sample,
                                           const float* noise,
                                           float* output,
                                           std::size_t count,
                                           float alpha_t,
                                           float alpha_prev,
                                           float eta,
                                           int prediction_type) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const float x = sample[index];
        const float prediction = __half2float(model_output[index]);
        const float beta_t = 1.0F - alpha_t;
        float predicted_original = 0.0F;
        float predicted_epsilon = 0.0F;
        if (prediction_type == 0) {
            predicted_original = (x - sqrtf(beta_t) * prediction) / sqrtf(alpha_t);
            predicted_epsilon = prediction;
        } else if (prediction_type == 1) {
            predicted_original = prediction;
            predicted_epsilon = (x - sqrtf(alpha_t) * predicted_original) / sqrtf(beta_t);
        } else {
            predicted_original = sqrtf(alpha_t) * x - sqrtf(beta_t) * prediction;
            predicted_epsilon = sqrtf(alpha_t) * prediction + sqrtf(beta_t) * x;
        }
        const float variance = (1.0F - alpha_prev) / (1.0F - alpha_t) *
                               (1.0F - alpha_t / alpha_prev);
        const float std_dev = eta * sqrtf(fmaxf(0.0F, variance));
        const float direction_scale = sqrtf(fmaxf(0.0F, 1.0F - alpha_prev - std_dev * std_dev));
        float value = sqrtf(alpha_prev) * predicted_original + direction_scale * predicted_epsilon;
        if (noise != nullptr && std_dev > 0.0F) value += std_dev * noise[index];
        output[index] = value;
    }
}

__global__ void ddim_cfg_step_kernel(const __half* model_output,
                                     const __half* sample,
                                     const __half* noise,
                                     __half* output,
                                     std::size_t batch,
                                     std::size_t per_batch,
                                     float guidance_scale,
                                     float alpha_t,
                                     float alpha_prev,
                                     float eta,
                                     int prediction_type) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = batch * per_batch;
    if (index < count) {
        const float x = __half2float(sample[index]);
        const float prediction = cfg_value(
            model_output, index, batch, per_batch, guidance_scale);
        const float beta_t = 1.0F - alpha_t;
        float predicted_original = 0.0F;
        float predicted_epsilon = 0.0F;
        if (prediction_type == 0) {
            predicted_original = (x - sqrtf(beta_t) * prediction) / sqrtf(alpha_t);
            predicted_epsilon = prediction;
        } else if (prediction_type == 1) {
            predicted_original = prediction;
            predicted_epsilon =
                (x - sqrtf(alpha_t) * predicted_original) / sqrtf(beta_t);
        } else {
            predicted_original = sqrtf(alpha_t) * x - sqrtf(beta_t) * prediction;
            predicted_epsilon = sqrtf(alpha_t) * prediction + sqrtf(beta_t) * x;
        }
        const float variance = (1.0F - alpha_prev) / (1.0F - alpha_t) *
                               (1.0F - alpha_t / alpha_prev);
        const float standard_deviation = eta * sqrtf(fmaxf(0.0F, variance));
        const float direction =
            sqrtf(fmaxf(0.0F, 1.0F - alpha_prev -
                              standard_deviation * standard_deviation)) *
            predicted_epsilon;
        float result = sqrtf(alpha_prev) * predicted_original + direction;
        if (noise != nullptr && standard_deviation != 0.0F) {
            result += standard_deviation * __half2float(noise[index]);
        }
        output[index] = __float2half_rn(result);
    }
}

__global__ void f32_to_scaled_f16_kernel(const float* input, __half* output,
                                         std::size_t count, float scale) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = __float2half_rn(input[index] * scale);
}

__global__ void finite_kernel(const __half* input, std::size_t count, int* failure) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count && !isfinite(__half2float(input[index]))) atomicExch(failure, 1);
}

__global__ void cast_f16_to_f32_kernel(const __half* input, float* output, std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = __half2float(input[index]);
}

__global__ void cast_f16_to_f32_scale_kernel(const __half* input, float* output,
                                              std::size_t count, float scale) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = __half2float(input[index]) * scale;
}

__global__ void cast_f32_to_f16_kernel(const float* input, __half* output, std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = __float2half_rn(input[index]);
}

__global__ void add_f32_kernel(const float* first, const float* second,
                               float* output, std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = first[index] + second[index];
}

__global__ void add_in_place_f32_kernel(float* destination, const float* source,
                                        std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) destination[index] += source[index];
}

__global__ void add_last_dim_bias_f32_kernel(float* destination, const float* bias,
                                             std::size_t count, std::size_t width) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) destination[index] += bias[index % width];
}

__global__ void scale_f32_kernel(const float* input, float* output,
                                 std::size_t count, float scale) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = input[index] * scale;
}

__global__ void scale_in_place_f32_kernel(float* values, std::size_t count, float scale) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) values[index] *= scale;
}

__global__ void silu_f32_kernel(const float* input, float* output, std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const float x = input[index];
        output[index] = x / (1.0F + expf(-x));
    }
}

__global__ void silu_in_place_f32_kernel(float* values, std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const float x = values[index];
        values[index] = x / (1.0F + expf(-x));
    }
}

__global__ void group_norm_f32_kernel(const float* input,
                                      const float* weight,
                                      const float* bias,
                                      float* output,
                                      std::size_t batch,
                                      std::size_t channels,
                                      std::size_t height,
                                      std::size_t width,
                                      std::size_t groups,
                                      float epsilon,
                                      bool apply_silu) {
    extern __shared__ float shared[];
    const std::size_t group_index = blockIdx.x;
    if (group_index >= batch * groups) return;
    const std::size_t batch_index = group_index / groups;
    const std::size_t group = group_index % groups;
    const std::size_t channels_per_group = channels / groups;
    const std::size_t spatial = height * width;
    const std::size_t count = channels_per_group * spatial;
    const std::size_t channel_begin = group * channels_per_group;

    float sum = 0.0F;
    for (std::size_t local = threadIdx.x; local < count; local += blockDim.x) {
        const std::size_t channel = channel_begin + local / spatial;
        const std::size_t position = local % spatial;
        sum += input[(batch_index * channels + channel) * spatial + position];
    }
    const float mean = block_sum(sum, shared) / static_cast<float>(count);

    float squared_centered = 0.0F;
    for (std::size_t local = threadIdx.x; local < count; local += blockDim.x) {
        const std::size_t channel = channel_begin + local / spatial;
        const std::size_t position = local % spatial;
        const float centered =
            input[(batch_index * channels + channel) * spatial + position] - mean;
        squared_centered += centered * centered;
    }
    const float variance = fmaxf(
        0.0F, block_sum(squared_centered, shared) / static_cast<float>(count));
    const float inverse = rsqrtf(variance + epsilon);

    for (std::size_t local = threadIdx.x; local < count; local += blockDim.x) {
        const std::size_t channel = channel_begin + local / spatial;
        const std::size_t position = local % spatial;
        const std::size_t index = (batch_index * channels + channel) * spatial + position;
        float normalized = (input[index] - mean) * inverse * weight[channel] + bias[channel];
        if (apply_silu) normalized = normalized / (1.0F + expf(-normalized));
        output[index] = normalized;
    }
}

__global__ void flatten_spatial_f32_kernel(const float* input,
                                           float* output,
                                           std::size_t batch,
                                           std::size_t channels,
                                           std::size_t height,
                                           std::size_t width) {
    const std::size_t count = batch * channels * height * width;
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const std::size_t x = index % width;
        const std::size_t y = (index / width) % height;
        const std::size_t channel = (index / (width * height)) % channels;
        const std::size_t batch_index = index / (width * height * channels);
        const std::size_t sequence = y * width + x;
        output[(batch_index * height * width + sequence) * channels + channel] = input[index];
    }
}

__global__ void unflatten_spatial_f32_kernel(const float* input,
                                             float* output,
                                             std::size_t batch,
                                             std::size_t channels,
                                             std::size_t height,
                                             std::size_t width) {
    const std::size_t count = batch * channels * height * width;
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const std::size_t x = index % width;
        const std::size_t y = (index / width) % height;
        const std::size_t channel = (index / (width * height)) % channels;
        const std::size_t batch_index = index / (width * height * channels);
        const std::size_t sequence = y * width + x;
        output[index] = input[(batch_index * height * width + sequence) * channels + channel];
    }
}

__global__ void nearest_upsample_f32_kernel(const float* input,
                                            float* output,
                                            std::size_t batch,
                                            std::size_t channels,
                                            std::size_t input_height,
                                            std::size_t input_width,
                                            std::size_t output_height,
                                            std::size_t output_width) {
    const std::size_t count = batch * channels * output_height * output_width;
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const std::size_t x = index % output_width;
        const std::size_t y = (index / output_width) % output_height;
        const std::size_t channel = (index / (output_width * output_height)) % channels;
        const std::size_t batch_index = index / (output_width * output_height * channels);
        const std::size_t mapped_y = y * input_height / output_height;
        const std::size_t source_y = mapped_y < input_height ? mapped_y : input_height - 1;
        const std::size_t mapped_x = x * input_width / output_width;
        const std::size_t source_x = mapped_x < input_width ? mapped_x : input_width - 1;
        output[index] = input[((batch_index * channels + channel) * input_height + source_y) * input_width + source_x];
    }
}

__global__ void generic_online_attention_f32_kernel(const float* query,
                                                    const float* key,
                                                    const float* value,
                                                    const std::int32_t* key_mask,
                                                    float* output,
                                                    std::size_t batch,
                                                    std::size_t query_sequence,
                                                    std::size_t key_sequence,
                                                    std::size_t heads,
                                                    std::size_t head_dimension,
                                                    bool causal) {
    extern __shared__ float shared[];
    const unsigned warp_count = (blockDim.x + 31U) / 32U;
    float* warp_sums = shared;
    float* scalar = shared + warp_count;
    const std::size_t block = blockIdx.x;
    const std::size_t query_position = block % query_sequence;
    const std::size_t head = (block / query_sequence) % heads;
    const std::size_t batch_index = block / (query_sequence * heads);
    if (batch_index >= batch) return;

    const std::size_t dimension = threadIdx.x;
    const std::size_t query_base =
        (batch_index * query_sequence + query_position) * heads * head_dimension +
        head * head_dimension;
    const float query_value = dimension < head_dimension ? query[query_base + dimension] : 0.0F;
    float accumulator = 0.0F;
    float running_max = -CUDART_INF_F;
    float running_sum = 0.0F;
    const std::size_t causal_limit = query_position + 1 < key_sequence
        ? query_position + 1 : key_sequence;
    const std::size_t visible_keys = causal ? causal_limit : key_sequence;
    const float scale = rsqrtf(static_cast<float>(head_dimension));

    for (std::size_t key_position = 0; key_position < visible_keys; ++key_position) {
        const bool visible = key_mask == nullptr ||
            key_mask[batch_index * key_sequence + key_position] != 0;
        const std::size_t key_base =
            (batch_index * key_sequence + key_position) * heads * head_dimension +
            head * head_dimension;
        const float product = visible && dimension < head_dimension
            ? query_value * key[key_base + dimension] : 0.0F;
        const float dot = block_sum(product, warp_sums);
        if (threadIdx.x == 0) {
            const float score = visible ? dot * scale : -CUDART_INF_F;
            const float next_max = fmaxf(running_max, score);
            const float alpha = isfinite(running_max) ? expf(running_max - next_max) : 0.0F;
            const float beta = isfinite(score) ? expf(score - next_max) : 0.0F;
            running_sum = running_sum * alpha + beta;
            running_max = next_max;
            scalar[0] = alpha;
            scalar[1] = beta;
        }
        __syncthreads();
        if (dimension < head_dimension) {
            accumulator = accumulator * scalar[0] + scalar[1] * value[key_base + dimension];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) scalar[0] = running_sum;
    __syncthreads();
    if (dimension < head_dimension) {
        output[query_base + dimension] = accumulator / fmaxf(scalar[0], 1.0e-20F);
    }
}

__global__ void finite_f32_kernel(const float* input, std::size_t count, int* failure) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count && !isfinite(input[index])) atomicExch(failure, 1);
}

} // namespace

Tensor Ops::layer_norm(const Tensor& input, const Tensor& weight,
                       const Tensor& bias, float epsilon) const {
    auto profile = profile_scope("ops/layer_norm/w" + std::to_string(input.shape().back()));
    if (input.type() != ScalarType::Float16 || weight.type() != ScalarType::Float16 ||
        bias.type() != ScalarType::Float16 || input.rank() < 2 || weight.rank() != 1 ||
        bias.rank() != 1 || weight.size(0) != input.shape().back() ||
        bias.size(0) != input.shape().back()) {
        throw CudaError("CUDA layer norm shape/type mismatch");
    }
    Tensor output = Tensor::allocate(*runtime_, input.shape(), ScalarType::Float16, input.role());
    const std::size_t width = input.shape().back();
    const std::size_t rows = input.elements() / width;
    switch (width) {
    case 320:
        layer_norm_fixed_kernel<320><<<static_cast<unsigned>(rows), kThreads, 8 * sizeof(float), runtime_->stream()>>>(
            input.half_data(), weight.half_data(), bias.half_data(), output.half_data(), rows, epsilon);
        break;
    case 640:
        layer_norm_fixed_kernel<640><<<static_cast<unsigned>(rows), kThreads, 8 * sizeof(float), runtime_->stream()>>>(
            input.half_data(), weight.half_data(), bias.half_data(), output.half_data(), rows, epsilon);
        break;
    case 768:
        layer_norm_fixed_kernel<768><<<static_cast<unsigned>(rows), kThreads, 8 * sizeof(float), runtime_->stream()>>>(
            input.half_data(), weight.half_data(), bias.half_data(), output.half_data(), rows, epsilon);
        break;
    case 1280:
        layer_norm_fixed_kernel<1280><<<static_cast<unsigned>(rows), kThreads, 8 * sizeof(float), runtime_->stream()>>>(
            input.half_data(), weight.half_data(), bias.half_data(), output.half_data(), rows, epsilon);
        break;
    case 2048:
        layer_norm_fixed_kernel<2048><<<static_cast<unsigned>(rows), kThreads, 8 * sizeof(float), runtime_->stream()>>>(
            input.half_data(), weight.half_data(), bias.half_data(), output.half_data(), rows, epsilon);
        break;
    default:
        layer_norm_kernel<<<static_cast<unsigned>(rows), kThreads, 8 * sizeof(float), runtime_->stream()>>>(
            input.half_data(), weight.half_data(), bias.half_data(), output.half_data(), rows, width, epsilon);
        break;
    }
    SDXL_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor Ops::group_norm_nchw(const Tensor& input, const Tensor& weight,
                            const Tensor& bias, std::size_t groups, float epsilon) const {
    auto profile = profile_scope("ops/group_norm/c" + std::to_string(input.size(1)) +
                                 "_h" + std::to_string(input.size(2)) +
                                 "_w" + std::to_string(input.size(3)));
    if ((input.type() != ScalarType::Float16 && input.type() != ScalarType::Float32) ||
        weight.type() != input.type() || bias.type() != input.type() || input.rank() != 4 ||
        weight.rank() != 1 || bias.rank() != 1 || groups == 0 || input.size(1) % groups != 0 ||
        weight.size(0) != input.size(1) || bias.size(0) != input.size(1)) {
        throw CudaError("CUDA group norm shape/type mismatch");
    }
    Tensor output = Tensor::allocate(*runtime_, input.shape(), input.type(), input.role());
    const std::size_t blocks = input.size(0) * groups;
    if (input.type() == ScalarType::Float16) {
        group_norm_kernel<<<static_cast<unsigned>(blocks), kThreads, 8 * sizeof(float), runtime_->stream()>>>(
            input.half_data(), weight.half_data(), bias.half_data(), output.half_data(),
            input.size(0), input.size(1), input.size(2), input.size(3), groups, epsilon, false);
    } else {
        group_norm_f32_kernel<<<static_cast<unsigned>(blocks), kThreads, 8 * sizeof(float), runtime_->stream()>>>(
            input.float_data(), weight.float_data(), bias.float_data(), output.float_data(),
            input.size(0), input.size(1), input.size(2), input.size(3), groups, epsilon, false);
    }
    SDXL_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor Ops::group_norm_silu_nchw(const Tensor& input, const Tensor& weight,
                                 const Tensor& bias, std::size_t groups, float epsilon) const {
    auto profile = profile_scope("ops/group_norm_silu/c" + std::to_string(input.size(1)) +
                                 "_h" + std::to_string(input.size(2)) +
                                 "_w" + std::to_string(input.size(3)));
    if ((input.type() != ScalarType::Float16 && input.type() != ScalarType::Float32) ||
        weight.type() != input.type() || bias.type() != input.type() || input.rank() != 4 ||
        weight.rank() != 1 || bias.rank() != 1 || groups == 0 || input.size(1) % groups != 0 ||
        weight.size(0) != input.size(1) || bias.size(0) != input.size(1)) {
        throw CudaError("CUDA fused group norm + SiLU shape/type mismatch");
    }
    Tensor output = Tensor::allocate(*runtime_, input.shape(), input.type(), input.role());
    const std::size_t blocks = input.size(0) * groups;
    if (input.type() == ScalarType::Float16) {
        group_norm_kernel<<<static_cast<unsigned>(blocks), kThreads, 8 * sizeof(float), runtime_->stream()>>>(
            input.half_data(), weight.half_data(), bias.half_data(), output.half_data(),
            input.size(0), input.size(1), input.size(2), input.size(3), groups, epsilon, true);
    } else {
        group_norm_f32_kernel<<<static_cast<unsigned>(blocks), kThreads, 8 * sizeof(float), runtime_->stream()>>>(
            input.float_data(), weight.float_data(), bias.float_data(), output.float_data(),
            input.size(0), input.size(1), input.size(2), input.size(3), groups, epsilon, true);
    }
    SDXL_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor Ops::cast(const Tensor& input, ScalarType destination_type,
                 TensorRole destination_role) const {
    if ((input.type() != ScalarType::Float16 && input.type() != ScalarType::Float32) ||
        (destination_type != ScalarType::Float16 && destination_type != ScalarType::Float32)) {
        throw CudaError("CUDA cast supports only float16 and float32");
    }
    if (input.type() == destination_type) {
        Tensor output = Tensor::allocate(*runtime_, input.shape(), input.type(), destination_role);
        // Tensor::copy_from intentionally requires matching semantic roles so that
        // accidental model/sampler/VAE assignment is rejected. A cast is the
        // explicit role-conversion boundary, however, and same-dtype casts still
        // need a real device-to-device payload copy when the role changes (for
        // example FP32 SamplerState -> FP32 VAE before decode).
        if (input.role() == destination_role) {
            output.copy_from(*runtime_, input);
        } else {
            validate_same_runtime(input, output);
            SDXL_CUDA_CHECK(cudaMemcpyAsync(output.data(), input.data(), input.bytes(),
                                            cudaMemcpyDeviceToDevice, runtime_->stream()));
        }
        return output;
    }
    Tensor output = Tensor::allocate(*runtime_, input.shape(), destination_type, destination_role);
    if (input.type() == ScalarType::Float16) {
        cast_f16_to_f32_kernel<<<blocks_for(input.elements()), kThreads, 0, runtime_->stream()>>>(
            input.half_data(), output.float_data(), input.elements());
    } else {
        cast_f32_to_f16_kernel<<<blocks_for(input.elements()), kThreads, 0, runtime_->stream()>>>(
            input.float_data(), output.half_data(), input.elements());
    }
    SDXL_CUDA_CHECK(cudaGetLastError());
    return output;
}


Tensor Ops::cast_scale(const Tensor& input, ScalarType destination_type,
                       TensorRole destination_role, float scale_value) const {
    if (input.type() == ScalarType::Float16 && destination_type == ScalarType::Float32) {
        Tensor output = Tensor::allocate(*runtime_, input.shape(), destination_type, destination_role);
        cast_f16_to_f32_scale_kernel<<<blocks_for(input.elements()), kThreads, 0, runtime_->stream()>>>(
            input.half_data(), output.float_data(), input.elements(), scale_value);
        SDXL_CUDA_CHECK(cudaGetLastError());
        return output;
    }
    Tensor output = cast(input, destination_type, destination_role);
    scale_in_place(output, scale_value);
    return output;
}

Tensor Ops::add(const Tensor& first, const Tensor& second) const {
    validate_same_runtime(first, second);
    if (first.shape() != second.shape() || first.type() != second.type() ||
        (first.type() != ScalarType::Float16 && first.type() != ScalarType::Float32)) {
        throw CudaError("CUDA add mismatch");
    }
    Tensor output = Tensor::allocate(*runtime_, first.shape(), first.type(), first.role());
    if (first.type() == ScalarType::Float16) {
        add_kernel<<<blocks_for(first.elements()), kThreads, 0, runtime_->stream()>>>(
            first.half_data(), second.half_data(), output.half_data(), first.elements());
    } else {
        add_f32_kernel<<<blocks_for(first.elements()), kThreads, 0, runtime_->stream()>>>(
            first.float_data(), second.float_data(), output.float_data(), first.elements());
    }
    SDXL_CUDA_CHECK(cudaGetLastError());
    return output;
}


Tensor Ops::add_silu(const Tensor& first, const Tensor& second) const {
    validate_same_runtime(first, second);
    if (first.shape() != second.shape() || first.type() != ScalarType::Float16 ||
        second.type() != ScalarType::Float16 || first.role() != second.role()) {
        throw CudaError("CUDA fused add+SiLU requires matching FP16 tensors");
    }
    Tensor output = Tensor::allocate(*runtime_, first.shape(), ScalarType::Float16, first.role());
    add_silu_kernel<<<blocks_for(first.elements()), kThreads, 0, runtime_->stream()>>>(
        first.half_data(), second.half_data(), output.half_data(), first.elements());
    SDXL_CUDA_CHECK(cudaGetLastError());
    return output;
}

void Ops::add_in_place(Tensor& destination, const Tensor& source) const {
    validate_same_runtime(destination, source);
    if (destination.shape() != source.shape() || destination.type() != source.type() ||
        (destination.type() != ScalarType::Float16 && destination.type() != ScalarType::Float32)) {
        throw CudaError("CUDA add_in_place mismatch");
    }
    if (destination.type() == ScalarType::Float16) {
        add_in_place_kernel<<<blocks_for(destination.elements()), kThreads, 0, runtime_->stream()>>>(
            destination.half_data(), source.half_data(), destination.elements());
    } else {
        add_in_place_f32_kernel<<<blocks_for(destination.elements()), kThreads, 0, runtime_->stream()>>>(
            destination.float_data(), source.float_data(), destination.elements());
    }
    SDXL_CUDA_CHECK(cudaGetLastError());
}


void Ops::add_last_dim_bias_in_place(Tensor& destination, const Tensor& bias) const {
    validate_same_runtime(destination, bias);
    if (destination.type() != bias.type() ||
        (destination.type() != ScalarType::Float16 && destination.type() != ScalarType::Float32) ||
        destination.rank() < 1 || bias.rank() != 1 || destination.shape().back() != bias.size(0)) {
        throw CudaError("CUDA last-dimension bias mismatch");
    }
    if (destination.type() == ScalarType::Float16) {
        add_last_dim_bias_kernel<<<blocks_for(destination.elements()), kThreads, 0, runtime_->stream()>>>(
            destination.half_data(), bias.half_data(), destination.elements(), bias.size(0));
    } else {
        add_last_dim_bias_f32_kernel<<<blocks_for(destination.elements()), kThreads, 0, runtime_->stream()>>>(
            destination.float_data(), bias.float_data(), destination.elements(), bias.size(0));
    }
    SDXL_CUDA_CHECK(cudaGetLastError());
}


void Ops::bias_activation_in_place(Tensor& destination, const Tensor& bias,
                                   LinearActivation activation) const {
    if (destination.type() != ScalarType::Float16 || bias.type() != ScalarType::Float16 ||
        destination.rank() < 2 || bias.rank() != 1 ||
        destination.shape().back() != bias.size(0) ||
        activation == LinearActivation::None || activation == LinearActivation::GEGLU) {
        throw CudaError("fused linear bias activation requires matching FP16 tensors and SiLU/GELU/QuickGELU");
    }
    bias_activation_in_place_kernel<<<blocks_for(destination.elements()), kThreads, 0, runtime_->stream()>>>(
        destination.half_data(), bias.half_data(), destination.elements(), bias.size(0),
        static_cast<int>(activation));
    SDXL_CUDA_CHECK(cudaGetLastError());
}

Tensor Ops::bias_geglu(const Tensor& projected, const Tensor& bias) const {
    if (projected.type() != ScalarType::Float16 || bias.type() != ScalarType::Float16 ||
        projected.rank() < 2 || bias.rank() != 1 || projected.shape().back() != bias.size(0) ||
        bias.size(0) % 2 != 0) {
        throw CudaError("fused linear bias GEGLU requires matching even-width FP16 tensors");
    }
    const std::size_t width = bias.size(0) / 2;
    const std::size_t rows = projected.elements() / bias.size(0);
    std::vector<std::size_t> shape = projected.shape();
    shape.back() = width;
    Tensor output = Tensor::allocate(*runtime_, shape, ScalarType::Float16, projected.role());
    bias_geglu_kernel<<<blocks_for(output.elements()), kThreads, 0, runtime_->stream()>>>(
        projected.half_data(), bias.half_data(), output.half_data(), rows, width);
    SDXL_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor Ops::scale(const Tensor& tensor, float scale_value) const {
    if (tensor.type() != ScalarType::Float16 && tensor.type() != ScalarType::Float32) {
        throw CudaError("CUDA scale expects float16 or float32");
    }
    Tensor output = Tensor::allocate(*runtime_, tensor.shape(), tensor.type(), tensor.role());
    if (tensor.type() == ScalarType::Float16) {
        scale_kernel<<<blocks_for(tensor.elements()), kThreads, 0, runtime_->stream()>>>(
            tensor.half_data(), output.half_data(), tensor.elements(), scale_value);
    } else {
        scale_f32_kernel<<<blocks_for(tensor.elements()), kThreads, 0, runtime_->stream()>>>(
            tensor.float_data(), output.float_data(), tensor.elements(), scale_value);
    }
    SDXL_CUDA_CHECK(cudaGetLastError());
    return output;
}

void Ops::scale_in_place(Tensor& tensor, float scale_value) const {
    if (tensor.type() == ScalarType::Float16) {
        scale_in_place_kernel<<<blocks_for(tensor.elements()), kThreads, 0, runtime_->stream()>>>(
            tensor.half_data(), tensor.elements(), scale_value);
    } else if (tensor.type() == ScalarType::Float32) {
        scale_in_place_f32_kernel<<<blocks_for(tensor.elements()), kThreads, 0, runtime_->stream()>>>(
            tensor.float_data(), tensor.elements(), scale_value);
    } else {
        throw CudaError("CUDA scale expects float16 or float32");
    }
    SDXL_CUDA_CHECK(cudaGetLastError());
}

Tensor Ops::add_spatial_bias(const Tensor& nchw, const Tensor& batch_channels) const {
    if (nchw.rank() != 4 || batch_channels.rank() != 2 ||
        nchw.size(0) != batch_channels.size(0) || nchw.size(1) != batch_channels.size(1)) {
        throw CudaError("CUDA spatial bias mismatch");
    }
    Tensor output = Tensor::allocate(*runtime_, nchw.shape(), ScalarType::Float16, nchw.role());
    const std::size_t spatial = nchw.size(2) * nchw.size(3);
    add_spatial_bias_kernel<<<blocks_for(nchw.elements()), kThreads, 0, runtime_->stream()>>>(
        nchw.half_data(), batch_channels.half_data(), output.half_data(),
        nchw.size(0), nchw.size(1), spatial);
    SDXL_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor Ops::silu(const Tensor& input) const {
    if (input.type() != ScalarType::Float16 && input.type() != ScalarType::Float32) {
        throw CudaError("CUDA SiLU expects float16 or float32");
    }
    Tensor output = Tensor::allocate(*runtime_, input.shape(), input.type(), input.role());
    if (input.type() == ScalarType::Float16) {
        silu_kernel<<<blocks_for(input.elements()), kThreads, 0, runtime_->stream()>>>(
            input.half_data(), output.half_data(), input.elements());
    } else {
        silu_f32_kernel<<<blocks_for(input.elements()), kThreads, 0, runtime_->stream()>>>(
            input.float_data(), output.float_data(), input.elements());
    }
    SDXL_CUDA_CHECK(cudaGetLastError());
    return output;
}

void Ops::silu_in_place(Tensor& input) const {
    if (input.type() == ScalarType::Float16) {
        silu_in_place_kernel<<<blocks_for(input.elements()), kThreads, 0, runtime_->stream()>>>(
            input.half_data(), input.elements());
    } else if (input.type() == ScalarType::Float32) {
        silu_in_place_f32_kernel<<<blocks_for(input.elements()), kThreads, 0, runtime_->stream()>>>(
            input.float_data(), input.elements());
    } else {
        throw CudaError("CUDA SiLU expects float16 or float32");
    }
    SDXL_CUDA_CHECK(cudaGetLastError());
}

Tensor Ops::quick_gelu(const Tensor& input) const {
    Tensor output = Tensor::allocate(*runtime_, input.shape(), ScalarType::Float16, input.role());
    quick_gelu_kernel<<<blocks_for(input.elements()), kThreads, 0, runtime_->stream()>>>(
        input.half_data(), output.half_data(), input.elements());
    SDXL_CUDA_CHECK(cudaGetLastError());
    return output;
}


Tensor Ops::gelu(const Tensor& input) const {
    Tensor output = Tensor::allocate(*runtime_, input.shape(), ScalarType::Float16, input.role());
    gelu_kernel<<<blocks_for(input.elements()), kThreads, 0, runtime_->stream()>>>(
        input.half_data(), output.half_data(), input.elements());
    SDXL_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor Ops::geglu(const Tensor& input) const {
    if (input.rank() < 2 || input.shape().back() % 2 != 0) throw CudaError("CUDA GEGLU width mismatch");
    const std::size_t width = input.shape().back() / 2;
    const std::size_t rows = input.elements() / (2 * width);
    std::vector<std::size_t> shape = input.shape();
    shape.back() = width;
    Tensor output = Tensor::allocate(*runtime_, shape, ScalarType::Float16, input.role());
    geglu_kernel<<<blocks_for(output.elements()), kThreads, 0, runtime_->stream()>>>(
        input.half_data(), output.half_data(), rows, width);
    SDXL_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor Ops::flatten_spatial(const Tensor& input) const {
    if (input.rank() != 4 ||
        (input.type() != ScalarType::Float16 && input.type() != ScalarType::Float32)) {
        throw CudaError("CUDA flatten_spatial expects float NCHW");
    }
    Tensor output = Tensor::allocate(*runtime_,
        {input.size(0), input.size(2) * input.size(3), input.size(1)}, input.type(), input.role());
    if (input.type() == ScalarType::Float16) {
        flatten_spatial_kernel<<<blocks_for(input.elements()), kThreads, 0, runtime_->stream()>>>(
            input.half_data(), output.half_data(), input.size(0), input.size(1), input.size(2), input.size(3));
    } else {
        flatten_spatial_f32_kernel<<<blocks_for(input.elements()), kThreads, 0, runtime_->stream()>>>(
            input.float_data(), output.float_data(), input.size(0), input.size(1), input.size(2), input.size(3));
    }
    SDXL_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor Ops::unflatten_spatial(const Tensor& input, std::size_t height, std::size_t width) const {
    if (input.rank() != 3 || input.size(1) != height * width ||
        (input.type() != ScalarType::Float16 && input.type() != ScalarType::Float32)) {
        throw CudaError("CUDA unflatten_spatial mismatch");
    }
    Tensor output = Tensor::allocate(*runtime_,
        {input.size(0), input.size(2), height, width}, input.type(), input.role());
    if (input.type() == ScalarType::Float16) {
        unflatten_spatial_kernel<<<blocks_for(output.elements()), kThreads, 0, runtime_->stream()>>>(
            input.half_data(), output.half_data(), input.size(0), input.size(2), height, width);
    } else {
        unflatten_spatial_f32_kernel<<<blocks_for(output.elements()), kThreads, 0, runtime_->stream()>>>(
            input.float_data(), output.float_data(), input.size(0), input.size(2), height, width);
    }
    SDXL_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor Ops::concat_channels(const Tensor& first, const Tensor& second) const {
    if (first.rank() != 4 || second.rank() != 4 || first.size(0) != second.size(0) ||
        first.size(2) != second.size(2) || first.size(3) != second.size(3)) {
        throw CudaError("CUDA concat_channels mismatch");
    }
    Tensor output = Tensor::allocate(*runtime_,
        {first.size(0), first.size(1) + second.size(1), first.size(2), first.size(3)},
        ScalarType::Float16, first.role());
    concat_channels_kernel<<<blocks_for(output.elements()), kThreads, 0, runtime_->stream()>>>(
        first.half_data(), second.half_data(), output.half_data(), first.size(0), first.size(1),
        second.size(1), first.size(2) * first.size(3));
    SDXL_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor Ops::concat_last_dim(const Tensor& first, const Tensor& second) const {
    if (first.rank() != second.rank() || first.rank() < 2) throw CudaError("CUDA concat_last_dim rank mismatch");
    for (std::size_t dimension = 0; dimension + 1 < first.rank(); ++dimension) {
        if (first.size(dimension) != second.size(dimension)) throw CudaError("CUDA concat_last_dim shape mismatch");
    }
    const std::size_t rows = first.elements() / first.shape().back();
    std::vector<std::size_t> shape = first.shape();
    shape.back() += second.shape().back();
    Tensor output = Tensor::allocate(*runtime_, shape, ScalarType::Float16, first.role());
    concat_last_kernel<<<blocks_for(output.elements()), kThreads, 0, runtime_->stream()>>>(
        first.half_data(), second.half_data(), output.half_data(), rows,
        first.shape().back(), second.shape().back());
    SDXL_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor Ops::concat_batch(const Tensor& first, const Tensor& second) const {
    if (first.rank() != second.rank() || first.rank() == 0) throw CudaError("CUDA concat_batch rank mismatch");
    for (std::size_t dimension = 1; dimension < first.rank(); ++dimension) {
        if (first.size(dimension) != second.size(dimension)) throw CudaError("CUDA concat_batch shape mismatch");
    }
    std::vector<std::size_t> shape = first.shape();
    shape[0] += second.size(0);
    Tensor output = Tensor::allocate(*runtime_, shape, first.type(), first.role());
    SDXL_CUDA_CHECK(cudaMemcpyAsync(output.data(), first.data(), first.bytes(),
                                    cudaMemcpyDeviceToDevice, runtime_->stream()));
    SDXL_CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(output.data()) + first.bytes(),
                                    second.data(), second.bytes(), cudaMemcpyDeviceToDevice,
                                    runtime_->stream()));
    return output;
}

Tensor Ops::repeat_batch(const Tensor& input, std::size_t repeats) const {
    if (input.rank() == 0 || repeats == 0) throw CudaError("CUDA repeat_batch invalid arguments");
    std::vector<std::size_t> shape = input.shape();
    shape[0] *= repeats;
    Tensor output = Tensor::allocate(*runtime_, shape, input.type(), input.role());
    const std::size_t per_batch = input.elements() / input.size(0);
    repeat_batch_kernel<<<blocks_for(output.elements()), kThreads, 0, runtime_->stream()>>>(
        input.half_data(), output.half_data(), per_batch, input.size(0), repeats);
    SDXL_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor Ops::nearest_upsample(const Tensor& input, std::size_t target_height,
                             std::size_t target_width) const {
    if (input.rank() != 4 || target_height == 0 || target_width == 0 ||
        (input.type() != ScalarType::Float16 && input.type() != ScalarType::Float32)) {
        throw CudaError("CUDA nearest upsample expects float NCHW");
    }
    Tensor output = Tensor::allocate(*runtime_,
        {input.size(0), input.size(1), target_height, target_width}, input.type(), input.role());
    if (input.type() == ScalarType::Float16) {
        nearest_upsample_kernel<<<blocks_for(output.elements()), kThreads, 0, runtime_->stream()>>>(
            input.half_data(), output.half_data(), input.size(0), input.size(1), input.size(2), input.size(3),
            target_height, target_width);
    } else {
        nearest_upsample_f32_kernel<<<blocks_for(output.elements()), kThreads, 0, runtime_->stream()>>>(
            input.float_data(), output.float_data(), input.size(0), input.size(1), input.size(2), input.size(3),
            target_height, target_width);
    }
    SDXL_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor Ops::embedding(const Tensor& token_ids, const Tensor& token_embedding,
                      const Tensor& position_embedding) const {
    if (token_ids.type() != ScalarType::Int32 || token_ids.rank() != 2 ||
        token_embedding.rank() != 2 || position_embedding.rank() != 2 ||
        token_embedding.size(1) != position_embedding.size(1) ||
        token_ids.size(1) > position_embedding.size(0)) throw CudaError("CUDA embedding mismatch");
    Tensor output = Tensor::allocate(*runtime_,
        {token_ids.size(0), token_ids.size(1), token_embedding.size(1)}, ScalarType::Float16, token_embedding.role());
    embedding_kernel<<<blocks_for(output.elements()), kThreads, 0, runtime_->stream()>>>(
        token_ids.int32_data(), token_embedding.half_data(), position_embedding.half_data(),
        output.half_data(), token_ids.size(0), token_ids.size(1), token_embedding.size(1), token_embedding.size(0));
    SDXL_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor Ops::pool_eos(const Tensor& hidden, const Tensor& token_ids) const {
    if (hidden.rank() != 3 || token_ids.rank() != 2 || hidden.size(0) != token_ids.size(0) ||
        hidden.size(1) != token_ids.size(1)) throw CudaError("CUDA EOS pooling mismatch");
    Tensor output = Tensor::allocate(*runtime_, {hidden.size(0), hidden.size(2)}, ScalarType::Float16, hidden.role());
    pool_eos_kernel<<<blocks_for(output.elements()), kThreads, 0, runtime_->stream()>>>(
        hidden.half_data(), token_ids.int32_data(), output.half_data(), hidden.size(0), hidden.size(1), hidden.size(2));
    SDXL_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor Ops::attention(const Tensor& query, const Tensor& key, const Tensor& value,
                      std::size_t heads, bool causal, const Tensor* key_mask) const {
    if (query.rank() != 3 || key.rank() != 3 || value.shape() != key.shape() ||
        query.type() != key.type() || value.type() != key.type() ||
        (query.type() != ScalarType::Float16 && query.type() != ScalarType::Float32) ||
        query.size(0) != key.size(0) || query.size(2) != key.size(2) || heads == 0 ||
        query.size(2) % heads != 0) throw CudaError("CUDA attention mismatch");
    const std::size_t head_dimension = query.size(2) / heads;
    if (head_dimension == 0 || head_dimension > 1024) {
        throw CudaError("CUDA attention head dimension must be in [1,1024]");
    }
    if (key_mask != nullptr && (key_mask->type() != ScalarType::Int32 || key_mask->rank() != 2 ||
                                key_mask->size(0) != key.size(0) || key_mask->size(1) != key.size(1))) {
        throw CudaError("CUDA attention key mask mismatch");
    }

    const AttentionBackend requested = runtime_->options().attention_backend;
    const bool fp16_sm80 = query.type() == ScalarType::Float16 &&
        runtime_->device_properties().major >= 8;
    const bool cudnn_eligible = fp16_sm80 && key_mask == nullptr &&
        head_dimension % 8 == 0 && head_dimension <= 256 &&
        cudnn_frontend_sdpa_compiled();
    const bool flash_eligible = fp16_sm80 && head_dimension == 64;
    if (requested == AttentionBackend::CuDnnSDPA && !cudnn_eligible) {
        throw CudaError(
            "cudnn-sdpa requires a build with cudnn_frontend.h, FP16 SM80+, no explicit key mask, and head dimension <=256 divisible by 8");
    }
    if (requested == AttentionBackend::FlashSM80 && !flash_eligible) {
        throw CudaError("flash-sm80 attention requires FP16, head dimension 64, and SM80+");
    }

    // The tiny 77-token CLIP path remains on the lower-overhead warp kernel in
    // auto mode. Large spatial and cross-attention buckets prefer NVIDIA's
    // cuDNN FlashAttention-2 SDPA graph when the optional frontend is present,
    // then the in-tree SM80 tiled kernel, then the reference online kernel.
    const bool large_attention_shape = query.size(1) >= 256 || key.size(1) >= 256;
    const bool use_cudnn = requested == AttentionBackend::CuDnnSDPA ||
        (requested == AttentionBackend::Auto && large_attention_shape && cudnn_eligible);
    const bool use_flash = !use_cudnn && flash_eligible &&
        (requested == AttentionBackend::FlashSM80 ||
         (requested == AttentionBackend::Auto && large_attention_shape));
    const char* backend_label = use_cudnn ? "cudnn-sdpa" :
        (use_flash ? "flash-sm80" : "warp-online");

    std::ostringstream attention_label;
    attention_label << "ops/attention/" << backend_label
                    << "/q" << query.size(1) << "_k" << key.size(1)
                    << "_h" << heads << "_d" << head_dimension
                    << (causal ? "_causal" : "_full");
    auto profile = profile_scope(attention_label.str());

    Tensor output = Tensor::allocate(*runtime_, query.shape(), query.type(), query.role());
    if (use_cudnn) {
        launch_cudnn_frontend_sdpa(
            *runtime_, query, key, value, output, heads, causal);
        return output;
    }
    if (use_flash) {
        launch_flash_attention_sm80_hdim64(
            *runtime_, query, key, value, key_mask, output, heads, causal);
        return output;
    }

    const std::size_t blocks = query.size(0) * heads * query.size(1);
    if (query.type() == ScalarType::Float16 && head_dimension == 64) {
        const auto launch = [&](auto fixed_keys) {
            constexpr int fixed = decltype(fixed_keys)::value;
            online_attention_kernel<fixed><<<static_cast<unsigned>(blocks), 32, 0, runtime_->stream()>>>(
                query.half_data(), key.half_data(), value.half_data(),
                key_mask == nullptr ? nullptr : key_mask->int32_data(), output.half_data(),
                query.size(0), query.size(1), key.size(1), heads, head_dimension, causal);
        };
        switch (key.size(1)) {
        case 77: launch(std::integral_constant<int, 77>{}); break;
        case 256: launch(std::integral_constant<int, 256>{}); break;
        case 1024: launch(std::integral_constant<int, 1024>{}); break;
        case 4096: launch(std::integral_constant<int, 4096>{}); break;
        case 16384: launch(std::integral_constant<int, 16384>{}); break;
        default: launch(std::integral_constant<int, 0>{}); break;
        }
    } else {
        const unsigned threads = attention_threads(head_dimension);
        const unsigned warps = (threads + 31U) / 32U;
        const std::size_t shared_bytes = static_cast<std::size_t>(warps + 2U) * sizeof(float);
        if (query.type() == ScalarType::Float16) {
            generic_online_attention_kernel<<<static_cast<unsigned>(blocks), threads, shared_bytes, runtime_->stream()>>>(
                query.half_data(), key.half_data(), value.half_data(),
                key_mask == nullptr ? nullptr : key_mask->int32_data(), output.half_data(),
                query.size(0), query.size(1), key.size(1), heads, head_dimension, causal);
        } else {
            generic_online_attention_f32_kernel<<<static_cast<unsigned>(blocks), threads, shared_bytes, runtime_->stream()>>>(
                query.float_data(), key.float_data(), value.float_data(),
                key_mask == nullptr ? nullptr : key_mask->int32_data(), output.float_data(),
                query.size(0), query.size(1), key.size(1), heads, head_dimension, causal);
        }
    }
    SDXL_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor Ops::timestep_embedding_scalar(float timestep, std::size_t batch,
                                      std::size_t dimension) const {
    Tensor output = Tensor::allocate(*runtime_, {batch, dimension}, ScalarType::Float16, TensorRole::Model);
    timestep_scalar_kernel<<<blocks_for(output.elements()), kThreads, 0, runtime_->stream()>>>(
        timestep, output.half_data(), batch, dimension);
    SDXL_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor Ops::timestep_embedding_values(const Tensor& values, std::size_t dimension) const {
    if (values.type() != ScalarType::Float16 && values.type() != ScalarType::Float32) {
        throw CudaError("CUDA timestep values must be float16 or float32");
    }
    if (values.rank() == 0) throw CudaError("CUDA timestep values cannot be scalar-rank");
    const std::size_t rows = values.elements();
    std::vector<std::size_t> shape = values.shape();
    if (shape.size() == 1) shape.push_back(dimension);
    else {
        const std::size_t last = shape.back();
        shape.pop_back();
        shape.push_back(last * dimension);
    }
    Tensor output = Tensor::allocate(*runtime_, shape, ScalarType::Float16, values.role());
    timestep_values_kernel<<<blocks_for(output.elements()), kThreads, 0, runtime_->stream()>>>(
        values.data(), values.type() == ScalarType::Float16 ? 0 : 1,
        output.half_data(), rows, dimension);
    SDXL_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor Ops::classifier_free_guidance(const Tensor& model_output, std::size_t batch,
                                     float guidance_scale, float guidance_rescale) const {
    if (model_output.rank() == 0 || model_output.size(0) != batch * 2) {
        throw CudaError("CUDA CFG expects negative/positive batch concatenation");
    }
    std::vector<std::size_t> shape = model_output.shape();
    shape[0] = batch;
    Tensor guided = Tensor::allocate(*runtime_, shape, ScalarType::Float16, model_output.role());
    const std::size_t per_batch = guided.elements() / batch;
    cfg_kernel<<<blocks_for(guided.elements()), kThreads, 0, runtime_->stream()>>>(
        model_output.half_data(), guided.half_data(), batch, per_batch, guidance_scale);
    SDXL_CUDA_CHECK(cudaGetLastError());
    if (guidance_rescale > 0.0F) {
        Tensor ratios = Tensor::allocate(*runtime_, {batch}, ScalarType::Float16, TensorRole::Model);
        guidance_stats_kernel<<<static_cast<unsigned>(batch), kThreads, 16 * sizeof(float), runtime_->stream()>>>(
            guided.half_data(), model_output.half_data(), ratios.half_data(), batch, per_batch);
        SDXL_CUDA_CHECK(cudaGetLastError());
        guidance_rescale_kernel<<<blocks_for(guided.elements()), kThreads, 0, runtime_->stream()>>>(
            guided.half_data(), ratios.half_data(), batch, per_batch, guidance_rescale);
        SDXL_CUDA_CHECK(cudaGetLastError());
    }
    return guided;
}

Tensor Ops::euler_scale_input(const Tensor& sample, float sigma) const {
    if (sample.type() != ScalarType::Float16 && sample.type() != ScalarType::Float32) {
        throw CudaError("CUDA Euler scale expects an FP16 or FP32 tensor");
    }
    // The UNet remains FP16/FP8, while ComfyUI-compatible sampler state may
    // remain FP32. Cast only at the model-input boundary.
    Tensor output = Tensor::allocate(*runtime_, sample.shape(), ScalarType::Float16, TensorRole::Model);
    const float inverse = 1.0F / std::sqrt(sigma * sigma + 1.0F);
    if (sample.type() == ScalarType::Float16) {
        euler_scale_kernel<<<blocks_for(sample.elements()), kThreads, 0, runtime_->stream()>>>(
            sample.half_data(), output.half_data(), sample.elements(), inverse);
    } else {
        euler_scale_f32_to_half_kernel<<<blocks_for(sample.elements()), kThreads, 0, runtime_->stream()>>>(
            sample.float_data(), output.half_data(), sample.elements(), inverse);
    }
    SDXL_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor Ops::euler_scale_repeat_input(const Tensor& sample, float sigma,
                                            std::size_t repeats) const {
    if ((sample.type() != ScalarType::Float16 && sample.type() != ScalarType::Float32) ||
        sample.rank() == 0 || repeats == 0) {
        throw CudaError("CUDA Euler scale+repeat expects FP16/FP32 input and positive repeat count");
    }
    std::vector<std::size_t> shape = sample.shape();
    shape[0] *= repeats;
    Tensor output = Tensor::allocate(*runtime_, shape, ScalarType::Float16, TensorRole::Model);
    const float inverse = 1.0F / std::sqrt(sigma * sigma + 1.0F);
    if (sample.type() == ScalarType::Float16) {
        euler_scale_repeat_kernel<<<blocks_for(output.elements()), kThreads, 0, runtime_->stream()>>>(
            sample.half_data(), output.half_data(), sample.elements(), repeats, inverse);
    } else {
        euler_scale_repeat_f32_to_half_kernel<<<blocks_for(output.elements()), kThreads, 0, runtime_->stream()>>>(
            sample.float_data(), output.half_data(), sample.elements(), repeats, inverse);
    }
    SDXL_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor Ops::predicted_original(const Tensor& model_output, const Tensor& sample,
                               float sigma, int prediction_type) const {
    if (model_output.type() != ScalarType::Float16 ||
        (sample.type() != ScalarType::Float16 && sample.type() != ScalarType::Float32) ||
        model_output.shape() != sample.shape()) {
        throw CudaError("CUDA predicted-original conversion expects FP16 model output and matching FP16/FP32 state");
    }
    Tensor output = Tensor::allocate(*runtime_, sample.shape(), sample.type(), sample.role());
    if (sample.type() == ScalarType::Float16) {
        predicted_original_kernel<<<blocks_for(sample.elements()), kThreads, 0, runtime_->stream()>>>(
            model_output.half_data(), sample.half_data(), output.half_data(), sample.elements(),
            sigma, prediction_type);
    } else {
        predicted_original_f16_f32_kernel<<<blocks_for(sample.elements()), kThreads, 0, runtime_->stream()>>>(
            model_output.half_data(), sample.float_data(), output.float_data(), sample.elements(),
            sigma, prediction_type);
    }
    SDXL_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor Ops::batch_slice(const Tensor& input, std::size_t start_batch,
                              std::size_t batch_count) const {
    if (input.type() != ScalarType::Float16 || input.rank() == 0 || batch_count == 0 ||
        start_batch + batch_count > input.size(0)) {
        throw CudaError("CUDA batch_slice received an invalid FP16 batch range");
    }
    std::vector<std::size_t> shape = input.shape();
    shape[0] = batch_count;
    Tensor output = Tensor::allocate(*runtime_, shape, ScalarType::Float16, input.role());
    const std::size_t per_batch = input.elements() / input.size(0);
    SDXL_CUDA_CHECK(cudaMemcpyAsync(output.data(), input.half_data() + start_batch * per_batch,
                                   output.bytes(), cudaMemcpyDeviceToDevice, runtime_->stream()));
    return output;
}

Tensor Ops::combine(const Tensor& a, float ca, const Tensor* b, float cb,
                    const Tensor* c, float cc, const Tensor* noise, float cn) const {
    if (a.type() != ScalarType::Float16 && a.type() != ScalarType::Float32) {
        throw CudaError("CUDA combine expects FP16 or FP32 tensors");
    }
    auto valid = [&](const Tensor* value) {
        return value == nullptr || (value->type() == a.type() && value->shape() == a.shape());
    };
    if (!valid(b) || !valid(c) || !valid(noise)) throw CudaError("CUDA combine tensor shape/type mismatch");
    Tensor output = Tensor::allocate(*runtime_, a.shape(), a.type(), a.role());
    if (a.type() == ScalarType::Float16) {
        combine_half_kernel<<<blocks_for(a.elements()), kThreads, 0, runtime_->stream()>>>(
            a.half_data(), b == nullptr ? nullptr : b->half_data(),
            c == nullptr ? nullptr : c->half_data(), noise == nullptr ? nullptr : noise->half_data(),
            output.half_data(), a.elements(), ca, cb, cc, cn);
    } else {
        combine_float_kernel<<<blocks_for(a.elements()), kThreads, 0, runtime_->stream()>>>(
            a.float_data(), b == nullptr ? nullptr : b->float_data(),
            c == nullptr ? nullptr : c->float_data(), noise == nullptr ? nullptr : noise->float_data(),
            output.float_data(), a.elements(), ca, cb, cc, cn);
    }
    SDXL_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor Ops::dpmpp_2m_step(const Tensor& denoised, const Tensor& sample,
                          const Tensor* old_denoised, float sigma,
                          float sigma_next, float sigma_previous) const {
    if ((sample.type() != ScalarType::Float16 && sample.type() != ScalarType::Float32) ||
        denoised.type() != sample.type() || denoised.shape() != sample.shape() ||
        (old_denoised != nullptr &&
         (old_denoised->type() != sample.type() ||
          old_denoised->shape() != sample.shape()))) {
        throw CudaError("CUDA DPM++ 2M step expects matching FP16 or FP32 tensors");
    }
    if (!(sigma > 0.0F) || sigma_next < 0.0F) {
        throw CudaError("CUDA DPM++ 2M step received invalid sigmas");
    }

    float sample_coefficient = 0.0F;
    float denoised_coefficient = 1.0F;
    float old_coefficient = 0.0F;
    if (sigma_next > 0.0F) {
        const double ratio = static_cast<double>(sigma_next) / static_cast<double>(sigma);
        sample_coefficient = static_cast<float>(ratio);
        denoised_coefficient = static_cast<float>(1.0 - ratio);
        if (old_denoised != nullptr) {
            if (!(sigma_previous > sigma)) {
                throw CudaError("CUDA DPM++ 2M history sigma must exceed the current sigma");
            }
            const double time = -std::log(static_cast<double>(sigma));
            const double time_next = -std::log(static_cast<double>(sigma_next));
            const double time_previous = -std::log(static_cast<double>(sigma_previous));
            const double step_size = time_next - time;
            const double previous_step_size = time - time_previous;
            const double history_ratio = previous_step_size / step_size;
            if (!(history_ratio > 0.0) || !std::isfinite(history_ratio)) {
                throw CudaError("CUDA DPM++ 2M history ratio is invalid");
            }
            denoised_coefficient = static_cast<float>(
                (1.0 - ratio) * (1.0 + 1.0 / (2.0 * history_ratio)));
            old_coefficient = static_cast<float>(
                -(1.0 - ratio) / (2.0 * history_ratio));
        }
    }

    Tensor output = Tensor::allocate(*runtime_, sample.shape(), sample.type(), sample.role());
    if (sample.type() == ScalarType::Float16) {
        dpmpp_2m_step_kernel<<<blocks_for(sample.elements()), kThreads, 0, runtime_->stream()>>>(
            denoised.half_data(), sample.half_data(),
            old_denoised == nullptr ? nullptr : old_denoised->half_data(),
            output.half_data(), sample.elements(), sample_coefficient,
            denoised_coefficient, old_coefficient);
    } else {
        dpmpp_2m_step_float_kernel<<<blocks_for(sample.elements()), kThreads, 0, runtime_->stream()>>>(
            denoised.float_data(), sample.float_data(),
            old_denoised == nullptr ? nullptr : old_denoised->float_data(),
            output.float_data(), sample.elements(), sample_coefficient,
            denoised_coefficient, old_coefficient);
    }
    SDXL_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor Ops::euler_step(const Tensor& model_output, const Tensor& sample,
                       float sigma, float sigma_next, int prediction_type) const {
    if (model_output.type() != ScalarType::Float16 || model_output.shape() != sample.shape() ||
        (sample.type() != ScalarType::Float16 && sample.type() != ScalarType::Float32)) {
        throw CudaError("CUDA Euler step expects FP16 model output and matching FP16/FP32 state");
    }
    Tensor output = Tensor::allocate(*runtime_, sample.shape(), sample.type(), sample.role());
    if (sample.type() == ScalarType::Float16) {
        euler_step_kernel<<<blocks_for(sample.elements()), kThreads, 0, runtime_->stream()>>>(
            model_output.half_data(), sample.half_data(), output.half_data(), sample.elements(),
            sigma, sigma_next - sigma, prediction_type);
    } else {
        euler_step_f16_f32_kernel<<<blocks_for(sample.elements()), kThreads, 0, runtime_->stream()>>>(
            model_output.half_data(), sample.float_data(), output.float_data(), sample.elements(),
            sigma, sigma_next - sigma, prediction_type);
    }
    SDXL_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor Ops::euler_cfg_step(const Tensor& model_output, const Tensor& sample,
                           std::size_t batch, float guidance_scale,
                           float sigma, float sigma_next,
                           int prediction_type) const {
    auto profile = profile_scope("ops/cfg_euler_step");
    if (sample.type() != ScalarType::Float16 || model_output.type() != ScalarType::Float16 ||
        sample.rank() == 0 || sample.size(0) != batch ||
        model_output.rank() != sample.rank() || model_output.size(0) != batch * 2) {
        throw CudaError("CUDA fused Euler CFG step shape/type mismatch");
    }
    for (std::size_t dimension = 1; dimension < sample.rank(); ++dimension) {
        if (model_output.size(dimension) != sample.size(dimension)) {
            throw CudaError("CUDA fused Euler CFG step shape mismatch");
        }
    }
    Tensor output = Tensor::allocate(*runtime_, sample.shape(), ScalarType::Float16, sample.role());
    const std::size_t per_batch = sample.elements() / batch;
    euler_cfg_step_kernel<<<blocks_for(sample.elements()), kThreads, 0, runtime_->stream()>>>(
        model_output.half_data(), sample.half_data(), output.half_data(), batch, per_batch,
        guidance_scale, sigma, sigma_next - sigma, prediction_type);
    SDXL_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor Ops::ddim_step(const Tensor& model_output, const Tensor& sample,
                      float alpha_t, float alpha_prev, float eta,
                      int prediction_type, const Tensor* noise) const {
    if (model_output.type() != ScalarType::Float16 || model_output.shape() != sample.shape() ||
        (sample.type() != ScalarType::Float16 && sample.type() != ScalarType::Float32) ||
        (noise != nullptr && (noise->shape() != sample.shape() || noise->type() != sample.type()))) {
        throw CudaError("CUDA DDIM step expects FP16 model output and matching FP16/FP32 state/noise");
    }
    Tensor output = Tensor::allocate(*runtime_, sample.shape(), sample.type(), sample.role());
    if (sample.type() == ScalarType::Float16) {
        ddim_step_kernel<<<blocks_for(sample.elements()), kThreads, 0, runtime_->stream()>>>(
            model_output.half_data(), sample.half_data(), noise == nullptr ? nullptr : noise->half_data(),
            output.half_data(), sample.elements(), alpha_t, alpha_prev, eta, prediction_type);
    } else {
        ddim_step_f16_f32_kernel<<<blocks_for(sample.elements()), kThreads, 0, runtime_->stream()>>>(
            model_output.half_data(), sample.float_data(), noise == nullptr ? nullptr : noise->float_data(),
            output.float_data(), sample.elements(), alpha_t, alpha_prev, eta, prediction_type);
    }
    SDXL_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor Ops::ddim_cfg_step(const Tensor& model_output, const Tensor& sample,
                          std::size_t batch, float guidance_scale,
                          float alpha_t, float alpha_prev, float eta,
                          int prediction_type, const Tensor* noise) const {
    auto profile = profile_scope("ops/cfg_ddim_step");
    if (sample.type() != ScalarType::Float16 || model_output.type() != ScalarType::Float16 ||
        sample.rank() == 0 || sample.size(0) != batch ||
        model_output.rank() != sample.rank() || model_output.size(0) != batch * 2 ||
        (noise != nullptr && noise->shape() != sample.shape())) {
        throw CudaError("CUDA fused DDIM CFG step shape/type mismatch");
    }
    for (std::size_t dimension = 1; dimension < sample.rank(); ++dimension) {
        if (model_output.size(dimension) != sample.size(dimension)) {
            throw CudaError("CUDA fused DDIM CFG step shape mismatch");
        }
    }
    Tensor output = Tensor::allocate(*runtime_, sample.shape(), ScalarType::Float16, sample.role());
    const std::size_t per_batch = sample.elements() / batch;
    ddim_cfg_step_kernel<<<blocks_for(sample.elements()), kThreads, 0, runtime_->stream()>>>(
        model_output.half_data(), sample.half_data(),
        noise == nullptr ? nullptr : noise->half_data(), output.half_data(), batch, per_batch,
        guidance_scale, alpha_t, alpha_prev, eta, prediction_type);
    SDXL_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor Ops::random_normal(std::vector<std::size_t> shape, std::uint64_t seed, float scale,
                          ScalarType type, TensorRole role) const {
    if (type != ScalarType::Float16 && type != ScalarType::Float32) {
        throw CudaError("CUDA random_normal supports only FP16 and FP32");
    }
    Tensor output = Tensor::allocate(*runtime_, std::move(shape), type, role);
    random_normal_into(output, seed, scale);
    return output;
}

void Ops::random_normal_into(Tensor& output, std::uint64_t seed, float scale) const {
    if (runtime_ == nullptr ||
        (output.type() != ScalarType::Float16 && output.type() != ScalarType::Float32) ||
        (output.role() != TensorRole::Model && output.role() != TensorRole::SamplerState) ||
        output.runtime_state() != runtime_->state()) {
        throw CudaError("CUDA random_normal_into expects a runtime-owned FP16/FP32 model or sampler-state tensor");
    }
    const std::size_t count = output.elements();
    const std::size_t groups = (count + 3U) / 4U;
    if (output.type() == ScalarType::Float16) {
        random_normal_half_kernel<<<blocks_for(groups), kThreads, 0, runtime_->stream()>>>(
            output.half_data(), count, static_cast<unsigned long long>(seed), scale);
    } else {
        random_normal_float_kernel<<<blocks_for(groups), kThreads, 0, runtime_->stream()>>>(
            output.float_data(), count, static_cast<unsigned long long>(seed), scale);
    }
    SDXL_CUDA_CHECK(cudaGetLastError());
}

void Ops::random_normal_batch_into(Tensor& output,
                                   std::span<const std::uint64_t> seeds,
                                   float scale) const {
    if (runtime_ == nullptr ||
        (output.type() != ScalarType::Float16 && output.type() != ScalarType::Float32) ||
        (output.role() != TensorRole::Model && output.role() != TensorRole::SamplerState) ||
        output.runtime_state() != runtime_->state() || output.rank() == 0 || seeds.empty() ||
        output.size(0) != seeds.size()) {
        throw CudaError("CUDA random_normal_batch_into expects FP16/FP32 model/sampler-state [batch,...] output and one seed per batch item");
    }
    std::vector<std::int32_t> seed_words(seeds.size() * 2);
    for (std::size_t index = 0; index < seeds.size(); ++index) {
        seed_words[index * 2] = static_cast<std::int32_t>(seeds[index] & 0xffffffffULL);
        seed_words[index * 2 + 1] = static_cast<std::int32_t>(seeds[index] >> 32U);
    }
    Tensor device_seeds = Tensor::from_host_i32(
        *runtime_, {seeds.size(), 2}, seed_words);
    const std::size_t per_batch = output.elements() / output.size(0);
    const std::size_t groups = ((per_batch + 3U) / 4U) * output.size(0);
    if (output.type() == ScalarType::Float16) {
        random_normal_batch_half_kernel<<<blocks_for(groups), kThreads, 0, runtime_->stream()>>>(
            output.half_data(), per_batch, output.size(0),
            reinterpret_cast<const unsigned long long*>(device_seeds.data()), scale);
    } else {
        random_normal_batch_float_kernel<<<blocks_for(groups), kThreads, 0, runtime_->stream()>>>(
            output.float_data(), per_batch, output.size(0),
            reinterpret_cast<const unsigned long long*>(device_seeds.data()), scale);
    }
    SDXL_CUDA_CHECK(cudaGetLastError());
}

void Ops::check_finite(const Tensor& tensor, std::string_view name) const {
#if defined(SDXL_ENABLE_FINITE_CHECKS)
    if (tensor.type() != ScalarType::Float16 && tensor.type() != ScalarType::Float32) {
        throw CudaError("finite check expects a floating-point tensor");
    }
    int* device_failure = nullptr;
    int host_failure = 0;
    SDXL_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&device_failure), sizeof(int), runtime_->stream()));
    SDXL_CUDA_CHECK(cudaMemsetAsync(device_failure, 0, sizeof(int), runtime_->stream()));
    if (tensor.type() == ScalarType::Float16) {
        finite_kernel<<<blocks_for(tensor.elements()), kThreads, 0, runtime_->stream()>>>(
            tensor.half_data(), tensor.elements(), device_failure);
    } else {
        finite_f32_kernel<<<blocks_for(tensor.elements()), kThreads, 0, runtime_->stream()>>>(
            tensor.float_data(), tensor.elements(), device_failure);
    }
    SDXL_CUDA_CHECK(cudaGetLastError());
    SDXL_CUDA_CHECK(cudaMemcpyAsync(&host_failure, device_failure, sizeof(int),
                                    cudaMemcpyDeviceToHost, runtime_->stream()));
    SDXL_CUDA_CHECK(cudaFreeAsync(device_failure, runtime_->stream()));
    runtime_->synchronize();
    if (host_failure != 0) throw CudaError(std::string(name) + " contains NaN or infinity");
#else
    (void)tensor;
    (void)name;
    throw CudaError(
        "finite tracing is disabled in this build; configure with -DSDXL_ENABLE_FINITE_CHECKS=ON");
#endif
}

} // namespace sdxl::cuda
