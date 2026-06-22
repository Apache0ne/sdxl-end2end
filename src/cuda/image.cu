#include "sdxl/cuda/image.hpp"
#include "runtime_internal.hpp"

#include <cuda_fp16.h>

#include <cmath>
#include <limits>
#include <utility>

namespace sdxl::cuda {
namespace {

constexpr unsigned kThreads = 256;

[[nodiscard]] unsigned blocks_for(std::size_t count) {
    return static_cast<unsigned>((count + kThreads - 1) / kThreads);
}

__global__ void nchw_f16_to_rgb8_kernel(const __half* decoded,
                                         std::uint8_t* output,
                                         std::size_t batch,
                                         std::size_t height,
                                         std::size_t width) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = batch * height * width * 3;
    if (index >= count) return;

    const std::size_t channel = index % 3;
    const std::size_t pixel = index / 3;
    const std::size_t x = pixel % width;
    const std::size_t y = (pixel / width) % height;
    const std::size_t batch_index = pixel / (width * height);
    const std::size_t source = ((batch_index * 3 + channel) * height + y) * width + x;

    float value = __half2float(decoded[source]) * 0.5F + 0.5F;
    value = value < 0.0F ? 0.0F : (value > 1.0F ? 1.0F : value);
    output[index] = static_cast<std::uint8_t>(value * 255.0F + 0.5F);
}

__global__ void nchw_f32_to_rgb8_kernel(const float* decoded,
                                         std::uint8_t* output,
                                         std::size_t batch,
                                         std::size_t height,
                                         std::size_t width) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = batch * height * width * 3;
    if (index >= count) return;
    const std::size_t channel = index % 3;
    const std::size_t pixel = index / 3;
    const std::size_t x = pixel % width;
    const std::size_t y = (pixel / width) % height;
    const std::size_t batch_index = pixel / (width * height);
    const std::size_t source = ((batch_index * 3 + channel) * height + y) * width + x;
    float value = decoded[source] * 0.5F + 0.5F;
    value = value < 0.0F ? 0.0F : (value > 1.0F ? 1.0F : value);
    output[index] = static_cast<std::uint8_t>(value * 255.0F + 0.5F);
}

} // namespace

ImageConverter::ImageConverter(const Runtime& runtime) : runtime_(&runtime) {}

std::vector<RGBImage> ImageConverter::to_rgb8(const Tensor& decoded) const {
    if (runtime_ == nullptr) throw CudaError("CUDA image converter has no runtime");
    if ((decoded.type() != ScalarType::Float16 && decoded.type() != ScalarType::Float32) ||
        decoded.rank() != 4 || decoded.size(1) != 3) {
        throw CudaError("CUDA image conversion expects float16 or float32 [B,3,H,W]");
    }
    const std::size_t batch = decoded.size(0);
    const std::size_t height = decoded.size(2);
    const std::size_t width = decoded.size(3);
    if (width > std::numeric_limits<std::uint32_t>::max() ||
        height > std::numeric_limits<std::uint32_t>::max()) {
        throw CudaError("image dimensions exceed PNG limits");
    }
    const std::size_t per_image = height * width * 3;
    const std::size_t total = batch * per_image;

    std::uint8_t* device_pixels = nullptr;
    SDXL_CUDA_CHECK(cudaMallocAsync(
        reinterpret_cast<void**>(&device_pixels), total, runtime_->stream()));
    if (decoded.type() == ScalarType::Float16) {
        nchw_f16_to_rgb8_kernel<<<blocks_for(total), kThreads, 0, runtime_->stream()>>>(
            decoded.half_data(), device_pixels, batch, height, width);
    } else {
        nchw_f32_to_rgb8_kernel<<<blocks_for(total), kThreads, 0, runtime_->stream()>>>(
            decoded.float_data(), device_pixels, batch, height, width);
    }
    SDXL_CUDA_CHECK(cudaGetLastError());

    std::vector<std::uint8_t> host(total);
    SDXL_CUDA_CHECK(cudaMemcpyAsync(
        host.data(), device_pixels, total, cudaMemcpyDeviceToHost, runtime_->stream()));
    SDXL_CUDA_CHECK(cudaFreeAsync(device_pixels, runtime_->stream()));
    runtime_->synchronize();

    std::vector<RGBImage> images;
    images.reserve(batch);
    for (std::size_t item = 0; item < batch; ++item) {
        RGBImage image;
        image.width = width;
        image.height = height;
        const auto begin = host.begin() + static_cast<std::ptrdiff_t>(item * per_image);
        image.pixels.assign(begin, begin + static_cast<std::ptrdiff_t>(per_image));
        images.push_back(std::move(image));
    }
    return images;
}

} // namespace sdxl::cuda
