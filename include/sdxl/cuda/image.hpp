#pragma once

#include "sdxl/cuda/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace sdxl::cuda {

struct RGBImage {
    std::size_t width = 0;
    std::size_t height = 0;
    std::vector<std::uint8_t> pixels; // tightly packed RGBRGB...
};

class ImageConverter final {
public:
    explicit ImageConverter(const Runtime& runtime);

    // Converts [B,3,H,W] float16 or float32 decoder output to host RGB8 images.
    // SDXL/VAE postprocessing is: clamp(sample / 2 + 0.5, 0, 1), then round to u8.
    [[nodiscard]] std::vector<RGBImage> to_rgb8(const Tensor& decoded) const;

private:
    const Runtime* runtime_ = nullptr;
};

// Dependency-free PNG writer (8-bit RGB, non-interlaced). The writer emits a
// standards-compliant zlib stream using stored DEFLATE blocks, avoiding libpng/zlib.
void write_png(const std::filesystem::path& path, const RGBImage& image);

} // namespace sdxl::cuda
