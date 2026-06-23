#pragma once

#include "sdxl/cuda/tensor.hpp"
#include "sdxl/sdxl.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace sdxl::cuda {

enum class FP8Format : std::uint8_t {
    Auto,
    E4M3,
    E5M2
};

enum class FP8Backend : std::uint8_t {
    Auto,
    Native,
    WeightOnly
};

struct WeightLoadStats {
    std::size_t tensors = 0;
    std::size_t source_bytes = 0;
    std::size_t device_bytes = 0;
    std::size_t fp8_e4m3_tensors = 0;
    std::size_t fp8_e5m2_tensors = 0;
    std::size_t int8_tensors = 0;
    std::size_t int8_convrot_tensors = 0;
    std::size_t fp16_tensors = 0;
    std::size_t fp32_tensors = 0;
    std::size_t tensorwide_scaled_tensors = 0;
    std::size_t channel_scaled_tensors = 0;

    [[nodiscard]] std::size_t fp8_tensors() const noexcept {
        return fp8_e4m3_tensors + fp8_e5m2_tensors;
    }
};

struct FP8WeightLoadOptions {
    bool quantize_linear_weights = true;
    bool preserve_native_fp8 = true;
    FP8Format format = FP8Format::Auto;
    FP8ScaleMode scale_mode = FP8ScaleMode::TensorWide;
    FP8Backend backend = FP8Backend::Auto;
};


struct INT8WeightLoadOptions {
    bool quantize_floating_weights = true;
    bool require_prequantized = false;
    bool strict = false;
    bool enable_convrot = true;
    std::size_t convrot_group_size = 256;
};

struct FP8CacheOptions {
    std::filesystem::path path;
    std::string key;
    bool read = true;
    bool write = true;
};

struct FP8CacheStats {
    bool hit = false;
    bool written = false;
    std::size_t tensors_loaded = 0;
    std::size_t bytes_loaded = 0;
    std::size_t bytes_written = 0;
};

class WeightStore final {
public:
    WeightStore(const Runtime& runtime, const SDXLModel& model);

    WeightStore(const WeightStore&) = delete;
    WeightStore& operator=(const WeightStore&) = delete;

    [[nodiscard]] WeightLoadStats load_prefix(
        std::string_view prefix, ScalarType destination_type = ScalarType::Float16);
    [[nodiscard]] WeightLoadStats load_prefixes(
        const std::vector<std::string>& prefixes,
        ScalarType destination_type = ScalarType::Float16);
    [[nodiscard]] WeightLoadStats load_unet_fp8(FP8WeightLoadOptions options = {},
                                                const FP8CacheOptions* cache = nullptr,
                                                FP8CacheStats* cache_stats = nullptr);
    [[nodiscard]] WeightLoadStats load_prefixes_int8(
        const std::vector<std::string>& prefixes,
        INT8WeightLoadOptions options = {});
    void unload_prefix(std::string_view prefix);
    void clear();

    [[nodiscard]] bool contains(std::string_view logical_name) const;
    [[nodiscard]] const Tensor& get(std::string_view logical_name) const;
    [[nodiscard]] std::size_t device_bytes() const noexcept { return device_bytes_; }
    [[nodiscard]] const FP8WeightLoadOptions& active_unet_fp8_options() const noexcept {
        return active_unet_fp8_options_;
    }
    [[nodiscard]] const INT8WeightLoadOptions& active_int8_options() const noexcept {
        return active_int8_options_;
    }

private:
    [[nodiscard]] Tensor upload_tensor(const TensorView& source,
                                       ScalarType destination_type,
                                       TensorRole role,
                                       FP8ScaleMode scale_mode = FP8ScaleMode::TensorWide) const;

    const Runtime* runtime_ = nullptr;
    const SDXLModel* model_ = nullptr;
    std::unordered_map<std::string, Tensor> tensors_;
    std::size_t device_bytes_ = 0;
    [[nodiscard]] bool load_fp8_cache(const FP8CacheOptions& cache,
                                      WeightLoadStats& stats,
                                      FP8CacheStats& cache_stats);
    void write_fp8_cache(const FP8CacheOptions& cache,
                         FP8CacheStats& cache_stats) const;

    FP8WeightLoadOptions active_unet_fp8_options_{};
    INT8WeightLoadOptions active_int8_options_{};
};

} // namespace sdxl::cuda
