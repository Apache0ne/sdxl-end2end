#pragma once

#include "sdxl/cuda/runtime.hpp"
#include "sdxl/text_encoder.hpp"

#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace sdxl::cuda {

enum class ScalarType : std::uint8_t {
    Float8E4M3,
    Float8E5M2,
    Float16,
    Float32,
    Int32
};

enum class TensorRole : std::uint8_t {
    Model,
    VAE,
    FP8ScaleMetadata,
    HostInterop
};

enum class FP8ScaleMode : std::uint8_t {
    None,
    TensorWide,
    PerOutputChannel
};

// Physical byte order for rank-2 FP8 matrices. RowMajorNK matches the
// SafeTensors/cuBLASLt logical [N,K] layout. KMajorKN stores the same logical
// matrix transposed as [K,N], which gives the SM70-SM86 weight-only WMMA
// kernel coalesced 16-column tile loads without expanding the matrix.
enum class FP8StorageLayout : std::uint8_t {
    RowMajorNK,
    KMajorKN
};

[[nodiscard]] std::size_t scalar_size(ScalarType type) noexcept;
[[nodiscard]] const char* scalar_name(ScalarType type) noexcept;
[[nodiscard]] const char* tensor_role_name(TensorRole role) noexcept;
[[nodiscard]] bool is_fp8(ScalarType type) noexcept;

struct DeviceAllocation final {
    std::shared_ptr<RuntimeState> runtime;
    void* pointer = nullptr;
    std::size_t bytes = 0;
    bool persistent = false;

    ~DeviceAllocation();
};

class Tensor final {
public:
    Tensor() = default;

    [[nodiscard]] static Tensor allocate(const Runtime& runtime,
                                         std::vector<std::size_t> shape,
                                         ScalarType type = ScalarType::Float16,
                                         TensorRole role = TensorRole::Model);
    [[nodiscard]] static Tensor zeros(const Runtime& runtime,
                                      std::vector<std::size_t> shape,
                                      ScalarType type = ScalarType::Float16,
                                      TensorRole role = TensorRole::Model);
    // Model weights are long-lived and must not consume the temporary tensor
    // slab. Persistent allocations are released directly when the component is
    // unloaded and are intentionally excluded from temporary-arena churn stats.
    [[nodiscard]] static Tensor allocate_persistent(
        const Runtime& runtime, std::vector<std::size_t> shape,
        ScalarType type = ScalarType::Float16,
        TensorRole role = TensorRole::Model);
    [[nodiscard]] static Tensor zeros_persistent(
        const Runtime& runtime, std::vector<std::size_t> shape,
        ScalarType type = ScalarType::Float16,
        TensorRole role = TensorRole::Model);
    [[nodiscard]] static Tensor from_host_f32(const Runtime& runtime,
                                             const FloatTensor& host,
                                             ScalarType type = ScalarType::Float16,
                                             TensorRole role = TensorRole::Model);
    [[nodiscard]] static Tensor from_host_i32(const Runtime& runtime,
                                             std::vector<std::size_t> shape,
                                             std::span<const std::int32_t> values);

    [[nodiscard]] Tensor view(std::vector<std::size_t> shape,
                              std::size_t element_offset = 0) const;
    [[nodiscard]] Tensor reshape(std::vector<std::size_t> shape) const;

    [[nodiscard]] bool defined() const noexcept { return data_ != nullptr; }
    [[nodiscard]] void* data() noexcept { return data_; }
    [[nodiscard]] const void* data() const noexcept { return data_; }
    [[nodiscard]] std::uint8_t* fp8_data() noexcept { return static_cast<std::uint8_t*>(data_); }
    [[nodiscard]] const std::uint8_t* fp8_data() const noexcept { return static_cast<const std::uint8_t*>(data_); }
    [[nodiscard]] __half* half_data() noexcept { return static_cast<__half*>(data_); }
    [[nodiscard]] const __half* half_data() const noexcept { return static_cast<const __half*>(data_); }
    [[nodiscard]] float* float_data() noexcept { return static_cast<float*>(data_); }
    [[nodiscard]] const float* float_data() const noexcept { return static_cast<const float*>(data_); }
    [[nodiscard]] std::int32_t* int32_data() noexcept { return static_cast<std::int32_t*>(data_); }
    [[nodiscard]] const std::int32_t* int32_data() const noexcept { return static_cast<const std::int32_t*>(data_); }

    [[nodiscard]] const std::vector<std::size_t>& shape() const noexcept { return shape_; }
    [[nodiscard]] ScalarType type() const noexcept { return type_; }
    [[nodiscard]] TensorRole role() const noexcept { return role_; }
    [[nodiscard]] std::size_t rank() const noexcept { return shape_.size(); }
    [[nodiscard]] std::size_t size(std::size_t dimension) const { return shape_.at(dimension); }
    [[nodiscard]] std::size_t elements() const noexcept;
    [[nodiscard]] std::size_t bytes() const noexcept { return elements() * scalar_size(type_); }
    [[nodiscard]] bool contiguous() const noexcept { return true; }
    [[nodiscard]] std::shared_ptr<RuntimeState> runtime_state() const noexcept { return runtime_; }

    [[nodiscard]] bool has_dequant_scale() const noexcept { return dequant_scale_data_ != nullptr; }
    [[nodiscard]] const float* dequant_scale_data() const noexcept { return dequant_scale_data_; }
    [[nodiscard]] std::size_t dequant_scale_count() const noexcept { return dequant_scale_count_; }
    [[nodiscard]] FP8ScaleMode dequant_scale_mode() const noexcept { return dequant_scale_mode_; }
    [[nodiscard]] FP8StorageLayout fp8_storage_layout() const noexcept { return fp8_storage_layout_; }
    void set_fp8_storage_layout(FP8StorageLayout layout);
    void attach_dequant_scale(const Tensor& scale, FP8ScaleMode mode = FP8ScaleMode::TensorWide);

    [[nodiscard]] FloatTensor to_host_f32(const Runtime& runtime) const;
    void copy_from(const Runtime& runtime, const Tensor& source);

private:
    Tensor(std::shared_ptr<DeviceAllocation> allocation,
           std::shared_ptr<RuntimeState> runtime,
           void* data,
           std::vector<std::size_t> shape,
           ScalarType type,
           TensorRole role);

    std::shared_ptr<DeviceAllocation> allocation_;
    std::shared_ptr<RuntimeState> runtime_;
    void* data_ = nullptr;
    std::vector<std::size_t> shape_;
    ScalarType type_ = ScalarType::Float16;
    TensorRole role_ = TensorRole::Model;
    std::shared_ptr<DeviceAllocation> dequant_scale_allocation_;
    const float* dequant_scale_data_ = nullptr;
    std::size_t dequant_scale_count_ = 0;
    FP8ScaleMode dequant_scale_mode_ = FP8ScaleMode::None;
    FP8StorageLayout fp8_storage_layout_ = FP8StorageLayout::RowMajorNK;
};

[[nodiscard]] std::size_t element_count(std::span<const std::size_t> shape);
void validate_same_runtime(const Tensor& first, const Tensor& second);

} // namespace sdxl::cuda
