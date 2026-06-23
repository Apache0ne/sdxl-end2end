#include "sdxl/cuda/ops.hpp"
#include "sdxl/cuda/profiler.hpp"
#include "runtime_internal.hpp"
#include "fp8_internal.hpp"
#include "int8_convrot.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <limits>
#include <mutex>
#include <sstream>
#include <vector>

namespace sdxl::cuda {
namespace {

class LtMatmulDesc final {
public:
    LtMatmulDesc(cublasComputeType_t compute_type, cudaDataType_t scale_type) {
        SDXL_CUBLAS_CHECK(cublasLtMatmulDescCreate(&value, compute_type, scale_type));
    }
    ~LtMatmulDesc() { if (value != nullptr) cublasLtMatmulDescDestroy(value); }
    cublasLtMatmulDesc_t value = nullptr;
};

class LtLayout final {
public:
    LtLayout(cudaDataType_t data_type,
             std::uint64_t rows,
             std::uint64_t columns,
             std::int64_t leading_dimension,
             cublasLtOrder_t order = CUBLASLT_ORDER_ROW) {
        SDXL_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
            &value, data_type, rows, columns, leading_dimension));
        SDXL_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
            value, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    }
    ~LtLayout() { if (value != nullptr) cublasLtMatrixLayoutDestroy(value); }
    cublasLtMatrixLayout_t value = nullptr;
};

class LtPreference final {
public:
    explicit LtPreference(std::size_t workspace_bytes) {
        SDXL_CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&value));
        SDXL_CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
            value, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_bytes, sizeof(workspace_bytes)));
    }
    ~LtPreference() { if (value != nullptr) cublasLtMatmulPreferenceDestroy(value); }
    cublasLtMatmulPreference_t value = nullptr;
};

[[nodiscard]] cudaDataType_t cublas_data_type(ScalarType type) {
    switch (type) {
    case ScalarType::Float8E4M3: return CUDA_R_8F_E4M3;
    case ScalarType::Float8E5M2: return CUDA_R_8F_E5M2;
    case ScalarType::Int8: return CUDA_R_8I;
    case ScalarType::Float16: return CUDA_R_16F;
    case ScalarType::Float32: return CUDA_R_32F;
    case ScalarType::Int32: break;
    }
    throw CudaError("cuBLASLt operation requires a floating-point tensor");
}

[[nodiscard]] cudnnDataType_t cudnn_data_type(ScalarType type) {
    switch (type) {
    case ScalarType::Float16: return CUDNN_DATA_HALF;
    case ScalarType::Float32: return CUDNN_DATA_FLOAT;
    case ScalarType::Float8E4M3:
    case ScalarType::Float8E5M2:
    case ScalarType::Int8:
    case ScalarType::Int32: break;
    }
    throw CudaError("cuDNN convolution requires float16 or float32 tensors");
}

[[nodiscard]] int checked_int(std::size_t value, const char* name) {
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw CudaError(std::string(name) + " exceeds cuDNN int range");
    }
    return static_cast<int>(value);
}

[[nodiscard]] std::string linear_profile_label(const Tensor& input, const Tensor& weight) {
    const std::size_t k = input.shape().back();
    const std::size_t m = input.elements() / k;
    const std::size_t n = weight.size(0);
    std::ostringstream stream;
    stream << "ops/linear/" << scalar_name(weight.type()) << "/m" << m
           << "_n" << n << "_k" << k;
    return stream.str();
}

[[nodiscard]] std::string convolution_profile_label(const Tensor& input,
                                                     const Tensor& weight,
                                                     int stride_y,
                                                     int stride_x) {
    std::ostringstream stream;
    stream << "ops/convolution/" << scalar_name(input.type())
           << "/n" << input.size(0) << "_c" << input.size(1)
           << "_h" << input.size(2) << "_w" << input.size(3)
           << "_k" << weight.size(0) << "_r" << weight.size(2)
           << "_s" << weight.size(3) << "_stride" << stride_y << 'x' << stride_x;
    return stream.str();
}


[[nodiscard]] std::shared_ptr<MatmulDescriptors> create_matmul_descriptors(
    cublasComputeType_t compute_type,
    cudaDataType_t scale_type,
    cublasOperation_t trans_a,
    cublasOperation_t trans_b,
    cudaDataType_t a_type, std::uint64_t a_rows, std::uint64_t a_columns,
    std::int64_t a_ld, cublasLtOrder_t a_order,
    cudaDataType_t b_type, std::uint64_t b_rows, std::uint64_t b_columns,
    std::int64_t b_ld, cublasLtOrder_t b_order,
    cudaDataType_t c_type, std::uint64_t c_rows, std::uint64_t c_columns,
    std::int64_t c_ld, cublasLtOrder_t c_order,
    std::size_t workspace_bytes) {
    auto descriptors = std::make_shared<MatmulDescriptors>();
    SDXL_CUBLAS_CHECK(cublasLtMatmulDescCreate(
        &descriptors->operation, compute_type, scale_type));
    SDXL_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        descriptors->operation, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
    SDXL_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        descriptors->operation, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));
    SDXL_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
        &descriptors->a, a_type, a_rows, a_columns, a_ld));
    SDXL_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
        &descriptors->b, b_type, b_rows, b_columns, b_ld));
    SDXL_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
        &descriptors->c, c_type, c_rows, c_columns, c_ld));
    SDXL_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
        &descriptors->d, c_type, c_rows, c_columns, c_ld));
    SDXL_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        descriptors->a, CUBLASLT_MATRIX_LAYOUT_ORDER, &a_order, sizeof(a_order)));
    SDXL_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        descriptors->b, CUBLASLT_MATRIX_LAYOUT_ORDER, &b_order, sizeof(b_order)));
    SDXL_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        descriptors->c, CUBLASLT_MATRIX_LAYOUT_ORDER, &c_order, sizeof(c_order)));
    SDXL_CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        descriptors->d, CUBLASLT_MATRIX_LAYOUT_ORDER, &c_order, sizeof(c_order)));
    SDXL_CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&descriptors->preference));
    SDXL_CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        descriptors->preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_bytes, sizeof(workspace_bytes)));
    return descriptors;
}

[[nodiscard]] Tensor native_fp8_linear(const Runtime& runtime,
                                       const Tensor& input,
                                       const Tensor& weight) {
    const std::size_t k = input.shape().back();
    const std::size_t n = weight.size(0);
    const std::size_t m = input.elements() / k;
    if (weight.size(1) != k || input.type() != ScalarType::Float16 ||
        !is_fp8(weight.type()) || !weight.has_dequant_scale() ||
        weight.dequant_scale_mode() != FP8ScaleMode::TensorWide) {
        throw CudaError("native FP8 linear received incompatible tensors or scaling metadata");
    }

    Tensor input_fp8 = quantize_tensor_fp8(runtime, input, weight.type());
    std::vector<std::size_t> output_shape = input.shape();
    output_shape.back() = n;
    Tensor output = Tensor::allocate(runtime, output_shape, ScalarType::Float16,
                                     TensorRole::Model);

    // NVIDIA requires CUBLAS_COMPUTE_32F and CUDA_R_32F scales for native FP8.
    // This is internal accumulation/metadata; C and D remain FP16 and no FP32
    // model activation tensor is materialized.
    const cudaDataType_t fp8_type = cublas_data_type(weight.type());
    const int variant = weight.type() == ScalarType::Float8E4M3 ? 100 : 101;
    const MatmulKey key{m, n, k, variant};
    MatmulPlan plan{};
    {
        std::lock_guard lock(runtime.state()->cublas_mutex);
        const auto cached = runtime.state()->matmul_plans.find(key);
        if (cached != runtime.state()->matmul_plans.end()) {
            plan = cached->second;
        } else {
            plan.descriptors = create_matmul_descriptors(
                CUBLAS_COMPUTE_32F, CUDA_R_32F, CUBLAS_OP_T, CUBLAS_OP_N,
                fp8_type, k, n, static_cast<std::int64_t>(k), CUBLASLT_ORDER_COL,
                fp8_type, k, m, static_cast<std::int64_t>(k), CUBLASLT_ORDER_COL,
                CUDA_R_16F, n, m, static_cast<std::int64_t>(n), CUBLASLT_ORDER_COL,
                runtime.options().cublas_workspace_bytes);
            cublasLtMatmulHeuristicResult_t heuristic{};
            int returned = 0;
            SDXL_CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
                runtime.cublas_lt(), plan.descriptors->operation,
                plan.descriptors->a, plan.descriptors->b,
                plan.descriptors->c, plan.descriptors->d,
                plan.descriptors->preference, 1, &heuristic, &returned));
            if (returned == 0 || heuristic.state != CUBLAS_STATUS_SUCCESS) {
                throw CudaError("cuBLASLt found no native FP8 algorithm for this projection");
            }
            plan.algorithm = heuristic.algo;
            plan.workspace_bytes = heuristic.workspaceSize;
            runtime.state()->matmul_plans.emplace(key, plan);
        }
        if (!plan.descriptors) throw CudaError("cached native FP8 plan has no descriptors");
        const float* a_scale = weight.dequant_scale_data();
        const float* b_scale = input_fp8.dequant_scale_data();
        SDXL_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            plan.descriptors->operation, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            &a_scale, sizeof(a_scale)));
        SDXL_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            plan.descriptors->operation, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            &b_scale, sizeof(b_scale)));
        constexpr float alpha = 1.0F;
        constexpr float beta = 0.0F;
        SDXL_CUBLAS_CHECK(cublasLtMatmul(
            runtime.cublas_lt(), plan.descriptors->operation, &alpha,
            weight.data(), plan.descriptors->a,
            input_fp8.data(), plan.descriptors->b, &beta,
            output.data(), plan.descriptors->c,
            output.data(), plan.descriptors->d,
            &plan.algorithm, runtime.state()->cublas_workspace,
            plan.workspace_bytes, runtime.stream()));
    }
    return output;
}

} // namespace

Ops::Ops(const Runtime& runtime) : runtime_(&runtime) {}

Tensor Ops::quantize_fp8(const Tensor& input, ScalarType fp8_type) const {
    if (runtime_ == nullptr) throw CudaError("CUDA Ops has no runtime");
    return quantize_tensor_fp8(*runtime_, input, fp8_type);
}

Tensor Ops::quantize_e4m3(const Tensor& input) const {
    return quantize_fp8(input, ScalarType::Float8E4M3);
}

Tensor Ops::quantize_e5m2(const Tensor& input) const {
    return quantize_fp8(input, ScalarType::Float8E5M2);
}

Tensor Ops::linear(const Tensor& input, const Tensor& weight, const Tensor* bias) const {
    if (runtime_ == nullptr) throw CudaError("CUDA Ops has no runtime");
    auto profile = profile_scope(linear_profile_label(input, weight));
    if (weight.type() == ScalarType::Int8) {
        if ((input.type() != ScalarType::Float16 && input.type() != ScalarType::Float32) ||
            input.role() != TensorRole::Model || input.rank() < 2 || weight.rank() != 2) {
            throw CudaError("INT8 linear requires FP16/FP32 model activations and a rank-2 INT8 weight");
        }
        const std::size_t n = weight.size(0);
        if (bias != nullptr && (bias->type() != input.type() ||
                                bias->role() != input.role() ||
                                bias->rank() != 1 || bias->size(0) != n)) {
            throw CudaError("INT8 linear bias mismatch");
        }
        Tensor output = int8_convrot_linear(*runtime_, input, weight);
        if (bias != nullptr) add_last_dim_bias_in_place(output, *bias);
        return output;
    }
    if (is_fp8(weight.type())) {
        if (input.type() != ScalarType::Float16 || input.role() != TensorRole::Model ||
            input.rank() < 2 || weight.rank() != 2) {
            throw CudaError("FP8 UNet linear requires FP16 model activations and a rank-2 FP8 weight");
        }
        const std::size_t k = input.shape().back();
        const std::size_t n = weight.size(0);
        const std::size_t m = input.elements() / k;
        if (weight.size(1) != k) throw CudaError("FP8 UNet linear feature mismatch");
        if (bias != nullptr && (bias->type() != ScalarType::Float16 ||
                                bias->role() != TensorRole::Model ||
                                bias->rank() != 1 || bias->size(0) != n)) {
            throw CudaError("FP8 UNet linear bias mismatch");
        }
        Tensor output;
        const bool aligned_native_shape = (m % 16 == 0) && (n % 16 == 0) && (k % 16 == 0);
        const bool native_scaling = weight.dequant_scale_mode() == FP8ScaleMode::TensorWide;
        const bool native_layout = weight.fp8_storage_layout() == FP8StorageLayout::RowMajorNK;
        const bool native_candidate =
            runtime_->fp8_execution_mode() == FP8ExecutionMode::NativeTensorCore &&
            aligned_native_shape && native_scaling && native_layout;
        if (native_candidate) {
            try {
                output = native_fp8_linear(*runtime_, input, weight);
            } catch (const CudaError&) {
                if (runtime_->options().fp8_backend == FP8BackendPreference::NativeOnly) throw;
                output = fp8_weight_only_linear(*runtime_, input, weight);
            }
        } else {
            if (runtime_->options().fp8_backend == FP8BackendPreference::NativeOnly) {
                throw CudaError("native-only FP8 cannot execute this shape or scale mode");
            }
            output = fp8_weight_only_linear(*runtime_, input, weight);
        }
        if (bias != nullptr) add_last_dim_bias_in_place(output, *bias);
        return output;
    }

    if ((input.type() != ScalarType::Float16 && input.type() != ScalarType::Float32) ||
        weight.type() != input.type() || weight.role() != input.role()) {
        throw CudaError("CUDA linear expects matching FP16/FP32 tensors and roles, or FP8 weights");
    }
    if (input.rank() < 2 || weight.rank() != 2) throw CudaError("CUDA linear rank mismatch");
    const std::size_t k = input.shape().back();
    const std::size_t n = weight.size(0);
    if (weight.size(1) != k) throw CudaError("CUDA linear feature mismatch");
    const std::size_t m = input.elements() / k;
    if (bias != nullptr) {
        validate_same_runtime(input, *bias);
        if (bias->type() != input.type() || bias->role() != input.role() ||
            bias->rank() != 1 || bias->size(0) != n) {
            throw CudaError("CUDA linear bias mismatch");
        }
    }

    std::vector<std::size_t> output_shape = input.shape();
    output_shape.back() = n;
    Tensor output = Tensor::allocate(*runtime_, output_shape, input.type(), input.role());

    const cudaDataType_t data_type = cublas_data_type(input.type());
    const bool model_fp16_accum = input.type() == ScalarType::Float16 &&
        input.role() == TensorRole::Model &&
        runtime_->options().non_vae_accumulation == NonVAEAccumulation::Float16;
    const cublasComputeType_t compute_type = model_fp16_accum
        ? CUBLAS_COMPUTE_16F
        : (input.type() == ScalarType::Float32 && runtime_->options().allow_tf32
            ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F);
    const cudaDataType_t scale_type = model_fp16_accum ? CUDA_R_16F : CUDA_R_32F;
    const int variant = static_cast<int>(input.type()) * 10 + (model_fp16_accum ? 1 : 2);
    const MatmulKey key{m, n, k, variant};
    MatmulPlan plan{};
    {
        std::lock_guard lock(runtime_->state()->cublas_mutex);
        const auto cached = runtime_->state()->matmul_plans.find(key);
        if (cached != runtime_->state()->matmul_plans.end()) {
            plan = cached->second;
        } else {
            plan.descriptors = create_matmul_descriptors(
                compute_type, scale_type, CUBLAS_OP_N, CUBLAS_OP_T,
                data_type, m, k, static_cast<std::int64_t>(k), CUBLASLT_ORDER_ROW,
                data_type, n, k, static_cast<std::int64_t>(k), CUBLASLT_ORDER_ROW,
                data_type, m, n, static_cast<std::int64_t>(n), CUBLASLT_ORDER_ROW,
                runtime_->options().cublas_workspace_bytes);
            cublasLtMatmulHeuristicResult_t heuristic{};
            int returned = 0;
            SDXL_CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
                runtime_->cublas_lt(), plan.descriptors->operation,
                plan.descriptors->a, plan.descriptors->b,
                plan.descriptors->c, plan.descriptors->d,
                plan.descriptors->preference, 1, &heuristic, &returned));
            if (returned == 0 || heuristic.state != CUBLAS_STATUS_SUCCESS) {
                throw CudaError("cuBLASLt found no valid algorithm for linear projection");
            }
            plan.algorithm = heuristic.algo;
            plan.workspace_bytes = heuristic.workspaceSize;
            runtime_->state()->matmul_plans.emplace(key, plan);
        }
        if (!plan.descriptors) throw CudaError("cached cuBLASLt plan has no descriptors");
        if (model_fp16_accum) {
            const __half alpha = __float2half(1.0F);
            const __half beta = __float2half(0.0F);
            SDXL_CUBLAS_CHECK(cublasLtMatmul(
                runtime_->cublas_lt(), plan.descriptors->operation, &alpha,
                input.data(), plan.descriptors->a,
                weight.data(), plan.descriptors->b, &beta,
                output.data(), plan.descriptors->c,
                output.data(), plan.descriptors->d,
                &plan.algorithm, runtime_->state()->cublas_workspace,
                plan.workspace_bytes, runtime_->stream()));
        } else {
            constexpr float alpha = 1.0F;
            constexpr float beta = 0.0F;
            SDXL_CUBLAS_CHECK(cublasLtMatmul(
                runtime_->cublas_lt(), plan.descriptors->operation, &alpha,
                input.data(), plan.descriptors->a,
                weight.data(), plan.descriptors->b, &beta,
                output.data(), plan.descriptors->c,
                output.data(), plan.descriptors->d,
                &plan.algorithm, runtime_->state()->cublas_workspace,
                plan.workspace_bytes, runtime_->stream()));
        }
    }
    if (bias != nullptr) add_last_dim_bias_in_place(output, *bias);
    return output;
}

Tensor Ops::linear_activation(const Tensor& input, const Tensor& weight,
                              const Tensor& bias, LinearActivation activation) const {
    if (activation == LinearActivation::None) return linear(input, weight, &bias);
    Tensor projected = linear(input, weight, nullptr);
    if (activation == LinearActivation::GEGLU) {
        return bias_geglu(projected, bias);
    }
    bias_activation_in_place(projected, bias, activation);
    return projected;
}

Tensor Ops::convolution_nchw(const Tensor& input,
                             const Tensor& weight,
                             const Tensor* bias,
                             int stride_y,
                             int stride_x,
                             int pad_y,
                             int pad_x) const {
    if (runtime_ == nullptr) throw CudaError("CUDA Ops has no runtime");
    auto profile = profile_scope(convolution_profile_label(input, weight, stride_y, stride_x));
    if ((input.type() != ScalarType::Float16 && input.type() != ScalarType::Float32) ||
        weight.type() != input.type() || weight.role() != input.role() ||
        input.rank() != 4 || weight.rank() != 4) {
        throw CudaError("CUDA convolution expects matching FP16/FP32 NCHW tensors and roles");
    }
    if (input.size(1) != weight.size(1)) throw CudaError("CUDA convolution channel mismatch");
    if (bias != nullptr && (bias->type() != input.type() || bias->role() != input.role() ||
                            bias->rank() != 1 || bias->size(0) != weight.size(0))) {
        throw CudaError("CUDA convolution bias mismatch");
    }

    const int n = checked_int(input.size(0), "batch");
    const int c = checked_int(input.size(1), "input channels");
    const int h = checked_int(input.size(2), "input height");
    const int w = checked_int(input.size(3), "input width");
    const int k = checked_int(weight.size(0), "output channels");
    const int r = checked_int(weight.size(2), "kernel height");
    const int s = checked_int(weight.size(3), "kernel width");
    const cudnnDataType_t data_type = cudnn_data_type(input.type());
    const bool model_fp16_accum = input.type() == ScalarType::Float16 &&
        input.role() == TensorRole::Model &&
        runtime_->options().non_vae_accumulation == NonVAEAccumulation::Float16;
    const cudnnDataType_t compute_type = model_fp16_accum ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT;
    const int variant = static_cast<int>(input.type()) * 10 + (model_fp16_accum ? 1 : 2);
    const ConvolutionKey convolution_key{
        n, c, h, w, k, r, s, stride_y, stride_x, pad_y, pad_x, variant};

    ConvolutionPlan convolution_plan{};
    {
        std::lock_guard lock(runtime_->state()->cudnn_mutex);
        const auto cached = runtime_->state()->convolution_plans.find(convolution_key);
        if (cached != runtime_->state()->convolution_plans.end()) {
            convolution_plan = cached->second;
        } else {
            auto descriptors = std::make_shared<ConvolutionDescriptors>();
            SDXL_CUDNN_CHECK(cudnnCreateTensorDescriptor(&descriptors->input));
            SDXL_CUDNN_CHECK(cudnnCreateTensorDescriptor(&descriptors->output));
            SDXL_CUDNN_CHECK(cudnnCreateTensorDescriptor(&descriptors->bias));
            SDXL_CUDNN_CHECK(cudnnCreateFilterDescriptor(&descriptors->filter));
            SDXL_CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&descriptors->convolution));

            SDXL_CUDNN_CHECK(cudnnSetTensor4dDescriptor(
                descriptors->input, CUDNN_TENSOR_NCHW, data_type, n, c, h, w));
            SDXL_CUDNN_CHECK(cudnnSetFilter4dDescriptor(
                descriptors->filter, data_type, CUDNN_TENSOR_NCHW, k, c, r, s));
            SDXL_CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
                descriptors->convolution, pad_y, pad_x, stride_y, stride_x, 1, 1,
                CUDNN_CROSS_CORRELATION, compute_type));
            const cudnnMathType_t math_type = input.role() == TensorRole::VAE &&
                                              runtime_->options().allow_tf32
                ? CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION : CUDNN_TENSOR_OP_MATH;
            SDXL_CUDNN_CHECK(cudnnSetConvolutionMathType(
                descriptors->convolution, math_type));

            SDXL_CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(
                descriptors->convolution, descriptors->input, descriptors->filter,
                &convolution_plan.output_n, &convolution_plan.output_c,
                &convolution_plan.output_h, &convolution_plan.output_w));
            SDXL_CUDNN_CHECK(cudnnSetTensor4dDescriptor(
                descriptors->output, CUDNN_TENSOR_NCHW, data_type,
                convolution_plan.output_n, convolution_plan.output_c,
                convolution_plan.output_h, convolution_plan.output_w));
            SDXL_CUDNN_CHECK(cudnnSetTensor4dDescriptor(
                descriptors->bias, CUDNN_TENSOR_NCHW, data_type,
                1, convolution_plan.output_c, 1, 1));

            std::array<cudnnConvolutionFwdAlgoPerf_t,
                       CUDNN_CONVOLUTION_FWD_ALGO_COUNT> algorithms{};
            int returned = 0;
            SDXL_CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
                runtime_->cudnn(), descriptors->input, descriptors->filter,
                descriptors->convolution, descriptors->output,
                static_cast<int>(algorithms.size()), &returned, algorithms.data()));
            bool found = false;
            for (int index = 0; index < returned; ++index) {
                const auto& algorithm = algorithms[static_cast<std::size_t>(index)];
                if (algorithm.status != CUDNN_STATUS_SUCCESS) continue;
                if (runtime_->options().deterministic &&
                    algorithm.determinism != CUDNN_DETERMINISTIC) continue;
                if (algorithm.memory > runtime_->options().cudnn_workspace_limit_bytes) continue;
                convolution_plan.algorithm = algorithm.algo;
                convolution_plan.workspace_bytes = algorithm.memory;
                found = true;
                break;
            }
            if (!found) {
                throw CudaError("cuDNN found no convolution algorithm within the workspace limit");
            }
            convolution_plan.descriptors = std::move(descriptors);
            runtime_->state()->convolution_plans.emplace(convolution_key, convolution_plan);
        }
    }

    if (!convolution_plan.descriptors) {
        throw CudaError("cached cuDNN convolution plan has no descriptors");
    }
    Tensor output = Tensor::allocate(*runtime_,
        {static_cast<std::size_t>(convolution_plan.output_n),
         static_cast<std::size_t>(convolution_plan.output_c),
         static_cast<std::size_t>(convolution_plan.output_h),
         static_cast<std::size_t>(convolution_plan.output_w)},
        input.type(), input.role());

    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    {
        std::lock_guard lock(runtime_->state()->cudnn_mutex);
        void* workspace = runtime_->state()->ensure_cudnn_workspace(
            convolution_plan.workspace_bytes);
        const auto& descriptors = *convolution_plan.descriptors;
        SDXL_CUDNN_CHECK(cudnnConvolutionForward(
            runtime_->cudnn(), &alpha,
            descriptors.input, input.data(), descriptors.filter, weight.data(),
            descriptors.convolution, convolution_plan.algorithm,
            workspace, convolution_plan.workspace_bytes,
            &beta, descriptors.output, output.data()));
        if (bias != nullptr) {
            SDXL_CUDNN_CHECK(cudnnAddTensor(
                runtime_->cudnn(), &alpha, descriptors.bias, bias->data(),
                &alpha, descriptors.output, output.data()));
        }
    }
    return output;
}

} // namespace sdxl::cuda
