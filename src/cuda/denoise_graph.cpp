#include "sdxl/cuda/denoise_graph.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace sdxl::cuda {
namespace {

[[nodiscard]] Tensor clone_tensor(const Runtime& runtime, const Tensor& source) {
    Tensor clone = Tensor::allocate(
        runtime, source.shape(), source.type(), source.role());
    clone.copy_from(runtime, source);
    return clone;
}

[[nodiscard]] std::vector<float> graph_timesteps(const DenoiseOptions& options) {
    SigmaSchedule schedule(options.scheduler_config);
    schedule.set_timesteps(options.inference_steps, options.scheduler);
    return schedule.timesteps();
}

[[nodiscard]] float graph_initial_noise_scale(const DenoiseOptions& options) {
    if (options.sampler == SamplerKind::DDIM) return 1.0F;
    SigmaSchedule schedule(options.scheduler_config);
    schedule.set_timesteps(options.inference_steps, options.scheduler);
    const float sigma = schedule.initial_noise_sigma();
    const float model_sigma_max = schedule.training_sigmas().back();
    const float scale = std::max({1.0F, std::abs(sigma), std::abs(model_sigma_max)});
    const bool max_denoise = sigma > model_sigma_max ||
        std::abs(sigma - model_sigma_max) <= 1.0e-5F * scale;
    if (max_denoise &&
        options.sampler_config.initial_noise_scaling ==
            InitialNoiseScaling::ComfyUIMaxDenoise) {
        // ComfyUI ModelSamplingDiscrete::noise_scaling(..., max_denoise=true).
        return std::sqrt(1.0F + sigma * sigma);
    }
    return sigma;
}

[[nodiscard]] ScalarType graph_state_type(const DenoiseOptions& options) {
    return options.sampler_config.state_precision == SamplerStatePrecision::Float32
        ? ScalarType::Float32 : ScalarType::Float16;
}

} // namespace

DenoiseGraph::DenoiseGraph(const Runtime& runtime,
                           const SDXLModel& model,
                           const WeightStore& weights,
                           const ClassifierFreeConditioning& conditioning,
                           DenoiseOptions options,
                           UNetOptions unet_options)
    : runtime_(&runtime),
      options_(std::move(options)),
      timesteps_(graph_timesteps(options_)),
      initial_noise_scale_(graph_initial_noise_scale(options_)) {
    if (conditioning.batch_size == 0) {
        throw CudaError("CUDA Graph conditioning batch cannot be zero");
    }
    if (options_.profile_steps) {
        throw CudaError("per-step event profiling cannot be captured into a reusable CUDA Graph");
    }
    if (options_.sampler_config.noise_device == NoiseDevice::CPU) {
        throw CudaError(
            "CUDA Graph replay does not support CPU KSampler noise; use --noise-device gpu or disable --cuda-graph");
    }
    const bool stochastic_sampler =
        (options_.sampler == SamplerKind::DDIM && options_.ddim_eta != 0.0F) ||
        (options_.sampler == SamplerKind::DPMppSDE &&
         options_.sampler_config.eta > 0.0F && options_.sampler_config.s_noise > 0.0F) ||
        (options_.sampler == SamplerKind::EulerAncestral &&
         options_.sampler_config.eta > 0.0F && options_.sampler_config.s_noise > 0.0F) ||
        (options_.sampler == SamplerKind::DPMpp2SAncestralCFGpp &&
         options_.sampler_config.eta > 0.0F && options_.sampler_config.s_noise > 0.0F) ||
        (options_.sampler == SamplerKind::Euler && options_.sampler_config.s_churn > 0.0F);
    if (stochastic_sampler) {
        throw CudaError(
            "reusable CUDA Graphs require a deterministic sampler because stochastic noise must change with each seed");
    }
    if (unet_options.check_finite_output) {
        throw CudaError("finite-check tracing cannot be captured into a reusable CUDA Graph");
    }

    fixed_conditioning_.batch_size = conditioning.batch_size;
    fixed_conditioning_.classifier_free = conditioning.classifier_free;
    fixed_conditioning_.prompt_embeds = clone_tensor(runtime, conditioning.prompt_embeds);
    fixed_conditioning_.pooled_text_embeds = clone_tensor(
        runtime, conditioning.pooled_text_embeds);

    Denoiser denoiser(runtime, model, weights, unet_options);
    initial_latents_ = Tensor::allocate(
        runtime,
        {conditioning.batch_size, 4, options_.height / 8, options_.width / 8},
        graph_state_type(options_), TensorRole::SamplerState);
    Ops ops(runtime);
    ops.random_normal_into(initial_latents_, options_.seed, initial_noise_scale_);
    time_ids_ = denoiser.create_time_ids(
        conditioning.batch_size, options_, conditioning.classifier_free);
    output_latents_ = Tensor::allocate(
        runtime, initial_latents_.shape(), graph_state_type(options_), TensorRole::SamplerState);
    runtime.synchronize();

    runtime.begin_graph_allocation_capture();
    bool stream_capture_started = false;
    try {
        SDXL_CUDA_CHECK(cudaStreamBeginCapture(
            runtime.stream(), cudaStreamCaptureModeThreadLocal));
        stream_capture_started = true;
        denoiser.denoise_into(
            fixed_conditioning_, initial_latents_, time_ids_, output_latents_, options_);
        SDXL_CUDA_CHECK(cudaStreamEndCapture(runtime.stream(), &graph_));
        stream_capture_started = false;
        runtime.end_graph_allocation_capture();
        if (graph_ == nullptr) throw CudaError("CUDA Graph capture returned a null graph");
        SDXL_CUDA_CHECK(cudaGraphInstantiate(
            &executable_, graph_, nullptr, nullptr, 0));
    } catch (...) {
        if (stream_capture_started) {
            cudaGraph_t aborted = nullptr;
            cudaStreamEndCapture(runtime.stream(), &aborted);
            if (aborted != nullptr) cudaGraphDestroy(aborted);
        }
        runtime.end_graph_allocation_capture();
        if (executable_ != nullptr) cudaGraphExecDestroy(executable_);
        if (graph_ != nullptr) cudaGraphDestroy(graph_);
        executable_ = nullptr;
        graph_ = nullptr;
        throw;
    }
}

DenoiseGraph::~DenoiseGraph() {
    if (runtime_ != nullptr) cudaStreamSynchronize(runtime_->stream());
    if (executable_ != nullptr) cudaGraphExecDestroy(executable_);
    if (graph_ != nullptr) cudaGraphDestroy(graph_);
}

DenoiseResult DenoiseGraph::launch(
    const ClassifierFreeConditioning& conditioning,
    std::uint64_t seed) {
    return launch(conditioning, std::span<const std::uint64_t>(&seed, 1));
}

DenoiseResult DenoiseGraph::launch(
    const ClassifierFreeConditioning& conditioning,
    std::span<const std::uint64_t> seeds) {
    if (runtime_ == nullptr || executable_ == nullptr) {
        throw CudaError("CUDA denoising graph is not initialized");
    }
    if (conditioning.batch_size != fixed_conditioning_.batch_size ||
        conditioning.classifier_free != fixed_conditioning_.classifier_free ||
        conditioning.prompt_embeds.shape() != fixed_conditioning_.prompt_embeds.shape() ||
        conditioning.pooled_text_embeds.shape() !=
            fixed_conditioning_.pooled_text_embeds.shape()) {
        throw CudaError("CUDA Graph conditioning shape changed; rebuild the graph for the new batch/shape");
    }
    if (seeds.empty() || (seeds.size() != 1 && seeds.size() != conditioning.batch_size)) {
        throw CudaError("CUDA Graph seed count must be one or match the conditioning batch");
    }
    fixed_conditioning_.prompt_embeds.copy_from(
        *runtime_, conditioning.prompt_embeds);
    fixed_conditioning_.pooled_text_embeds.copy_from(
        *runtime_, conditioning.pooled_text_embeds);
    Ops ops(*runtime_);
    if (seeds.size() == conditioning.batch_size) {
        ops.random_normal_batch_into(initial_latents_, seeds, initial_noise_scale_);
    } else {
        ops.random_normal_into(initial_latents_, seeds.front(), initial_noise_scale_);
    }
    SDXL_CUDA_CHECK(cudaGraphLaunch(executable_, runtime_->stream()));
    return DenoiseResult{output_latents_, timesteps_, {}};
}

} // namespace sdxl::cuda
