#include "sdxl/cuda/denoise_graph.hpp"

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
    if (options.scheduler == SchedulerKind::EulerDiscrete) {
        EulerDiscreteScheduler scheduler(options.scheduler_config);
        scheduler.set_timesteps(options.inference_steps);
        return scheduler.timesteps();
    }
    if (options.inference_steps == 0 ||
        options.inference_steps > options.scheduler_config.training_timesteps) {
        throw CudaError("invalid DDIM graph step count");
    }
    const std::size_t ratio =
        options.scheduler_config.training_timesteps / options.inference_steps;
    std::vector<float> timesteps(options.inference_steps);
    for (std::size_t index = 0; index < options.inference_steps; ++index) {
        timesteps[index] = static_cast<float>(
            (options.inference_steps - 1 - index) * ratio);
    }
    return timesteps;
}

[[nodiscard]] float graph_initial_noise_scale(const DenoiseOptions& options) {
    if (options.scheduler != SchedulerKind::EulerDiscrete) return 1.0F;
    EulerDiscreteScheduler scheduler(options.scheduler_config);
    scheduler.set_timesteps(options.inference_steps);
    return scheduler.initial_noise_sigma();
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
    if (options_.scheduler == SchedulerKind::DDIM && options_.ddim_eta != 0.0F) {
        throw CudaError("reusable DDIM CUDA Graphs currently require eta=0 so seed-dependent noise remains external");
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
        ScalarType::Float16, TensorRole::Model);
    Ops ops(runtime);
    ops.random_normal_into(initial_latents_, options_.seed, initial_noise_scale_);
    time_ids_ = denoiser.create_time_ids(
        conditioning.batch_size, options_, conditioning.classifier_free);
    output_latents_ = Tensor::allocate(
        runtime, initial_latents_.shape(), ScalarType::Float16, TensorRole::Model);
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
