#pragma once

#include "sdxl/cuda/denoiser.hpp"

#include <cuda_runtime_api.h>

#include <span>
#include <vector>

namespace sdxl::cuda {

// Reusable fixed-shape denoising graph. Construct only after one normal denoise
// has warmed cuBLASLt/cuDNN plans and persistent workspaces.
class DenoiseGraph final {
public:
    DenoiseGraph(const Runtime& runtime,
                 const SDXLModel& model,
                 const WeightStore& weights,
                 const ClassifierFreeConditioning& conditioning,
                 DenoiseOptions options,
                 UNetOptions unet_options = {});
    ~DenoiseGraph();

    DenoiseGraph(const DenoiseGraph&) = delete;
    DenoiseGraph& operator=(const DenoiseGraph&) = delete;
    DenoiseGraph(DenoiseGraph&&) = delete;
    DenoiseGraph& operator=(DenoiseGraph&&) = delete;

    [[nodiscard]] DenoiseResult launch(
        const ClassifierFreeConditioning& conditioning,
        std::uint64_t seed);
    [[nodiscard]] DenoiseResult launch(
        const ClassifierFreeConditioning& conditioning,
        std::span<const std::uint64_t> seeds);

    [[nodiscard]] const DenoiseOptions& options() const noexcept { return options_; }

private:
    const Runtime* runtime_ = nullptr;
    DenoiseOptions options_;
    ClassifierFreeConditioning fixed_conditioning_;
    Tensor initial_latents_;
    Tensor time_ids_;
    Tensor output_latents_;
    std::vector<float> timesteps_;
    float initial_noise_scale_ = 1.0F;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t executable_ = nullptr;
};

} // namespace sdxl::cuda
