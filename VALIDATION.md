# Validation record

## Environment boundary

This environment does not provide NVIDIA `nvcc`, CUDA/cuDNN runtime libraries, or an NVIDIA GPU. Validation therefore separates deterministic CPU/reference execution and strict CUDA-facing host checks from the required Windows CUDA 13.3/cuDNN 9.23 build and RTX 3060 image comparison.

The native parity changes do not add Python, PyTorch, LibTorch, or a neural CPU fallback. The CPU code is used only for deterministic noise generation and reference tests; CLIP, UNet, sampler tensor operations, VAE, and image conversion remain CUDA/cuDNN in the production executable.

## Fresh CPU-reference build

A fresh C++20 CPU-reference build was configured from this exact source snapshot with warnings treated as errors. All six tests passed:

- `sdxl_tokenizer_test`;
- `sdxl_text_encoder_test`;
- `sdxl_unet_scheduler_test`;
- `sdxl_builtin_tokenizer_test`;
- `sdxl_scheduler_parity_test`;
- `sdxl_int8_metadata_test`.

Result:

```text
100% tests passed, 0 tests failed out of 6
```

## ComfyUI parity regression coverage

`tests/scheduler_parity_test.cpp` checks:

- CPU/FP32/Comfy startup-scaling defaults;
- parsing of CPU/GPU noise, FP16/FP32 state, and startup-scaling options;
- the first PyTorch CPU Float32 `randn` values for a fixed seed;
- exact four-step SDXL normal-scheduler timesteps and sigma references;
- nearest integer log-sigma timestep selection for a DPM++ SDE midpoint;
- ComfyUI partial-denoise schedule slicing;
- DPM++ SDE midpoint/final coefficient construction;
- Brownian interval orientation, variance weights, and overlapping-interval composition.


## Strict CUDA-facing host checks

The current versions of these changed translation units passed C++20 syntax checking with `-Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror` against CUDA/cuDNN API stubs:

- `src/cuda_main.cpp`;
- `src/cuda/denoiser.cpp`;
- `src/cuda/denoise_graph.cpp`;
- `src/cuda/engine.cpp`.

This catches host-interface, type, ownership, and warning regressions. Stub compilation does not validate CUDA device code or real SDK ABI behavior.

## CUDA operator source coverage

`tests/cuda/cuda_ops_test.cu` now includes source-level coverage for:

- FP32 `TensorRole::SamplerState` allocation;
- FP32 state to FP16/model-role scaled input conversion;
- FP32 predicted-original conversion;
- FP32 Euler/DDIM/combine/DPM++ 2M arithmetic;
- FP16 and FP32 random-normal output;
- existing normalization, attention, FP8, CUDA Graph, VAE, RGB, and PNG paths.

The changed CUDA kernels were manually audited for matching declarations, scalar-type dispatch, output roles, and launch signatures. They still require real NVCC compilation and execution.

## DPM++ SDE validation

The DPM++ SDE implementation follows the standard SDXL EPS reduction of the current ComfyUI/k-diffusion equations:

- midpoint lambda is `lambda_s + r * (lambda_t - lambda_s)`;
- midpoint and final ancestral decompositions both start from the original sigma;
- the second-order denoised mixture is `(1 - 1/(2r)) * d1 + 1/(2r) * d2`;
- the terminal zero-sigma transition returns the predicted clean sample;
- midpoint UNet sigma is converted to the nearest training log-sigma timestep.

All midpoint and full-step stochastic queries use one finite Brownian path over the complete known query set. Elementary increments are independent; each interval query is their signed normalized sum. This gives exact interval variance and the correct covariance for overlapping queries. The random realization is not promised to be bit-for-bit identical to `torchsde.BrownianTree`.

## ComfyUI numerical runtime changes

The known-good ComfyUI KSampler run exposed several runtime behaviors that were previously different:

1. initial noise is generated on CPU as Float32 with the PyTorch MT19937/Box-Muller sequence;
2. full-denoise empty-latent startup uses `sqrt(1 + sigma_max^2)`;
3. sampler state and DPM++ history remain Float32 on CUDA;
4. only the scaled UNet input is cast to FP16;
5. `--comfyui-parity` defaults checkpoint execution to FP16 unless `--precision` is explicit;
6. generated sigma calls use nearest integer training timesteps.

These are implemented and covered by CPU/unit or host checks. Final image quality and seed parity require the target GPU build.


## Native INT8 ConvRot validation coverage

`tests/int8_metadata_test.cpp` creates a real SafeTensors checkpoint fragment and verifies that the C++ loader preserves I8 weights, per-output-row FP32 scales, regular ConvRot metadata, and the matching zero-copy Q/K/V scale slices for fused OpenCLIP `in_proj_weight`.

`tests/cuda/cuda_ops_test.cu` covers both native on-the-fly ConvRot quantization and direct prequantized checkpoint execution. The test path rotates weights and activations, performs I8 x I8 accumulation into INT32, applies row scales, and checks the result against a floating identity projection. It also verifies that the prequantized `weight` + `weight_scale` + `comfy_quant` layout produces the same result.

The production kernel uses cuBLASLt INT8 Tensor Core GEMM when a plan is available and a native CUDA `__dp4a` fallback otherwise. It never expands an INT8 weight matrix to FP16 for inference. A real CUDA 13.x build and RTX 3060 run are still required to validate toolkit ABI, kernel launches, performance, and final image quality.

Recommended RTX 3060 smoke tests:

```powershell
# On-the-fly ConvRot of a normal floating SDXL checkpoint
.\build-cuda13\Release\sdxl_cuda_denoise.exe `
  "D:\models\model.safetensors" 1024 1024 4 1.0 1234 `
  "a cinematic photograph of a city at night" "blurry" "int8_otf.png" `
  --sampler dpmpp_sde --scheduler normal --denoise 1 --comfyui-parity `
  --memory balanced --precision int8-convrot --int8-strict

# Direct execution of a converted full SDXL ConvRot checkpoint
.\build-cuda13\Release\sdxl_cuda_denoise.exe `
  "D:\models\model-int8-convrot.safetensors" 1024 1024 4 1.0 1234 `
  "a cinematic photograph of a city at night" "blurry" "int8_prequant.png" `
  --sampler dpmpp_sde --scheduler normal --denoise 1 --comfyui-parity `
  --memory balanced --precision int8-convrot-prequantized-strict
```

## Required Windows/RTX 3060 acceptance sequence

```bat
BUILD_WINDOWS_CUDA13.bat 86
ctest --test-dir build-cuda13 -C Release --output-on-failure
```

Then run the direct CLI path:

```powershell
.\build-cuda13\Release\sdxl_cuda_denoise.exe `
  "D:\models\model.safetensors" 1024 1024 4 1.0 788897633613589 `
  "a cinematic photograph of a city at night" "blurry" "output.png" `
  --sampler dpmpp_sde --scheduler normal --denoise 1 --comfyui-parity `
  --memory balanced --precision fp16 --profile --profile-json comfy_parity_trace.json
```

Acceptance checks:

1. Startup log reports `precision: fp16`, `sampler: dpmpp_sde`, `scheduler: normal`, `noise-device: cpu`, `sampler-state: fp32`, and `initial-noise: comfyui-max-denoise`.
2. The four normal-scheduler timesteps are `999, 666, 333, 0`.
3. The run finishes eagerly; stochastic CPU/Brownian parity mode must not silently reuse a CUDA Graph noise realization.
4. Compare the same CLI settings and seed against ComfyUI for composition, exposure, detail, and latent/image statistics.
5. Run an A/B with `--sampler-state fp16` to isolate solver rounding.
6. Run an A/B with `--initial-noise-scaling sigma` to isolate startup scaling.
7. Run an A/B with `--precision fp8-auto` to isolate model quantization.
8. Run an A/B with `--noise-device gpu` to confirm that a different seeded composition is expected, not a sampler failure.
9. Run `sdxl_cuda_ops_test` to validate all changed FP32 kernels on the real toolkit/device.

No exact image-parity or performance claim is made until those target-machine checks complete.


## Same-dtype tensor-role cast regression

`tests/cuda/cuda_ops_test.cu` now covers the exact ComfyUI parity handoff from an FP32 `SamplerState` tensor to an FP32 `VAE` tensor. `Ops::cast` performs a device-to-device copy when the scalar type is unchanged but the destination role differs, while `Tensor::copy_from` retains its strict shape/type/role validation for ordinary assignment. This path requires real NVCC/CUDA execution on the Windows target.
