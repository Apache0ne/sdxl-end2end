# ComfyUI KSampler parity from the CLI

The project does not import or execute ComfyUI workflow JSON. The matching
KSampler behavior is selected directly with normal command-line arguments.
This keeps the executable Python-free and makes every active setting visible in
the command that produced the image.

A direct equivalent of the tested ComfyUI text-to-image path is:

```powershell
.\build-cuda13\Release\sdxl_cuda_denoise.exe `
  "D:\models\model.safetensors" 1024 1024 4 1.0 788897633613589 `
  "a cinematic photograph of a city at night" "blurry" "output.png" `
  --sampler dpmpp_sde --scheduler normal --denoise 1 `
  --comfyui-parity --memory balanced --precision fp16
```

`--comfyui-parity` selects the numerical policy used by the standard ComfyUI
KSampler path unless an individual setting is explicitly overridden:

- PyTorch-compatible CPU Float32 initial noise;
- FP32 sampler latent and history tensors;
- FP16 casts only at the UNet boundary;
- full-denoise startup scaling `sqrt(1 + sigma_max^2)`;
- nearest integer log-sigma timestep mapping;
- interval-consistent Brownian noise for DPM++ SDE.

The same path can use native ConvRot INT8 model execution without changing the
sampler recipe:

```powershell
.\build-cuda13\Release\sdxl_cuda_denoise.exe `
  "D:\models\model.safetensors" 1024 1024 4 1.0 788897633613589 `
  "a cinematic photograph of a city at night" "blurry" "output_int8.png" `
  --sampler dpmpp_sde --scheduler normal --denoise 1 `
  --comfyui-parity --memory balanced --precision int8-convrot
```

## Full-denoise startup scaling

For an empty latent at the model maximum sigma, ComfyUI uses:

```text
x = noise * sqrt(1 + sigma_max^2)
```

The `comfyui` initial-noise mode implements this behavior. The alternate
`sigma` mode uses `noise * sigma_max`.

## CPU noise

The CPU path implements the contiguous Float32 PyTorch `randn` sequence in C++:
MT19937, PyTorch's 24-bit uniform conversion, and its 16-value Box-Muller
layout. No LibTorch dependency is introduced.

## DPM++ SDE Brownian path

The runtime builds one deterministic Brownian path over all sigma and midpoint
queries for a generation. Overlapping interval queries therefore share the
correct increments instead of receiving unrelated random tensors.

## FP32 sampler-state to VAE handoff

ComfyUI parity finishes denoising in an FP32 `SamplerState` tensor while the
force-upcast VAE expects an FP32 `VAE` tensor. `Ops::cast` performs an explicit
device-to-device same-dtype role conversion, avoiding the stricter
`Tensor::copy_from` role check.

## Boundaries

Exact pixel identity additionally depends on checkpoint bytes, tokenizer data,
CUDA/cuDNN algorithms, quantization mode, and the Brownian RNG realization. At
CFG 1 the unconditional branch is mathematically unnecessary, so the negative
prompt does not affect ordinary CFG at exactly that value.
