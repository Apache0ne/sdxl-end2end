# Native SDXL W8A8 INT8 ConvRot runtime

The INT8 path is implemented in C++20, CUDA, and cuBLASLt. It does not load
Python, PyTorch, Triton, LibTorch, or ComfyUI at runtime.

## What is quantized

Every eligible rank-2 Linear weight in the SDXL UNet and both text encoders can
execute as W8A8:

1. floating checkpoints are optionally rotated and quantized once during
   component upload;
2. prequantized I8 checkpoints are loaded directly with per-output-row FP32
   scales and `comfy_quant` metadata;
3. activations are rotated group-wise and dynamically quantized per row;
4. the matrix product uses I8 inputs and INT32 accumulation;
5. a CUDA epilogue applies activation and weight scales and returns FP16 or
   FP32 activations.

UNet convolutions, normalization parameters, biases, embeddings, and the VAE
remain in their normal floating-point formats. The standard SDXL VAE remains
FP32 because it uses `force_upcast=true`.

## Verified Tensor Core execution policy

The runtime no longer treats every successful cuBLASLt INT8 call as proof of a
Tensor Core path. It requests up to 32 cuBLASLt heuristic candidates and checks
each selected algorithm's numerical implementation flags. A fast-path call is
counted as `verified cuBLASLt IMMA` only when cuBLASLt reports all of:

- integer Tensor Core/IMMA implementation;
- signed 8-bit integer input;
- signed 32-bit integer accumulation.

There are two execution policies:

### Compatibility policy

Profiles without `strict` or `tensorcore` prefer verified cuBLASLt IMMA. If no
verified plan exists, or a selected plan fails, they may use the native CUDA
DP4A kernel. The output log reports every fallback.

### Tensor-Core-only policy

`*-strict`, `*-tensorcore`, or `--int8-tensor-cores-only` require every INT8
Linear call to use a verified cuBLASLt IMMA algorithm. A missing plan or failed
matmul is a fatal error. These modes never silently use DP4A.

Therefore, a completed strict run with:

```text
INT8 Linear calls N: verified cuBLASLt IMMA N, DP4A fallback 0,
IMMA plan misses 0, IMMA execution failures 0
```

proves that every INT8 Linear call in that run used the verified integer Tensor
Core path. It does not claim that convolutions, normalization, embeddings, or
the FP32 VAE are INT8 Linear operations.

## ConvRot

The default group size is 256. The regular normalized H4 Kronecker transform is
applied as:

```text
W_rot = W @ H^T
x_rot = x @ H
```

so the floating result is preserved before quantization while activation and
weight outliers are spread across each group.

Valid group sizes are 4, 16, 64, and 256.

## Precision profiles

```text
int8-convrot
int8-convrot-strict
int8-convrot-tensorcore
int8-convrot-prequantized
int8-convrot-prequantized-strict
int8-convrot-prequantized-tensorcore
int8-row
int8-row-strict
int8-convrot-unet-only
int8-convrot-unet-only-strict
```

`strict` combines two requirements:

- every eligible Linear weight must already satisfy the selected upload policy;
- every executed INT8 Linear must use verified cuBLASLt IMMA, with no DP4A
  fallback.

## Exact prequantized checkpoint command

```powershell
.\build-cuda13\Release\sdxl_cuda_denoise.exe `
  "D:\creapromptHyperSDXL_v1.2_FULL_ConvRot_INT8.safetensors" `
  1024 1024 4 1.0 788897633613589 `
  "a cinematic photograph of a city at night" "blurry" `
  "int8_convrot_prequant.png" `
  --sampler dpmpp_sde --scheduler normal --denoise 1 `
  --comfyui-parity --memory balanced `
  --precision int8-convrot-prequantized-strict
```

A successful run must print `verified-cuBLASLt-IMMA-only` during startup and a
zero DP4A/miss/failure count in the final INT8 statistics.

## On-the-fly conversion command

```powershell
.\build-cuda13\Release\sdxl_cuda_denoise.exe `
  "D:\StabilityMatrix\Models\StableDiffusion\creapromptLightning_creapromtHypersdxlV1.safetensors" `
  1024 1024 4 1.0 788897633613589 `
  "a cinematic photograph of a city at night" "blurry" `
  "int8_convrot_from_normal.png" `
  --sampler dpmpp_sde --scheduler normal --denoise 1 `
  --comfyui-parity --memory balanced --precision int8-convrot
```

This compatibility profile still uses real W8A8 weights/activations, but it may
report DP4A fallbacks. Add `--int8-tensor-cores-only` to make any fallback a
hard error.

## CLI controls

```text
--precision PROFILE
--int8-group-size N            4, 16, 64, or 256
--int8-strict                  prequantization/upload strictness + IMMA-only
--int8-tensor-cores-only       hard-error instead of DP4A fallback
--int8-prequantized-only       require I8 weights plus weight_scale metadata
--int8-unet-only               leave both text encoders in FP16
```

## Prequantized checkpoint layout

Each quantized Linear uses:

```text
<prefix>.weight        I8 [out_features, in_features]
<prefix>.weight_scale  F32 [out_features, 1] or [out_features]
<prefix>.comfy_quant   U8 JSON bytes
```

The metadata JSON accepts these equivalent group-size names:

```json
{"convrot":true,"convrot_groupsize":256,"per_row":true}
```

```json
{"convrot":true,"convrot_group_size":256,"per_row":true}
```

```json
{"convrot":true,"group_size":256,"per_row":true}
```

Fused OpenCLIP QKV scale vectors receive the same zero-copy Q/K/V row slice as
the corresponding weight.

## Supported full-checkpoint text-encoder wrappers

The single-file loader now binds all of these forms explicitly before attempting
suffix lookup:

```text
conditioner.embedders.0.transformer.text_model.*
conditioner.embedders.0.model.transformer.text_model.*
conditioner.embedders.0.model.text_model.*

conditioner.embedders.1.model.transformer.text_model.*
conditioner.embedders.1.model.text_model.*
conditioner.embedders.1.transformer.text_model.*
```

The important converted OpenCLIP-bigG form is:

```text
conditioner.embedders.1.model.transformer.text_model.*
```

Without that exact prefix, the first 196 OpenCLIP parameters collide by suffix
with CLIP-L names and cannot be bound safely. The loader now resolves all 517
OpenCLIP parameters and all 196 CLIP-L parameters directly for the provided
4,660-key reference inventory.
