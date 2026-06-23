# Native SDXL W8A8 INT8 ConvRot runtime

The INT8 path is implemented in C++20 and CUDA. It does not load Python,
PyTorch, Triton, or ComfyUI at runtime.

## Execution

For every eligible rank-2 Linear weight in the SDXL UNet and both text
encoders:

1. floating checkpoints are optionally rotated and quantized once when the
   component is uploaded;
2. prequantized I8 checkpoints are loaded directly with their per-output-row
   FP32 scales and `comfy_quant` metadata;
3. activations are rotated group-wise and dynamically quantized per row;
4. cuBLASLt performs I8 x I8 with INT32 accumulation on Tensor Cores;
5. a native CUDA epilogue applies activation and weight scales and returns
   FP16 or FP32 activations;
6. unsupported cuBLASLt shapes use a native CUDA DP4A fallback without
   expanding INT8 weights.

UNet convolutions, normalization parameters, biases, embeddings, and the VAE
remain in their normal floating-point formats. The standard VAE remains FP32.

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

## CLI

On-the-fly full SDXL INT8 ConvRot:

```powershell
.\build-cuda13\Release\sdxl_cuda_denoise.exe `
  "D:\models\model.safetensors" 1024 1024 4 1.0 1234 `
  "a cinematic photograph of a city at night" "blurry" "int8.png" `
  --sampler dpmpp_sde --scheduler normal --denoise 1 --comfyui-parity `
  --memory balanced --precision int8-convrot
```

Prequantized-only execution:

```text
--precision int8-convrot --int8-prequantized-only --int8-strict
```

Controls:

```text
--precision int8-convrot       ConvRot W8A8 for UNet and both CLIPs
--precision int8-row           row-wise W8A8 without rotation
--int8-group-size N            4, 16, 64, or 256
--int8-strict                  prohibit floating linear fallback
--int8-prequantized-only       require I8 checkpoint weights and scales
--int8-unet-only               leave both text encoders in FP16
```

## Prequantized checkpoint layout

Each quantized Linear uses:

```text
<prefix>.weight        I8 [out_features, in_features]
<prefix>.weight_scale  F32 [out_features, 1] or [out_features]
<prefix>.comfy_quant   U8 JSON bytes
```

The metadata JSON accepts:

```json
{"convrot":true,"convrot_groupsize":256,"per_row":true}
```

Fused OpenCLIP QKV scale vectors receive the same zero-copy Q/K/V row slice as
the corresponding weight.
