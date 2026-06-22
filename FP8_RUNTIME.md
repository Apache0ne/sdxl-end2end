# FP8 UNet runtime and packed cache

## Scope

Eligible rank-2 UNet matrices use E4M3 or E5M2 FP8 storage. UNet convolutions, normalization
parameters, and biases remain FP16. Both SDXL text encoders remain FP16. The standard SDXL VAE
remains FP32.

## Runtime profiles

```text
fp8-auto
fp8-e4m3
fp8-e5m2
fp8-e4m3-channel
fp8-e5m2-channel
fp8-native-e4m3
fp8-native-e5m2
fp8-weight-e4m3
fp8-weight-e5m2
fp16
```

`fp8-auto` selects checkpoint-native FP8 where compatible, native cuBLASLt FP8 on supported GPUs,
and weight-only execution otherwise.

## Scaling

Tensor-wide quantization uses:

```text
amax = max(abs(weight))
dequant_scale = max(amax / fp8_finite_max, 1e-12)
weight_fp8 = saturating_fp8(weight / dequant_scale)
```

Per-output-channel profiles calculate one scale per logical output row. Scale grouping is kept
separate from physical storage indexing so tensor-wide and per-channel matrices use the same correct
logical row mapping.

## Native FP8

On SM89 and newer, compatible shapes use cuBLASLt FP8 Tensor Core GEMMs. Activations can be
quantized dynamically, inputs use device scale pointers, accumulation is FP32 internally, and output
storage remains FP16.

The FP32 accumulation is part of the native NVIDIA matmul contract; it is not persistent FP32 model
storage.

## Ampere weight-only path

On SM80/SM86, the GPU has no native FP8 Tensor Core MMA. The custom path keeps weights compressed
and decodes each tile into FP16 shared memory before FP16 WMMA.

The optimized SM86 layout is:

```text
logical matrix:  [N,K]
physical bytes:  K-major [K,N]
```

For each CTA:

1. all four warps cooperatively load one 16x16 FP16 activation tile;
2. each warp reads one adjacent 16x16 K-major FP8 weight tile;
3. tensor-wide or output-channel scale is applied while decoding;
4. FP16 WMMA executes with an FP16 accumulator/output policy;
5. no complete FP16 matrix is created.

This removes the old four-times redundant activation load and the old full-row-stride FP8 byte
access.

## Source checkpoint types

Eligible source matrices may be FP16, BF16, FP32, FP64, E4M3FN, or E5M2. Checkpoint-native FP8 can
be copied directly for a compatible native backend or repacked directly into K-major bytes for the
Ampere weight-only backend. No full FP32 or persistent FP16 expansion is required.

## Packed sidecar cache C3

Cache magic:

```text
SDXLF8C3
```

C3 records the FP8 physical storage layout for every matrix. On SM80/SM86, rank-2 matrices must be
K-major. On native FP8 backends, they remain row-major for cuBLASLt.

The cache identity includes:

- canonical model path;
- all relevant model file sizes and modification times;
- FP8 format, scaling mode, and backend;
- GPU SM architecture;
- CUDA runtime and cuDNN versions;
- cache format version and K-major policy.

Each entry validates name, rank, logical shape, FP8 type, scale count/mode, storage layout, and byte
count. Writes use a temporary file followed by rename.

Older C1/C2 sidecars are intentionally rejected and rebuilt once. This is necessary because their
row-major bytes cannot be consumed efficiently by the new SM86 kernel.

## Correctness regression

The CUDA operator test now assigns a nonzero pattern to a rank-2 UNet weight and verifies:

- logical values immediately after tensor-wide K-major packing;
- the declared backend-specific storage layout;
- logical values after sidecar write, unload, and reload;
- E5M2 per-output-channel loading separately.

This specifically prevents the former error where tensor-wide scale grouping collapsed every
K-major write into output row zero.

## Residency

- low may upload packed UNet data per request;
- balanced/high uploads once and keeps the UNet resident;
- jobs/server mode combines process residency with cross-process sidecar reuse.
