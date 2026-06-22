# FlashAttention and RTX 3060 optimization release

## Baseline addressed

The reported RTX 3060 baseline was:

```text
1024x1024, four Euler steps, cfg=1.0, fp8-auto, balanced mode
Wall: 83404.8 ms
Driver allocations: 1599
Exact-cache hits: 5502
Slab suballocations: 2034
```

This release deliberately separates three independent optimizations so their gains can be measured:

1. `cfg=1.0` conditional-only execution;
2. attention backend replacement;
3. SM86 FP8 weight layout and WMMA data movement.

## Highest-impact change for this exact command

For standard classifier-free guidance:

```text
unconditional + cfg * (conditional - unconditional)
```

`cfg=1` equals the conditional prediction exactly. Unless `--force-cfg` is supplied, the runtime now avoids the negative CLIP branch, doubled UNet batch, and CFG combination. This is expected to be more important for the reported command than PNG or host launch overhead.

## Attention backends

The engine now exposes:

```text
--attention auto
--attention cudnn-sdpa
--attention flash-sm80
--attention warp-online
```

`auto` keeps tiny 77-token CLIP attention on the low-overhead warp kernel. Large eligible UNet attention prefers the optional NVIDIA cuDNN Frontend SDPA graph, then the in-tree FP16 SM80 head-dimension-64 FlashAttention-style kernel, and finally the previous warp-online implementation.

The upstream Dao-AILab extension was audited but is not copied into this raw C++ runtime because its normal API is tied to PyTorch/ATen/C10. The in-tree kernel uses the same tiled/online-softmax principles without that dependency. NVIDIA cuDNN SDPA is included as a second, vendor-maintained FlashAttention-2 implementation for direct A/B measurement.

## SM86 FP8 changes

On RTX 3060, FP8 is compressed weight storage feeding FP16 WMMA rather than native FP8 MMA. The weight-only backend now:

- stores logical `[N,K]` matrices physically as coalesced `[K,N]` bytes;
- shares each 16x16 activation tile across four adjacent output-channel warps;
- decodes only the current FP8 tile into FP16 shared memory;
- supports tensor-wide and per-output-channel scale metadata;
- never expands and retains a complete FP16 UNet matrix.

The sidecar format is now `SDXLF8C3`. Existing sidecars are rejected and rebuilt once.

## Required first test

```bat
BUILD_WINDOWS_CUDA13.bat 86
ATTENTION_BENCHMARK.bat
BENCHMARK_FLASH_CFG1.bat D:\models\myModelXL.safetensors
```

The first post-update generation rebuilds the C3 FP8 cache. Use the second and later runs for steady-state comparisons.

## Acceptance rule

Do not force one attention backend globally until the RTX 3060 benchmark records latency and numerical error for each SDXL bucket. `auto` is conservative, but the supplied CSV and traces are the source of truth.
