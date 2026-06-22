# FlashAttention and SDXL CUDA-kernel audit

## Baseline that motivated this pass

The measured RTX 3060 SM86 run was:

```text
1024x1024, batch 1, 4 Euler steps, cfg 1.0, fp8-auto, balanced memory
wall time: 83404.8 ms
driver allocations: 1599
exact-cache hits: 5502
slab suballocations: 2034
```

The run proved the full model path worked, but it also exposed three high-impact issues:

1. `cfg=1.0` still executed the unconditional CLIP and UNet branch, doubling the dominant model work.
2. the Ampere FP8 WMMA kernel reread one activation tile independently in four warps;
3. cached FP8 matrices were stored in logical `[N,K]` order even though the Ampere kernel consumes
   adjacent `[K,N]` tiles, producing strided byte loads.

All three are corrected in this tree.

## Upstream Dao-AILab FlashAttention result

Repository audited:

```text
https://github.com/Dao-AILab/flash-attention
```

Upstream FlashAttention-2 has highly optimized SM80 forward kernels, including an FP16
head-dimension-64 specialization suitable for the principal SDXL UNet attention width. It is not,
however, a direct raw-C++ Windows library:

- its public extension entry point is built around PyTorch, ATen, C10, and Torch tensor objects;
- the standard install and test path is Linux-first;
- its generated CUDA source depends on its CUTLASS-based internal headers and build system;
- directly copying the Python extension would reintroduce exactly the LibTorch/Python dependency
  this project intentionally removed.

The project therefore does not vendor the PyTorch binding. It uses two production-compatible paths:

1. NVIDIA cuDNN Frontend SDPA, whose implementation uses the FlashAttention-2 algorithm and has an
   official C++ graph API for Ampere;
2. a raw-CUDA in-tree SM80 forward kernel specialized for FP16 head dimension 64, used when the
   frontend is unavailable or loses an A/B benchmark.

The original warp-online kernel remains available as the correctness fallback.

## NVIDIA cuDNN SDPA integration

Repository:

```text
https://github.com/NVIDIA/cudnn-frontend
```

`FETCH_CUDNN_FRONTEND.bat` fetches a pinned source commit into `third_party/cudnn-frontend`.
`BUILD_WINDOWS_CUDA13.bat` attempts this automatically unless
`SDXL_SKIP_CUDNN_FRONTEND=1` is set.

The C++ backend:

- presents the engine's contiguous `[B,S,H,D]` storage to cuDNN as logical BHSD with explicit
  strides, so no Q/K/V transpose copies are needed;
- uses FP16 Q/K/V/O and FP32 intermediate/compute data types;
- disables training statistics;
- caches one built graph and workspace size per `(B,Q,K,H,D,causal)` key;
- uses the existing persistent cuDNN workspace;
- is eligible for FP16/BF16 attention on SM80 and newer;
- is selected only for unmasked UNet attention. CLIP's explicit padding mask remains on the
  in-tree path.

## In-tree SM80 FlashAttention-style kernel

`src/cuda/flash_attention_sm80.cu` is a forward-only kernel for contiguous FP16 tensors with
head dimension 64. It is independent of PyTorch and LibTorch.

The kernel:

- processes 32 query rows per CTA;
- streams 16 K/V rows per iteration;
- uses FP16 WMMA Tensor Cores for both `Q*K^T` and `P*V`;
- applies an online numerically stable softmax;
- stores running max/sum and output accumulation without allocating `[B,H,Q,K]`;
- supports causal and key-mask operation;
- specializes key lengths 77, 256, 1024, 4096, and 16384;
- retains the previous one-warp-per-query implementation as a selectable fallback.

This is intentionally benchmark-selectable with `--attention flash-sm80`; it is not described as
universally faster than cuDNN SDPA without target measurements.

## CFG=1 fast path

For the standard diffusion CFG equation,

```text
prediction = unconditional + cfg * (conditional - unconditional)
```

`cfg=1` simplifies exactly to the conditional prediction. The engine now skips:

- negative-prompt tokenization and both negative CLIP passes;
- negative conditioning buffers;
- doubled UNet model input;
- CFG combine kernels.

Use `--force-cfg` only for A/B testing or nonstandard workflows that deliberately require the
unconditional branch at `cfg<=1`.

## Ampere FP8 layout and WMMA improvements

On SM86, FP8 is a weight-storage format feeding FP16 WMMA because the RTX 3060 does not provide
native FP8 Tensor Cores.

The optimized path now:

- packs each rank-2 matrix physically in K-major `[K,N]` order;
- stores the layout identifier in cache format `SDXLF8C3`;
- automatically invalidates older row-major sidecars;
- loads each 16x16 activation tile once per CTA and shares it across four output-channel warps;
- reads adjacent FP8 K-major bytes for each KxN weight tile;
- applies tensor-wide or per-output-channel scales while decoding;
- never materializes a complete FP16 copy of the matrix.

A corrected pack-index regression verifies logical tensor values before and after sidecar reload.

## Runtime selection

```text
--attention auto
```

For large UNet attention shapes, auto selects:

1. cuDNN Frontend SDPA when compiled and eligible;
2. in-tree `flash-sm80` for FP16 head dimension 64;
3. `warp-online` fallback.

The 77-token CLIP path stays on the lower-overhead warp kernel in auto mode. Explicit backends are:

```text
--attention cudnn-sdpa
--attention flash-sm80
--attention warp-online
```

## Required target benchmarks

Run:

```bat
ATTENTION_BENCHMARK.bat
BENCHMARK_FLASH_CFG1.bat D:\models\myModelXL.safetensors
```

The operator benchmark reports latency and numerical error versus the warp-online reference. The
end-to-end benchmark separates the gain from CFG bypass from the gain from each attention backend.
No backend should become the permanent forced default until these target-machine results are
recorded.
