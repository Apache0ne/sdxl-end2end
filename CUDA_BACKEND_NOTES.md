# CUDA backend implementation notes

## Build compatibility

The project supports the installed CUDA 13.3/cuDNN 9.23 layout:

```text
C:\Program Files\NVIDIA\CUDNN\v9.23\include\13.3
C:\Program Files\NVIDIA\CUDNN\v9.23\lib\13.3\x64
```

The cuDNN include directory is public because the public runtime header includes `cudnn.h`. The
static CUDA library enables separable compilation and device-symbol resolution, preventing final
executable failures involving `__cudaRegisterLinkedBinary_*`.

`BUILD_WINDOWS_CUDA13.bat` builds into `build-cuda13` and attempts to fetch the pinned NVIDIA
cuDNN Frontend headers. Set `SDXL_SKIP_CUDNN_FRONTEND=1` to omit that optional backend.

## Precision policy

Persistent storage remains intentionally mixed precision:

- CLIP-L and OpenCLIP-bigG: FP16 weights/activations;
- eligible UNet rank-2 weights: E4M3 or E5M2 FP8;
- UNet convolution, normalization, and bias tensors: FP16;
- UNet activations, residuals, latents, CFG, and scheduler samples: FP16;
- standard SDXL VAE: FP32 because it is force-upcast;
- FP8 scales: small FP32 metadata;
- final image: RGB8.

FP32 register/shared-memory accumulation in normalization, attention, scheduler, cuBLASLt, cuDNN,
and SDPA is permitted for stability. It is not persistent FP32 model storage.

## Stable normalization

LayerNorm and GroupNorm use centered two-pass variance:

1. reduce the mean;
2. reduce `(x - mean)^2`;
3. normalize with `rsqrt(variance + epsilon)`.

This permanently replaces the cancellation-prone `E[x^2] - E[x]^2` implementation that caused the
late-UNet NaN.

Normal Release builds do not scan every tensor for NaN/Inf. Diagnostic scans require a
`debug-finite` build and `--finite-checks`.

## Attention dispatch

The runtime exposes:

```text
auto
cudnn-sdpa
flash-sm80
warp-online
```

### cuDNN Frontend SDPA

When `cudnn_frontend.h` is available, the engine caches one FP16 SDPA graph per
`(B,Q,K,H,D,causal)` shape. The graph uses FP32 intermediate/compute precision and the existing
persistent cuDNN workspace. Explicit BSHD strides avoid Q/K/V layout-copy kernels.

This backend is used only when there is no explicit key mask. CLIP padding-mask attention therefore
remains in-tree.

### In-tree SM80 flash kernel

The FP16 head-dimension-64 forward kernel uses WMMA Tensor Cores for QK and P*V, online softmax,
and 32-query/16-key tiles. It supports causal/key-mask use and never materializes `[B,H,Q,K]`.

### Warp-online fallback

The original one-warp-per-query implementation remains available for correctness comparison,
short sequences, masked unsupported shapes, FP32 VAE attention, and any shape outside the optimized
support surface.

In auto mode, tiny 77-token CLIP attention uses warp-online; large eligible UNet shapes prefer
cuDNN SDPA, then flash-sm80.

## Sampler and scheduler dispatch

Sampler and sigma schedule are independent request properties. `dpmpp_2m` with `normal` is the
default. Euler and DDIM remain selectable, and all samplers can use `normal`, `karras`,
`exponential`, `sgm_uniform`, `simple`, `ddim_uniform`, `ddim_trailing`, `beta`,
`linear_quadratic`, `kl_optimal`, or `gits`. DPM++ 2M stores one previous predicted-original FP16 tensor per request. Sampler and
scheduler are included in the CUDA Graph key.

## CFG=1 conditional-only path

When guidance is at most 1 and `--force-cfg` is absent, the engine does not create or execute the
unconditional branch. Prompt embeddings, time IDs, model input, and UNet output use batch `B`
instead of `2B`. Euler/DDIM update directly from the conditional prediction.

## SM86 FP8 execution

The RTX 3060 uses weight-only FP8 storage feeding FP16 WMMA. Cache format C3 stores matrices
physically as K-major `[K,N]` while retaining logical `[N,K]` shapes. Four output-channel warps share
one activation tile, and each warp reads adjacent K-major FP8 bytes for its KxN tile.

Native SM89+ FP8 remains on cuBLASLt and uses ordinary logical row-major storage expected by that
backend.

## Persistent execution and allocation

`SDXLEngine` owns checkpoint mappings, tokenizers, CUDA handles, resident weights, cuBLASLt plans,
cuDNN convolution/SDPA plans, workspaces, the temporary slab, and optional CUDA Graphs.

- low: sequential component staging;
- balanced: resident FP8 UNet;
- high: resident CLIP+UNet+VAE;
- server/jobs: repeated requests in one process.

Temporary tensors use a coalescing slab and exact-size fallback cache. Persistent model weights use
a separate allocator and separate counters. cuDNN uses one grow-only persistent workspace.

## Profiling

CUDA event pairs are inserted asynchronously and resolved at request completion. Instrumented scopes
include checkpoint/tokenizer work, uploads, each denoise step, every down/mid/up and transformer
block, attention backend/shape, GEMM, convolution, VAE, RGB transfer, and actual image writing.

## Plan caches

cuBLASLt operation descriptors, layouts, preferences, selected algorithms, and workspaces are cached
by full shape/precision key. cuDNN convolution descriptors/algorithms and cuDNN SDPA graphs are also
cached by shape and policy.

## No CPU fallback

The normal executables link the CUDA executor only. CUDA, cuBLASLt, cuDNN, SDPA, cache, or shape
failures are fatal; neural operations are never redirected to the scalar CPU reference path.

## Ancestral and SDE sampler controls

The CUDA denoiser implements DPM++ SDE, Euler ancestral, and DPM++ 2S ancestral
CFG++ with FP16 device-resident state updates. `--noise-device gpu` uses the
runtime CUDA normal generator; `--noise-device cpu` generates the requested
noise on the host and uploads only the noise tensor. This is not a neural CPU
fallback. DPM++ 2S ancestral CFG++ forces the negative/unconditional branch even
when CFG is 1 because the CFG++ correction requires it.


## ComfyUI-compatible sampler state

The UNet remains an FP16/FP8 CUDA model, but sampler state is a separate tensor role. `TensorRole::SamplerState` may be FP32 and is never accepted as a persistent model weight. `Ops::euler_scale_input` divides FP32 state by `sqrt(1+sigma^2)` and writes an FP16 `TensorRole::Model` tensor for the UNet. Model output is converted back into FP32 predicted-clean/state updates. This keeps cuDNN/cuBLASLt execution unchanged while matching ComfyUI KSampler numerical behavior.

CPU noise is generated in host Float32 and uploaded. GPU mode uses CUDA Philox. The full-denoise parity scale is `sqrt(1+sigma_max^2)`; `--initial-noise-scaling sigma` retains the previous behavior for A/B testing. Stochastic CPU/Brownian parity execution is eager rather than CUDA Graph captured.
