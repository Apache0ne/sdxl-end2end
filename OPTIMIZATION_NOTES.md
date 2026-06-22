# Optimization implementation notes

## Baseline and priority

The target RTX 3060 baseline was 83.4 seconds for one 1024x1024, four-step Euler image at
`cfg=1.0`, balanced mode, and `fp8-auto`. This pass concentrates on work that can dominate that
specific run rather than adding generic micro-optimizations blindly.

The highest-impact corrections are:

1. exact CFG=1 bypass, removing the negative CLIP/UNet branch;
2. cached NVIDIA cuDNN Frontend SDPA and an in-tree SM80 Tensor Core attention backend;
3. K-major coalesced SM86 FP8 weight storage;
4. one shared activation tile per four-warps FP8 WMMA CTA;
5. backend and CFG A/B scripts with real CUDA-event timings and numerical comparisons.

## Measurement

Profiling uses CUDA event pairs recorded on the inference stream. Scopes do not synchronize
individually. Event resolution occurs once at request completion, and Chrome trace records use
actual event-relative timestamps.

Host-only timers cover checkpoint mapping, tokenizer work, FP8 cache file processing, and image
encoding. The CLI always records top-level stages; `--profile` additionally enables every block,
GEMM, convolution, normalization, and attention bucket.

The benchmark CSV contains configured attention backend, CFG-bypass state, cuDNN/flash/warp
attention totals, linear/convolution totals, cache status, graph status, and per-request allocator
deltas.

## CFG=1 specialization

For standard CFG, scale 1 simplifies exactly to the conditional prediction. Unless `--force-cfg`
is present, `cfg<=1` now uses:

- one prompt batch rather than negative+positive concatenation;
- one CLIP branch;
- one UNet batch;
- direct Euler/DDIM scheduler update without CFG combine.

This is particularly important for Lightning/Turbo-style low-CFG generation.

## Attention

### `cudnn-sdpa`

The optional NVIDIA cuDNN Frontend backend builds and caches an FP16 SDPA graph for each
`(B,Q,K,H,D,causal)` shape. It exposes the engine's BSHD allocation through explicit BHSD strides,
uses FP32 intermediate/compute precision, generates no training statistics, and reuses the
persistent cuDNN workspace.

### `flash-sm80`

The in-tree raw-CUDA forward kernel targets FP16 head dimension 64 on SM80+:

- 32 query rows per CTA;
- 16 K/V rows streamed per iteration;
- WMMA Tensor Cores for both QK and probability-times-value;
- online stable softmax;
- no complete score tensor;
- fixed 77/256/1024/4096/16384 key buckets.

### `warp-online`

The original one-warp-per-query kernel remains the reference and unsupported-shape fallback.

`--attention auto` uses the lower-overhead warp path for tiny 77-token CLIP attention and prefers
cuDNN SDPA, then flash-sm80, for large eligible UNet buckets. `ATTENTION_BENCHMARK.bat` reports both
latency and error against the reference so a target GPU can choose from measured results.

## FP8 cache and SM86 WMMA

Ampere has no native FP8 Tensor Core MMA, so FP8 is retained as compressed weight storage and each
tile is decoded into FP16 WMMA shared memory.

Cache format C3 stores logical `[N,K]` matrices physically as K-major `[K,N]` on the weight-only
backend. This matches the KxN tile consumed by WMMA and changes strided single-byte reads into
adjacent reads. Old sidecars are rejected automatically.

Four output-channel warps in one CTA now share a single cooperatively loaded 16x16 activation tile.
The previous implementation loaded that same activation tile once per warp. Tensor-wide and
per-output-channel scale indexing are kept separate from physical layout indexing; a regression
checks logical values before and after cache reload.

## Residency

`SDXLEngine` owns one mapped model, one CUDA runtime, one weight store, plan caches, the GPU arena,
and optional denoising graph.

- Low mode unloads CLIP after conditioning, UNet after denoising, and VAE after decode.
- Balanced retains the FP8 UNet.
- High retains CLIP, UNet, and VAE.

The server preloads its selected resident components before reporting ready.

## Allocator

The persistent slab uses best-fit suballocation and adjacent-range coalescing. Balanced CLI mode
reserves 1024 MiB by default; low/high reserve 512 MiB unless overridden. Model weights use a
separate persistent allocation path, so temporary-arena statistics are not polluted by weight
residency.

Requests that do not fit the slab use the exact-size fallback cache before any driver allocation.
Batch seed uploads are arena-backed.

## CUDA Graph

A complete fixed-shape denoising loop is captured after eager plan warmup. cuBLASLt, cuDNN
convolution, and cuDNN SDPA plans therefore exist before capture. Stable conditioning and latent
buffers are updated before replay.

The graph key includes resolution, steps, batch, scheduler, guidance, resolved CFG mode, guidance
rescale, and DDIM eta.

## Linear layers

cuBLASLt descriptors, layouts, preferences, algorithms, and workspace requirements are cached by
matrix shape and precision. Post-GEMM fusions cover bias+SiLU, GELU, QuickGELU, and GEGLU.

The SM86 FP8-weight kernel now combines coalesced K-major tiles with shared activation loads. Native
SM89+ FP8 remains on cuBLASLt.

## Convolutions

cuDNN input/output/bias/filter/convolution descriptors, selected forward algorithms, output shapes,
and workspace sizes are cached by full NCHW/kernel/stride/padding/precision key. One persistent
workspace grows only when a later plan needs more memory.

## Fusions

The hot path includes:

- GroupNorm + SiLU;
- bias + SiLU/GELU/QuickGELU/GEGLU;
- time-conditioning add + SiLU;
- Euler scale + CFG repeat;
- CFG + Euler/DDIM scheduler update;
- VAE cast + latent scaling.

## Remaining target-only work

No universal kernel is guaranteed to win every SDXL bucket. After running the included target
benchmarks, the next work should be selected from measured traces:

- tune or disable cuDNN SDPA per shape if its graph loses to flash-sm80;
- adjust the SM80 query/key tile geometry for the 64x64 self-attention bucket;
- increase/decrease the slab based on allocation deltas and free VRAM;
- benchmark batch two at 512/768 before attempting batch two at 1024;
- use Nsight Compute to identify whether the remaining SM86 FP8 projection bottleneck is global
  bandwidth, FP8 conversion, shared-memory pressure, or WMMA occupancy.
