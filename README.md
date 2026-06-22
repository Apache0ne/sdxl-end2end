### Only human input is here : How human runs this project using powershell:

<p align="center">
  <img src="assets/output.png" alt="Output Image" width="500">
</p>

IF you never ran this project before: 1st Agent prompt: ```Audit this project: "Path to the project folder on users system". make a compact list of installs I need to run the project and check what I already have and do not have installed with web links to the needed installs I dont have. ```

IF you already have the needed installs from 1st Agent prompt: 2nd Agent prompt: ```I have the new build of this project: send me the commands for me to build it and run it: "Path to the project folder on users system" ```

ADD this context to help with models understanding of the CLI:
```
cd path to folder of the project 

$env:CUDNN_ROOT="the users CUDNN path"
$env:PATH="The users NVIDIA GPU Computing Toolkit path"

& "The users cmake exe path" -S . -B build-cuda13 -G "Visual Studio 17 2022" -A x64 -T cuda=" NVIDIA GPU Computing Toolkit path of the user" -DSDXL_CUDA_ARCHITECTURES=86 -DSDXL_CUDA_FAST_MATH=ON -DSDXL_BUILD_TESTS=OFF -DSDXL_BUILD_CPU_REFERENCE=OFF

& "The users cmake exe path" --build build-cuda13 --config Release --target sdxl_cuda_denoise --parallel

.\build-cuda13\Release\sdxl_cuda_denoise.exe "The users path to their .safetensors file they are using" 1024 1024 4 1.0 1234 euler "a cinematic photograph of a city at night" "blurry" "output.png" 0 0.0 fp8-auto

```

# Production-style pure C++/CUDA SDXL runtime

This project is a Python-free, LibTorch-free SDXL Base 1.0 inference engine written in C++20 and CUDA. It accepts a complete SDXL `.safetensors` checkpoint or a Diffusers component directory and produces RGB PNG files end to end.

The standard SDXL tokenizers are embedded in the executable. A normal run needs only the model path, prompt, and generation settings.

## Implemented pipeline

- strict SDXL Base 1.0 architecture and shape validation
- memory-mapped SafeTensors loading
- complete single-file/LDM SDXL key conversion
- Diffusers single-file and sharded component loading
- embedded CLIP BPE vocabulary and merge table
- FP16 CLIP-L and OpenCLIP-bigG CUDA execution
- mixed FP8/FP16 SDXL UNet CUDA execution
- Euler Discrete and DDIM denoising
- classifier-free guidance and guidance rescaling
- FP32 force-upcast standard SDXL VAE
- CUDA RGB8 conversion
- dependency-free PNG writer and optional raw RGB output

There is no neural-network CPU fallback in the normal target.

## Production runtime additions

- asynchronous CUDA-event profiler with nested per-block/per-op timings
- Chrome trace JSON export with real event start times
- low, balanced, and high VRAM modes
- persistent `SDXLEngine` API
- preloaded stdin generation server
- packed FP8 sidecar cache with validation and invalidation keys
- coalescing persistent GPU slab plus exact-size fallback cache
- cached cuBLASLt algorithms, descriptors, layouts, and preferences
- cached cuDNN convolution algorithms, descriptors, and workspace
- reusable fixed-shape CUDA Graph denoising
- selectable cuDNN SDPA, in-tree SM80 FlashAttention-style, and warp-online attention backends
- exact CFG=1 conditional-only fast path with optional `--force-cfg` A/B override
- SM86 K-major packed FP8 cache and shared-activation WMMA tiles
- per-image batch seeds during eager and CUDA Graph execution
- background PNG encoding and raw-RGB output
- benchmark CSV and automated benchmark matrix

## Precision policy

| Component | Storage/execution policy |
|---|---|
| CLIP-L | FP16 weights and activations |
| OpenCLIP-bigG | FP16 weights and activations |
| eligible UNet rank-2 weights | E4M3 or E5M2 FP8 |
| UNet convolutions, norms, biases | FP16 |
| UNet activations, residuals, latents, scheduler, CFG | FP16 |
| FP8 scale metadata | small FP32 tensors required by FP8 APIs |
| standard SDXL VAE | FP32 because `force_upcast=true` |
| final image | RGB8 |

FP32 accumulation inside native FP8 cuBLASLt and FP32 reductions inside normalization kernels do not create persistent FP32 copies of CLIP or the UNet.

## FP8 profiles

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

On SM89 and newer, aligned projections can use native cuBLASLt FP8 Tensor Cores. On SM86 RTX 3060, FP8 weights stay compressed and are decoded tile-by-tile into FP16 WMMA shared memory. The complete UNet is never expanded and retained as FP16 merely to execute a projection.

## Accepted model layouts

### Complete single-file model

```text
D:\models\myModelXL.safetensors
```

It must contain CLIP-L, OpenCLIP-bigG, UNet, and VAE weights for an SDXL Base-compatible architecture. Common original prefixes are mapped automatically:

```text
model.diffusion_model.*
conditioner.embedders.0.*
conditioner.embedders.1.*
first_stage_model.*
```

### Diffusers directory

```text
stable-diffusion-xl-base-1.0\
├── text_encoder\
├── text_encoder_2\
├── unet\
└── vae\
```

Each component can contain one SafeTensors file or shards referenced by a `*.safetensors.index.json` file.

## Embedded tokenizers

The executable generates the standard 49,408-entry CLIP vocabulary from the embedded official BPE merge order.

| Encoder | BOS | EOS | padding |
|---|---:|---:|---:|
| CLIP-L | 49406 | 49407 | 49407 |
| OpenCLIP-bigG | 49406 | 49407 | 0 |

An unusual checkpoint with a custom vocabulary can override the embedded assets:

```bat
--tokenizer-dir D:\models\custom_tokenizers
```

## Windows requirements

- Windows 11
- Visual Studio 2022 C++ tools
- CMake 3.28+
- CUDA Toolkit 13.3
- cuDNN 9.23 for CUDA 13.3

The default cuDNN search supports this installation:

```text
C:\Program Files\NVIDIA\CUDNN\v9.23\include\13.3
C:\Program Files\NVIDIA\CUDNN\v9.23\lib\13.3\x64
```

## Build for RTX 3060

```bat
cd D:\path\to\sdxl_raw_cpp_cuda_flashattention
BUILD_WINDOWS_CUDA13.bat 86
```

The script uses `build-cuda13`, attempts to fetch the pinned NVIDIA cuDNN Frontend headers, configures CUDA separable compilation and device-symbol resolution for the static CUDA library, builds all targets, and runs CTest. Set `SDXL_SKIP_CUDNN_FRONTEND=1` to build only the in-tree attention backend.

A diagnostic finite-check build is available when tracking numerical failures:

```bat
BUILD_WINDOWS_CUDA13.bat 86 debug-finite
```

Normal Release builds do not scan tensors for NaN/Inf after every stage.

## One-shot generation

```bat
build-cuda13\Release\sdxl_cuda_denoise.exe ^
  D:\models\myModelXL.safetensors ^
  1024 1024 30 5.0 1234 euler ^
  "a cinematic photograph of a city at night" ^
  "blurry, low quality" ^
  output.png ^
  --memory balanced ^
  --precision fp8-auto ^
  --attention auto ^
  --profile ^
  --profile-json output_trace.json
```

At `cfg <= 1.0`, the engine now runs only the conditional CLIP/UNet branch. Add `--force-cfg` only when deliberately benchmarking the old doubled-batch path.

## Memory modes

```text
--memory low
```

Stages CLIP, UNet, and VAE sequentially. This minimizes residency and preserves the original low-VRAM behavior.

```text
--memory balanced
```

Keeps the FP8 UNet resident across requests. CLIP and VAE are staged as needed. This is the recommended starting mode for an RTX 3060 12 GB.

```text
--memory high
```

Keeps CLIP, UNet, and VAE resident together. Use only when available VRAM is sufficient.

Use `--preload` to upload the components selected by the memory policy before the first job.

## Persistent jobs in one process

`jobs.tsv`:

```text
first prompt<TAB>negative prompt<TAB>first.png<TAB>1234<TAB>1
second prompt<TAB>negative prompt<TAB>second.png<TAB>1235<TAB>1
```

Run:

```bat
build-cuda13\Release\sdxl_cuda_denoise.exe ^
  D:\models\myModelXL.safetensors ^
  1024 1024 30 5.0 1234 euler ^
  --jobs jobs.tsv ^
  --memory balanced ^
  --preload ^
  --cuda-graph ^
  --profile
```

The first matching request warms cuBLASLt/cuDNN plans and captures the complete fixed-shape denoising loop. Later requests with the same shape, batch, scheduler, step count, CFG, rescale, and eta replay the graph. Prompt buffers and per-image seeded latents are updated before replay.

## Warm server

```bat
build-cuda13\Release\sdxl_cuda_server.exe ^
  D:\models\myModelXL.safetensors ^
  --memory balanced ^
  --precision fp8-auto ^
  --arena-reserve-mib 1024
```

The server maps the checkpoint once and preloads the components selected by the memory mode before it reports `ready`. It accepts tab-separated commands over stdin. See `SERVER_PROTOCOL.md`.

## Packed FP8 cache

The first FP8 run can quantize eligible UNet matrices and save a `.sdxlfp8` sidecar. Later processes upload those packed bytes directly. On SM80/SM86, cache version C3 stores rank-2 matrices physically in coalesced K-major order. Older sidecars are rejected and rebuilt once.

The cache identity includes:

- canonical model path
- SafeTensors/JSON file sizes and modification times
- precision profile and scale mode
- SM architecture
- CUDA runtime version
- cuDNN version
- cache format version

Default location:

```text
<model-directory>\.sdxl_cuda_cache\
```

Options:

```text
--fp8-cache-dir PATH
--no-fp8-cache
```

## Persistent GPU arena

```text
--arena-reserve-mib N
--arena-cache-mib N
```

The reserve is one persistent coalescing device slab. Temporary tensors suballocate and return ranges without CUDA driver calls. A fallback exact-size cache handles requests that do not fit the slab. Per-request allocation deltas are written to benchmark CSV.

## CUDA Graphs

```text
--cuda-graph
```

CUDA Graph replay requires balanced or high mode because the UNet must remain resident. The graph key includes every setting that changes the fixed denoising schedule. Timesteps are captured once for that fixed graph; only conditioning and seeded initial latents are updated between requests.

## Profiling

```bat
--profile --profile-json trace.json
```

The profiler records without synchronizing after every scope. All event pairs are resolved together at the end.

Measured categories include:

- checkpoint mapping
- tokenizer host work
- CLIP host upload preparation, GPU upload, and encoding
- UNet cache/read/pack/upload
- every denoise step
- every UNet down, middle, and up block
- transformer and attention blocks
- every linear projection and convolution shape
- normalization shape buckets
- VAE upload, blocks, attention, and decode
- RGB conversion/download
- actual PNG or raw-RGB file write time

Chrome trace output contains real event start times, so nested scopes overlap correctly instead of being serialized artificially.

## Attention backends and FlashAttention audit

`--attention auto` uses the best eligible backend per shape:

1. NVIDIA cuDNN Frontend SDPA for large unmasked UNet attention when the optional headers were present at build time;
2. the in-tree raw-CUDA `flash-sm80` Tensor Core kernel for FP16 head dimension 64;
3. the original `warp-online` kernel as a correctness fallback.

The tiny 77-token CLIP path remains on the lower-overhead warp kernel in auto mode. The standard Dao-AILab FlashAttention Python extension was audited but is not vendored because its public entry point is PyTorch/ATen/C10 based. See `FLASH_ATTENTION_AUDIT.md`.

Fetch/reconfigure explicitly when needed:

```bat
FETCH_CUDNN_FRONTEND.bat
BUILD_WINDOWS_CUDA13.bat 86
```

Benchmark all backends and compare their output error against the reference:

```bat
ATTENTION_BENCHMARK.bat
BENCHMARK_FLASH_CFG1.bat D:\models\myModelXL.safetensors
```

## Fused and specialized paths

- stable two-pass FP32-reduction LayerNorm and GroupNorm
- fixed-width LayerNorm kernels for 320, 640, 768, 1280, and 2048
- GroupNorm + SiLU
- GEMM bias + SiLU/GELU/QuickGELU
- GEMM bias + GEGLU
- time-conditioning add + SiLU
- Euler input scale + CFG batch repeat
- CFG + Euler step
- CFG + DDIM step
- VAE FP16-to-FP32 cast + latent scaling
- cuDNN Frontend FlashAttention-2 SDPA graph cache for eligible SM80+ shapes
- Tensor Core QK and PV in-tree SM80 head-dimension-64 attention
- warp-specialized head-dimension-64 online-softmax fallback
- fixed key-sequence buckets for 77, 256, 1024, 4096, and 16384
- generic memory-efficient FP32 VAE attention

Neither optimized attention path materializes the full Q×K score tensor. The backend remains selectable because cuDNN SDPA and the in-tree kernel can win on different SDXL shape buckets.

## Batch generation

```bat
--batch 2
```

A single prompt can be duplicated into a batch with distinct sequential seeds. Larger batches naturally create larger GEMMs and can improve throughput, but 1024×1024 batches may exceed 12 GB depending on memory mode and workspaces.

## Image output

PNG is encoded on a background thread by default. Use:

```text
--sync-png
```

for synchronous benchmarking, or:

```text
--raw-rgb
```

to write the already converted interleaved RGB8 bytes without PNG compression.

## Benchmark matrix

```bat
BENCHMARK_MATRIX.bat D:\models\myModelXL.safetensors
```

The PowerShell-backed matrix compares:

- cold FP16 and FP8 process runs
- 512 and 1024 resolutions
- Euler and DDIM
- persistent balanced mode first and second image
- CUDA Graph warm replay
- preloaded server first and second image

Results are stored in `benchmark_results\benchmark.csv`, Chrome traces, and a server log.

## Important options

```text
--memory low|balanced|high
--precision PROFILE
--attention auto|cudnn-sdpa|flash-sm80|warp-online
--force-cfg
--preload
--batch N
--cuda-graph
--profile
--profile-json FILE
--benchmark-csv FILE
--fp8-cache-dir PATH
--no-fp8-cache
--arena-reserve-mib N
--arena-cache-mib N
--guidance-rescale F
--ddim-eta F
--raw-rgb
--sync-png
--finite-checks
--tokenizer-dir PATH
--device N
```

## Major source files

| File | Purpose |
|---|---|
| `src/cuda/engine.cpp` | persistent model engine and memory policies |
| `src/server_main.cpp` | preloaded stdin server |
| `src/cuda/profiler.cpp` | asynchronous CUDA event profiler and trace export |
| `src/cuda/runtime.cpp` | CUDA handles, plan caches, persistent GPU arena |
| `src/cuda/weights.cu` | uploads and packed FP8 cache |
| `src/cuda/ops_blas.cpp` | persistent cuBLASLt/cuDNN plans |
| `src/cuda/ops_kernels.cu` | attention dispatch, fused normalization, CFG, scheduler, and RNG kernels |
| `src/cuda/flash_attention_sm80.cu` | raw FP16 Tensor Core FlashAttention-style forward kernel |
| `src/cuda/cudnn_sdpa.cpp` | optional cached NVIDIA cuDNN Frontend SDPA graphs |
| `src/cuda/fp8_kernels.cu` | K-major SM86 FP8-weight WMMA projection kernel |
| `src/cuda/denoise_graph.cpp` | reusable complete-loop CUDA Graph |
| `src/cuda/text_encoder.cpp` | FP16 dual CLIP execution |
| `src/cuda/unet.cpp` | mixed FP8/FP16 SDXL UNet |
| `src/cuda/vae.cpp` | force-upcast FP32 VAE |
| `src/cuda/async_image_writer.cpp` | background PNG/raw image output |
| `BENCHMARK_MATRIX.ps1` | reproducible baseline matrix |
| `BENCHMARK_FLASH_CFG1.ps1` | CFG bypass and attention backend end-to-end A/B matrix |
| `ATTENTION_BENCHMARK.bat` | shape-level attention speed and numerical-error comparison |

## Validation boundary

The source/reference tests and host/CUDA syntax checks can run in the provided development environment. Final `nvcc` linking and GPU execution of the new production paths must be run on the target Windows CUDA 13.3/cuDNN 9.23 machine. See `VALIDATION.md` for the exact completed and pending checks.
