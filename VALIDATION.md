# Validation record

## Environment boundary

This environment does not provide NVIDIA `nvcc`, CUDA/cuDNN runtime libraries, or an NVIDIA GPU.
Validation therefore separates current-source deterministic execution and strict host/structural
checks from target-machine CUDA compilation and benchmarking.

The user previously confirmed that the pre-audit project completed a full RTX 3060 run and wrote a
1024x1024 PNG. The new attention, CFG-bypass, and K-major FP8 changes have not been performance-run
on that GPU in this environment.

## Current-source deterministic rebuild

A fresh CPU-reference CMake build was made from this exact source snapshot with C++20 and warning
enabled targets. All four tests passed:

- external CLIP tokenizer fixture;
- embedded standard SDXL tokenizer fixture;
- dual CLIP text-encoder fixture;
- 2,641-parameter SDXL architecture, UNet, Euler, DDIM, CFG, and denoising fixture.

Result:

```text
100% tests passed, 0 tests failed out of 4
```

## Strict CUDA-facing host checks

The following current translation units passed C++20 syntax checking with `-Wall -Wextra
-Wpedantic -Werror` against CUDA/cuBLASLt/cuDNN API stubs:

- `src/cuda/profiler.cpp`;
- `src/cuda/async_image_writer.cpp`;
- `src/cuda/engine.cpp`;
- `src/cuda/text_encoder.cpp`;
- `src/cuda/unet.cpp`;
- `src/cuda/denoiser.cpp`;
- `src/cuda/denoise_graph.cpp`;
- `src/cuda/vae.cpp`;
- `src/cuda/cudnn_sdpa.cpp` with the optional frontend macro disabled;
- `src/cuda_main.cpp`;
- `src/server_main.cpp`.

`runtime.cpp` and the complete cuBLASLt/cuDNN implementation require broader real SDK declarations
than the minimal stub set and are target-build items.

## CUDA structural checks

The current new device paths were transformed into parseable C++ structural fixtures and passed
strict parsing:

- SM80 WMMA FlashAttention-style QK/PV kernel;
- SM86 E4M3/E5M2 weight-only WMMA kernel with shared activation tiles;
- K-major FP8 checkpoint-to-device packing kernel.

These checks catch malformed templates, indexing declarations, and host/device interface mistakes.
They do not replace NVCC device compilation or execution.

## Regression coverage added

`tests/cuda/cuda_ops_test.cu` now covers:

- stable two-pass LayerNorm with large-offset FP16 values;
- fused GroupNorm+SiLU and fused GEMM activations;
- an explicit `FlashSM80` runtime compared numerically with `WarpOnline`;
- generic FP32 VAE attention;
- CFG and fused Euler/DDIM scheduler updates;
- conditional-only no-CFG denoising at guidance 1;
- persistent slab allocation and coalescing;
- FP8 E4M3/E5M2 execution;
- backend-specific FP8 physical layout;
- nonzero tensor-wide K-major logical-value verification;
- sidecar unload/reload logical-value verification;
- E5M2 per-output-channel scaling;
- batch-two CUDA Graph seed/input updates;
- VAE decode, RGB conversion, and PNG creation.

The corrected K-major pack calculation deliberately derives `output_channel` from the logical matrix
column count, not from scale-reduction grouping. This prevents tensor-wide scaling from collapsing
all writes into row zero.

## Optional cuDNN Frontend boundary

The source follows NVIDIA's official C++ SDPA graph pattern and compiles when
`SDXL_HAS_CUDNN_FRONTEND_SDPA` is absent. The actual pinned frontend headers could not be downloaded
or compiled in this network-restricted container. `FETCH_CUDNN_FRONTEND.bat` pins the exact commit
and the Windows build must validate the macro-enabled path.

## Target-machine commands

```bat
BUILD_WINDOWS_CUDA13.bat 86
ctest --test-dir build-cuda13 -C Release --output-on-failure
ATTENTION_BENCHMARK.bat
BENCHMARK_FLASH_CFG1.bat D:\StabilityMatrix\Models\StableDiffusion\creapromptLightning_creapromtHypersdxlV1.safetensors
```

The first run after this update should report an FP8 cache miss/rebuild because cache C3 uses the new
K-major SM86 byte layout. Later runs should report a hit.

## Required real-GPU acceptance checks

1. Confirm CMake prints `cuDNN Frontend SDPA enabled` after the automatic pinned fetch.
2. Run `sdxl_cuda_ops_test`; verify the explicit flash-versus-warp and K-major cache regressions pass.
3. Run `ATTENTION_BENCHMARK.bat`; inspect both latency and max/RMS error for each shape.
4. Run the supplied cfg=1 end-to-end matrix and compare forced CFG against bypass independently from
   attention backend changes.
5. Confirm `--attention auto` records `ops/attention/cudnn-sdpa/...` or
   `ops/attention/flash-sm80/...` for large UNet buckets.
6. Confirm a standard `cfg=1.0` request reports `cfg_bypassed` and uses batch one in UNet traces.
7. Confirm the second process run reports an FP8 cache C3 hit.
8. Confirm temporary driver-allocation deltas approach zero after warmup in persistent jobs/server
   mode.
9. Compare output images and latent statistics against `--attention warp-online --force-cfg` before
   making a measured backend permanent for a shape bucket.

No exact speedup is claimed until these tests run on the RTX 3060. The included benchmarks are
specifically designed to isolate the gain from CFG bypass, attention replacement, and FP8 layout.
