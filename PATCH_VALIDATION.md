# Patch validation: prequantized INT8 loader and Tensor Core enforcement

Validation date: 2026-06-23

## Completed in the available environment

A fresh C++20 CPU/reference configuration was built from this patched source
with GCC 14.2.0. All seven tests passed:

```text
1/7 sdxl_tokenizer_test .............. Passed
2/7 sdxl_text_encoder_test ........... Passed
3/7 sdxl_unet_scheduler_test ......... Passed
4/7 sdxl_builtin_tokenizer_test ...... Passed
5/7 sdxl_scheduler_parity_test ....... Passed
6/7 sdxl_int8_metadata_test .......... Passed
7/7 sdxl_int8_key_layout_test ........ Passed

100% tests passed, 0 tests failed out of 7
```

The two repair-specific tests also passed when compiled independently with:

```text
-Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
```

Repair-specific results:

```text
INT8 checkpoint metadata test passed
INT8 reference key layout test passed: 196 CLIP-L + 517 OpenCLIP parameters,
1008 quantized Linear scale tensors
```

## What these tests prove

- the nested OpenCLIP-bigG prefix is selected before suffix fallback;
- the exact 4,660-key inventory contains direct sources for all 713 required
  text-encoder parameters;
- I8 dtypes, per-row scales, ConvRot metadata, and Q/K/V scale slicing survive
  loader binding;
- all previous CPU/reference tests remain passing.

## Required target validation

This environment has no NVIDIA `nvcc`, CUDA 13.3/cuDNN 9.23 libraries, or RTX
3060. The following must therefore be executed on the Windows target:

```bat
BUILD_WINDOWS_CUDA13.bat 86
ctest --test-dir build-cuda13 -C Release --output-on-failure
```

Then run the exact strict prequantized command documented in
`INT8_CONVROT.md`. Acceptance requires the run to finish and print equal INT8
Linear/verified IMMA counts with zero DP4A fallbacks, plan misses, and execution
failures. Strict mode stops at the first unsupported M/N/K shape instead of
silently using DP4A.
