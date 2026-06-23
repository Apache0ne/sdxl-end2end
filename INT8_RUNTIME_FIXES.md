# INT8 prequantized runtime repair

## Reproduced failure

The converted full checkpoint failed before CUDA execution with:

```text
SDXL checkpoint validation failed: 196 missing, 0 shape mismatches.
First issue: text_encoder_2.text_model.embeddings.token_embedding.weight
```

The checkpoint inventory stores OpenCLIP-bigG under:

```text
conditioner.embedders.1.model.transformer.text_model.*
```

The loader previously tried only:

```text
conditioner.embedders.1.model.*
conditioner.embedders.1.transformer.*
```

Suffix fallback could not safely choose between CLIP-L and OpenCLIP names for
the first 12 transformer layers. That ambiguity covered exactly 196 required
parameters.

## Loader repair

`src/sdxl.cpp` now checks the nested wrapper first:

```text
conditioner.embedders.1.model.transformer.*
```

The equivalent nested CLIP-L wrapper is also accepted. The metadata parser now
accepts `convrot_groupsize`, `convrot_group_size`, or `group_size`.

Regression coverage:

- `tests/int8_metadata_test.cpp` creates deliberately ambiguous dual-encoder
  suffixes and verifies that OpenCLIP binds to the nested encoder-2 source;
- `tests/int8_key_layout_test.cpp` validates the complete provided 4,660-key
  inventory and confirms exact keys for 196 CLIP-L plus 517 OpenCLIP parameters;
- the reference inventory is stored under `reference/` and contains no model
  tensor bytes.

## Tensor Core enforcement repair

The old runtime tried cuBLASLt and then silently used DP4A when no plan was
returned or execution failed. That was valid W8A8 arithmetic, but it did not
prove Tensor Core execution.

The new runtime:

1. requests up to 32 cuBLASLt heuristic algorithms;
2. filters/validates algorithms using cuBLASLt numerical implementation flags;
3. counts a call as fast-path only when it is verified IMMA with I8 input and
   I32 accumulation;
4. exposes per-run counters for INT8 calls, verified IMMA calls, DP4A fallbacks,
   plan misses, and execution failures;
5. makes every strict/tensorcore profile fail instead of falling back.

This is an enforcement guarantee, not a promise that every possible shape has a
supported cuBLASLt IMMA plan. On an unsupported shape, strict mode reports the
exact M/N/K shape and stops. A completed strict run with zero misses/failures
and matching INT8/IMMA counts is the acceptance condition.

## Files changed

- `src/sdxl.cpp`
- `include/sdxl/cuda/weights.hpp`
- `include/sdxl/cuda/runtime.hpp`
- `include/sdxl/cuda/engine.hpp`
- `src/cuda/runtime_internal.hpp`
- `src/cuda/runtime.cpp`
- `src/cuda/engine.cpp`
- `src/cuda/int8_convrot.cu`
- `src/cuda_main.cpp`
- `tests/int8_metadata_test.cpp`
- `tests/int8_key_layout_test.cpp`
- `CMakeLists.txt`
- INT8 documentation and validation records
