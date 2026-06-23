# Raw C++ SDXL architecture, weights, tokenizer, and text encoders

This is a C++20 SDXL Base 1.0 implementation stage with no LibTorch, ATen,
Python, ONNX Runtime, tokenizers library, or third-party JSON library.

## Implemented

### SDXL architecture and weight loading

- CLIP-L text encoder: 12 layers, width 768, MLP width 3072, 12 heads.
- OpenCLIP ViT-bigG text encoder: 32 layers, width 1280, MLP width 5120,
  20 heads, and 1280-wide text projection.
- SDXL UNet parameter graph.
- SDXL AutoencoderKL parameter graph.
- Memory-mapped SafeTensors reader.
- Diffusers component directories and sharded SafeTensors indexes.
- Original single-file SDXL checkpoints.
- Original checkpoint mappings for UNet, VAE, CLIP-L, and OpenCLIP bigG.
- Zero-copy packed OpenCLIP Q/K/V slices and transposed projection views.
- Exact expected-shape validation.

### CLIP tokenization

- Includes built-in standard SDXL CLIP-L and OpenCLIP-bigG tokenizer profiles.
- Compiles one shared official BPE merge table into the executable.
- Generates the exact 49,408-entry CLIP vocabulary from the merge order.
- Preserves CLIP-L padding ID 49407 and OpenCLIP-bigG padding ID 0.
- Can optionally read external `vocab.json`, `merges.txt`, and `tokenizer_config.json` files.
- CLIP byte-to-Unicode encoding.
- Byte-pair encoding and merge-rank cache.
- CLIP token splitting for words, single numeric characters, punctuation runs,
  contractions, and start/end special tokens.
- NFC normalization and invariant Unicode lowercase on Windows through Win32.
- Unicode whitespace collapse.
- BOS/EOS insertion, truncation, EOS-preserving truncation, and max-length
  padding.
- Batched token IDs and attention masks.

### Both SDXL text encoders

- Token and position embeddings.
- Pre-normalized transformer layers.
- Multi-head causal self-attention.
- Q, K, V, and output projections.
- LayerNorm with epsilon `1e-5`.
- QuickGELU for CLIP-L.
- exact GELU for OpenCLIP bigG.
- Residual connections and MLP blocks.
- Final LayerNorm.
- SDXL-compatible EOS pooling by token-ID argmax.
- OpenCLIP 1280-wide text projection.
- Penultimate hidden-state capture.
- Concatenation of CLIP-L 768 and OpenCLIP 1280 hidden states into the SDXL
  `[batch, 77, 2048]` prompt conditioning tensor.
- OpenCLIP projected pooled conditioning `[batch, 1280]`.
- Positive and negative prompt batch helper.
- F16, BF16, F32, F64, E4M3FN, and E5M2 mapped weight reads.
- Multithreaded CPU execution and optional AVX2 dot products.

The text encoder math executes in float32. FP16/BF16 checkpoint values are
converted as they are packed into each CPU matrix block. This avoids converting
all text-encoder weights into a second multi-gigabyte float32 copy.

### Samplers and schedulers

The CPU reference includes DPM++ 2M, Euler, and DDIM. DPM++ 2M with the normal sigma schedule is
the default. The same eleven schedules and `--sampler` / `--scheduler` names as the CUDA executable
are supported, allowing deterministic reference checks for every combination.

## Not implemented yet

- Textual-inversion token injection.
- LoRA application.
- Prompt weighting syntax.
- CUDA text-encoder kernels.
- UNet forward execution.
- VAE decoding and image conversion.

## Build on Windows 11

Use a Visual Studio 2022 x64 Developer Command Prompt:

```bat
cd D:\path\to\sdxl_raw_cpp_arch_loader
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

AVX2 is enabled by default. Disable it for an older CPU with:

```bat
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DSDXL_ENABLE_AVX2=OFF
```

Run the included tests:

```bat
ctest --test-dir build -C Release --output-on-failure
```

## Inspect weights

Diffusers model directory:

```bat
build\Release\sdxl_inspect.exe D:\models\stable-diffusion-xl-base-1.0
```

Single-file SDXL checkpoint:

```bat
build\Release\sdxl_inspect.exe D:\models\sd_xl_base_1.0.safetensors
```

## Execute tokenization and both text encoders

The standard SDXL tokenizers are embedded, so a single-file checkpoint needs no tokenizer folder:

```bat
build\Release\sdxl_text_encode.exe ^
  D:\models\custom_sdxl.safetensors ^
  "a cinematic photograph of a city at night"
```

An unusual checkpoint with a custom vocabulary can override the embedded assets:

```bat
build\Release\sdxl_text_encode.exe ^
  D:\models\custom_sdxl.safetensors ^
  "a cinematic photograph of a city at night" ^
  --tokenizer-dir D:\models\custom_tokenizers
```

This executable is a correctness-oriented CPU path. OpenCLIP bigG is very large, so a complete CPU
forward pass remains computationally expensive.

## C++ API

```cpp
#include "sdxl/sdxl.hpp"
#include "sdxl/text_encoder.hpp"

sdxl::SDXLModel model;
sdxl::SDXLWeightLoader loader;
loader.load(model, R"(D:\models\stable-diffusion-xl-base-1.0)");

sdxl::TextEncoderExecutionOptions options;
options.thread_count = 0; // use hardware_concurrency
options.use_attention_mask = false; // matches the normal SDXL pipeline

sdxl::SDXLTextConditioner conditioner =
    sdxl::SDXLTextConditioner::builtin_sdxl(model, options);

sdxl::SDXLPromptConditioning result = conditioner.encode(
    "a cinematic photograph of a city at night");

// result.prompt_embeds.shape        == [1, 77, 2048]
// result.pooled_prompt_embeds.shape == [1, 1280]
```

Classifier-free prompt inputs can be prepared with:

```cpp
auto pair = conditioner.encode_classifier_free(
    {"a cinematic photograph of a city at night"},
    {"blurry, low quality"});
```

## Main source files

- `include/sdxl/tokenizer.hpp`
- `src/tokenizer.cpp`
- `include/sdxl/text_encoder.hpp`
- `src/text_encoder.cpp`
- `src/text_main.cpp`
- `tests/tokenizer_test.cpp`
- `tests/text_encoder_test.cpp`
