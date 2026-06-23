# Third-party notices

## CLIP BPE merge data

The embedded BPE merge table is the standard CLIP tokenizer data used by SDXL. It was obtained from
the Stability AI SDXL Base 1.0 tokenizer assets, which use the OpenAI CLIP BPE vocabulary.

- SDXL repository: `stabilityai/stable-diffusion-xl-base-1.0`
- embedded asset: `tokenizer_2/merges.txt`
- embedded byte length: `524619`
- SHA-256: `9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a`
- tokenizer reference: `openai/CLIP/clip/simple_tokenizer.py`

The project generates the standard CLIP `vocab.json` ordering from the byte encoder and BPE merge
order instead of embedding a duplicate vocabulary JSON file.

## NVIDIA cuDNN Frontend

The project can optionally fetch NVIDIA's header-based cuDNN Frontend repository at build time:

- repository: `https://github.com/NVIDIA/cudnn-frontend`
- pinned commit: `9782b855ddecefe1646b00bb0cfd9870c381e391`
- license: MIT

The fetched source is placed in `third_party/cudnn-frontend` and is not included in this source ZIP.
Its C++ SDPA graph API is used to access NVIDIA's FlashAttention-2-based cuDNN implementation.

## Dao-AILab FlashAttention audit

The public FlashAttention repository was reviewed as an implementation and shape-specialization
reference:

- repository: `https://github.com/Dao-AILab/flash-attention`
- license: BSD-3-Clause

No PyTorch/ATen/C10 extension source from that repository is redistributed in this project. The
in-tree raw-CUDA SM80 attention kernel is an independent implementation of tiled Tensor Core
attention with online softmax.


## DPM++ 2M algorithm reference

The deterministic DPM++ 2M update was implemented from the public k-diffusion sampling formulation:

- repository: `https://github.com/crowsonkb/k-diffusion`
- file: `k_diffusion/sampling.py`, function `sample_dpmpp_2m`
- license: BSD-3-Clause

No Python, PyTorch, or k-diffusion runtime source is linked into this project.

## ComfyUI GITS scheduler and k-diffusion sampler formulas

The GITS coefficient tables and log-linear interpolation behavior are derived from
`Comfy-Org/ComfyUI/comfy_extras/nodes_gits.py`, which identifies the original
GITS implementation in `zju-pi/diff-sampler`. The Euler ancestral, DPM-Solver++
SDE, and DPM-Solver++ 2S ancestral CFG++ formulas are independently implemented
in C++/CUDA from the corresponding ComfyUI k-diffusion sampler definitions.
No Python runtime code is linked or shipped.

## Hyper-SDXL fixed-step scheduler recipe

The `ddim_trailing` scheduler and `--hyper-sdxl` preset reproduce the published
fixed-step Hyper-SDXL 2/4/8-step inference recipe:

- model card: `ByteDance/Hyper-SD`
- scheduler reference: Hugging Face Diffusers `DDIMScheduler.set_timesteps`
- spacing: `round(arange(num_train_timesteps, 0, -N/steps)) - 1`
- sampler settings: DDIM, trailing spacing, eta zero, guidance zero
- inherited SDXL scheduler settings: `clip_sample=false`, `set_alpha_to_one=false`

The implementation is independent C++ and does not redistribute Diffusers or
Hyper-SD Python source.


## PyTorch CPU random-number parity

The standalone C++ CPU-noise implementation follows the algorithms documented in PyTorch's BSD-licensed ATen sources: `CPUGeneratorImpl`, `MT19937RNGEngine`, `TransformationHelper`, and `native/cpu/DistributionTemplates`. It reimplements the MT19937 state transition, Float32 uniform transform, and contiguous Float32 Box-Muller layout needed for seed-compatible initial latent generation. No PyTorch, ATen, C10, or LibTorch code is linked into the executable.

The MT19937 engine source also carries the original Matsumoto/Nishimura permissive notice; the project preserves its attribution and disclaimer through this notice and the upstream provenance.

## ConvRot / INT8 implementation reference

The native C++/CUDA W8A8 implementation was developed from the mathematical
behavior documented in the user-supplied INT8 Fast reference: per-row weight
and activation quantization, regular group-wise Hadamard ConvRot, INT32 GEMM,
and scale epilogue. The supplied `convrot.py` identifies its regular Hadamard
implementation as MIT-derived from ComfyUI-ZImage-Triton. No Python, Triton, or
PyTorch source is compiled into or loaded by this runtime.
