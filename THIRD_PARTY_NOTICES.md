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
