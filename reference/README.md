# INT8 checkpoint reference inventory

`creapromptHyperSDXL_v1.2_FULL_ConvRot_INT8_keys.txt` is a key-only inventory
provided from the converted full SDXL ConvRot INT8 checkpoint used to reproduce
the prequantized loader failure. It does **not** contain model tensor data.

Inventory summary:

- declared tensors: 4,660;
- CLIP-L wrapper: `conditioner.embedders.0.transformer.*`;
- OpenCLIP-bigG wrapper: `conditioner.embedders.1.model.transformer.*`;
- UNet wrapper: `model.diffusion_model.*`;
- VAE wrapper: `first_stage_model.*`;
- quantized Linear metadata: `.weight`, `.weight_scale`, and `.comfy_quant`.

The host-only `sdxl_int8_key_layout_test` checks that every required CLIP-L and
OpenCLIP-bigG parameter has an exact, non-suffix-ambiguous source key in this
inventory.
