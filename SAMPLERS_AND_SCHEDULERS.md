# Samplers and sigma schedulers

The runtime separates the **sampler algorithm** from the **sigma scheduler**. The existing positional `steps` value is the only step-count input. Scheduler-specific controls never require a second steps argument.

Defaults:

```text
--sampler dpmpp_2m --scheduler normal
```

## Samplers

| CLI name | Algorithm | Controls |
|---|---|---|
| `dpmpp_2m` | DPM-Solver++ 2M | deterministic; no sampler-specific controls |
| `dpmpp_sde` | stochastic two-stage DPM-Solver++ SDE | `--eta`, `--s-noise`, `--r`, `--noise-device` |
| `euler` | Euler discrete with optional Karras churn | `--s-churn`, `--s-tmin`, `--s-tmax`, `--s-noise`, `--noise-device` |
| `euler_ancestral` / `SamplerEulerAncestral` | Euler ancestral | `--eta`, `--s-noise`, `--noise-device` |
| `dpmpp_2s_ancestral_cfg_pp` | DPM-Solver++ 2S ancestral with CFG++ correction | `--eta`, `--s-noise`, `--noise-device`; forces the unconditional branch |
| `ddim` | DDIM | `--ddim-eta`, `--noise-device` |

`dpmpp_sde` uses an interval-consistent Brownian tree in sigma space, matching
ComfyUI/k-diffusion's default Brownian transform. Every midpoint and full-step
query is reconstructed from the same elementary Brownian increments and divided
by the square root of the queried interval length. Overlapping intervals therefore
have the exact Brownian covariance instead of receiving unrelated random tensors.
`--noise-device cpu` builds both the initial latent and Brownian increments on the
CPU before upload; `--noise-device gpu` uses deterministic GPU streams.

Sampler controls:

```text
--eta F                  ancestral/SDE stochasticity (default 1.0)
--s-noise F              injected-noise multiplier (default 1.0)
--r F                    DPM++ SDE internal stage ratio (default 0.5, range 0..1)
--noise-device cpu|gpu   initial latent and stochastic sampler noise device (default cpu)
--s-churn F              Euler churn amount (default 0)
--s-tmin F               minimum sigma eligible for churn (default 0)
--s-tmax F               maximum sigma eligible for churn (default effectively unlimited)
--ddim-eta F             DDIM stochasticity (default 0)
--sampler-state fp32|fp16 sampler latent/history precision (default fp32)
--initial-noise-scaling comfyui|sigma  full-denoise startup scale (default comfyui)
```

Non-Brownian stochastic samplers derive each step/stage noise stream from the requested image seed. DPM++ SDE instead derives deterministic elementary Brownian increments and reuses them for every overlapping interval query. Results are repeatable for a fixed build, device choice, and command line.

## ComfyUI KSampler parity path

A normal ComfyUI text-to-image `KSampler` does more than choose a sampler and a scheduler. For SDXL EPS models it also:

1. creates the initial latent with PyTorch CPU Float32 `randn`;
2. moves noise, latent, and sigma tensors to CUDA as Float32;
3. scales a full-denoise empty latent by `sqrt(1 + sigma_max^2)`;
4. divides the Float32 state by `sqrt(1 + sigma^2)` and casts only the UNet input to the model dtype;
5. computes predicted-clean samples and every solver update in Float32;
6. maps sigma-valued midpoint calls to the nearest training log-sigma timestep;
7. uses one Brownian path for overlapping DPM++ SDE interval queries.

The native parity controls are:

```text
--comfyui-parity
--noise-device cpu
--sampler-state fp32
--initial-noise-scaling comfyui
--precision fp16
```

The direct CLI equivalent is:

```powershell
... model.safetensors 1024 1024 4 1 788897633613589 `
  "a cinematic photograph of a city at night" "blurry" output.png `
  --sampler dpmpp_sde --scheduler normal --denoise 1 `
  --comfyui-parity
```

At CFG 1 the negative branch is mathematically unnecessary and ComfyUI may optimize it away; the CLI negative prompt is retained, but ordinary CFG cannot make it affect the result at exactly 1.0.

## Schedulers

| CLI name | Controls |
|---|---|
| `normal` | `--training-timesteps`, `--beta-start`, `--beta-end` |
| `karras` | `--karras-rho` plus training schedule controls |
| `exponential` | training schedule controls |
| `sgm_uniform` | training schedule controls |
| `simple` | training schedule controls |
| `ddim_uniform` | training schedule controls |
| `ddim_trailing` / `hyper_sdxl` | exact Diffusers DDIM trailing timestep spacing used by fixed-step Hyper-SDXL 2/4/8-step LoRAs |
| `beta` | `--alpha`, `--beta` plus training schedule controls |
| `linear_quadratic` | `--linear-quadratic-threshold` plus training schedule controls |
| `kl_optimal` | training schedule controls |
| `gits` / `GITSScheduler` | `--coeff`, `--denoise`; exact ComfyUI GITS noise-level tables |

Scheduler controls:

```text
--training-timesteps N
--beta-start F
--beta-end F
--karras-rho F
--alpha F                 beta scheduler alpha (alias: --beta-alpha)
--beta F                  beta scheduler beta (alias: --beta-beta)
--linear-quadratic-threshold F
--coeff F                 GITS coefficient, 0.80..1.50 in 0.05 increments
--denoise F               GITS denoise fraction, 0..1
```

GITS uses the exact coefficient/step tables from ComfyUI's `GITSScheduler`. For more than 20 steps, the final 21-point table for the selected coefficient is log-linearly interpolated to `steps + 1`. `--denoise` keeps the final `round(steps * denoise)` transitions. The terminal sigma is always zero.


## Hyper-SDXL fixed-step recipe

The official fixed-step Hyper-SDXL 2-, 4-, and 8-step LoRAs use a DDIM
sampler with Diffusers `timestep_spacing="trailing"`, DDIM eta zero, and
guidance scale zero. The runtime exposes that exact recipe as:

```powershell
... 1024 1024 4 1.0 1234 "prompt" "" output.png --hyper-sdxl
```

`--hyper-sdxl` requires the positional step count to be `2`, `4`, or `8`,
overrides the sampler to `ddim`, the scheduler to `ddim_trailing`, DDIM eta to
`0`, guidance/rescale to `0`, and disables forced CFG. Passing positional CFG
`1.0` is fine because the preset replaces it. The fixed-step preset uses Diffusers-compatible `set_alpha_to_one=true` final
handling. An explicit non-preset DDIM command keeps the project default unless
`--ddim-set-alpha-to-one` is supplied.

The equivalent explicit form is:

```powershell
... 1024 1024 4 0 1234 "prompt" "" output.png `
  --sampler ddim --scheduler ddim_trailing --ddim-eta 0
```

For 1000 training timesteps, the exact trailing schedules begin with:

- 2 steps: `999, 499`
- 4 steps: `999, 749, 499, 249`
- 8 steps: `999, 874, 749, 624, 499, 374, 249, 124`

The official unified 1-step Hyper-SDXL LoRA uses TCD scheduling, while the
standalone 1-step UNet uses an LCM-style recipe starting at timestep 800. Those
are different algorithms and are not silently substituted by the fixed-step
`--hyper-sdxl` preset.

## Examples

DPM++ SDE with GPU noise:

```powershell
.\build-cuda13\Release\sdxl_cuda_denoise.exe model.safetensors 1024 1024 30 5.0 1234 `
  "prompt" "negative" output.png `
  --sampler dpmpp_sde --scheduler karras `
  --eta 1.0 --s-noise 1.0 --r 0.5 --noise-device gpu --karras-rho 7
```

DPM++ SDE with CPU-created initial latent and Brownian noise:

```powershell
.\build-cuda13\Release\sdxl_cuda_denoise.exe model.safetensors 1024 1024 30 5.0 1234 `
  "prompt" "negative" output_cpu_noise.png `
  --sampler dpmpp_sde --scheduler normal --noise-device cpu
```

Exact GITS scheduler using the command's existing 20 steps:

```powershell
.\build-cuda13\Release\sdxl_cuda_denoise.exe model.safetensors 1024 1024 20 5.0 1234 `
  "prompt" "negative" output.png `
  --sampler dpmpp_2m --scheduler gits --coeff 1.20 --denoise 0.42
```

Euler ancestral:

```powershell
... --sampler euler_ancestral --scheduler normal --eta 1 --s-noise 1 --noise-device gpu
```

DPM++ 2S ancestral CFG++:

```powershell
... --sampler dpmpp_2s_ancestral_cfg_pp --scheduler beta --alpha 0.6 --beta 0.6 --eta 1 --s-noise 1
```

Euler with churn:

```powershell
... --sampler euler --scheduler normal --s-churn 5 --s-tmin 0 --s-tmax 999 --s-noise 1
```

## CUDA Graph restriction

A graph cannot safely replay changing stochastic noise. `dpmpp_sde`, `euler_ancestral`, and `dpmpp_2s_ancestral_cfg_pp` therefore require `--eta 0` or `--s-noise 0` when `--cuda-graph` is enabled. Deterministic `dpmpp_2m`, Euler without churn, and deterministic DDIM remain graph-compatible.
