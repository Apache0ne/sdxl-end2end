# Warm server stdin protocol

Start the resident engine:

```bat
build-cuda13\Release\sdxl_cuda_server.exe ^
  D:\models\myModelXL.safetensors ^
  --memory balanced ^
  --precision fp8-auto ^
  --attention auto
```

The build attempts to include NVIDIA cuDNN Frontend SDPA. If unavailable, `auto` uses the in-tree
SM80 flash kernel for large eligible UNet attention shapes.

The server prints one JSON `ready` line after checkpoint mapping and selected component preloading.
It includes the configured attention policy.

## Generate

Send one tab-separated line:

```text
generate<TAB>prompt<TAB>negative<TAB>output.png<TAB>seed<TAB>width<TAB>height<TAB>steps<TAB>cfg<TAB>euler|ddim<TAB>batch<TAB>graph<TAB>profile<TAB>force_cfg
```

Example PowerShell:

```powershell
$commands = @(
  "generate`tA cinematic city`tlow quality`tfirst.png`t1234`t1024`t1024`t4`t1.0`teuler`t1`t1`t1`t0",
  "generate`tA mountain lake`tlow quality`tsecond.png`t1235`t1024`t1024`t4`t1.0`teuler`t1`t1`t1`t0",
  "stats",
  "flush",
  "quit"
)
$commands | .\build-cuda13\Release\sdxl_cuda_server.exe D:\models\myModelXL.safetensors --memory balanced --attention auto
```

At `cfg<=1`, `force_cfg=0` bypasses the unconditional CLIP and UNet branch. The response includes
`cfg_bypassed:true`. Set the final field to `1` only for an explicit old-path A/B comparison.

The first graph-enabled request for a setting executes eagerly and builds the graph. A later matching
request reports `graph_replay:true`. The graph key includes the resolved CFG mode, so forced and
bypassed requests never share an incompatible graph.

## Commands

```text
stats
```

Returns temporary-arena and persistent-weight allocation counters separately.

```text
flush
```

Waits for background image encoding and returns completed image count and total image-write time.

```text
trim
```

Releases fallback cached temporary allocations. The coalescing slab, plan caches, and resident model
components remain allocated.

```text
quit
```

Flushes queued images and exits.

## Memory modes

- `low`: components stage sequentially.
- `balanced`: FP8 UNet is preloaded and retained.
- `high`: CLIP, UNet, and VAE are preloaded and retained.

Use `--no-preload` only when startup latency should be deferred to the first request.
