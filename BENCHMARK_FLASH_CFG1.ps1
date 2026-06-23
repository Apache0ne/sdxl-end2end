param(
    [Parameter(Mandatory=$true)][string]$Model,
    [string]$Exe = ".\build-cuda13\Release\sdxl_cuda_denoise.exe",
    [string]$OutDir = ".\flash_cfg1_benchmark"
)

$ErrorActionPreference = "Stop"
if (!(Test-Path $Exe)) { throw "Missing executable: $Exe" }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$Csv = Join-Path $OutDir "results.csv"
if (Test-Path $Csv) { Remove-Item $Csv -Force }

$Prompt = "a cinematic photograph of a city at night"
$Negative = "blurry"
$Common = @($Model, "1024", "1024", "4", "1.0", "1234", $Prompt, $Negative)
$Runtime = @("--sampler", "dpmpp_2m", "--scheduler", "normal", "--memory", "balanced", "--precision", "fp8-auto", "--arena-reserve-mib", "1024", "--sync-png", "--profile", "--benchmark-csv", $Csv)

function Run-Case([string]$Name, [string[]]$Extra) {
    $Output = Join-Path $OutDir ($Name + ".png")
    $Trace = Join-Path $OutDir ($Name + ".json")
    Write-Host "`n=== $Name ==="
    & $Exe @Common $Output @Runtime "--profile-json" $Trace @Extra
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE" }
}

# Warm the filesystem and create the backend-specific packed FP8 cache before
# collecting comparisons. The output is intentionally not included in the CSV.
$Warm = Join-Path $OutDir "warmup.png"
Write-Host "=== packed-cache warmup ==="
& $Exe @Common $Warm "--memory" "balanced" "--precision" "fp8-auto" "--attention" "warp-online" "--sync-png"
if ($LASTEXITCODE -ne 0) { throw "warmup failed" }

Run-Case "01_legacy_warp_forced_cfg" @("--attention", "warp-online", "--force-cfg")
Run-Case "02_cfg1_bypass_warp" @("--attention", "warp-online")
Run-Case "03_flash_forced_cfg" @("--attention", "flash-sm80", "--force-cfg")
Run-Case "04_flash_cfg1_bypass" @("--attention", "flash-sm80")
Run-Case "05_auto_cfg1_bypass" @("--attention", "auto")

# This final case is optional. It succeeds only when the project was configured
# with NVIDIA cudnn-frontend. Failure does not invalidate the in-tree kernel.
try {
    Run-Case "06_cudnn_sdpa_cfg1_bypass" @("--attention", "cudnn-sdpa")
} catch {
    Write-Warning "cuDNN Frontend SDPA unavailable: $($_.Exception.Message)"
}

Write-Host "`nResults: $Csv"
Import-Csv $Csv | Select-Object sampler,scheduler,attention,cfg_bypassed,wall_ms,denoise_ms,attention_ms,cudnn_attention_ms,flash_attention_ms,warp_attention_ms,linear_ms,convolution_ms,temp_driver_alloc_delta | Format-Table -AutoSize
