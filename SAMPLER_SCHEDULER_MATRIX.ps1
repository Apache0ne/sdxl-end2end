param(
    [Parameter(Mandatory=$true)][string]$Model,
    [string]$Exe = ".\build-cuda13\Release\sdxl_cuda_denoise.exe",
    [string]$OutputDir = ".\sampler_scheduler_matrix",
    [int]$Size = 512,
    [int]$Steps = 4,
    [double]$Cfg = 1.0,
    [UInt64]$Seed = 1234,
    [string]$Prompt = "a cinematic photograph of a city at night",
    [string]$Negative = "blurry, low quality"
)

$ErrorActionPreference = "Stop"
if (!(Test-Path $Exe)) { throw "Missing executable: $Exe" }
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$Csv = Join-Path $OutputDir "results.csv"
if (Test-Path $Csv) { Remove-Item $Csv -Force }

$Samplers = @("dpmpp_2m", "dpmpp_sde", "euler", "euler_ancestral", "dpmpp_2s_ancestral_cfg_pp", "ddim")
$Schedulers = @(
    "normal", "karras", "exponential", "sgm_uniform", "simple",
    "ddim_uniform", "ddim_trailing", "beta", "linear_quadratic", "kl_optimal", "gits"
)

foreach ($Sampler in $Samplers) {
    foreach ($Scheduler in $Schedulers) {
        $Tag = "${Sampler}_${Scheduler}".Replace("-", "_")
        $Output = Join-Path $OutputDir "$Tag.png"
        $Trace = Join-Path $OutputDir "$Tag.trace.json"
        Write-Host "`n=== sampler=$Sampler scheduler=$Scheduler ==="
        & $Exe $Model $Size $Size $Steps $Cfg $Seed $Prompt $Negative $Output `
            --sampler $Sampler --scheduler $Scheduler --eta 1 --s-noise 1 --r 0.5 `
            --memory balanced --precision fp8-auto --attention auto `
            --profile --profile-json $Trace --benchmark-csv $Csv --sync-png
        if ($LASTEXITCODE -ne 0) {
            throw "sampler=$Sampler scheduler=$Scheduler failed with exit code $LASTEXITCODE"
        }
    }
}

Write-Host "`nSampler/scheduler matrix complete: $Csv"
Import-Csv $Csv |
    Select-Object sampler,scheduler,wall_ms,denoise_ms,attention_ms,linear_ms,convolution_ms |
    Format-Table -AutoSize
