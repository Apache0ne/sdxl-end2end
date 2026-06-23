param(
    [Parameter(Mandatory=$true)][string]$Model,
    [string]$BuildDir = ".\build-cuda13\Release",
    [string]$OutputDir = ".\benchmark_results",
    [int]$Steps = 4,
    [double]$Cfg = 1.0,
    [UInt64]$Seed = 1234,
    [string]$Prompt = "a cinematic photograph of a city at night",
    [string]$Negative = "blurry, low quality"
)

$ErrorActionPreference = "Stop"
$cli = Join-Path $BuildDir "sdxl_cuda_denoise.exe"
$server = Join-Path $BuildDir "sdxl_cuda_server.exe"
if (!(Test-Path $cli)) { throw "Missing executable: $cli" }
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$csv = Join-Path $OutputDir "benchmark.csv"
if (Test-Path $csv) { Remove-Item $csv }

function Invoke-ColdCase([int]$Size, [string]$Sampler, [string]$Precision) {
    $tag = "cold_${Size}_${Sampler}_normal_${Precision}".Replace("-", "_")
    $png = Join-Path $OutputDir "$tag.png"
    $trace = Join-Path $OutputDir "$tag.trace.json"
    & $cli $Model $Size $Size $Steps $Cfg $Seed $Prompt $Negative $png `
        --sampler $Sampler --scheduler normal --memory low --precision $Precision --attention auto --profile --profile-json $trace `
        --benchmark-csv $csv --sync-png
    if ($LASTEXITCODE -ne 0) { throw "Cold benchmark failed: $tag" }
}

# Cold process baselines: filesystem/model mapping, uploads, execution and PNG.
foreach ($size in @(512, 1024)) {
    foreach ($sampler in @("dpmpp_2m", "euler", "ddim")) {
        foreach ($precision in @("fp16", "fp8-auto")) {
            Invoke-ColdCase $size $sampler $precision
        }
    }
}

# Warm persistent CLI: first request warms plans/captures, second request replays.
$jobs = Join-Path $OutputDir "warm_jobs.tsv"
@(
    "$Prompt`t$Negative`t$(Join-Path $OutputDir 'warm_first.png')`t$Seed`t1",
    "$Prompt`t$Negative`t$(Join-Path $OutputDir 'warm_second.png')`t$($Seed + 1)`t1"
) | Set-Content -Encoding UTF8 $jobs
& $cli $Model 1024 1024 $Steps $Cfg $Seed --jobs $jobs `
    --sampler dpmpp_2m --scheduler normal --memory balanced --precision fp8-auto --attention auto --preload --cuda-graph --profile `
    --benchmark-csv $csv --sync-png
if ($LASTEXITCODE -ne 0) { throw "Warm persistent CLI benchmark failed" }

# Warm server: the engine and selected weights stay alive across requests.
if (Test-Path $server) {
    $serverLog = Join-Path $OutputDir "server.log"
    $commands = @(
        "generate`t$Prompt`t$Negative`t$(Join-Path $OutputDir 'server_first.png')`t$Seed`t1024`t1024`t$Steps`t$Cfg`tdpmpp_2m`tnormal`t1`t1`t1`t0",
        "generate`t$Prompt`t$Negative`t$(Join-Path $OutputDir 'server_second.png')`t$($Seed + 1)`t1024`t1024`t$Steps`t$Cfg`tdpmpp_2m`tnormal`t1`t1`t1`t0",
        "stats",
        "quit"
    )
    $commands | & $server $Model --memory balanced --precision fp8-auto --arena-reserve-mib 1024 2>&1 |
        Tee-Object -FilePath $serverLog
    if ($LASTEXITCODE -ne 0) { throw "Persistent server benchmark failed" }
}

Write-Host "Benchmark matrix complete. CSV: $csv"
