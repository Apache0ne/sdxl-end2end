@echo off
setlocal
if "%~1"=="" (
  echo Usage: BENCHMARK_FLASH_CFG1.bat ^<full-sdxl-model.safetensors^>
  exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0BENCHMARK_FLASH_CFG1.ps1" -Model "%~1"
exit /b %ERRORLEVEL%
