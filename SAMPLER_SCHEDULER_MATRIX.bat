@echo off
setlocal
if "%~1"=="" (
  echo Usage: SAMPLER_SCHEDULER_MATRIX.bat ^<full-sdxl-model.safetensors^> [size] [steps]
  exit /b 1
)
set "SIZE=%~2"
set "STEPS=%~3"
if "%SIZE%"=="" set "SIZE=512"
if "%STEPS%"=="" set "STEPS=4"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0SAMPLER_SCHEDULER_MATRIX.ps1" -Model "%~1" -Size %SIZE% -Steps %STEPS%
exit /b %ERRORLEVEL%
