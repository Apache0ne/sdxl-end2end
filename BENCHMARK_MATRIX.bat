@echo off
setlocal
if "%~1"=="" (
  echo Usage: BENCHMARK_MATRIX.bat ^<model.safetensors-or-directory^> [build-release-dir] [output-dir]
  exit /b 2
)
set "MODEL=%~1"
set "BUILD=%~2"
set "OUT=%~3"
if "%BUILD%"=="" set "BUILD=build-cuda13\Release"
if "%OUT%"=="" set "OUT=benchmark_results"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0BENCHMARK_MATRIX.ps1" -Model "%MODEL%" -BuildDir "%BUILD%" -OutputDir "%OUT%"
exit /b %ERRORLEVEL%
