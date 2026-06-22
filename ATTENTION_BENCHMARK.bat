@echo off
setlocal
set EXE=build-cuda13\Release\sdxl_cuda_attention_bench.exe
if not exist "%EXE%" (
  echo Missing %EXE%. Build the project first.
  exit /b 1
)
echo === 77-token full-attention bucket ===
"%EXE%" 77 77 20 2 100
echo.
echo === SDXL cross-attention at 64x64 ===
"%EXE%" 4096 77 10 2 20
echo.
echo === SDXL self-attention at 32x32 ===
"%EXE%" 1024 1024 20 2 20
echo.
echo === SDXL self-attention at 64x64 ===
"%EXE%" 4096 4096 10 2 5
endlocal
