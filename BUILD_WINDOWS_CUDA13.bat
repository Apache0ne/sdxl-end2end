@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
if not exist "%CUDA_ROOT%\bin\nvcc.exe" (
  echo ERROR: CUDA 13.3 nvcc was not found at:
  echo   %CUDA_ROOT%\bin\nvcc.exe
  exit /b 1
)

if "%CUDNN_ROOT%"=="" set "CUDNN_ROOT=C:\Program Files\NVIDIA\CUDNN\v9.23"
if not exist "%CUDNN_ROOT%\include\13.3\cudnn.h" (
  echo ERROR: cuDNN 9.23 CUDA 13.3 headers were not found at:
  echo   %CUDNN_ROOT%\include\13.3\cudnn.h
  echo Set CUDNN_ROOT to the installed cuDNN v9.23 directory.
  exit /b 1
)
if exist "%CUDNN_ROOT%\bin\13.3\x64" set "PATH=%CUDNN_ROOT%\bin\13.3\x64;%PATH%"
if exist "%CUDNN_ROOT%\bin\13.3" set "PATH=%CUDNN_ROOT%\bin\13.3;%PATH%"
if exist "%CUDNN_ROOT%\bin" set "PATH=%CUDNN_ROOT%\bin;%PATH%"
set "PATH=%CUDA_ROOT%\bin;%PATH%"

set "ARCH=%~1"
if "%ARCH%"=="" set "ARCH=86"

set "BUILD_DIR=build-cuda13"
if not "%SDXL_BUILD_DIR%"=="" set "BUILD_DIR=%SDXL_BUILD_DIR%"

if /I not "%SDXL_SKIP_CUDNN_FRONTEND%"=="1" (
  if not exist "%~dp0third_party\cudnn-frontend\include\cudnn_frontend.h" (
    echo Fetching pinned NVIDIA cuDNN Frontend for the optional SDPA backend...
    call "%~dp0FETCH_CUDNN_FRONTEND.bat"
    if errorlevel 1 echo WARNING: cuDNN Frontend fetch failed; the build will use the in-tree flash-sm80 kernel.
  )
)

set "FINITE=OFF"
if /I "%~2"=="debug-finite" set "FINITE=ON"

echo Building optimized mixed-precision SDXL CUDA 13.3 / cuDNN 9.23 for SM %ARCH%.

cmake -S . -B "%BUILD_DIR%" ^
  -G "Visual Studio 17 2022" -A x64 ^
  -T cuda="%CUDA_ROOT%" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCUDAToolkit_ROOT="%CUDA_ROOT%" ^
  -DCUDNN_ROOT="%CUDNN_ROOT%" ^
  -DSDXL_CUDA_ARCHITECTURES=%ARCH% ^
  -DSDXL_CUDA_FAST_MATH=ON ^
  -DSDXL_ENABLE_PROFILING=ON ^
  -DSDXL_ENABLE_FINITE_CHECKS=%FINITE% ^
  -DSDXL_BUILD_TESTS=ON ^
  -DSDXL_BUILD_CPU_REFERENCE=OFF
if errorlevel 1 exit /b 1

cmake --build "%BUILD_DIR%" --config Release --parallel
if errorlevel 1 exit /b 1

ctest --test-dir "%BUILD_DIR%" -C Release --output-on-failure
if errorlevel 1 exit /b 1

echo.
echo CUDA build and operator tests completed successfully for SM %ARCH% in %BUILD_DIR%.
if /I "%FINITE%"=="ON" echo Finite tracing was compiled for this build.
