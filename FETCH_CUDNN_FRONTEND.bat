@echo off
setlocal
set ROOT=%~dp0
set DEST=%ROOT%third_party\cudnn-frontend
set COMMIT=9782b855ddecefe1646b00bb0cfd9870c381e391

where git >nul 2>nul
if errorlevel 1 (
  echo Git is required to fetch NVIDIA cudnn-frontend.
  exit /b 1
)

if exist "%DEST%\include\cudnn_frontend.h" (
  echo cuDNN Frontend already exists at %DEST%
  exit /b 0
)

if exist "%DEST%" rmdir /s /q "%DEST%"
mkdir "%DEST%"
pushd "%DEST%"
git init
if errorlevel 1 goto :failed
git remote add origin https://github.com/NVIDIA/cudnn-frontend.git
if errorlevel 1 goto :failed
git fetch --depth 1 origin %COMMIT%
if errorlevel 1 goto :failed
git checkout --detach FETCH_HEAD
if errorlevel 1 goto :failed
popd

echo Installed pinned NVIDIA cudnn-frontend commit %COMMIT%
echo Reconfigure CMake so cudnn-sdpa is enabled.
exit /b 0

:failed
popd
rmdir /s /q "%DEST%" 2>nul
echo Failed to fetch NVIDIA cudnn-frontend.
exit /b 1
