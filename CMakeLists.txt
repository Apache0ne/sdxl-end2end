cmake_minimum_required(VERSION 3.28)
project(sdxl_raw_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

set(SDXL_CUDA_ARCHITECTURES "86" CACHE STRING
    "CUDA architectures to compile, for example 75 Turing, 86 Ampere, 89 Ada, or 90 Hopper")
set(CMAKE_CUDA_ARCHITECTURES "${SDXL_CUDA_ARCHITECTURES}")

option(SDXL_CUDA_FAST_MATH "Use CUDA fast math for inference kernels" ON)
option(SDXL_BUILD_TESTS "Build CUDA operator tests" ON)
option(SDXL_BUILD_CPU_REFERENCE "Also build the original slow CPU executors" OFF)
option(SDXL_ENABLE_FINITE_CHECKS "Compile expensive CUDA NaN/Inf tracing support" OFF)
option(SDXL_ENABLE_PROFILING "Compile CUDA event profiling support" ON)
option(SDXL_ENABLE_CUDNN_FRONTEND_SDPA "Enable optional cuDNN Frontend FlashAttention-2 SDPA when cudnn_frontend.h is available" ON)
set(CUDNN_FRONTEND_ROOT "" CACHE PATH "Optional NVIDIA cudnn-frontend source/install root")

find_package(CUDAToolkit 13.0 REQUIRED)
find_package(Threads REQUIRED)

set(_CUDNN_HINTS
    "${CUDNN_ROOT}"
    "$ENV{CUDNN_ROOT}"
    "$ENV{CUDNN_PATH}"
    "${CUDAToolkit_ROOT}"
    "C:/Program Files/NVIDIA/CUDNN/v9.23"
    "C:/Program Files/NVIDIA/CUDNN/v9"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0")
find_path(CUDNN_INCLUDE_DIR cudnn.h
    HINTS ${_CUDNN_HINTS}
    PATH_SUFFIXES include include/13.3 include/13.0 include/12.9)
find_library(CUDNN_LIBRARY
    NAMES cudnn cudnn64_9
    HINTS ${_CUDNN_HINTS}
    PATH_SUFFIXES lib lib/x64 lib/13.3/x64 lib/13.0/x64 lib/12.9/x64 bin)
if(NOT CUDNN_INCLUDE_DIR OR NOT CUDNN_LIBRARY)
    message(FATAL_ERROR
        "cuDNN 9 was not found. Set CUDNN_ROOT to the cuDNN root. CUDA 13.3 installs are expected under include/13.3 and lib/13.3/x64.")
endif()

set(CUDNN_FRONTEND_INCLUDE_DIR "")
if(SDXL_ENABLE_CUDNN_FRONTEND_SDPA)
    find_path(CUDNN_FRONTEND_INCLUDE_DIR cudnn_frontend.h
        HINTS
            "${CUDNN_FRONTEND_ROOT}"
            "$ENV{CUDNN_FRONTEND_ROOT}"
            "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cudnn-frontend"
            "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cudnn_frontend"
            "${CUDNN_ROOT}"
            "$ENV{CUDNN_ROOT}"
        PATH_SUFFIXES include)
    if(CUDNN_FRONTEND_INCLUDE_DIR)
        message(STATUS "cuDNN Frontend SDPA enabled: ${CUDNN_FRONTEND_INCLUDE_DIR}")
    else()
        message(STATUS "cudnn_frontend.h not found; using the in-tree SM80 FlashAttention-style kernel")
    endif()
endif()

add_library(sdxl_cuda STATIC
    src/json.cpp
    src/safetensors.cpp
    src/model.cpp
    src/sdxl.cpp
    src/tokenizer.cpp
    src/resources/sdxl_tokenizer_data.cpp
    src/scheduler.cpp
    src/png.cpp
    src/cuda/runtime.cpp
    src/cuda/profiler.cpp
    src/cuda/engine.cpp
    src/cuda/async_image_writer.cpp
    src/cuda/tensor.cu
    src/cuda/weights.cu
    src/cuda/fp8_kernels.cu
    src/cuda/ops_blas.cpp
    src/cuda/ops_kernels.cu
    src/cuda/flash_attention_sm80.cu
    src/cuda/cudnn_sdpa.cpp
    src/cuda/text_encoder.cpp
    src/cuda/unet.cpp
    src/cuda/denoiser.cpp
    src/cuda/denoise_graph.cpp
    src/cuda/vae.cpp
    src/cuda/image.cu)

target_include_directories(sdxl_cuda
    PUBLIC include "${CUDNN_INCLUDE_DIR}"
    PRIVATE src/cuda)
if(CUDNN_FRONTEND_INCLUDE_DIR)
    target_include_directories(sdxl_cuda PRIVATE "${CUDNN_FRONTEND_INCLUDE_DIR}")
    target_compile_definitions(sdxl_cuda PRIVATE
        SDXL_HAS_CUDNN_FRONTEND_SDPA=1
        CUDNN_FRONTEND_SKIP_JSON_LIB=1)
endif()
target_link_libraries(sdxl_cuda
    PUBLIC Threads::Threads CUDA::cudart CUDA::cublasLt CUDA::curand "${CUDNN_LIBRARY}")
target_compile_definitions(sdxl_cuda PUBLIC SDXL_CUDA_BACKEND=1)
if(SDXL_ENABLE_FINITE_CHECKS)
    target_compile_definitions(sdxl_cuda PUBLIC SDXL_ENABLE_FINITE_CHECKS=1)
endif()
if(SDXL_ENABLE_PROFILING)
    target_compile_definitions(sdxl_cuda PUBLIC SDXL_ENABLE_PROFILING=1)
endif()
set_target_properties(sdxl_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    target_compile_options(sdxl_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive- /EHsc /utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4,-permissive-,-EHsc,-utf-8 --extended-lambda --expt-relaxed-constexpr>)
else()
    target_compile_options(sdxl_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wconversion -Wshadow>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic --extended-lambda --expt-relaxed-constexpr>)
endif()
if(SDXL_CUDA_FAST_MATH)
    target_compile_options(sdxl_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
endif()

target_compile_options(sdxl_cuda PRIVATE
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo>)

add_executable(sdxl_cuda_denoise src/cuda_main.cpp)
target_link_libraries(sdxl_cuda_denoise PRIVATE sdxl_cuda)
set_target_properties(sdxl_cuda_denoise PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_executable(sdxl_cuda_inspect src/main.cpp)
target_link_libraries(sdxl_cuda_inspect PRIVATE sdxl_cuda)

add_executable(sdxl_cuda_server src/server_main.cpp)
target_link_libraries(sdxl_cuda_server PRIVATE sdxl_cuda)
set_target_properties(sdxl_cuda_server PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_executable(sdxl_cuda_attention_bench src/attention_bench.cu)
target_link_libraries(sdxl_cuda_attention_bench PRIVATE sdxl_cuda)
set_target_properties(sdxl_cuda_attention_bench PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

if(SDXL_BUILD_TESTS)
    enable_testing()
    add_executable(sdxl_cuda_ops_test tests/cuda/cuda_ops_test.cu)
    target_link_libraries(sdxl_cuda_ops_test PRIVATE sdxl_cuda)
    set_target_properties(sdxl_cuda_ops_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    add_test(NAME sdxl_cuda_ops_test COMMAND sdxl_cuda_ops_test)

    add_executable(sdxl_builtin_tokenizer_test tests/builtin_tokenizer_test.cpp)
    target_link_libraries(sdxl_builtin_tokenizer_test PRIVATE sdxl_cuda)
    add_test(NAME sdxl_builtin_tokenizer_test COMMAND sdxl_builtin_tokenizer_test)
endif()

if(SDXL_BUILD_CPU_REFERENCE)
    add_library(sdxl_cpu_reference STATIC
        src/text_encoder.cpp src/unet.cpp src/denoiser.cpp)
    target_include_directories(sdxl_cpu_reference PUBLIC include)
    target_link_libraries(sdxl_cpu_reference PUBLIC sdxl_cuda)
endif()

install(TARGETS sdxl_cuda sdxl_cuda_denoise sdxl_cuda_server sdxl_cuda_attention_bench sdxl_cuda_inspect)
