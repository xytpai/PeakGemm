cmake_minimum_required(VERSION 3.21)
project(PeakGemm LANGUAGES CXX)

# Set our targets before Torch configures GPU languages.
set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA architectures")
set(CMAKE_HIP_ARCHITECTURES gfx950 CACHE STRING "HIP architectures")
set(PEAKGEMM_HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
find_package(Torch REQUIRED)

set(PEAKGEMM_BACKEND "CUDA" CACHE STRING "GPU backend: CUDA or HIP")
set_property(CACHE PEAKGEMM_BACKEND PROPERTY STRINGS CUDA HIP)

if(PEAKGEMM_BACKEND STREQUAL "CUDA")
    enable_language(CUDA)
elseif(PEAKGEMM_BACKEND STREQUAL "HIP")
    enable_language(HIP)
else()
    message(FATAL_ERROR "PEAKGEMM_BACKEND must be CUDA or HIP")
endif()

set(PEAKGEMM_SOURCES csrc/hgemm.cpp csrc/rms_norm.cpp)
set_source_files_properties(${PEAKGEMM_SOURCES} PROPERTIES LANGUAGE ${PEAKGEMM_BACKEND})
add_library(PeakGemm SHARED ${PEAKGEMM_SOURCES})
target_include_directories(PeakGemm PRIVATE include ${TORCH_INCLUDE_DIRS})
target_link_libraries(PeakGemm PRIVATE ${TORCH_LIBRARIES})
separate_arguments(TORCH_CXX_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(PeakGemm PRIVATE ${TORCH_CXX_FLAGS}
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
set_target_properties(PeakGemm PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    HIP_ARCHITECTURES "${PEAKGEMM_HIP_ARCHITECTURES}"
    HIP_STANDARD 20
    HIP_STANDARD_REQUIRED ON
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH "$ORIGIN/../torch/lib")
