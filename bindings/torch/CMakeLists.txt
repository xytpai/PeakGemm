cmake_minimum_required(VERSION 3.18)
project(PeakGemm LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
get_filename_component(REPOSITORY_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../.. ABSOLUTE)

set(DEVICE_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/hgemm_sm80.cpp)

if(NOT TORCH_CUDA_ARCH_LIST)
    set(TORCH_CUDA_ARCH_LIST "8.0")
endif()
message(STATUS "PeakGemm CUDA architectures: ${TORCH_CUDA_ARCH_LIST}")
find_package(Torch REQUIRED)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++17 -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set_source_files_properties(${DEVICE_SOURCE} PROPERTIES LANGUAGE CUDA)

add_library(${PROJECT_NAME}_device SHARED ${DEVICE_SOURCE})
target_include_directories(${PROJECT_NAME}_device PRIVATE
    ${REPOSITORY_ROOT}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TORCH_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME}_device PRIVATE ${TORCH_LIBRARIES})

add_library(${PROJECT_NAME} MODULE register.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TORCH_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${TORCH_LIBRARIES}
    ${PROJECT_NAME}_device)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
