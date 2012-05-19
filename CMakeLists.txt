cmake_minimum_required(VERSION 2.6)

set(CMAKE_USER_MAKE_RULES_OVERRIDE "cmake/platform.cmake")

project(nvcuda-tools C)

find_package(CUDA QUIET REQUIRED COMPONENTS cuda)

include_directories(${CUDA_INCLUDE_DIR})

add_subdirectory(nvlock)
add_subdirectory(nvcuda-occupancy)
add_subdirectory(nvtop)
