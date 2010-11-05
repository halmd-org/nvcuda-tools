project(nvcuda-tools)

cmake_minimum_required(VERSION 2.6)

find_package(CUDA QUIET REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif(NOT CMAKE_BUILD_TYPE)

include_directories(
  ${CUDA_INCLUDE_DIR}
)
add_subdirectory(cufree)
add_subdirectory(nvlock)
