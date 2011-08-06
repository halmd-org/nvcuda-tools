project(nvcuda-tools)

cmake_minimum_required(VERSION 2.6)

find_package(CUDA QUIET REQUIRED)

if(CMAKE_BUILD_TYPE STREQUAL "${CMAKE_BUILD_TYPE_INIT}")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

include_directories(
  ${CUDA_INCLUDE_DIR}
)
add_subdirectory(cufree)
add_subdirectory(nvlock)
add_subdirectory(nvcuda-occupancy)
add_subdirectory(nvtop)
