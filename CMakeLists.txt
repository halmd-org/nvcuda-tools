project(nvcuda-tools C)

cmake_minimum_required(VERSION 2.6)

find_package(CUDA QUIET REQUIRED)

if(CMAKE_BUILD_TYPE STREQUAL "${CMAKE_BUILD_TYPE_INIT}")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_C_FLAGS STREQUAL "${CMAKE_C_FLAGS_INIT}")
    set(CMAKE_C_FLAGS "-Wall -std=c99" CACHE STRING
      "Flags used by the compiler during all build types." FORCE)
  endif()
endif()

if(CMAKE_C_PLATFORM_ID STREQUAL "Linux")
  if(CMAKE_EXE_LINKER_FLAGS_RELEASE STREQUAL "${CMAKE_EXE_LINKER_FLAGS_RELEASE_INIT}")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-Wl,-s" CACHE STRING
      "Flags used by the linker during release builds." FORCE)
  endif()
  if(CMAKE_MODULE_LINKER_FLAGS_RELEASE STREQUAL "${CMAKE_MODULE_LINKER_FLAGS_RELEASE_INIT}")
    set(CMAKE_MODULE_LINKER_FLAGS_RELEASE "-Wl,-s" CACHE STRING
      "Flags used by the linker during release builds." FORCE)
  endif()
  if(CMAKE_SHARED_LINKER_FLAGS_RELEASE STREQUAL "${CMAKE_SHARED_LINKER_FLAGS_RELEASE_INIT}")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "-Wl,-s" CACHE STRING
      "Flags used by the linker during release builds." FORCE)
  endif()
endif()

include_directories(
  ${CUDA_INCLUDE_DIR}
)
add_subdirectory(cufree)
add_subdirectory(nvlock)
add_subdirectory(nvcuda-occupancy)
add_subdirectory(nvtop)
