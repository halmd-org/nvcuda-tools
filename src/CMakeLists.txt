include_directories(
  ${CUDA_INCLUDE_DIR}
)

add_executable(nvlock
  nvlock_wrapper.c
)

add_library(libnvlock SHARED
  nvlock.c
)
target_link_libraries(libnvlock
  ${CUDA_LIBRARY}
  dl
)
set_target_properties(libnvlock PROPERTIES
  PREFIX ""
)

add_executable(cufree
  cufree.c
)
target_link_libraries(cufree
  ${CUDA_LIBRARY}
)

install(
  TARGETS
    libnvlock
    nvlock
    cufree
  RUNTIME
    DESTINATION bin
  LIBRARY
    DESTINATION lib
)
