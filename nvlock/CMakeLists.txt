add_library(nvlock SHARED
  nvlock.c
)
target_link_libraries(nvlock
  ${CUDA_LIBRARIES}
  dl
)
set_target_properties(nvlock PROPERTIES
  OUTPUT_NAME "cuda"
)
install(TARGETS nvlock
  LIBRARY DESTINATION "lib/nvlock"
)

configure_file(
  wrapper.c.in wrapper.c @ONLY
)
add_executable(wrapper
  ${CMAKE_CURRENT_BINARY_DIR}/wrapper.c
)
set_target_properties(wrapper PROPERTIES
  OUTPUT_NAME "nvlock"
)
install(TARGETS wrapper
  RUNTIME DESTINATION "bin"
)
