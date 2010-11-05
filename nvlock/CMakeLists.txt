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
install(TARGETS libnvlock nvlock
  RUNTIME DESTINATION "bin"
  LIBRARY DESTINATION "lib"
)
