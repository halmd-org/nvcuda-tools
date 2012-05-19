add_library(nvlock SHARED
  nvlock.c
)
target_link_libraries(nvlock
  ${CUDA_LIBRARY}
  dl
)
set_target_properties(nvlock PROPERTIES
  OUTPUT_NAME "cuda"
)
install(TARGETS nvlock nvlock
  LIBRARY DESTINATION "lib"
)
