cmake_minimum_required(VERSION 3.28)

project(CUTE-VISUALIZE VERSION 0.0.1 LANGUAGES CXX CUDA)

# Set the CUDA architecture to compile the code for
# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
set(CUTE_VISUALIZE cute_visualize)

add_executable(
    ${CUTE_VISUALIZE}
    cute_visualize.cu
)
target_include_directories(${CUTE_VISUALIZE} PRIVATE ${CUTLASS_INCLUDE_DIR} ${CUTLASS_UTILS_INCLUDE_DIR})
set_target_properties(${CUTE_VISUALIZE} PROPERTIES CUDA_ARCHITECTURES native)
target_compile_options(${CUTE_VISUALIZE} PRIVATE --expt-relaxed-constexpr)
