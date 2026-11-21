# The example project as a CMake library, and its kernels raised from the target's own flags.
#   cmake -S example -B build -DRAISE_BIN=<raise> -DAETHER_TRACE_INCLUDE=<dir> -DAETHER_IR=<dir>
#         -DCMAKE_CXX_COMPILER=<clang++> [-DMODEL_STIFF=ON]
#   cmake --build build --target example_model_raise
cmake_minimum_required(VERSION 3.21)
project(raise_example CXX)

set(CMAKE_CXX_STANDARD 20)
set(RAISE_BIN "" CACHE FILEPATH "The raise driver binary")
set(AETHER_TRACE_INCLUDE "" CACHE PATH "The folder that holds aether_probe/")
set(AETHER_IR "" CACHE PATH "A raptor checkout with the IR library built")
option(MODEL_STIFF "Clip the drag kernel earlier" OFF)

set(here "${CMAKE_CURRENT_SOURCE_DIR}")
add_library(example_model STATIC "${here}/physics/kernels.cpp")
target_include_directories(example_model PUBLIC
  "${here}/physics/include" "${here}/numerics/include" "${AETHER_TRACE_INCLUDE}")
if(MODEL_STIFF)
  target_compile_definitions(example_model PUBLIC MODEL_STIFF)
endif()

include("${here}/../cmake/AetherRaise.cmake")
aether_raise(
  TARGET example_model
  ENTRY "${here}/physics/kernels.cpp"
  KERNELS physics::drag physics::relax physics::band physics::gear
  ROOTS "${here}/physics" "${here}/numerics"
  OUT "${CMAKE_BINARY_DIR}/raised"
  RAISE "${RAISE_BIN}"
  TRACE_INCLUDE "${AETHER_TRACE_INCLUDE}"
  IR "${AETHER_IR}"
  LINK -lm)
