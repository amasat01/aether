# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# aether-dsc: the sealed payload (pure data, installed by the wheel packages) plus `aether._core`, the
# arrays binding. `_core` is PLAIN C++23 on every platform: no CUDA language, no nvcc, no numpy. On
# linux x86_64 the wheel also carries two CUDA plugins behind the C seam in src/seam.h, one per CUDA
# major, each the same g++-compiled TU (src/plugin.cpp, aether's CUDA-mode Chunk/fill/StreamTransport behind the seam) linked to that major's static cudart.
cmake_minimum_required(VERSION 3.20)
project(aether_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

get_property(_langs GLOBAL PROPERTY ENABLED_LANGUAGES)
if("CUDA" IN_LIST _langs)
  message(FATAL_ERROR "aether._core is CUDA-free: device memory lives in the plugins (src/plugin.cpp)")
endif()
# A CUDA language enabled later by anything else would also be a bug
function(_aether_refuse_cuda _id)
  if(_id STREQUAL "CUDA")
    message(FATAL_ERROR "aether._core is CUDA-free: enable_language(CUDA) is refused")
  endif()
endfunction()
variable_watch(CMAKE_CUDA_COMPILER _aether_refuse_cuda)

find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG QUIET)
if(NOT nanobind_FOUND)
  execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
                  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
  find_package(nanobind CONFIG REQUIRED PATHS "${nanobind_ROOT}" NO_DEFAULT_PATH)
endif()

# The aether header root: $AETHER_HEADER_ROOT, else the checkout this directory sits in, else the closure
# an sdist carries (_sdist_include/, copied by _build_backend.py).
set(_root "$ENV{AETHER_HEADER_ROOT}")
if(NOT _root)
  foreach(_c "${CMAKE_CURRENT_SOURCE_DIR}/.." "${CMAKE_CURRENT_SOURCE_DIR}/_sdist_include")
    if(EXISTS "${_c}/aether/typedefs.h")
      set(_root "${_c}")
      break()
    endif()
  endforeach()
endif()
if(NOT EXISTS "${_root}/aether/typedefs.h")
  message(FATAL_ERROR "aether headers not found: set AETHER_HEADER_ROOT to a checkout of aether")
endif()
message(STATUS "aether headers: ${_root}")

nanobind_add_module(_core NB_STATIC FREE_THREADED src/aether_core.cpp)
target_include_directories(_core PRIVATE "${_root}" src)
target_compile_definitions(_core PRIVATE AETHER_CPP_MODE=1)
target_link_libraries(_core PRIVATE ${CMAKE_DL_LIBS})
install(TARGETS _core LIBRARY DESTINATION aether)
install(FILES aether/__init__.py aether/_dtype.py DESTINATION aether)

# ------------------------------------------------------------------ the CUDA plugins (linux x86_64 only)
#   AETHER_PYTHON_PLUGINS = BUILD (default on linux x86_64 when a cudart is found) | OFF | <dir of prebuilt .so>
set(AETHER_PYTHON_PLUGINS "AUTO" CACHE STRING "BUILD, AUTO, OFF, or a directory holding prebuilt plugins")
if(DEFINED ENV{AETHER_PYTHON_PLUGINS})
  set(AETHER_PYTHON_PLUGINS "$ENV{AETHER_PYTHON_PLUGINS}")
endif()

function(_aether_plugin major)
  set(_hint "$ENV{AETHER_CUDA${major}_ROOT}")
  file(GLOB _cands "/usr/local/cuda-${major}.*" "/usr/local/cuda-${major}")
  find_library(_cudart_${major} NAMES cudart_static
    HINTS ${_hint} ${_cands} PATH_SUFFIXES lib64 lib targets/x86_64-linux/lib NO_DEFAULT_PATH)
  find_path(_cudainc_${major} cuda_runtime.h
    HINTS ${_hint} ${_cands} PATH_SUFFIXES include targets/x86_64-linux/include NO_DEFAULT_PATH)
  if(NOT _cudart_${major} OR NOT _cudainc_${major})
    message(STATUS "aether: no CUDA ${major} toolkit (libcudart_static.a + headers) found: skipping its plugin")
    return()
  endif()
  add_library(aether_cuda${major} MODULE src/plugin.cpp)
  set_target_properties(aether_cuda${major} PROPERTIES PREFIX "lib" OUTPUT_NAME "aether_cuda${major}"
    CXX_VISIBILITY_PRESET hidden POSITION_INDEPENDENT_CODE ON)
  target_include_directories(aether_cuda${major} PRIVATE src "${_root}")
  target_include_directories(aether_cuda${major} SYSTEM PRIVATE "${_cudainc_${major}}")
  target_compile_definitions(aether_cuda${major} PRIVATE AETHER_HAS_CUDA=1)
  target_compile_options(aether_cuda${major} PRIVATE -fno-gnu-unique)
  target_link_libraries(aether_cuda${major} PRIVATE "${_cudart_${major}}" ${CMAKE_DL_LIBS} rt pthread)
  target_link_options(aether_cuda${major} PRIVATE -Wl,--exclude-libs,ALL -Wl,--no-undefined -Wl,--as-needed)
  install(TARGETS aether_cuda${major} LIBRARY DESTINATION aether)
  message(STATUS "aether: CUDA ${major} plugin from ${_cudart_${major}}")
endfunction()

if(AETHER_PYTHON_PLUGINS STREQUAL "OFF")
  message(STATUS "aether: plugins OFF (CPU arrays only)")
elseif(IS_DIRECTORY "${AETHER_PYTHON_PLUGINS}")
  file(GLOB _prebuilt "${AETHER_PYTHON_PLUGINS}/libaether_cuda*.so")
  install(FILES ${_prebuilt} DESTINATION aether)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  _aether_plugin(12)
  _aether_plugin(13)
endif()
