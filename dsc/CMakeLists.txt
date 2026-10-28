# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# aether-dsc: the sealed payload (pure data, installed by the wheel packages) plus `aether._core`, the
# arrays binding. `_core` is PLAIN C++23 on every platform: no CUDA language, no nvcc, no numpy. On
# linux x86_64 the wheel also carries the CUDA plugin behind the C seam in src/seam.h (src/plugin.cpp: aether's CUDA-mode
# Chunk/fill/StreamTransport behind the seam), one library for CUDA 12 and 13 that talks to the driver API only.
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
find_package(OpenMP REQUIRED COMPONENTS CXX)   # the host FFT runs its lines on an OpenMP team
target_link_libraries(_core PRIVATE ${CMAKE_DL_LIBS} OpenMP::OpenMP_CXX)
install(TARGETS _core LIBRARY DESTINATION aether)

# ---------------------------------------------------------------- the OpenMP runtime: ONE, owned by this package
# aether-dsc ships the runtime the build linked, under a stable name in aether/_omp/, and `_core` finds it by a relative
# rpath. Wheel repair tools are told to leave it (auditwheel --exclude, delocate --exclude), so no other wheel (eagle,
# hawk's host kernels) vendors a second copy: two OpenMP runtimes in one process abort on macOS and oversubscribe on
# Linux. Dependents ask `aether_dsc.openmp()` for the file and link that one.
if(UNIX)
  set(_omp_real "")
  if(NOT APPLE)
    # In a gcc-toolset `libgomp.so` is a linker SCRIPT; ask the compiler for the real shared object.
    execute_process(COMMAND "${CMAKE_CXX_COMPILER}" -print-file-name=libgomp.so.1
                    OUTPUT_VARIABLE _omp_cand OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(IS_ABSOLUTE "${_omp_cand}" AND EXISTS "${_omp_cand}")
      file(REAL_PATH "${_omp_cand}" _omp_real)
    endif()
  endif()
  if(NOT _omp_real)
    foreach(_l IN LISTS OpenMP_CXX_LIBRARIES)
      if(_l MATCHES "lib(gomp|omp)[^/]*\\.(so|dylib)" AND EXISTS "${_l}")
        file(REAL_PATH "${_l}" _omp_real)
        break()
      endif()
    endforeach()
  endif()
  if(NOT _omp_real)
    message(FATAL_ERROR "no libgomp/libomp among OpenMP_CXX_LIBRARIES (${OpenMP_CXX_LIBRARIES}): cannot ship the runtime")
  endif()
  if(APPLE)
    set(_omp_name libomp.dylib)
    set_target_properties(_core PROPERTIES INSTALL_RPATH "@loader_path/_omp")
  else()
    set(_omp_name libgomp.so.1)
    set_target_properties(_core PROPERTIES INSTALL_RPATH "$ORIGIN/_omp")
  endif()
  # Refuse by name anything that is not a binary (a text linker script copied here fails at import: "invalid ELF header").
  file(READ "${_omp_real}" _omp_magic LIMIT 4 HEX)
  if(NOT (_omp_magic STREQUAL "7f454c46" OR _omp_magic STREQUAL "cffaedfe" OR _omp_magic STREQUAL "cafebabe"))
    message(FATAL_ERROR "the OpenMP runtime ${_omp_real} is not an ELF or Mach-O binary (magic ${_omp_magic}): a linker script?")
  endif()
  message(STATUS "OpenMP runtime shipped as aether/_omp/${_omp_name}: ${_omp_real}")
  install(FILES "${_omp_real}" DESTINATION aether/_omp RENAME ${_omp_name})
  if(APPLE)
    # Point `_core` and the shipped copy at @rpath/libomp.dylib (rewritten at install, below), and ship omp.h so a
    # dependent can compile OpenMP code against it.
    foreach(_d IN LISTS OpenMP_CXX_INCLUDE_DIRS)
      if(EXISTS "${_d}/omp.h")
        install(FILES "${_d}/omp.h" DESTINATION aether/_omp/include)
        break()
      endif()
    endforeach()
    install(CODE [[
      set(_dest "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/aether")
      execute_process(COMMAND install_name_tool -id @rpath/libomp.dylib "${_dest}/_omp/libomp.dylib" COMMAND_ERROR_IS_FATAL ANY)
      file(GLOB _cores "${_dest}/_core*.so")
      if(NOT _cores)
        message(FATAL_ERROR "no aether/_core*.so under ${_dest}: cannot point it at the shipped OpenMP runtime")
      endif()
      # Rewrite from what `_core` actually records (the linker wrote the runtime's own install name: an absolute Homebrew
      # path), then refuse a build that still has an OpenMP reference outside @rpath.
      foreach(_pass change verify)
        foreach(_c IN LISTS _cores)
          execute_process(COMMAND otool -L "${_c}" OUTPUT_VARIABLE _deps COMMAND_ERROR_IS_FATAL ANY)
          string(REPLACE "\n" ";" _lines "${_deps}")
          foreach(_ln IN LISTS _lines)
            if(_ln MATCHES "^[ \t]*([^ \t]*/lib(g|i)?omp[0-9]*\\.dylib)")
              set(_ref "${CMAKE_MATCH_1}")
              if(NOT _ref MATCHES "^@rpath/")
                if(_pass STREQUAL "change")
                  execute_process(COMMAND install_name_tool -change "${_ref}" @rpath/libomp.dylib "${_c}" COMMAND_ERROR_IS_FATAL ANY)
                else()
                  message(FATAL_ERROR "${_c} still links ${_ref}, not @rpath/libomp.dylib")
                endif()
              endif()
            endif()
          endforeach()
        endforeach()
      endforeach()
    ]])
  endif()
endif()
install(FILES aether/__init__.py aether/_dtype.py aether/fft.py DESTINATION aether)

# ------------------------------------------------------------------ the CUDA plugins (linux x86_64 and aarch64)
#   AETHER_PYTHON_PLUGINS = BUILD (default on linux x86_64 and aarch64 when nvcc is found) | OFF | <dir of prebuilt .so>
set(AETHER_PYTHON_PLUGINS "AUTO" CACHE STRING "BUILD, AUTO, OFF, or a directory holding prebuilt plugins")
if(DEFINED ENV{AETHER_PYTHON_PLUGINS})
  set(AETHER_PYTHON_PLUGINS "$ENV{AETHER_PYTHON_PLUGINS}")
endif()

# The plugin is ONE library for CUDA 12 and 13: nvcc (`-cudart none`) compiles plugin.cpp, g++ compiles driver_runtime.cpp
# (the CUDA runtime functions aether's headers call, defined over the driver API), and nvcc links them. It needs
# libcuda.so.1 at run time and no CUDA runtime library. No CMake CUDA language: `_core` stays CUDA-free.
function(_aether_plugin)
  set(_hints "$ENV{AETHER_CUDA_ROOT}")
  file(GLOB _cands "/usr/local/cuda-12.*" "/usr/local/cuda")
  list(SORT _cands COMPARE NATURAL ORDER DESCENDING)
  find_program(_aether_nvcc nvcc HINTS ${_hints} ${_cands} PATH_SUFFIXES bin)
  find_path(_aether_cuda_inc cuda_runtime.h HINTS ${_hints} ${_cands} PATH_SUFFIXES include targets/x86_64-linux/include
                                                                              targets/sbsa-linux/include targets/aarch64-linux/include)
  if(NOT _aether_nvcc OR NOT _aether_cuda_inc)
    message(STATUS "aether: no CUDA toolkit (nvcc + headers) found: skipping the CUDA plugin")
    return()
  endif()
  # SASS for every architecture this nvcc builds from sm_60 up, plus PTX for the newest
  execute_process(COMMAND "${_aether_nvcc}" --list-gpu-arch OUTPUT_VARIABLE _archs OUTPUT_STRIP_TRAILING_WHITESPACE)
  string(REGEX MATCHALL "compute_[0-9]+" _archs "${_archs}")
  set(_gencode "")
  set(_newest "")
  foreach(_a IN LISTS _archs)
    string(REPLACE "compute_" "" _n "${_a}")
    if(_n GREATER_EQUAL 60)
      list(APPEND _gencode -gencode arch=compute_${_n},code=sm_${_n})
      set(_newest ${_n})
    endif()
  endforeach()
  if(NOT _newest)
    message(STATUS "aether: ${_aether_nvcc} builds no sm_60+ architecture: skipping the CUDA plugin")
    return()
  endif()
  list(APPEND _gencode -gencode arch=compute_${_newest},code=compute_${_newest})
  add_library(aether_driver_runtime OBJECT src/driver_runtime.cpp)
  set_target_properties(aether_driver_runtime PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_VISIBILITY_PRESET hidden)
  target_include_directories(aether_driver_runtime PRIVATE "${_root}")
  target_include_directories(aether_driver_runtime SYSTEM PRIVATE "${_aether_cuda_inc}")
  # -fno-gnu-unique is a GCC flag (it keeps the plugin unloadable); clang rejects it
  set(_xcompiler "-fPIC,-fvisibility=hidden")
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(aether_driver_runtime PRIVATE -fno-gnu-unique)
    string(APPEND _xcompiler ",-fno-gnu-unique")
  endif()
  set(_plugin_o "${CMAKE_CURRENT_BINARY_DIR}/aether_plugin.o")
  set(_plugin_so "${CMAKE_CURRENT_BINARY_DIR}/libaether_cuda.so")
  set(_nvcc_common -ccbin "${CMAKE_CXX_COMPILER}" -std=c++20 -O2 -cudart none ${_gencode})
  add_custom_command(OUTPUT "${_plugin_o}"
    COMMAND "${_aether_nvcc}" ${_nvcc_common} -x cu -c -Xcompiler ${_xcompiler}
            -DAETHER_HAS_CUDA=1 -I "${CMAKE_CURRENT_SOURCE_DIR}/src" -I "${_root}" "${CMAKE_CURRENT_SOURCE_DIR}/src/plugin.cpp" -o "${_plugin_o}"
    DEPENDS src/plugin.cpp src/cufft_plan.h src/seam.h VERBATIM)
  # The version script keeps the export list to the seam alone: hidden visibility does not cover the libstdc++
  # template symbols some toolchains (manylinux gcc-toolset) leave in the plugin.
  add_custom_command(OUTPUT "${_plugin_so}"
    COMMAND "${_aether_nvcc}" ${_nvcc_common} -shared "${_plugin_o}" $<TARGET_OBJECTS:aether_driver_runtime>
            -Xlinker --version-script=${CMAKE_CURRENT_SOURCE_DIR}/src/plugin.map -Xlinker --exclude-libs,ALL -Xlinker --no-undefined
            -Xlinker --as-needed -ldl -lrt -lpthread -o "${_plugin_so}"
    DEPENDS "${_plugin_o}" aether_driver_runtime src/plugin.map COMMAND_EXPAND_LISTS VERBATIM)
  add_custom_target(aether_cuda ALL DEPENDS "${_plugin_so}")
  install(FILES "${_plugin_so}" DESTINATION aether)
  message(STATUS "aether: CUDA plugin (driver API, no CUDA runtime) with ${_aether_nvcc}")
endfunction()

if(AETHER_PYTHON_PLUGINS STREQUAL "OFF")
  message(STATUS "aether: plugins OFF (CPU arrays only)")
elseif(IS_DIRECTORY "${AETHER_PYTHON_PLUGINS}")
  file(GLOB _prebuilt "${AETHER_PYTHON_PLUGINS}/libaether_cuda*.so")
  install(FILES ${_prebuilt} DESTINATION aether)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|aarch64|arm64")
  _aether_plugin()
endif()
