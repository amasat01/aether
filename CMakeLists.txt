# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

option(AETHER_BUILD_TESTS "Whether to build tests" OFF)
option(AETHER_DEBUG_MODE "Whether to compile with debug features" OFF)
option(AETHER_WARN_AS_ERR "Whether to promote compilation warnings to errors" OFF)
option(AETHER_CPP_MODE "Whether to build aether as a pure C++ Package. Default is CUDA/C++" OFF)

# The `sanitize` custom target (tests/sanitize/). OFF by default; the
# sanitizer gate turns it on.
# Sanitizer build of the test targets (tests/sanitize/README.md): a -fsanitize= list such as
# `address,undefined`. Empty (default) changes nothing.
set(AETHER_SANITIZERS "" CACHE STRING "-fsanitize= list applied to the test targets and canaries (empty = off)")
option(AETHER_BUILD_SANITIZE "Register the \`sanitize\` custom target (valgrind/compute-sanitizer wrapper)" OFF)
set(AETHER_SANITIZE_TOOL "both" CACHE STRING "Which sanitizer(s) the sanitize target runs: valgrind|computesan|both")
option(AETHER_SANITIZE_MINIMAL "Run the sanitize target with AETHER_TEST_MINIMAL=1 (currently a no-op, no aether test reads it)" OFF)
option(AETHER_SANITIZE_FAIL_ON_REACHABLE "Escalate still-reachable leaks to failures" OFF)

# CUDA architectures, when CMAKE_CUDA_ARCHITECTURES is not given: AETHER_CUDA_ARCHS
# "native" (the default) builds for the GPUs in this machine only, the fastest
# build; "all" builds for every one of 61 70 75 80 86 89 90 100 120 that the nvcc
# in use still compiles (CUDA 13 drops the pre-Turing ones and adds Blackwell);
# one architecture or a list ("86", "75;86"; sm_86 is accepted) builds exactly those. With no GPU to query, "native"
# builds PTX for the oldest architecture nvcc compiles, which the driver
# compiles for any newer GPU at load. Runs BEFORE project() enables CUDA, so the
# compiler is located here (-DCMAKE_CUDA_COMPILER, then $CUDACXX, then PATH).
set(AETHER_CUDA_ARCHS "native" CACHE STRING
    "CUDA architectures when CMAKE_CUDA_ARCHITECTURES is unset: native, all, one (86) or a list (75;86)")
if(NOT AETHER_CPP_MODE AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(_aether_nvcc "")
  if(CMAKE_CUDA_COMPILER)
    set(_aether_nvcc "${CMAKE_CUDA_COMPILER}")
  elseif(NOT "$ENV{CUDACXX}" STREQUAL "")
    set(_aether_nvcc "$ENV{CUDACXX}")
  else()
    find_program(_aether_nvcc_found nvcc)
    set(_aether_nvcc "${_aether_nvcc_found}")
  endif()
  set(_aether_supported "")
  if(_aether_nvcc)
    execute_process(COMMAND "${_aether_nvcc}" --list-gpu-arch
                    OUTPUT_VARIABLE _aether_list RESULT_VARIABLE _aether_rc ERROR_QUIET)
    if(_aether_rc EQUAL 0)
      string(REGEX MATCHALL "compute_[0-9]+" _aether_found "${_aether_list}")
      foreach(_c IN LISTS _aether_found)
        string(REPLACE "compute_" "" _c "${_c}")
        list(APPEND _aether_supported ${_c})
      endforeach()
      list(SORT _aether_supported COMPARE NATURAL)
    endif()
  endif()
  set(CMAKE_CUDA_ARCHITECTURES "")
  if(AETHER_CUDA_ARCHS STREQUAL "all")
    foreach(_a 61 70 75 80 86 89 90 100 120)
      if(NOT _aether_supported OR _a IN_LIST _aether_supported)
        list(APPEND CMAKE_CUDA_ARCHITECTURES ${_a})
      endif()
    endforeach()
  elseif(AETHER_CUDA_ARCHS STREQUAL "native")
    set(_aether_caps "")
    find_program(_aether_smi nvidia-smi)
    if(_aether_smi)
      execute_process(COMMAND "${_aether_smi}" --query-gpu=compute_cap --format=csv,noheader
                      OUTPUT_VARIABLE _aether_smi_out RESULT_VARIABLE _aether_smi_rc ERROR_QUIET)
      if(_aether_smi_rc EQUAL 0)
        string(REGEX MATCHALL "[0-9]+\\.[0-9]+" _aether_caps "${_aether_smi_out}")
        list(TRANSFORM _aether_caps REPLACE "\\." "")
        list(REMOVE_DUPLICATES _aether_caps)
      endif()
    endif()
    if(_aether_supported)
      list(GET _aether_supported 0 _aether_min)
      list(GET _aether_supported -1 _aether_max)
    endif()
    foreach(_c IN LISTS _aether_caps)
      if(NOT _aether_supported OR _c IN_LIST _aether_supported)
        list(APPEND CMAKE_CUDA_ARCHITECTURES ${_c})
      elseif(_c GREATER _aether_max)
        # a GPU newer than this nvcc: PTX for its newest architecture
        list(APPEND CMAKE_CUDA_ARCHITECTURES ${_aether_max}-virtual)
      elseif(_c LESS _aether_min)
        message(FATAL_ERROR "This machine's GPU (sm_${_c}) is older than the oldest architecture "
                            "${_aether_nvcc} compiles (sm_${_aether_min}): use an older CUDA toolkit, or "
                            "pass AETHER_CUDA_ARCHS / CMAKE_CUDA_ARCHITECTURES for another target.")
      endif()
    endforeach()
    if(NOT CMAKE_CUDA_ARCHITECTURES AND _aether_supported)
      # no GPU here: portable PTX for the oldest architecture nvcc compiles, and
      # never below sm_60 (the double atomicAdd the headers use needs it)
      foreach(_a IN LISTS _aether_supported)
        if(NOT _a LESS 60)
          set(CMAKE_CUDA_ARCHITECTURES ${_a}-virtual)
          break()
        endif()
      endforeach()
    endif()
  else()
    # one architecture or several; "sm_86" and "compute_86" spell "86"
    string(REGEX REPLACE "(sm|compute)_" "" CMAKE_CUDA_ARCHITECTURES "${AETHER_CUDA_ARCHS}")
  endif()
  if(NOT CMAKE_CUDA_ARCHITECTURES)  # no nvcc to ask yet: the long-standing set
    set(CMAKE_CUDA_ARCHITECTURES 61 70 80 90)
  endif()
  message(STATUS "> Compiling for CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Device code is capped at C++20 (nvcc's device-code standard ceiling; see
# tests/langsplit/). Set BEFORE `project()` so it is the
# default for every target created afterwards; test targets additionally
# pin `CUDA_STANDARD 20` explicitly as a target property (tests/CMakeLists.txt)
# so the requirement holds even if a target overrides the directory default.
if(NOT AETHER_CPP_MODE AND NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if (AETHER_CPP_MODE)
  message(STATUS "> Configuring aether in C++ only mode")
  project(aether VERSION 0.2.1 LANGUAGES CXX)
else()
  message(STATUS "> Configuring aether in CUDA/C++ mode")
  project(aether VERSION 0.2.1 LANGUAGES CUDA CXX)
endif()

include(GNUInstallDirs)

if (NOT DEFINED AETHER_WARP_SIZE)
  set(AETHER_WARP_SIZE 32 CACHE STRING "Warp size assumed by aether's device code")
endif()

###########################

# include OpenMP
find_package(OpenMP REQUIRED)
message(STATUS "> Found OpenMP")

# Compile/link flags for CUDA/C++
if(NOT DEFINED AETHER_MARCH)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64)")
    set(AETHER_MARCH "x86-64-v3")        # AVX2 + FMA — runs on all modern x86
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64|arm64|ARM64)")
    set(AETHER_MARCH "")                  # compiler default (armv8-a baseline)
  else()
    set(AETHER_MARCH "")                  # unknown arch — let the compiler decide
  endif()
endif()
if(AETHER_MARCH)
  list(APPEND AETHER_MARCH_FLAGS -march=${AETHER_MARCH})
  message(STATUS "> Target CPU architecture: -march=${AETHER_MARCH}")
else()
  set(AETHER_MARCH_FLAGS "")
  message(STATUS "> Target CPU architecture: compiler default")
endif()
list(APPEND AETHER_COMMON_FLAGS -Wall -Wextra -O3 ${AETHER_MARCH_FLAGS} -ftemplate-backtrace-limit=100)
# CUDA only flags
if(NOT AETHER_CPP_MODE)
  list(APPEND AETHER_CUDA_FLAGS ${AETHER_COMMON_FLAGS} ${OpenMP_CXX_FLAGS} -Wno-deprecated-gpu-targets)
endif()

# C++ >= 23 is required (host side; device TUs are capped at C++20, see above)
list(APPEND AETHER_CXX_FLAGS ${AETHER_COMMON_FLAGS} ${OpenMP_CXX_FLAGS} -Wpedantic)

# Treat warnings as errors - for CI pipeline
if (AETHER_WARN_AS_ERR)
  list(APPEND AETHER_CXX_FLAGS -Werror)
  if(NOT AETHER_CPP_MODE)
    list(APPEND AETHER_CUDA_FLAGS -Werror=all-warnings)
  endif()
endif()

# Preprocessor variable definitions
list(APPEND AETHER_DEFINES AETHER_WARP_SIZE=${AETHER_WARP_SIZE})
if (AETHER_DEBUG_MODE)
  list(APPEND AETHER_DEFINES AETHER_DEBUG_MODE)
endif()

# Mark whether C++ only
if (AETHER_CPP_MODE)
  list(APPEND AETHER_DEFINES AETHER_CPP_MODE=1)
endif()

# Mark whether this build has a CUDA backend (L3): gates kDLCUDA/kDLCUDAHost
# support in aether::backend_enabled<> and every AETHER_HAS_CUDA-guarded
# code path (aether/{device/Device,err/Error,chunk/Chunk,chunk/Copy}.h).
if (NOT AETHER_CPP_MODE)
  list(APPEND AETHER_DEFINES AETHER_HAS_CUDA=1)
endif()

# Header only library
set(AETHER_LIB aether)
add_library(${AETHER_LIB} INTERFACE)
add_library(${AETHER_LIB}::${AETHER_LIB} ALIAS ${AETHER_LIB})
target_include_directories(${AETHER_LIB} INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(${AETHER_LIB} INTERFACE ${OpenMP_CXX_LIBRARIES})
# Common options
target_compile_features(${AETHER_LIB} INTERFACE cxx_std_23)
target_compile_definitions(${AETHER_LIB} INTERFACE ${AETHER_DEFINES})
target_compile_options(${AETHER_LIB} INTERFACE
  "$<$<COMPILE_LANGUAGE:CXX>:${AETHER_CXX_FLAGS}>")
target_link_options(${AETHER_LIB} INTERFACE
  "$<$<COMPILE_LANGUAGE:CXX>:${AETHER_CXX_FLAGS}>")
# CUDA-only options
if (NOT AETHER_CPP_MODE)
  # The CUDA headers aether includes (vector_types.h, cuda_runtime.h) must also
  # reach plain C++ files that include aether, not only .cu files: nvcc finds
  # them on its own, the host compiler only through this include directory.
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(${AETHER_LIB} INTERFACE CUDA::toolkit)
  target_compile_options(${AETHER_LIB} INTERFACE
    "$<$<COMPILE_LANGUAGE:CUDA>:${AETHER_CUDA_FLAGS}>")
  target_link_options(${AETHER_LIB} INTERFACE
    "$<$<COMPILE_LANGUAGE:CUDA>:${AETHER_CUDA_FLAGS}>")
endif()
###########################

# A header-only library must be includable from a fresh checkout WITHOUT
# running CMake at all, so
# aether/version.h is a COMMITTED header (no configure_file, no generated
# file in the source tree). AETHER_VERSION/AETHER_ABI are literals in that
# header, kept in sync with project(VERSION) by
# VersionTest.VersionMatchesCMakeProjectVersion (tests/CMakeLists.txt feeds
# the CMake value through a compile definition). Only the git revision is
# configure-time-computed: `git rev-parse HEAD` ("unknown" if absent),
# threaded to consumers as an INTERFACE compile definition
# (AETHER_GIT_REVISION) on the aether target itself — so a normal CMake
# consumer (in-tree test, or a downstream `find_package(aether)`, which
# picks it up via the exported target's INTERFACE_COMPILE_DEFINITIONS) sees
# the real sha, while a bare `-I<repo_root>` compile with no CMake
# (tests/headers/check_header_diet.sh) sees no such macro and
# aether::gitRevision() falls back to the "unknown" literal in the header
# itself — see aether/version.h.
set(AETHER_VERSION "${PROJECT_VERSION}")
find_package(Git QUIET)
set(AETHER_GIT_REVISION "unknown")
if (Git_FOUND)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE _AETHER_GIT_REV
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE _AETHER_GIT_RC)
  if (_AETHER_GIT_RC EQUAL 0 AND _AETHER_GIT_REV)
    set(AETHER_GIT_REVISION "${_AETHER_GIT_REV}")
  endif()
endif()
target_compile_definitions(${AETHER_LIB} INTERFACE AETHER_GIT_REVISION="${AETHER_GIT_REVISION}")
###########################

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

###########################

# Tests — enable_testing() lives HERE, at the ROOT, gated on AETHER_BUILD_TESTS.
# CMake only emits the root CTestTestfile.cmake (the one carrying the
# `subdirs("tests")` link that makes `ctest --test-dir build` see anything
# at all) when `enable_testing()` is called at the TOP level — calling it
# inside tests/CMakeLists.txt instead makes the root ctest silently report
# "Total Tests: 0" and exit 0. aether avoids that trap by calling it here.
# (ctest still is NOT the primary gate — tests/check_gate.sh is — but the
# root CTestTestfile.cmake this enables is at least non-vacuous.)
if (AETHER_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# The `sanitize` custom target (see tests/sanitize/
# README.md). A separate gate from AETHER_BUILD_TESTS's own block above
# because it needs the aether_tests TARGET to already exist
# ($<TARGET_FILE:aether_tests>, tests/sanitize/CMakeLists.txt).
if (AETHER_BUILD_TESTS AND AETHER_BUILD_SANITIZE)
  add_subdirectory(tests/sanitize)
endif()

###########################

# Installation voodoo. This allows
# `make install`, and downstream libraries can do
# `find_package(aether CONFIG REQUIRED)` as long as CMAKE_PREFIX_PATH
# includes the install prefix used when configuring aether.

# Copy includes to install location
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/aether" DESTINATION "include")
# Configure targets to be exported (in this case just our header lib)
install(TARGETS ${AETHER_LIB}
    EXPORT ${PROJECT_NAME}_Targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# Create version config file aetherConfigVersion.cmake
include(CMakePackageConfigHelpers)
write_basic_package_version_file("${PROJECT_NAME}ConfigVersion.cmake"
                                 VERSION ${PROJECT_VERSION}
                                 COMPATIBILITY SameMajorVersion)

# Create package config file aetherConfig.cmake
configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/${PROJECT_NAME}Config.cmake.in"
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION
  ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)

# Create target config file aetherTargets.cmake at install location
install(EXPORT ${PROJECT_NAME}_Targets
        FILE ${PROJECT_NAME}Targets.cmake
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)

# Finally, copy cmake config files to install location
install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
              "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)

###########################

# Uninstall target
add_custom_target("uninstall" COMMENT "Uninstall installed files")
add_custom_command(
    TARGET "uninstall"
    POST_BUILD
    COMMENT "Uninstall files with install_manifest.txt"
    COMMAND xargs rm -vf < install_manifest.txt || echo Nothing in
            install_manifest.txt to be uninstalled!
)
