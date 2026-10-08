# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

# tools/ulp_oracle/CMakeLists.txt — standalone build, deliberately not
# wired into aether's root CMakeLists.txt: this tool's whole point is to
# run independently of the aether/ tree, without requiring a full aether
# build. Configure/build it from a scratch directory outside
# aether/build*, e.g.:
#
#   cmake -S aether/tools/ulp_oracle -B /path/to/scratch/ulp_oracle_build
#   cmake --build /path/to/scratch/ulp_oracle_build
#
# See README.md for usage of the two resulting binaries (ulp_oracle,
# selftest).
cmake_minimum_required(VERSION 3.20)
project(ulp_oracle LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# MPFR / GMP live in the conda env for this workspace
# ($CONDA_PREFIX/include/mpfr.h, $CONDA_PREFIX/lib/libmpfr.so). Honor an
# explicit MPFR_ROOT override first (e.g. for a from-source MPFR), fall back
# to $CONDA_PREFIX, then let find_path/find_library search the usual system
# locations.
if(NOT DEFINED MPFR_ROOT AND DEFINED ENV{CONDA_PREFIX})
  set(MPFR_ROOT "$ENV{CONDA_PREFIX}")
endif()

find_path(MPFR_INCLUDE_DIR mpfr.h HINTS "${MPFR_ROOT}/include")
find_library(MPFR_LIBRARY NAMES mpfr HINTS "${MPFR_ROOT}/lib")
find_library(GMP_LIBRARY NAMES gmp HINTS "${MPFR_ROOT}/lib")

if(NOT MPFR_INCLUDE_DIR OR NOT MPFR_LIBRARY OR NOT GMP_LIBRARY)
  message(FATAL_ERROR
    "ulp_oracle needs MPFR + GMP (found include=${MPFR_INCLUDE_DIR} "
    "mpfr=${MPFR_LIBRARY} gmp=${GMP_LIBRARY}) — set -DMPFR_ROOT=... or "
    "activate the conda env that has them (CONDA_PREFIX).")
endif()

message(STATUS "> ulp_oracle: MPFR include=${MPFR_INCLUDE_DIR} lib=${MPFR_LIBRARY} gmp=${GMP_LIBRARY}")

add_library(ulp_oracle_core STATIC oracle_core.cpp)
target_include_directories(ulp_oracle_core PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${MPFR_INCLUDE_DIR}")
target_link_libraries(ulp_oracle_core PUBLIC "${MPFR_LIBRARY}" "${GMP_LIBRARY}")
target_compile_options(ulp_oracle_core PRIVATE -Wall -Wextra -O2)

add_executable(ulp_oracle ulp_oracle.cpp)
target_link_libraries(ulp_oracle PRIVATE ulp_oracle_core)
target_compile_options(ulp_oracle PRIVATE -Wall -Wextra -O2)

add_executable(selftest selftest.cpp)
target_link_libraries(selftest PRIVATE ulp_oracle_core m)
target_compile_options(selftest PRIVATE -Wall -Wextra -O2)

enable_testing()
add_test(NAME ulp_oracle_selftest COMMAND selftest)
