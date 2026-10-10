# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.14)

# Only require CUDA when building in CUDA mode; allow pure C++ tests in CPP mode
if (AETHER_CPP_MODE)
  project(aether_tests LANGUAGES CXX)
else()
  project(aether_tests LANGUAGES CUDA CXX)
endif()

# Supress a warning in newer CMake versions when using FetchContent
if (POLICY CMP0135)
	cmake_policy(SET CMP0135 NEW)
endif()

# Download GoogleTest; the ambient conda env carries no GTest CMake config
# package, so it is fetched directly rather than found.
include(FetchContent)
message(STATUS "> Downloading dependency: GoogleTest")
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
# Prevent GTest from being installed along with aether
option(INSTALL_GTEST "Enable installation of googletest." OFF)
FetchContent_MakeAvailable(googletest)
message(STATUS "> Downloading dependency: GoogleTest - done")

include(GoogleTest)

if(AETHER_CPP_MODE)
  # C++ TESTING - .cpp TEST FILES
  # tests/smoke/ joins the same aether_tests binary, globbed in explicitly
  # alongside the flat tests/ tree.
  file(GLOB TEST_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/smoke/test_*.cpp
  )
else()
  # CUDA TESTING - .cu TEST FILES
  file(GLOB TEST_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/smoke/test_*.cu
  )
endif()

list(SORT TEST_FILES COMPARE NATURAL)

foreach(file ${TEST_FILES})
    get_filename_component(filename ${file} NAME_WE)
    message(STATUS "> Found test source " ${file})
endforeach()

list(LENGTH TEST_FILES TEST_FILES_LEN)
message(STATUS "> Found " ${TEST_FILES_LEN} " test sources")

# test_PacketExpr.{cpp,cu} compares the packet/SIMD path (explicit AVX2/SSE2
# intrinsics) against the scalar RecursiveAssign reference for bit-identity
# (memcmp exact, never a tolerance). Both sides compute the same `a + k*b`
# shape from plain `+`/`*`, and this toolchain contracts that into a
# hardware FMA by default (confirmed empirically for both sides, see the
# test file's own docstring) — -ffp-contract=off (host) / -fmad=false
# (device, a no-op here since this file carries no device code) makes the
# comparison decidable by construction rather than by which compiler
# happens to keep both sides fused in step.
if (AETHER_CPP_MODE)
  if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_PacketExpr.cpp)
    set_source_files_properties(test_PacketExpr.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
  endif()
else()
  if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_PacketExpr.cu)
    set_source_files_properties(test_PacketExpr.cu PROPERTIES COMPILE_OPTIONS "-Xcompiler=-ffp-contract=off;-fmad=false")
  endif()
endif()

# test_Bundle.{cpp,cu} compares the bundle/DeviceBundle path against the
# scalar path for bit-identity, same trap as test_PacketExpr above — the
# quat-sandwich chain (QuatMul/QuatConj Hamilton products, sums-of-products)
# does get contracted into a hardware FMA under -march=x86-64-v3 -O3
# (confirmed empirically: diverges without this flag, exact with it — see
# test_Bundle.cpp's docstring), even though the bundle and scalar sides run
# the identical node formulas over different leaf types (DeviceBundle vs
# Item).
if (AETHER_CPP_MODE)
  if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_Bundle.cpp)
    set_source_files_properties(test_Bundle.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
  endif()
else()
  if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_Bundle.cu)
    set_source_files_properties(test_Bundle.cu PROPERTIES COMPILE_OPTIONS "-Xcompiler=-ffp-contract=off;-fmad=false")
  endif()
endif()

# test_ContractionGuard.{cpp,cu} prove the host error-free transformations
# (aether/accum/atomic.h) exact UNDER contraction, so they are built with
# -ffp-contract=fast explicitly (gcc's default, pinned here so a changed
# default cannot make the test vacuous; the file also checks its own TU
# contracts).
if (AETHER_CPP_MODE)
  set_source_files_properties(test_ContractionGuard.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=fast")
else()
  set_source_files_properties(test_ContractionGuard.cu PROPERTIES COMPILE_OPTIONS "-Xcompiler=-ffp-contract=fast")
endif()

# smoke/test_MiniRK.cu gates its DeviceBundle arm bit-exact against the
# scalar Item arm. Both arms call the same rk4Step, but that only fixes the
# rounding while the compiler keeps one shared out-of-line copy: once
# rk4Step is inlined into each arm, the host compiler contracts `a*b + c`
# into FMAs per call site, and the two inlined copies need not fuse the
# same products (confirmed: diverges in the last bit at -O3
# -march=x86-64-v3, exact with contraction off or with rk4Step noinline).
# Same remedy as test_Bundle above. Only the CUDA file carries that arm;
# test_MiniRK.cpp is gated against its reference with a tolerance.
if (NOT AETHER_CPP_MODE)
  if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/smoke/test_MiniRK.cu)
    set_source_files_properties(smoke/test_MiniRK.cu PROPERTIES COMPILE_OPTIONS "-Xcompiler=-ffp-contract=off;-fmad=false")
  endif()
endif()

# THE HOST-COMPILER TRANSLATION UNIT. In the CUDA build only, one extra
# source is compiled by the host compiler (g++) and linked into the same
# aether_tests binary as the nvcc-compiled .cu files: a CUDA-mode aether
# must be includable from a plain .cpp, and both compilers must agree on
# every type's layout — test_HostTuLayout.cu compares the two
# sizeof/alignof tables and passes g++-built objects into an
# nvcc-launched kernel.
#
# NOT named `test_*.cpp` on purpose: the CPP_MODE branch above globs
# `test_*.cpp`, and this harness only means something where the two
# compilers coexist. Every gtest case lives in the `.cu` partner, so only
# tests/expected_tests_cuda.txt gains rows.
#
# LANGUAGE CXX is set explicitly rather than relied on: it is what stops
# CMake from handing the file to nvcc if CUDA ever becomes the directory
# default, and handing it to nvcc would silently destroy the test's subject.
# It also must NOT inherit the CUDA-only flags applied further down.
if (NOT AETHER_CPP_MODE)
  set(AETHER_HOST_TU_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/hosttu/host_tu_probe.cpp)
  set_source_files_properties(${AETHER_HOST_TU_SOURCE} PROPERTIES LANGUAGE CXX)
  list(APPEND TEST_FILES ${AETHER_HOST_TU_SOURCE})
  message(STATUS "> host-compiler TU: " ${AETHER_HOST_TU_SOURCE})
endif()

# CPU packet math (aether/backend/cpu/simd/math/) accuracy gate: the packet
# code is evaluated in host-compiled kernel TUs, each built for ONE ISA so
# every width is exercised regardless of the suite's -march (widths 1+4 at
# the default target, width 8 with -mavx512f added and run only on CPUs that
# report AVX-512F, width 2 on plain x86-64 = SSE2 without FMA), plus the
# AETHER_HOST_VECTOR_MATH hook TU. test_PacketMath.{cpp,cu} (globbed above)
# holds the corpus, the std:: reference and the ULP checks. Joined in BOTH
# build modes, always as CXX (never handed to nvcc), and all with
# -ffp-contract=off so lanes at different widths stay bit-identical — except
# the hook TU, built with -ffp-contract=fast on purpose: the hook's symbols
# must be contraction-free whatever the consumer's TU asks for (test
# PacketMathHostVectorHook.HookUnderContractionMatchesKernels).
if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  set(AETHER_PM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/packetmath)
  set(AETHER_PM_SOURCES
    ${AETHER_PM_DIR}/pm_kernels_w1_w4.cpp
    ${AETHER_PM_DIR}/pm_kernels_w8.cpp
    ${AETHER_PM_DIR}/pm_kernels_w2.cpp
    ${AETHER_PM_DIR}/pm_hook.cpp)
  set_source_files_properties(${AETHER_PM_SOURCES} PROPERTIES LANGUAGE CXX)
  set_source_files_properties(${AETHER_PM_DIR}/pm_kernels_w1_w4.cpp PROPERTIES
    COMPILE_OPTIONS "-ffp-contract=off")
  set_source_files_properties(${AETHER_PM_DIR}/pm_kernels_w8.cpp PROPERTIES
    COMPILE_OPTIONS "-ffp-contract=off;-mavx512f")
  set_source_files_properties(${AETHER_PM_DIR}/pm_kernels_w2.cpp PROPERTIES
    COMPILE_OPTIONS "-ffp-contract=off;-march=x86-64;-mtune=generic")
  set_source_files_properties(${AETHER_PM_DIR}/pm_hook.cpp PROPERTIES
    COMPILE_OPTIONS "-ffp-contract=fast;-fopenmp-simd"
    COMPILE_DEFINITIONS "AETHER_HOST_VECTOR_MATH")
  list(APPEND TEST_FILES ${AETHER_PM_SOURCES})
  message(STATUS "> packet-math kernel TUs: " "${AETHER_PM_SOURCES}")
endif()

add_executable(aether_tests ${TEST_FILES})
target_link_libraries(aether_tests aether GTest::gtest_main)
target_include_directories(aether_tests PRIVATE ../)
# The packet-math faithful-rounding gate's committed hard-case set (read at
# run time by test_PacketMath_common.h).
target_compile_definitions(aether_tests PRIVATE
  AETHER_PM_HARD_SET="${CMAKE_CURRENT_SOURCE_DIR}/packetmath/faithful_hard.bin")

# Full-corpus faithful-rounding checker for the packet math (>= 1M inputs per
# function against a 128-bit mpmath reference regenerated by
# tools/packet_math_ref/gen_golden.py and pinned by md5): not part of `all`,
# built and run by `make faithful-gate`. Same kernel TUs (and per-file
# flags) as aether_tests.
if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  set(AETHER_PM_CHECK_MAIN ${AETHER_PM_DIR}/faithful_check_main.cpp)
  set_source_files_properties(${AETHER_PM_CHECK_MAIN} PROPERTIES LANGUAGE CXX COMPILE_OPTIONS "-ffp-contract=off")
  add_executable(aether_faithful_check EXCLUDE_FROM_ALL ${AETHER_PM_CHECK_MAIN}
    ${AETHER_PM_DIR}/pm_kernels_w1_w4.cpp ${AETHER_PM_DIR}/pm_kernels_w8.cpp ${AETHER_PM_DIR}/pm_kernels_w2.cpp)
  target_link_libraries(aether_faithful_check aether)
  target_include_directories(aether_faithful_check PRIVATE ../)
  target_compile_options(aether_faithful_check PRIVATE -Wall -Wextra -O3)
endif()

# test_Version.{cpp,cu} reads the expected version string at test-configure
# time via a compile definition rather than re-deriving it — AETHER_VERSION
# is the same variable root CMakeLists.txt's configure_file(aether/version.h.in
# ...) feeds into aether/version.h, so this is the identical value by
# construction, not a second, driftable copy. Not ${PROJECT_VERSION}: this
# file's own project(aether_tests ...) call above (no VERSION argument)
# shadows the root project's PROJECT_VERSION with an empty string in this
# scope.
target_compile_definitions(aether_tests PRIVATE AETHER_TEST_EXPECTED_VERSION="${AETHER_VERSION}")

# CUDA device TUs are capped at C++20 (nvcc's device-code ceiling); fix it
# explicitly on the test binary target so the requirement holds regardless
# of what the directory-scope CMAKE_CUDA_STANDARD default is.
if (NOT AETHER_CPP_MODE)
  set_target_properties(aether_tests PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON)
endif()

# --- ctest is deliberately not wired up here. Every gate here invokes the
# binary directly through check_gate.sh, which compares both an
# env-scrubbed listing AND the actual run against the committed name
# manifest in tests/expected_tests_{cuda,cpp}.txt — a count alone (what a
# bare `./aether_tests` or a naive ctest wiring would give you) cannot
# distinguish "everything passed" from "a filter silently ran nothing".

if (AETHER_CPP_MODE)
target_compile_options(
  aether_tests PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3 -Wsign-compare -ftemplate-backtrace-limit=100>
)
else()
target_compile_options(
  aether_tests PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Wall -Wextra -O3 -Wsign-compare -ftemplate-backtrace-limit=100 -Wno-deprecated-gpu-targets>
)
# The same warning discipline for the one host-compiled TU in this
# build — the CUDA generator expression above never reaches it.
target_compile_options(
  aether_tests PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3 -Wsign-compare -ftemplate-backtrace-limit=100>
)
endif()

# Sanitizer flags (empty AETHER_SANITIZERS = no-op) go LAST so their -O1 follows the -O3 above;
# the host/device canaries are built in the same configure. See sanitize/README.md.
include(${CMAKE_CURRENT_SOURCE_DIR}/sanitize/sanitizers.cmake)
aether_apply_sanitizers(aether_tests)
