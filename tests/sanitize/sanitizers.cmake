# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# AETHER_SANITIZERS (cache string, default empty = no effect): a -fsanitize= list, e.g.
# `address,undefined` for the CPP_MODE sanitize job. Appends
# `-fsanitize=<value> -fno-omit-frame-pointer -g` to compile + link of the test
# targets, and builds the argv-selected canaries in the SAME configure:
#   aether_canary     (host,  when AETHER_SANITIZERS is set)    tests/sanitize/canary.cpp
#   aether_canary_cu  (CUDA mode, with tests)                   tests/sanitize/canary.cu
# See tests/sanitize/README.md.

function(aether_apply_sanitizers target)
  if(NOT AETHER_SANITIZERS)
    return()
  endif()
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=${AETHER_SANITIZERS} -fno-omit-frame-pointer -g>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fsanitize=${AETHER_SANITIZERS},-fno-omit-frame-pointer>)
  target_link_options(${target} PRIVATE -fsanitize=${AETHER_SANITIZERS})
endfunction()

if(AETHER_SANITIZERS)
  add_executable(aether_canary ${CMAKE_CURRENT_LIST_DIR}/canary.cpp)
  aether_apply_sanitizers(aether_canary)
endif()

if(NOT AETHER_CPP_MODE)
  add_executable(aether_canary_cu ${CMAKE_CURRENT_LIST_DIR}/canary.cu)
endif()
