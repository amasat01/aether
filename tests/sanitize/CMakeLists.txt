# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

# Sanitize custom target: runs valgrind (+ compute-sanitizer when available)
# over the already-built `aether_tests` binary via the python driver.
# Only registered when AETHER_BUILD_SANITIZE=ON (root CMakeLists.txt).
#
# WIRING ONLY: this target is REGISTERED, never RUN automatically — no
# sanitizer run is part of any gate here.

find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_custom_target(sanitize
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/run_sanitize.py
            --binary $<TARGET_FILE:aether_tests>
            --tool ${AETHER_SANITIZE_TOOL}
            --suppressions ${CMAKE_CURRENT_SOURCE_DIR}/suppressions
            --log-dir ${CMAKE_BINARY_DIR}/sanitize
            --project-name AETHER
            $<$<BOOL:${AETHER_SANITIZE_MINIMAL}>:--minimal>
            $<$<BOOL:${AETHER_SANITIZE_FAIL_ON_REACHABLE}>:--fail-on-reachable>
    DEPENDS aether_tests
    USES_TERMINAL
    COMMENT "Running sanitize sweep on aether_tests"
    VERBATIM
)
