# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

# Subdirectory of cwd where build artifacts will be generated
builddir ?= build

# Parallel jobs to use for build and tests
j ?= 8

# Make command to use in the build directory
mkcmd ?= make --directory $(builddir) -j$(j) --no-print-directory

# Path to NVCC compiler
nvcc ?= $(shell which nvcc)

# Passed to the CMake variable AETHER_DEBUG_MODE
debug ?= ON

# Passed to CMake variable AETHER_BUILD_TESTS
test ?= ON

# Passed to CMake variable AETHER_CPP_MODE
cpp ?= OFF

# CUDA Architectures: native (the GPUs in this machine, the default), all (every
# architecture the nvcc in use compiles), or a list such as cudaarch="61;80".
# Passed to CMake variable AETHER_CUDA_ARCHS.
cudaarch ?= native
ARCH_FLAG = -DAETHER_CUDA_ARCHS="$(cudaarch)"

# Installation path used by dependencies
install_path ?= "${CONDA_PREFIX}"

# prefix path
prefix_path ?= "${CONDA_PREFIX}"

.PHONY: all
all: build test

.PHONY: build
build: configure
	$(mkcmd) all

# Decide the configure command at make-parse time (not inside a shell recipe)
ifeq ($(cpp),ON)
CONFIGURE_CMD = cmake -B $(builddir) \
		-DFETCHCONTENT_QUIET=ON \
		-DAETHER_BUILD_TESTS=$(test) -DAETHER_DEBUG_MODE=$(debug) -DAETHER_CPP_MODE=$(cpp) \
		-DCMAKE_INSTALL_PREFIX=$(install_path) .
else
ifneq ($(strip $(nvcc)),)
CONFIGURE_CMD = cmake -B $(builddir) -DCMAKE_CUDA_COMPILER=$(nvcc) $(ARCH_FLAG) \
		-DFETCHCONTENT_QUIET=ON \
		-DAETHER_BUILD_TESTS=$(test) -DAETHER_DEBUG_MODE=$(debug) -DAETHER_CPP_MODE=$(cpp) \
		-DCMAKE_INSTALL_PREFIX=$(install_path) .
else
# Fallback gracefully to CPU-only if nvcc is not available
CONFIGURE_CMD = cmake -B $(builddir) \
		-DFETCHCONTENT_QUIET=ON \
		-DAETHER_BUILD_TESTS=$(test) -DAETHER_DEBUG_MODE=$(debug) -DAETHER_CPP_MODE=ON \
		-DCMAKE_INSTALL_PREFIX=$(install_path) .
endif
endif

$(builddir)/Makefile: CMakeLists.txt tests/CMakeLists.txt
		mkdir -p $(builddir)
		$(CONFIGURE_CMD)
.PHONY: configure
configure: $(builddir)/Makefile

.PHONY: reconfigure
reconfigure:
	rm -f $(builddir)/Makefile
	make configure

install: build
	$(mkcmd) install

uninstall:
	$(mkcmd) uninstall

# Runs the suite through the INTEGRITY GATE, never the binary bare.
# `./$(builddir)/tests/aether_tests` on its own is VACUOUS: with a filter that matches
# nothing (or an ambient GTEST_FILTER) gtest prints "[  PASSED  ] 0 tests." and exits 0,
# so the exit code cannot distinguish "everything passed" from "nothing ran". The gate
# compares BOTH an env-scrubbed listing AND the actual run against the committed name
# manifest in tests/expected_tests_{cuda,cpp}.txt. This is the SAME command CI runs, so
# a red is reproducible before pushing.
test: configure
	$(mkcmd) aether_tests
	tests/check_gate.sh $(if $(filter ON,$(cpp)),cpp,cuda) $(builddir)/tests/aether_tests

# Regenerate the name manifest FROM YOUR BUILD after adding or removing tests, and commit
# the manifest diff in the SAME commit as the tests. --allow-removals is required when the
# re-mint would DELETE lines. Refused when $CI is set: a gate that regenerates its own
# expectation is a tautology.
.PHONY: remint
remint: configure
	$(mkcmd) aether_tests
	tests/check_gate.sh $(if $(filter ON,$(cpp)),cpp,cuda) $(builddir)/tests/aether_tests --remint


# Full-corpus faithful-rounding gate for the CPU packet math
# (aether/backend/cpu/simd/math/): >= 2^20 inputs per function scored at every
# width against a 128-bit mpmath reference. The corpus is regenerated (needs
# python3 + mpmath, ~10 min at j=4) into $(golden_dir) unless already there,
# then checked byte for byte against the committed md5 list, so the scored
# reference is the pinned one whatever produced it. The checker runs twice:
# it must be GREEN as is, and must catch a planted 2-ULP error in every
# function (non-vacuity).
golden_dir ?= $(builddir)/faithful_golden
.PHONY: faithful-gate
faithful-gate: configure
	$(mkcmd) aether_faithful_check
	if ! ( cd $(golden_dir) 2>/dev/null && md5sum --quiet --strict -c $(CURDIR)/tests/packetmath/faithful_golden.md5 ); then
		python3 tools/packet_math_ref/gen_golden.py --out $(golden_dir) --jobs $(j)
		( cd $(golden_dir) && md5sum --quiet --strict -c $(CURDIR)/tests/packetmath/faithful_golden.md5 )
	fi
	$(builddir)/tests/aether_faithful_check $(golden_dir)
	$(builddir)/tests/aether_faithful_check $(golden_dir) --plant

# docs: doxygen (docs/Doxyfile.in -> docs/_doxybuild/html)
# THEN Sphinx (docs/conf.py + docs/content/*.rst -> docs/_build/html), both
# with warnings promoted to errors. Invoked as `python3 -m sphinx`, not the
# bare `sphinx-build` command: a user-site `~/.local/bin/sphinx-build` on
# PATH shadows the environment's pinned Sphinx (docs/requirements.txt) with
# an older one; `python3 -m sphinx` resolves through the active interpreter
# instead and reliably hits the pinned version.
.PHONY: docs
docs:
	( cd docs && doxygen Doxyfile.in )
	python3 -m sphinx -W -b html docs docs/_build/html

.PHONY: clean
clean:
	rm -rf $(builddir)

# Special target that causes whole recipe to be run in one shell
.PHONY: .ONESHELL
.ONESHELL:
# Fail-fast for multi-line recipes under .ONESHELL: without -e, `test:`/`remint:` ran check_gate.sh
# against a STALE binary after a failed build and reported the gate's RC alone.
.SHELLFLAGS := -ec
