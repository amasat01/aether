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

# Passed to CMake variable CMAKE_CUDA_HOST_COMPILER when set (the gate sets $CXX)
cuda_host ?=

# Passed to CMake variable AETHER_WARN_AS_ERR (the gate sets ON)
warn ?= OFF

# Passed to CMake variable AETHER_SANITIZERS when set, e.g. address,undefined
sanitizers ?=

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
		-DAETHER_BUILD_TESTS=$(test) -DAETHER_WARN_AS_ERR=$(warn) $(if $(sanitizers),-DAETHER_SANITIZERS=$(sanitizers)) -DAETHER_DEBUG_MODE=$(debug) -DAETHER_CPP_MODE=$(cpp) \
		-DCMAKE_INSTALL_PREFIX=$(install_path) .
else
ifneq ($(strip $(nvcc)),)
CONFIGURE_CMD = cmake -B $(builddir) -DCMAKE_CUDA_COMPILER=$(nvcc) $(if $(cuda_host),-DCMAKE_CUDA_HOST_COMPILER=$(cuda_host)) $(ARCH_FLAG) \
		-DFETCHCONTENT_QUIET=ON \
		-DAETHER_BUILD_TESTS=$(test) -DAETHER_WARN_AS_ERR=$(warn) $(if $(sanitizers),-DAETHER_SANITIZERS=$(sanitizers)) -DAETHER_DEBUG_MODE=$(debug) -DAETHER_CPP_MODE=$(cpp) \
		-DCMAKE_INSTALL_PREFIX=$(install_path) .
else
# Fallback gracefully to CPU-only if nvcc is not available
CONFIGURE_CMD = cmake -B $(builddir) \
		-DFETCHCONTENT_QUIET=ON \
		-DAETHER_BUILD_TESTS=$(test) -DAETHER_WARN_AS_ERR=$(warn) $(if $(sanitizers),-DAETHER_SANITIZERS=$(sanitizers)) -DAETHER_DEBUG_MODE=$(debug) -DAETHER_CPP_MODE=ON \
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


# Local pre-release sanitizer gate (CUDA mode, GPU $(san_gpu)). Rows: compute-sanitizer memcheck /
# racecheck / synccheck and valgrind, each THROUGH tests/check_gate.sh via an exec wrapper
# (listing + run + manifest + tool exit in one command), initcheck triage-only, then the
# device canaries must go red. Prints its log dir; record it with the suppressed count in the
# release ledger. Needs a CUDA-mode build (cpp=OFF). The CPP_MODE sanitizer leg is CI's
# `sanitize-cpp` job (-DAETHER_SANITIZERS=address,undefined).
san_gpu ?= 1
.PHONY: sanitize-gate
sanitize-gate: configure
	$(mkcmd) aether_tests aether_canary_cu
	CUDA_VISIBLE_DEVICES=$(san_gpu) tests/sanitize/sanitize_gate.sh $(builddir)

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

# ------------------------------------------------------------------------------------------------
# The stateless gate: `make gate` (the CPU legs CI runs), `make gate-gpu` (needs a GPU), `make gate-all`.
# CI calls these targets. Everything is built under $(GATE_DIR) (gitignored) with a fresh venv made by
# $(PYTHON); GATE_REUSE=1 keeps the venv for the inner loop. Siblings are inputs, never guessed:
# RAPTOR_SRC (else PyPI), EAGLE_SRC (the layout header the payload seal reads; gate-wheel needs it),
# SITE_SRC (a checkout of amasat01.github.io; gate-docs needs it). Toolchains come from the environment and the gate fails loudly
# without them: CC and CXX (a C++23 GCC; CI uses gcc-13), CUDA_PATH (a CUDA 12 toolkit, for gate-clang and
# gate-gpu), clang and clang++ on PATH (gate-clang). Each leg is a target: `make gate-<leg>`.
# On a GPU machine run `make gate-gpu-build`, then `make gate-gpu` under the GPU lock.
# ------------------------------------------------------------------------------------------------
GATE_DIR    ?= build/gate
PYTHON      ?= python3
GATE_JOBS   ?= 2
GATE_REUSE  ?=
GATE_NECESSITY ?=
gate_debug  ?= ON OFF
PYTEST_ARGS ?=
RAPTOR_SRC  ?=
EAGLE_SRC   ?=
SITE_SRC    ?=
CUDA_PATH   ?=
gate_venv    = $(GATE_DIR)/venv
gate_py      = $(CURDIR)/$(gate_venv)/bin/python
gate_env     = export CMAKE_BUILD_PARALLEL_LEVEL=$(GATE_JOBS) PATH="$(CURDIR)/$(gate_venv)/bin:$$PATH"
# the tests that run against the INSTALLED wheel (the others read the source tree: dsc/aether has no _core)
gate_wheel_tests = test_arrays.py test_ft_payload.py test_fft.py test_openmp.py
gate_sub     = $(MAKE) j=$(GATE_JOBS) install_path=$(GATE_DIR)/prefix warn=ON

.PHONY: gate gate-gpu gate-gpu-build gate-all gate-venv gate-need-cxx gate-need-cuda
.PHONY: gate-names gate-lint gate-cpp gate-sanitize gate-hostgcc gate-clang gate-wheel gate-docs
gate: gate-lint gate-names gate-cpp gate-sanitize gate-hostgcc gate-clang gate-wheel gate-docs
gate-all: gate gate-gpu

# The throwaway environment: ruff (pinned in ci/constraints.txt) and pytest.
gate-venv:
	if [ -z "$(GATE_REUSE)" ] || [ ! -x "$(gate_py)" ]; then
	  rm -rf $(gate_venv)
	  mkdir -p $(GATE_DIR)
	  $(PYTHON) -m venv $(gate_venv)
	  $(gate_py) -m pip install --quiet -c ci/constraints.txt ruff pytest
	fi

gate-need-cxx:
	[ -n "$${CC:-}" ] && [ -n "$${CXX:-}" ] || { echo "gate: set CC and CXX to a C++23 compiler"; exit 1; }

gate-need-cuda:
	[ -n "$(CUDA_PATH)" ] && [ -x "$(CUDA_PATH)/bin/nvcc" ] || { echo "gate: set CUDA_PATH to a CUDA toolkit with bin/nvcc"; exit 1; }

# Every public name of the Python package is referenced by a test or on the shrink-only allowlist
gate-names:
	$(PYTHON) tools/public_names.py --src dsc --tests dsc/tests --allow ci/public_names_untested.txt --module aether --module aether.fft --module aether_dsc

# ruff, the C reference-core corpus and its C99 check, and the generator's own tests
gate-lint: gate-venv gate-need-cxx
	$(gate_env)
	$(gate_py) -m ruff check
	PYTHONPATH=dsc $(gate_py) tools/c_core/mint_golden.py --check
	tests/c_core/check_c99.sh
	PYTHONPATH=dsc $(gate_py) -m pytest -q -p no:cacheprovider dsc/tests/test_c_core.py $(PYTEST_ARGS)

# The CPP_MODE suite, debug on and off, warnings as errors, through the manifest gate
gate-cpp: gate-need-cxx
	for dbg in $(gate_debug); do
	  $(gate_sub) builddir=$(GATE_DIR)/cpp-$$dbg cpp=ON debug=$$dbg build test
	done

# The same suite under ASan and UBSan, with the non-vacuity checks
gate-sanitize: gate-need-cxx
	$(gate_sub) builddir=$(GATE_DIR)/san cpp=ON debug=ON sanitizers=address,undefined build
	tests/sanitize/cpp_gate.sh $(GATE_DIR)/san

# The host-vector-math reproducers with the compiler in $CXX (CI runs three GCC majors). The necessity step proves
# that THIS compiler carries the bug the rule works around: it holds for Ubuntu's g++-13/14 (CI sets GATE_NECESSITY=1
# there) and not for every vendor build (Red Hat's gcc-toolset-14.2.1 has it fixed), so it is not a default.
gate-hostgcc: gate-need-cxx
	tests/packetmath/check_hvm_masked.sh "$$CXX"
	if [ -n "$(GATE_NECESSITY)" ]; then tests/packetmath/check_hvm_masked.sh "$$CXX" --necessity; fi
	tests/hosttu/check_select_pta.sh "$$CXX"

# The arrays binding and its CUDA 12 plugin built with clang, zero warnings; the plugin exports only the seam
gate-clang: gate-venv gate-need-cuda
	$(gate_env)
	command -v clang >/dev/null && command -v clang++ >/dev/null || { echo "gate: clang and clang++ must be on PATH"; exit 1; }
	$(gate_py) -m pip install --quiet "nanobind>=3.1,<4" ninja
	CC=clang CXX=clang++ AETHER_CUDA_ROOT="$(CUDA_PATH)" cmake -S dsc -B $(GATE_DIR)/clang -G Ninja -DCMAKE_BUILD_TYPE=Release \
	  -DCMAKE_COMPILE_WARNING_AS_ERROR=ON -Dnanobind_DIR="$$($(gate_py) -c 'import nanobind; print(nanobind.cmake_dir())')" -DPython_EXECUTABLE="$(gate_py)"
	cmake --build $(GATE_DIR)/clang
	syms=$$(nm -D --defined-only $(GATE_DIR)/clang/libaether_cuda.so | awk '{print $$NF}')
	[ "$$syms" = aether_memory_v1 ] || { echo "the plugin exports more than the seam: $$syms"; exit 1; }

# The aether-dsc wheel (sealed payload, one OpenMP runtime) in a fresh venv, with every Python test
gate-wheel: gate-venv gate-need-cxx gate-need-cuda
	[ -n "$(EAGLE_SRC)" ] || { echo "gate: set EAGLE_SRC (the payload seal reads plugin/gref_layout.h)"; exit 1; }
	$(gate_env)
	$(gate_py) -m pip install --quiet cuda-bindings nvidia-cuda-nvrtc-cu12 nvidia-cuda-cccl-cu12 pyyaml
	( cd dsc && $(gate_py) -m aether_dsc.seal --layout-header "$(abspath $(EAGLE_SRC))/plugin/gref_layout.h" )
	PYTHONPATH=dsc $(gate_py) -m pytest -q -p no:cacheprovider dsc/tests/test_aether_dsc.py
	rm -rf $(GATE_DIR)/wheels $(GATE_DIR)/wheelenv
	export SETUPTOOLS_SCM_PRETEND_VERSION="$${SETUPTOOLS_SCM_PRETEND_VERSION:-0.0.0.dev0+g$$(git rev-parse --short HEAD)}"
	AETHER_CUDA_ROOT="$(CUDA_PATH)" $(gate_py) -m pip wheel --quiet --no-deps -w $(GATE_DIR)/wheels ./dsc
	PYTHON=$(PYTHON) RAPTOR_SRC="$(if $(RAPTOR_SRC),$(abspath $(RAPTOR_SRC)))" tools/wheel_check.sh $(GATE_DIR)/wheelenv $(GATE_DIR)/wheels dsc/tests $(or $(PYTEST_ARGS),$(gate_wheel_tests))

# Strict docs: manifesto copy, notebooks run, doxygen + Sphinx -W, every public name has a page
gate-docs: gate-venv
	[ -n "$(SITE_SRC)" ] || { echo "gate: set SITE_SRC to a checkout of amasat01.github.io (docs tools and capability data)"; exit 1; }
	$(gate_env)
	export RAPTOR_SITE="$(abspath $(SITE_SRC))"
	$(if $(RAPTOR_SRC),export MANIFESTO_SOURCE="$(abspath $(RAPTOR_SRC))/docs/content/principles.md")
	bash tools/manifesto_check.sh CONTRIBUTING.md
	grep -v '^raptor-docs' docs/requirements.txt > $(GATE_DIR)/docs-requirements.txt
	$(gate_py) -m pip install --quiet -r $(GATE_DIR)/docs-requirements.txt "$$RAPTOR_SITE"
	$(gate_py) -m raptor_docs.fresh --check --site "$$RAPTOR_SITE"
	$(MAKE) -C docs nbcheck
	$(MAKE) -C docs strict
	PYTHONPATH=dsc $(gate_py) tools/docs_inventory_check.py docs/_build/html/objects.inv aether_dsc --allow seal

# CUDA build (native architectures, warnings as errors) and the manifest gate on the GPU
gate-gpu-build: gate-need-cxx gate-need-cuda
	$(gate_sub) builddir=$(GATE_DIR)/cuda cpp=OFF nvcc=$(CUDA_PATH)/bin/nvcc cuda_host="$$CXX" debug=ON build

gate-gpu: gate-gpu-build
	$(gate_sub) builddir=$(GATE_DIR)/cuda cpp=OFF nvcc=$(CUDA_PATH)/bin/nvcc cuda_host="$$CXX" debug=ON test

.PHONY: clean
clean:
	rm -rf $(builddir)

# Special target that causes whole recipe to be run in one shell
.PHONY: .ONESHELL
.ONESHELL:
# Fail-fast for multi-line recipes under .ONESHELL: without -e, `test:`/`remint:` ran check_gate.sh
# against a STALE binary after a failed build and reported the gate's RC alone.
.SHELLFLAGS := -ec
