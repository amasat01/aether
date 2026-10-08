# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# docs/Makefile — aether Sphinx + Doxygen build.
#
# `html`/`strict`/`linkcheck` all depend on `doxygen` (docs/Doxyfile.in ->
# _doxybuild/xml), which the breathe-based C++ API pages
# (content/api/*.rst) read. Invoked as `python3 -m sphinx`, not the bare
# `sphinx-build` command: a user-site `~/.local/bin/sphinx-build` on PATH
# can shadow the environment's pinned Sphinx (requirements.txt) with an
# older one.
#
# Notebooks (content/tutorials/, content/examples/) are executed LOCALLY
# (`make nbexec`) and committed WITH outputs; `nb_execution_mode = "off"`
# in conf.py means the docs build itself never re-executes them. `make
# nbcheck` refuses a build unless every code cell in every notebook has
# run without error.

SPHINXOPTS    ?=
SPHINXBUILD   ?= python3 -m sphinx
SOURCEDIR     = .
BUILDDIR      = _build
DOXYBUILDDIR  = _doxybuild
NB_DIRS       = content/tutorials content/examples

.PHONY: help html strict linkcheck doxygen nbexec nbcheck clean

help:
	@echo "Targets: html strict linkcheck doxygen nbexec nbcheck clean"

doxygen:
	doxygen Doxyfile.in

html: doxygen
	$(SPHINXBUILD) -b html $(SPHINXOPTS) "$(SOURCEDIR)" "$(BUILDDIR)/html"

# `-W --keep-going`: every warning is an error, and the build keeps going
# so one run reports ALL of them rather than stopping at the first.
strict: doxygen
	$(SPHINXBUILD) -b html -W --keep-going $(SPHINXOPTS) "$(SOURCEDIR)" "$(BUILDDIR)/html"

linkcheck: doxygen
	$(SPHINXBUILD) -b linkcheck $(SPHINXOPTS) "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck"

# Execute every tutorial/example notebook IN PLACE so the filled notebooks
# are what gets committed (and what CI renders with nb_execution_mode =
# "off" — one execution per notebook, done locally, not in CI).
nbexec:
	@for d in $(NB_DIRS); do \
	  find "$$d" -name '*.ipynb' -not -path '*/.ipynb_checkpoints/*' -print0 2>/dev/null | \
	    xargs -0 -r -n1 jupyter nbconvert --to notebook --execute --inplace \
	      --ExecutePreprocessor.timeout=120; \
	done

# Refuse to proceed unless every code cell in every tutorial/example
# notebook has actually run (execution_count set, no empty/placeholder
# notebook slipped in unexecuted).
nbcheck:
	python3 _tools/nb_check_filled.py $(NB_DIRS)

clean:
	rm -rf $(BUILDDIR) $(DOXYBUILDDIR) content/api/generated content/python/generated
