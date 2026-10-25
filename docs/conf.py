# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

# docs/conf.py — Sphinx configuration for aether.
#
# Wired to `docs/Makefile` (`make html`/`make strict`), which runs `doxygen`
# first (Doxyfile.in -> docs/_doxybuild/xml) and then Sphinx. The C++ API
# reference is rendered as breathe pages under content/api/ (not linked
# Doxygen HTML). aether itself ships no Python package, but its sealed
# Python payload, `aether_dsc` (../dsc), gets a short autosummary page —
# hence the `sys.path` insert below.
#
# The pedagogic pages live under content/ and are toctree'd from index.md.

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "_ext"))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dsc")))

# -- Project information -----------------------------------------------------

project = "aether"
copyright = "2026, Alessandro Masat"
author = "Alessandro Masat"

# The version is the latest git tag (setuptools-scm), never written into the tree.
import setuptools_scm

release = version = setuptools_scm.get_version(root="..", relative_to=__file__, fallback_version="0.0.0")

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx_book_theme",
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "myst_nb",       # markdown pages + executed notebooks
    "breathe",       # Doxygen (C++ API) integration
    "raptor_docs.sphinx",  # family skin: fonts, css, favicons, theme defaults (pip install -r requirements.txt)
    # Registers `.. aether-example::` / `.. aether-example-end::`, the
    # marker pair tests/docs/check_examples.py scans for (docs/_ext/
    # aether_example.py) — no-op at render time.
    "aether_example",
]

# MyST options — math and rich fencing for the landing/narrative .md pages.
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

# -- Notebook execution (myst_nb) ---------------------------------------------
# Tutorials/examples are executed LOCALLY (`make nbexec`, GPU 1 in this
# workspace) and committed WITH their outputs; the docs build itself never
# re-executes them (`nb_execution_mode = "off"`) — `make nbcheck` is the gate
# that refuses an unfilled notebook before a publish.
nb_execution_mode = "off"
nb_execution_timeout = 120
nb_merge_streams = True
nb_output_stderr = "show"
nb_render_markdown_format = "myst"

# -----------------------------------------------------------------------------
# Python API reference (autosummary/autodoc) — aether_dsc only; aether's C++
# surface is breathe, not autodoc.
# -----------------------------------------------------------------------------

autosummary_generate = True
autosummary_imported_members = False
napoleon_google_docstring = True
napoleon_include_special_with_doc = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "show-inheritance": True,
    "exclude-members": "__weakref__",
}

# -----------------------------------------------------------------------------
# Intersphinx is loaded (for parity with the rest of the family) but left
# empty: no page here uses a cross-project role, and intersphinx fetches
# every configured inventory eagerly at build start, so an entry here
# would make `make strict` depend on a live, reachable URL at build time —
# a sibling (eagle/raptor) whose site does not exist yet until the
# family's first publish would make that worse, not better. Cross-family links on this
# site are plain URLs (see content/interop.rst).
# -----------------------------------------------------------------------------

intersphinx_mapping: dict = {}

# -----------------------------------------------------------------------------
# Breathe (C++ API from Doxygen XML)
# -----------------------------------------------------------------------------

breathe_projects = {"aether": os.path.abspath(os.path.join(os.path.dirname(__file__), "_doxybuild", "xml"))}
breathe_default_project = "aether"
breathe_default_members = ("members",)

templates_path = ["_templates"]
# overview.md is Doxygen's own USE_MDFILE_AS_MAINPAGE source (see
# Doxyfile.in) — excluded from Sphinx's source discovery so it is not
# rendered as an orphan page once myst_nb makes .md a Sphinx suffix too.
exclude_patterns = ["_templates", "_build", "_doxybuild", "_tools", "overview.md", "Thumbs.db", ".DS_Store"]

# Known, acknowledged doxygen/breathe limitation (see
# docs/doxygen_allowlist.txt's own comment on aether::math::fma and the
# bundle loader): doxygen 1.9.1's C++ parser does not track a C++20
# requires-clause written on its own line ahead of a declaration, which
# breathe then fails to re-parse as a C++ domain declaration. Not a wording
# fix — it is a parser limitation in the upstream tools, not this page's
# docstrings.
suppress_warnings = ["cpp.parse"]

# -- Options for HTML output --------------------------------------------------

html_theme = "sphinx_book_theme"
html_title = f"aether — one array, built once, runs on CPU or GPU ({version})"

html_static_path = ["_static"]
# The family skin and its css order come from raptor_docs.sphinx; this site only adds its accent.
html_css_files = ["site-accent.css"]

html_theme_options = {
    "repository_url": "https://github.com/amasat01/aether",
    # One transparent logo file works on light AND dark pages (RAPTOR brand kit).
    "logo": {
        "image_light": "_static/brand/family_aether.svg",
        "image_dark": "_static/brand/family_aether.svg",
        "alt_text": "aether",
    },
}
