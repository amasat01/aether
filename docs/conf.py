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
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "_ext"))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dsc")))

# -- Project information -----------------------------------------------------

project = "aether"
copyright = "2026, Alessandro Masat"
author = "Alessandro Masat"

# Read version from CMakeLists.txt's project(aether VERSION x.y.z ...).
_version = "unknown"
_cmake = os.path.join(os.path.dirname(__file__), "..", "CMakeLists.txt")
_pattern = re.compile(r"project\s*\(\s*aether\s+VERSION\s+(\d+\.\d+\.\d+)", re.IGNORECASE)
try:
    with open(_cmake) as _f:
        for _line in _f:
            _m = _pattern.search(_line)
            if _m:
                _version = _m.group(1)
                break
except FileNotFoundError:
    pass

version = _version
release = _version

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
# raptor-tokens.css (kit) -> site-accent.css (this site's --accent) ->
# raptor-theme.css (shared skin, derives everything from --accent) ->
# raptor-reveal.css (kit).
html_css_files = ["raptor-tokens.css", "site-accent.css", "raptor-theme.css", "raptor-reveal.css"]

html_theme_options = {
    "repository_url": "https://github.com/amasat01/aether",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "collapse_navigation": True,
    # Colab + download launch buttons on notebook pages (no Binder: not configured).
    "launch_buttons": {
        "colab_url": "https://colab.research.google.com",
        "notebook_interface": "classic",
    },
    # One transparent logo file works on light AND dark pages (RAPTOR brand kit).
    "logo": {
        "image_light": "_static/brand/family_aether.svg",
        "image_dark": "_static/brand/family_aether.svg",
        "alt_text": "aether",
    },
    # Family frame: purple "part of RAPTOR" chip in the footer, every site
    # (raptor-theme.css's .raptor-family-chip; theme's own extension point).
    "extra_footer": (
        '<div class="raptor-family-chip">part of '
        '<a href="https://amasat01.github.io/">RAPTOR</a></div>'
    ),
}

# RAPTOR brand: favicons + home-screen icon. Leave html_favicon unset: the
# hook below writes the icon links itself (SVG where supported, favicon.ico
# for Safari/older tools, 180 px icon for iOS home screens).
_RAPTOR_ICONS = [
    ("icon", "brand/favicon.ico", 'sizes="any"'),
    ("icon", "brand/favicon.svg", 'type="image/svg+xml"'),
    ("apple-touch-icon", "brand/app_icon_180.png", ""),
]


def _raptor_icons(app, pagename, templatename, context, doctree):
    pathto = context.get("pathto")
    if pathto is None:
        return
    links = "".join(
        f'<link rel="{rel}" href="{pathto("_static/" + path, 1)}" {extra}>\n'
        for rel, path, extra in _RAPTOR_ICONS
    )
    context["metatags"] = context.get("metatags", "") + links


def setup(app):
    app.connect("html-page-context", _raptor_icons)
