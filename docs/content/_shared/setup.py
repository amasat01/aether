# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# Run by every notebook's hidden first cell as `%run ../_shared/setup.py`
# -- IPython's `%run` executes this file and then merges its names into
# the notebook's own namespace, which is what makes that a single line:
# no `sys.path`/`import` plumbing left for a lesson cell to show.
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cpp_harness import DEVICE, plot_style, run_cpp, run_or_show, split_data  # noqa: F401

print("running on:", DEVICE)
