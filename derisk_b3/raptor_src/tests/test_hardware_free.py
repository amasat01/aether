# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The contracts package holds no hardware: no architecture name, no vendor, no machine, and no measured row."""

import re
from pathlib import Path

import raptor.ir

_ROOT = Path(raptor.ir.__file__).parent
_HARDWARE = re.compile(r"sm_[0-9]|cuda|nvidia|xeon|p2000", re.IGNORECASE)
_TABLES = ("_ROWS", "CLASSES", "ARCH_FEATURES", "DEFAULT_ARCH")


def test_no_hardware_name_under_raptor_ir():
    hits = [f"{p.name}:{n}: {line.strip()[:60]}" for p in sorted(_ROOT.rglob("*.py"))
            for n, line in enumerate(p.read_text().splitlines(), 1) if _HARDWARE.search(line)]
    assert not hits, hits


def test_no_measured_table_or_machine_default_under_raptor_ir():
    hits = [f"{p.name}: {name}" for p in sorted(_ROOT.rglob("*.py")) for name in _TABLES
            if re.search(rf"^{name}\b", p.read_text(), re.MULTILINE)]
    assert not hits, hits
