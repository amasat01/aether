# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""How a body names the slot a plane or a scalar is read through: the identifier a writer and a form's text share."""

from __future__ import annotations

#: Role -> the identifier prefix its binding carries. A prefix, not the
#: bare name, since ``(role, name)`` is the slot identity.
IDENT_PREFIX = {
    "mutable": "mut", "out": "out", "wide_out": "wout", "accum_out": "aout",
    "vec_in": "vin", "mat_in": "mtx", "wide_in": "win", "per_sample": "psc",
    "lookup": "lut", "terminated": "trm", "uniform": "uni", "nsamples": "nsm",
}


def binding_name(role: str, name: str) -> str:
    """The identifier the body reads a slot through (a view, or a by-value scalar)."""
    return f"{IDENT_PREFIX[role]}_{name}"
