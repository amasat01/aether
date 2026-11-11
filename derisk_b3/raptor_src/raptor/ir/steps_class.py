# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The rule that decides when a fused K-step reverse beats K step-by-step launches (``steps_reverse="auto"``).

The rows are an ARGUMENT: measured data is the compiler's, and a target with no row answers ``per_step``. A row is
``(state width, K, narrowest regime, widest regime)`` the fused launch won across, the regime being the batch per unit of
parallel capacity the caller reads from its target. The nearest measured width and K (log2 distance) is matched and fused is
chosen inside its regime."""

from __future__ import annotations

import math


def choose(rows, width: int, K: int, regime: float) -> str:
    """``"fused"`` when the row nearest ``(width, K)`` won at ``regime``, else ``"per_step"``."""
    if not rows:
        return "per_step"
    w, k, lo, hi = min(rows, key=lambda r: (abs(math.log2(width / r[0])) + abs(math.log2(K / r[1])), r))
    return "fused" if lo <= regime <= hi else "per_step"
