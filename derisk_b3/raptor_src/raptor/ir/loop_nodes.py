# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The loop family of IR nodes: a lowered bounded ``for`` and the
placeholders and reads its body uses."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .types import TensorType
from .nodes import HawkError, Loop, LoopCarry, LoopCount, LoopIndex, LoopValue, Node, Op, Sink, TapeRead, plain_in, plain_out  # LoopIndex, LoopCarry: generated from core.json

# --------------------------------------------------------------------------
# The LOOP family.
#
# A bounded ``for`` is lowered, not unrolled: unrolling blew ptxas's memory
# on a large cell, and unrolling is a transpiler's answer, not a code
# generator's. The body is traced ONCE against a symbolic index with
# explicit loop-carried values, and the emitter writes a plain ``for`` —
# whether it unrolls is the C++ compiler's decision, not HAWK's.
#
# The body is NOT in ``operands``: a loop body is a SCOPE, evaluated once
# per iteration inside braces, so hoisting its nodes into the flat walk
# order would re-unroll it. A :class:`Loop` carries its body in fields the
# walk does not traverse; its ``operands`` are only the values crossing
# INTO the scope (carried inits and loop-invariant boundary subexpressions),
# which keeps every body leaf reachable through the ordinary walk.


#: The op KIND of a lowered bounded ``for``.
LOOP_KIND = "loop"
#: The KIND of a loop's symbolic index — the body's own ``k``.
LOOP_INDEX_KIND = "loop_index"
#: The KIND of a loop-carried value READ at the start of an iteration.
LOOP_CARRY_KIND = "loop_carry"
#: The KIND that projects one carried slot's value OUT of a finished loop.
LOOP_VALUE_KIND = "loop_value"
#: The KIND that reads a forward loop's per-iteration carry out of its tape
#: — the ONE form a reverse loop needs. The tape is a bounded local array a
#: derived kernel declares at a compiler-visible size, not a runtime tape.
TAPE_READ_KIND = "tape_read"
#: The KIND of a loop-with-a-``break``'s per-sample iteration count.
LOOP_COUNT_KIND = "loop_count"








#: How wide ONE body-node key is, in hex characters (:func:`_body_keys`).
#: 128 bits of blake2b: the collision exposure of a structural key is the same
#: exposure the walk's own sha256 digest already accepts one level up, and 16 bytes
#: is where the key stops being the thing that dominates a body's memory.














from .. import _core as _rcore  # noqa: E402


#: the width of a structural body key (hex digits of the hash kept), fixed in the core
BODY_KEY_HEX = 32


def body_key_widths(loop: Loop) -> tuple[int, ...]:
    """The BYTE width of every structural key of ``loop``'s body, in body order (the diagnostic that sees an O(2**depth) key)."""
    return tuple(_rcore.body_key_widths(loop))


_rcore.set_loop(Loop)
_rcore.set_loop_carry(LoopCarry)
