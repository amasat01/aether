# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The fold-threshold queries of a lowering: pure functions over a number and the threshold a variant states."""

from __future__ import annotations

from math import prod

from .nodes import HawkError, Node, Op


def exceeds(n: int, limit: int | None) -> bool:
    """``n > limit`` for a fold threshold of the target's variant. A variant that names none is REFUSED by name where a threshold is
    read: no silent "nothing exceeds it", and no other target's number is borrowed."""
    if limit is None:
        raise HawkError("the variant names no fold threshold (form.rolled_fold_min), and this build needs one to choose between an "
                        "unrolled and a rolled site; name it in the variant, or build with emit_form='static' or 'rolled'")
    return n > limit


def rolled(node: Node, rolled_fold_min: int | None, form: str = "auto") -> bool:
    """Whether ``node`` is spelled as its ``aether::rt`` twin: a matrix-vector or
    matrix-matrix product whose fold length or result size passes
    ``rolled_fold_min`` (the variant's ``form.rolled_fold_min``; ``form`` is the build's ``emit_form``), or a ``dot``, ``sum`` or ``norm`` of a vector longer than it
    (under ``emit_form="rolled"`` every such site, whatever its size; under ``"static"``, none)."""
    if not isinstance(node, Op):
        return False
    if node.kind in ("mv", "mm"):
        fold, size = node.operands[0].ttype.shape[1], prod(node.ttype.shape)
    elif node.kind in ("dot", "sum", "norm") and len(node.operands[0].ttype.shape) == 1:
        fold, size = node.operands[0].ttype.shape[0], 1
    else:
        return False
    if form != "auto":
        return form == "rolled"
    return exceeds(fold, rolled_fold_min) or exceeds(size, rolled_fold_min)
