# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The block-shaped keyed sum, a kernel-model node: the sum over the lanes of a key of a product of per-row rank-1
values. A frontend emits it (the most structured form); whether a build keeps it as one block or expands it per entry
is a lowering decision and lives with the lowering passes."""

from __future__ import annotations

from .nodes import Node, Op, ReduceSamples, WideWrite
from .linalg import TEAM_MAX_DIM
from .types import TensorType


def factors(payload: Node) -> tuple | None:
    """``(u, v, rows)`` when ``payload`` is a sum over per-lane rows of rank-1 products, whose sum
    over the lanes of a key is ``sum_r u_r v_r^T``: ``outer(u, v)`` (a ``p x q`` block), ``u * v`` of
    a rank-1 ``u`` and a rank-0 ``v`` (a ``p`` vector: the ``q = 1`` block), and, for a per-lane
    ``rows x p`` matrix ``M``, ``transpose(M) @ M`` (``u = v = M``, a ``p x p`` block) and
    ``transpose(M) @ r`` (``v = r``, a ``p`` vector). ``rows`` is how many rows a lane holds (1 for
    the first two spellings); else ``None``."""
    if not isinstance(payload, Op):
        return None
    if payload.kind == "outer":
        return (*payload.operands, 1)
    if payload.kind == "mul":
        a, b = payload.operands
        if len(a.ttype.shape) == 1 and not b.ttype.shape:
            return a, b, 1
        if len(b.ttype.shape) == 1 and not a.ttype.shape:
            return b, a, 1
    if payload.kind in ("mm", "mv"):
        t, other = payload.operands
        if isinstance(t, Op) and t.kind == "transpose" and len(t.operands[0].ttype.shape) == 2:
            m = t.operands[0]
            if payload.kind == "mm" and other is m:
                return m, m, m.ttype.shape[0]
            if payload.kind == "mv" and other.ttype.shape == (m.ttype.shape[0],):
                return m, other, m.ttype.shape[0]
    return None


class BlockSum(ReduceSamples):
    """The sum over the lanes of a key of a product of per-row rank-1 values into a dense
    ``p x q`` block (or a ``p`` vector): ONE node where the per-entry form would be ``p * q``
    accumulates. It is a keyed :class:`~hawk.ir.nodes.ReduceSamples` to everything that reads it
    (the payload is its operand, so a derivative and a rebuild see the per-entry sum), and the
    split lowers it to a stage of per-row stores of the factors and a team stage that owns each
    key's block (:func:`hawk.passes.split.stages`). The sum is ordered: the rows of a key are added in
    lane order, whatever the policy."""

    __slots__ = ()

    @property
    def factors(self) -> tuple:
        """``(u, v, rows)`` of the payload (:func:`factors`); an ordered sum of a plain scalar or
        vector is ``(payload, None, 1)``: ``v`` is the constant 1."""
        payload = self.operands[0]
        return factors(payload) or (payload, None, 1)

    def stored(self, name: str, ttype: TensorType | None = None) -> tuple:
        return (BlockSumWrite(name, self),)

    def expanded(self, name: str) -> tuple:
        """The per-entry accumulates that store this sum to ``name`` (what a derivative sees)."""
        return ReduceSamples.stored(self, name)

    def plain(self) -> ReduceSamples:
        """The same sum as a plain :class:`~hawk.ir.nodes.ReduceSamples`."""
        if not self.kept:
            return ReduceSamples("sum", self.operands[0], policy=self.policy)
        return ReduceSamples("sum", self.operands[0], key=self.key, count=self.count, policy=self.policy, kept=self.kept)


class BlockSumWrite(WideWrite):
    """The store of a :class:`BlockSum` to the output plane ``name`` (``(*payload, *kept)``,
    payload first). The split turns it into the team stage that writes the plane."""

    __slots__ = ()

    @classmethod
    def from_state(cls, ttype, operands, state):
        return cls(state["name"], operands[0])

    def __init__(self, name: str, block: BlockSum) -> None:
        super().__init__(name, block, None, TensorType((), block.ttype.dtype))

    @property
    def block(self) -> BlockSum:
        return self.operands[0]

    def expanded(self) -> tuple:
        return self.block.expanded(self.name)


def ordered_case(payload: Node, policy: str) -> bool:
    return (policy == "ordered" and payload.ttype.dtype in ("f64", "f32") and len(payload.ttype.shape) <= 1
            and all(extent <= TEAM_MAX_DIM for extent in payload.ttype.shape))


def block_shaped(payload: Node, policy: str = "auto") -> bool:
    """Whether the keyed sum of ``payload`` CAN be one :class:`BlockSum`, by structure alone: a product of per-row
    rank-1 values of a float type within :data:`TEAM_MAX_DIM`; under ``policy="ordered"`` any floating scalar or vector
    payload is (rows of a key added in lane order: no atomics, the same bits on every launch)."""
    return ordered_case(payload, policy) or (
        factors(payload) is not None and payload.ttype.dtype in ("f64", "f32")
        and all(extent <= TEAM_MAX_DIM for extent in payload.ttype.shape))


def expand_stores(sinks) -> tuple:
    """``sinks`` with every :class:`BlockSumWrite` replaced by its per-entry accumulates."""
    out: list = []
    for sink in sinks:
        out.extend(sink.expanded() if isinstance(sink, BlockSumWrite) else (sink,))
    return tuple(out)
