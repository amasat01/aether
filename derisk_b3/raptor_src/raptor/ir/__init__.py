# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The kernel IR: what a unit IS and how to read it. L0 math core (typed expressions, loops, selects, op kinds with
typing), L1 kernel model (sample space, planes, sinks, cross-sample reads and reductions, the block-shaped keyed
sum), the node schema, the generic rebuild and THE canonical walk. Nothing here rewrites a unit, names a dialect or a
backend, or imports anything but the standard library and this package; form dialects live in their own packages and
plug in through their host's registry.

``IR_VERSION`` is PROVISIONAL: one integer, nothing reads it yet, and no compatibility is promised between two
values until the IR design declares a stable one."""

#: The IR definition's version (provisional, see the package docstring).
IR_VERSION = 1

# Internal names for the host's own modules/tests (`X as X` = not on `__all__`).
from .types import TensorType as TensorType
from .types import Wire as Wire
from .access import Access as Access
from .._core import keytext as key_text  # the key-text rule (GA/SPEC.md section 10), for a class that states its own text
from .block import BlockSum as BlockSum
from .block import BlockSumWrite as BlockSumWrite
from .block import factors as factors
from .access import infer
from .compound import Quantity as Quantity
from .compound import QuantitySpan as QuantitySpan
from .loop_nodes import Loop as Loop
from .loop_nodes import LoopCarry as LoopCarry
from .loop_nodes import LoopCount as LoopCount
from .loop_nodes import LoopIndex as LoopIndex
from .loop_nodes import LoopValue as LoopValue
from .loop_nodes import TapeRead as TapeRead
from .loops import ACCUMULATOR as ACCUMULATOR
from .loops import RECURRENCE as RECURRENCE
from .loops import LoopClass as LoopClass
from .loops import build_loop as build_loop
from .loops import classify_loop as classify_loop
from .nodes import DISPATCH_KIND as DISPATCH_KIND
from .nodes import DISPATCH_POLICIES as DISPATCH_POLICIES
from .nodes import AccumWrite, Assign, At, Const, Op
from .nodes import Dispatch as Dispatch
from .nodes import HawkError as HawkError
from .nodes import LanePosition as LanePosition
from .nodes import PlaneExtent as PlaneExtent
from .nodes import Leaf as Leaf
from .nodes import Node as Node
from .nodes import Primitive as Primitive
from .nodes import RoleConst as RoleConst
from .nodes import SampleIndex as SampleIndex
from .nodes import SampleSum as SampleSum
from .nodes import Select as Select
from .nodes import Sink as Sink
from .nodes import TapeWrite as TapeWrite
from .nodes import WideWrite as WideWrite
from .ops import OP_KINDS as OP_KINDS
from .ops import make
from .ops import result_type as result_type
from .segment import OFFSETS_DTYPE as OFFSETS_DTYPE
from .segment import OFFSETS_ROLE as OFFSETS_ROLE
from .segment import SegmentInfo as SegmentInfo
from .segment import ClassifyUnit as ClassifyUnit
from .segment import SegmentUnit as SegmentUnit
from .segment import classify_name as classify_name
from .walk import DispatchInfo as DispatchInfo
from .walk import Walk as Walk
from .walk import canonical
from .walk import canonical_nodes as canonical_nodes
from .walk import static_selectors as static_selectors

__all__ = [
    "AccumWrite", "Assign", "At", "Const", "Op", "canonical", "infer", "make",
]
