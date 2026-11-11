"""The walk-time verifier checks operand counts and the result-type rule of every kind, from core.json (and from a dialect kind's
declaration). Nodes are FORGED here (the constructors already refuse these shapes), so the core's verifier is what is under test."""
import pytest
from raptor.ir import nodes
from raptor.ir.types import TensorType
from raptor.ir.walk import canonical

F64, F32 = TensorType((), "f64"), TensorType((), "f32")


def _forge(good, **over):
    """A copy of ``good`` (same class) with slots replaced, set without running the constructor."""
    bad = object.__new__(type(good))
    for cls in type(good).__mro__:
        for s in getattr(cls, "__slots__", ()):
            if hasattr(good, s):
                object.__setattr__(bad, s, over.get(s, getattr(good, s)))
    return bad


def _select():
    c = nodes.Leaf("vocab_read", "lookup", "c", TensorType((), "bool"))
    a = nodes.Leaf("vocab_read", "lookup", "a", F64)
    b = nodes.Leaf("vocab_read", "lookup", "b", F64)
    return nodes.Select(c, a, b, F64), (c, a, b)


def _sink(v):
    return nodes.Sink("assign_out", "out", "y", v) if False else nodes.Sink("sink", "out", "y", v)


def test_good_select_walks():
    s, _ = _select()
    assert canonical([_sink(s)]).digest


def test_wrong_operand_count_is_refused_by_name():
    s, (c, a, b) = _select()
    bad = _forge(s, operands=(c, a))
    with pytest.raises(nodes.HawkError, match=r"\(select\): takes 3\.\.3 operand\(s\), got 2"):
        canonical([_sink(bad)])


def test_wrong_result_type_is_refused_by_name():
    s, _ = _select()
    bad = _forge(s, ttype=F32)
    with pytest.raises(nodes.HawkError, match=r"\(select\): type f32 \[\] is not what its result rule derives"):
        canonical([_sink(bad)])


def test_fixed_result_rule_of_finish():
    a = nodes.Leaf("vocab_read", "lookup", "a", TensorType((), "bool"))
    f = nodes.Finish("done", a)
    assert canonical([f]).digest
    with pytest.raises(nodes.HawkError, match=r"\(finish\): type f64 \[\]"):
        canonical([_forge(f, ttype=F64)])
