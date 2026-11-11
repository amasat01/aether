"""The key-text rule (GA/SPEC.md section 10) against Python's repr: 10^6 floats, every code point, strings, ints, tuples."""
import math, random, struct
import pytest
from raptor import _core

kt = _core.keytext


def _floats():
    r = random.Random(7)
    edge = [0.0, -0.0, 1.0, -1.0, 0.1, 1e15, 1e16, 1e17, 9999999999999998.0, 1e-4, 1e-5, 0.00012345, 5e-324, -5e-324, 2.2250738585072014e-308,
            2.225073858507201e-308, 1.7976931348623157e308, math.inf, -math.inf, math.nan, 123456789012345678.0, 1.5e300, 100.0, 1e22, 1e23, 0.3, 2.5, 1/3]
    yield from edge
    for _ in range(500_000):                      # random bit patterns: every exponent, subnormals included
        x = struct.unpack("<d", struct.pack("<Q", r.getrandbits(64)))[0]
        yield x
    for _ in range(300_000):                      # human-scale values
        yield r.uniform(-1e6, 1e6) * 10 ** r.randint(-8, 8)
    for _ in range(199_972):                      # short decimals and integers
        yield float(r.randint(-10**r.randint(1, 18), 10**r.randint(1, 18))) / 10 ** r.randint(0, 12)


def test_floats_equal_python_repr():
    n = bad = 0
    for x in _floats():
        n += 1
        if kt(x) != repr(x):
            bad += 1
            assert bad > 5 or False, (x, kt(x), repr(x))
    assert n >= 10**6 and bad == 0


def test_every_code_point_alone():
    for cp in range(0x110000):
        if 0xD800 <= cp < 0xE000:
            continue                                # surrogates are not str-encodable to UTF-8 here
        s = chr(cp)
        assert kt(s) == repr(s), hex(cp)


def test_strings_non_ascii_and_quotes():
    for s in ["", "a'b", 'a"b', "a'\"b", "café", "日本語", "\U0001f600x", "tab\there\n", "\x00\x7f\x80\xa0\xad", " ​﻿",
              "\\", "'", '"', "é", "\U000e0001", "\U0010ffff"]:
        assert kt(s) == repr(s), s
    r = random.Random(3)
    for _ in range(20000):
        s = "".join(chr(c) for c in (r.choice([r.randint(0, 0x2ff), r.randint(0x2000, 0x2fff), r.randint(0x10000, 0x1ffff), r.randint(0, 0x10ffff)])
                                      for _ in range(r.randint(0, 8))) if not 0xD800 <= c < 0xE000)
        assert kt(s) == repr(s)


def test_ints_bools_none_tuples_lists():
    for v in [0, -1, 2**63, -2**63, 2**100, True, False, None, (), (1,), (1, 2), ((),), (1, "a", None, 2.5, (True,)), [], [1], [(1, "x"), (2, "y")],
              ("leaf", "x'y", (1,)), (0.0, -0.0, 1e-7)]:
        assert kt(v) == repr(v), v


def test_other_kinds_are_refused_by_name():
    for v in [b"x", {1: 2}, {1}, 1 + 2j, object(), range(2), True.__class__]:
        with pytest.raises(Exception, match="key text: a value of type"):
            kt(v)
    with pytest.raises(Exception, match="beyond 128 bits"):
        kt(2**200)


def test_hook_only_for_declared_types():
    import json
    from raptor.ir import key_text
    assert json.loads(_core.manifest())["key_text"] == ["RawBlock"]

    class Stranger:
        def __key_text__(self):
            return "<x>"

    with pytest.raises(Exception, match="type Stranger has a __key_text__ method but is not declared"):
        key_text(Stranger())

    class RawBlock:                      # a declared name: the hook is used
        def __key_text__(self):
            return "<RawBlock a>"

    assert key_text((RawBlock(),)) == "(<RawBlock a>,)"

    class RawBlock2(RawBlock):           # a subclass is another type: refused by name
        pass
    with pytest.raises(Exception, match="RawBlock2 has a __key_text__ method but is not declared"):
        key_text(RawBlock2())
