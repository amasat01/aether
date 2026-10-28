# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""aether's C reference core: a generator of plain C99 functions.

Each function reproduces, bit for bit, what aether's expression templates
evaluate on the host (see docs/content/numerics_contract.rst). The rules
that make that true:

* folds (dot, norm, sum) are right-nested, highest index outermost, with no
  zero seed: ``e[N-1] + (e[N-2] + (... + e[0]))``;
* ``x / s`` is ``r = 1/s`` once, then ``r * x_k``;
* ``neg`` is ``x * -1``; ``norm`` is ``sqrt(ssq)`` (no hypot);
* nothing promotes: a ``f32`` text uses the ``f``-suffixed libm and only
  ``aether_c_f32_t`` values.

Consumers: aether's gtest suite pins the corpus header minted from these
texts against the C++ templates; hawk imports this module, requests
functions by ``(kind, shape, width)``, resolves ``deps`` transitively and
inlines the texts verbatim into standalone C files. The module only uses the
standard library. Every ``function``/``struct``/``real_typedef`` result is
memoised by name in one registry, which ``emit`` reads.
"""

from dataclasses import dataclass
from typing import Callable, Iterable

CORE_VERSION = "aether-c/1"
WIDTHS = ("f64", "f32")

# width -> (C type, libm suffix)
_WIDTH = {"f64": ("double", ""), "f32": ("float", "f")}


@dataclass(frozen=True)
class CFunction:
    name: str
    text: str
    deps: tuple[str, ...]
    libm: tuple[str, ...]


_REGISTRY: dict[str, CFunction] = {}


def _check_width(width: str) -> None:
    if width not in _WIDTH:
        raise ValueError(f"unknown width {width!r}; expected one of {WIDTHS}")


_RANK: dict[str, int] = {}  # 0 width typedef, 1 struct typedef, 2 function


def _memo(item: CFunction, rank: int) -> CFunction:
    _RANK.setdefault(item.name, rank)
    return _REGISTRY.setdefault(item.name, item)


def real_type(width: str) -> str:
    _check_width(width)
    return _WIDTH[width][0]


def _t(width: str) -> str:
    return f"aether_c_{width}_t"


def real_typedef(width: str) -> CFunction:
    _check_width(width)
    name = _t(width)
    guard = name.upper()
    text = f"#ifndef {guard}\n#define {guard}\ntypedef {_WIDTH[width][0]} {name};\n#endif\n"
    return _memo(CFunction(name, text, (), ()), 0)


def _check_shape(shape) -> None:
    """Every extent an ``int`` (not ``bool``) >= 1; raised before anything is memoised."""
    if not all(type(n) is int and n >= 1 for n in shape):
        raise ValueError(f"invalid shape {shape!r}: every extent must be an int >= 1")


def _struct_name(shape: tuple[int, ...], width: str) -> str:
    _check_shape(shape)
    if len(shape) == 1:
        return f"aether_c_{width}_vec{shape[0]}_t"
    if len(shape) == 2:
        return f"aether_c_{width}_mat{shape[0]}x{shape[1]}_t"
    raise ValueError(f"no struct for shape {shape!r}")


def struct(shape: tuple[int, ...], width: str) -> CFunction:
    shape = tuple(shape)
    name = _struct_name(shape, width)  # validates the shape before anything is memoised
    real = real_typedef(width)
    if len(shape) == 1:
        body = f"{real.name} v[{shape[0]}];"
    else:
        body = f"{real.name} m[{shape[0]}][{shape[1]}];"
    guard = name.upper()
    text = f"#ifndef {guard}\n#define {guard}\ntypedef struct {{ {body} }} {name};\n#endif\n"
    return _memo(CFunction(name, text, (real.name,), ()), 1)


# ---- per-family formatters; each returns (body lines, libm) ----
# ``cells`` are the component accessors in row-major order ("v[k]" or "m[r][c]"),
# so one formatter serves a vector kind and its matrix mirror; ``x`` names the
# operand of a unary kind.

def _fold(terms: list[str]) -> str:
    """Right-nested, highest index outermost, no seed."""
    if len(terms) == 1:
        return terms[0]
    return f"({terms[-1]} + {_fold(terms[:-1])})"


def _cells(shape: tuple[int, ...]) -> list[str]:
    if len(shape) == 1:
        return [f"v[{k}]" for k in range(shape[0])]
    return [f"m[{r}][{c}]" for r in range(shape[0]) for c in range(shape[1])]


def _fold_kind(op: str, cells: list[str], w: str, x: str) -> tuple[list[str], tuple[str, ...]]:
    if op == "vec_dot":
        return [f"return {_fold([f'(a.{c} * b.{c})' for c in cells])};"], ()
    if op == "vec_norm":
        sq = _fold([f"({x}.{c} * {x}.{c})" for c in cells])
        return [f"return sqrt{_WIDTH[w][1]}({sq});"], (f"sqrt{_WIDTH[w][1]}",)
    return [f"return {_fold([f'{x}.{c}' for c in cells])};"], ()


def _out_body(cells: list[str], exprs: list[str]) -> list[str]:
    return [f"out.{c} = {e};" for c, e in zip(cells, exprs)]


def _cwise_binary(kind: str, cells: list[str], w: str, x: str) -> tuple[list[str], tuple[str, ...]]:
    s = _WIDTH[w][1]
    if kind in _ARITH:
        op = _ARITH[kind]
        return _out_body(cells, [f"(a.{c} {op} b.{c})" for c in cells]), ()
    fn = f"{_LIBM_BINARY[kind]}{s}"
    return _out_body(cells, [f"{fn}(a.{c}, b.{c})" for c in cells]), (fn,)


def _cwise_unary(kind: str, cells: list[str], w: str, x: str) -> tuple[list[str], tuple[str, ...]]:
    t = _t(w)
    if kind == "vec_neg":
        return _out_body(cells, [f"({x}.{c} * ({t})-1.0)" for c in cells]), ()
    if kind == "vec_rsqrt":
        fn = f"sqrt{_WIDTH[w][1]}"
        return _out_body(cells, [f"(({t})1.0 / {fn}({x}.{c}))" for c in cells]), (fn,)
    fn = f"{_LIBM_UNARY[kind]}{_WIDTH[w][1]}"
    return _out_body(cells, [f"{fn}({x}.{c})" for c in cells]), (fn,)


def _scale(kind: str, cells: list[str], w: str, x: str) -> tuple[list[str], tuple[str, ...]]:
    t = _t(w)
    if kind == "vec_scale":
        return _out_body(cells, [f"(s * a.{c})" for c in cells]), ()
    pre = [f"const {t} r = (({t})1.0 / s);"]
    return pre + _out_body(cells, [f"(r * a.{c})" for c in cells]), ()


def _cross(kind: str, cells: list[str], w: str, x: str) -> tuple[list[str], tuple[str, ...]]:
    return _out_body(cells, [
        "((a.v[1] * b.v[2]) - (a.v[2] * b.v[1]))",
        "((a.v[2] * b.v[0]) - (a.v[0] * b.v[2]))",
        "((a.v[0] * b.v[1]) - (a.v[1] * b.v[0]))",
    ]), ()


_ARITH = {"vec_add": "+", "vec_sub": "-", "vec_mul": "*", "vec_cwise_div": "/"}
_LIBM_BINARY = {"vec_min": "fmin", "vec_max": "fmax", "vec_pow": "pow"}
_LIBM_UNARY = {"vec_abs": "fabs", "vec_sqrt": "sqrt", "vec_exp": "exp", "vec_log": "log",
               "vec_sin": "sin", "vec_cos": "cos", "vec_tan": "tan", "vec_tanh": "tanh",
               "vec_asin": "asin", "vec_acos": "acos", "vec_atan": "atan"}

# kind -> (arg layout, result, body formatter); layout "fold2"/"bin"=(a,b), "un"=(a) or (m),
# "scalar"=(a, s); result "vec" or "real". Insertion order here IS the order kinds() reports.
_VEC_KINDS: dict[str, tuple[str, str, Callable]] = {}


def _register(names: Iterable[str], layout: str, result: str, fmt: Callable) -> None:
    for n in names:
        _VEC_KINDS[n] = (layout, result, fmt)


_register(("vec_dot",), "fold2", "real", _fold_kind)
_register(("vec_norm", "vec_sum"), "un", "real", _fold_kind)
_register(("vec_cross",), "fold2", "vec", _cross)
_register(tuple(_ARITH), "fold2", "vec", _cwise_binary)
_register(("vec_scale", "vec_div_s"), "scalar", "vec", _scale)
_register(("vec_min", "vec_max", "vec_pow"), "fold2", "vec", _cwise_binary)
_register(("vec_neg", "vec_abs", "vec_sqrt", "vec_rsqrt", "vec_exp", "vec_log", "vec_sin",
           "vec_cos", "vec_tan", "vec_tanh", "vec_asin", "vec_acos", "vec_atan"),
          "un", "vec", _cwise_unary)
assert set(_LIBM_BINARY) | set(_ARITH) | set(_LIBM_UNARY) <= set(_VEC_KINDS)

_VEC_ORDER = tuple(_VEC_KINDS)
# matrix mirrors: every elementwise vector kind (not the folds or cross) over m[r][c]
_MAT_CWISE = tuple("mat_" + k[4:] for k, v in _VEC_KINDS.items()
                   if v[1] == "vec" and k != "vec_cross")
_MAT_OTHER = ("mat_sum", "mat_mv", "mat_mm", "mat_transpose", "vec_outer")
_QUAT = ("quat_mul", "quat_conj", "quat_recip", "quat_rotate", "quat_as_pure", "quat_as_vec3")
_SCALARS = ("rsqrt", "sign", "clip", "remainder")
_RAYS = ("ray_tri", "ray_aabb", "ray_sphere")


def kinds() -> tuple[str, ...]:
    return _VEC_ORDER + _MAT_CWISE + _MAT_OTHER + _QUAT + _SCALARS + _RAYS


def _render(kind: str, shape: tuple[int, ...], width: str, ret: str, name: str,
            params: str, body: list[str], deps: tuple[str, ...], libm: tuple[str, ...]) -> CFunction:
    head = f"/* aether-c: {kind} {shape!r} {width} {CORE_VERSION} */\n"
    lines = "".join(f"    {ln}\n" for ln in body)
    text = f"{head}static inline {ret} {name}({params})\n{{\n{lines}}}\n"
    return _memo(CFunction(name, text, deps, tuple(sorted(set(libm)))), 2)


def _deps(width: str, *shapes: tuple[int, ...]) -> tuple[str, ...]:
    """Width typedef first, then each struct type in order of first use."""
    out = [real_typedef(width).name]
    for sh in shapes:
        n = struct(sh, width).name
        if n not in out:
            out.append(n)
    return tuple(out)


def _require_rank(kind: str, shape: tuple[int, ...], rank: int) -> None:
    if len(shape) != rank:
        raise ValueError(f"{kind} needs a rank-{rank} shape, got {shape!r}")


def _vector_function(kind: str, shape: tuple[int, ...], width: str) -> CFunction:
    """A vector kind, or (kind ``mat_*``) its elementwise matrix mirror."""
    is_mat = kind.startswith("mat_")
    base = "vec_" + kind[4:] if is_mat else kind
    _require_rank(kind, shape, 2 if is_mat else 1)
    if kind == "vec_cross" and shape != (3,):
        raise ValueError(f"vec_cross is defined only for shape (3,), got {shape!r}")
    layout, result, fmt = _VEC_KINDS[base]
    t, st = _t(width), _struct_name(shape, width)
    x = "m" if is_mat else "a"
    body, libm = fmt(base, _cells(shape), width, x)
    params = {"fold2": f"{st} a, {st} b", "un": f"{st} {x}", "scalar": f"{st} a, {t} s"}[layout]
    if result == "vec":
        body = [f"{st} out;"] + body + ["return out;"]
    prefix = st[len(f"aether_c_{width}_"):-len("_t")]
    name = f"aether_c_{width}_{prefix}_{kind[4:]}"
    return _render(kind, shape, width, st if result == "vec" else t, name, params, body,
                   _deps(width, shape), libm)


def _matrix_function(kind: str, shape: tuple[int, ...], shape2, width: str) -> CFunction:
    t, w = _t(width), width
    if kind == "vec_outer":
        _require_rank(kind, shape, 1)
        if shape2 is None or len(shape2) != 1:
            raise ValueError(f"vec_outer needs shape2 of rank 1, got shape={shape!r} shape2={shape2!r}")
        out_shape = (shape[0], shape2[0])
        ut, vt, ot = _struct_name(shape, w), _struct_name(shape2, w), _struct_name(out_shape, w)
        body = [f"{ot} out;"] + _out_body(_cells(out_shape), [
            f"(u.v[{r}] * v.v[{c}])" for r in range(shape[0]) for c in range(shape2[0])]) + ["return out;"]
        name = f"aether_c_{w}_vec{shape[0]}_outer_vec{shape2[0]}"
        return _render(kind, shape, w, ot, name, f"{ut} u, {vt} v", body,
                       _deps(w, shape, shape2, out_shape), ())
    _require_rank(kind, shape, 2)
    rows, cols = shape
    mt = _struct_name(shape, w)
    if kind == "mat_sum":
        terms = [f"(m.{c} * ({t})1.0)" for c in _cells(shape)]
        return _render(kind, shape, w, t, f"aether_c_{w}_mat{rows}x{cols}_sum", f"{mt} m",
                       [f"return {_fold(terms)};"], _deps(w, shape), ())
    if kind == "mat_transpose":
        out_shape = (cols, rows)
        ot = _struct_name(out_shape, w)
        body = [f"{ot} out;"] + _out_body(_cells(out_shape), [
            f"m.m[{r}][{c}]" for c in range(cols) for r in range(rows)]) + ["return out;"]
        return _render(kind, shape, w, ot, f"aether_c_{w}_mat{rows}x{cols}_transpose", f"{mt} m",
                       body, _deps(w, shape, out_shape), ())
    if kind == "mat_mv":
        vt, ot = _struct_name((cols,), w), _struct_name((rows,), w)
        body = [f"{ot} out;"] + _out_body(_cells((rows,)), [
            _fold([f"(m.m[{r}][{c}] * v.v[{c}])" for c in range(cols)]) for r in range(rows)]) + ["return out;"]
        return _render(kind, shape, w, ot, f"aether_c_{w}_mat{rows}x{cols}_mv", f"{mt} m, {vt} v",
                       body, _deps(w, shape, (cols,), (rows,)), ())
    # mat_mm
    if shape2 is None or len(shape2) != 2 or shape2[0] != cols:
        raise ValueError(f"mat_mm needs shape2 (K, C) with K == {cols}, got shape={shape!r} shape2={shape2!r}")
    out_shape = (rows, shape2[1])
    rt, ot = _struct_name(shape2, w), _struct_name(out_shape, w)
    body = [f"{ot} out;"] + _out_body(_cells(out_shape), [
        _fold([f"(l.m[{i}][{k}] * r.m[{k}][{j}])" for k in range(cols)])
        for i in range(rows) for j in range(shape2[1])]) + ["return out;"]
    name = f"aether_c_{w}_mat{rows}x{cols}_mm_mat{shape2[0]}x{shape2[1]}"
    return _render(kind, shape, w, ot, name, f"{mt} l, {rt} r", body,
                   _deps(w, shape, shape2, out_shape), ())


def _quat_function(kind: str, shape, shape2, width: str) -> CFunction:
    """Quaternions are vec4 values laid out (w, x, y, z); a 3-vector is a vec3."""
    if shape is not None or shape2 is not None:
        raise ValueError(f"{kind} takes no shape, got shape={shape!r} shape2={shape2!r}")
    t, w = _t(width), width
    q4, q3 = (4,), (3,)
    st4, st3 = _struct_name(q4, w), _struct_name(q3, w)
    conj = lambda k: f"(q.v[{k}] * ({t})-1.0)"
    pre: list[str] = []
    if kind == "quat_mul":
        # (l_i * r_j) term indices per output component, in aether's left-to-right order
        pairs = [[(0, 0), (1, 1), (2, 2), (3, 3)], [(0, 1), (1, 0), (2, 3), (3, 2)],
                 [(0, 2), (1, 3), (2, 0), (3, 1)], [(0, 3), (1, 2), (2, 1), (3, 0)]]
        signs = [("-", "-", "-"), ("+", "+", "-"), ("-", "+", "+"), ("+", "-", "+")]
        exprs = []
        for ps, sg in zip(pairs, signs):
            e = f"(l.v[{ps[0][0]}] * r.v[{ps[0][1]}])"
            for (i, j), op in zip(ps[1:], sg):
                e = f"({e} {op} (l.v[{i}] * r.v[{j}]))"
            exprs.append(e)
        params, ret, shapes, cells = f"{st4} l, {st4} r", st4, (q4,), _cells(q4)
    elif kind == "quat_conj":
        exprs = ["q.v[0]"] + [conj(k) for k in (1, 2, 3)]
        params, ret, shapes, cells = f"{st4} q", st4, (q4,), _cells(q4)
    elif kind == "quat_recip":
        ssq = _fold([f"(q.v[{k}] * q.v[{k}])" for k in range(4)])
        pre = [f"const {t} r = (({t})1.0 / {ssq});"]
        exprs = ["(r * q.v[0])"] + [f"(r * {conj(k)})" for k in (1, 2, 3)]
        params, ret, shapes, cells = f"{st4} q", st4, (q4,), _cells(q4)
    elif kind == "quat_rotate":
        # component i uses j = i+1, k = i+2 (mod 3); u_j is q.v[j+1]
        for i, n in enumerate("xyz"):
            j, k = (i + 1) % 3, (i + 2) % 3
            pre.append(f"const {t} t{n} = (({t})2.0 * ((q.v[{j + 1}] * v.v[{k}]) - (q.v[{k + 1}] * v.v[{j}])));")
        exprs = []
        for i in range(3):
            j, k = (i + 1) % 3, (i + 2) % 3
            exprs.append(f"((v.v[{i}] + (q.v[0] * t{'xyz'[i]})) + "
                         f"((q.v[{j + 1}] * t{'xyz'[k]}) - (q.v[{k + 1}] * t{'xyz'[j]})))")
        params, ret, shapes, cells = f"{st4} q, {st3} v", st3, (q4, q3), _cells(q3)
    elif kind == "quat_as_pure":
        exprs = [f"({t})0.0"] + [f"v.v[{k}]" for k in range(3)]
        params, ret, shapes, cells = f"{st3} v", st4, (q3, q4), _cells(q4)
    else:  # quat_as_vec3
        exprs = [f"q.v[{k}]" for k in (1, 2, 3)]
        params, ret, shapes, cells = f"{st4} q", st3, (q4, q3), _cells(q3)
    body = pre + [f"{ret} out;"] + _out_body(cells, exprs) + ["return out;"]
    return _render(kind, q4, w, ret, f"aether_c_{w}_quat_{kind[5:]}", params, body,
                   _deps(w, *shapes), ())


def _scalar_function(kind: str, width: str) -> CFunction:
    t, s = _t(width), _WIDTH[width][1]
    real = real_typedef(width)
    sq, fm, cs = f"sqrt{s}", f"fmod{s}", f"copysign{s}"
    if kind == "rsqrt":
        params, body, libm = f"{t} x", [f"return (({t})1.0 / {sq}(x));"], (sq,)
    elif kind == "sign":
        params, libm = f"{t} x", ()
        body = [f"return (x > ({t})0.0) ? ({t})1.0 : ((x < ({t})0.0) ? ({t})-1.0 : ({t})0.0);"]
    elif kind == "clip":
        params, libm = f"{t} x, {t} lo, {t} hi", ()
        body = [f"const {t} t = ((x >= lo) || (x != x)) ? x : lo;",
                "return ((t <= hi) || (t != t)) ? t : hi;"]
    else:
        params, libm = f"{t} a, {t} b", (fm, cs)
        body = [f"{t} m = {fm}(a, b);",
                f"if (m != ({t})0.0) {{",
                f"    if ((b < ({t})0.0) != (m < ({t})0.0)) {{",
                "        m += b;",
                "    }",
                "} else {",
                f"    m = {cs}(({t})0.0, b);",
                "}",
                "return m;"]
    return _render(kind, (), width, t, f"aether_c_{width}_{kind}", params, body, (real.name,), libm)


def _ray_function(kind: str, shape, shape2, width: str) -> CFunction:
    """Ray primitives of ``aether/geometry/Ray.h``: a vec4 ``[hit, a, b, c]`` result, safe denominators.

    Same expressions, same order as the header: dot is ``(x*x + y*y) + z*z`` and every
    division or ``sqrt`` sees a guarded operand before the select.
    """
    if shape is not None or shape2 is not None:
        raise ValueError(f"{kind} takes no shape, got shape={shape!r} shape2={shape2!r}")
    t, w, s = _t(width), width, _WIDTH[width][1]
    v3, v4 = _struct_name((3,), w), _struct_name((4,), w)
    c = lambda x: f"(({t}){x})"
    lines: list[str] = []

    def sub(name: str, a: str, b: str) -> None:
        lines.extend(f"const {t} {name}{k} = ({a}.v[{k}] - {b}.v[{k}]);" for k in range(3))

    def dot(a: str, b: str) -> str:
        return f"((({a}0 * {b}0) + ({a}1 * {b}1)) + ({a}2 * {b}2))"

    def cross(name: str, a: str, b: str) -> None:
        lines.extend([f"const {t} {name}0 = (({a}1 * {b}2) - ({a}2 * {b}1));",
                      f"const {t} {name}1 = (({a}2 * {b}0) - ({a}0 * {b}2));",
                      f"const {t} {name}2 = (({a}0 * {b}1) - ({a}1 * {b}0));"])

    def vecs(name: str, v: str) -> str:
        lines.extend(f"const {t} {name}{k} = {v}.v[{k}];" for k in range(3))
        return name

    libm: tuple[str, ...] = ()
    if kind == "ray_tri":
        params = f"{v3} o, {v3} d, {v3} a, {v3} b, {v3} c, {t} tmax"
        d = vecs("d", "d")
        sub("e1_", "b", "a"); sub("e2_", "c", "a")
        cross("p", d, "e2_")
        lines += [f"const {t} det = {dot('e1_', 'p')};",
                  f"const int ok = ((det > {c(0.0)}) || (det < {c(0.0)})) ? 1 : 0;",
                  f"const {t} dets = (ok != 0) ? det : {c(1.0)};",
                  f"const {t} inv = (ok != 0) ? ({c(1.0)} / dets) : {c(0.0)};"]
        sub("s", "o", "a")
        lines.append(f"const {t} u = (inv * {dot('s', 'p')});")
        cross("q", "s", "e1_")
        lines += [f"const {t} v = (inv * {dot(d, 'q')});",
                  f"const {t} t = (inv * {dot('e2_', 'q')});",
                  f"const int hit = (ok && (u >= {c(0.0)}) && (v >= {c(0.0)}) && ((u + v) <= {c(1.0)})"
                  f" && (t >= {c(0.0)}) && (t <= tmax)) ? 1 : 0;"]
        out = ["t", "u", "v"]
    elif kind == "ray_aabb":
        params = f"{v3} o, {v3} d, {v3} lo, {v3} hi, {t} tmax"
        libm = (f"copysign{s}",)
        for k in range(3):
            lines += [f"const int ok{k} = ((d.v[{k}] > {c(0.0)}) || (d.v[{k}] < {c(0.0)})) ? 1 : 0;",
                      f"const {t} ds{k} = (ok{k} != 0) ? d.v[{k}] : {c(1.0)};",
                      f"const {t} inv{k} = (ok{k} != 0) ? ({c(1.0)} / ds{k}) : copysign{s}(({t})1e18, d.v[{k}]);",
                      f"const {t} a{k} = ((lo.v[{k}] - o.v[{k}]) * inv{k});",
                      f"const {t} b{k} = ((hi.v[{k}] - o.v[{k}]) * inv{k});",
                      f"const {t} n{k} = (a{k} < b{k}) ? a{k} : b{k};",
                      f"const {t} f{k} = (a{k} < b{k}) ? b{k} : a{k};"]
        lines += [f"const {t} tin1 = (n1 > n0) ? n1 : n0;", f"const {t} tout1 = (f1 < f0) ? f1 : f0;",
                  f"const {t} tin = (n2 > tin1) ? n2 : tin1;", f"const {t} tout = (f2 < tout1) ? f2 : tout1;",
                  f"const int hit = ((tin <= tout) && (tout >= {c(0.0)}) && (tin <= tmax)) ? 1 : 0;"]
        out = ["tin", "tout", c(0.0)]
    else:  # ray_sphere
        params = f"{v3} o, {v3} d, {v3} c, {t} r, {t} tmax"
        libm = (f"sqrt{s}",)
        d = vecs("d", "d")
        sub("oc", "o", "c")
        lines += [f"const {t} qa = {dot('d', 'd')};", f"const {t} hb = {dot('oc', 'd')};",
                  f"const {t} qc = ({dot('oc', 'oc')} - (r * r));",
                  f"const {t} disc = ((hb * hb) - (qa * qc));",
                  f"const int ok = ((qa > {c(0.0)}) && (disc >= {c(0.0)})) ? 1 : 0;",
                  f"const {t} qas = (qa > {c(0.0)}) ? qa : {c(1.0)};",
                  f"const {t} ds = (ok != 0) ? disc : {c(1.0)};",
                  f"const {t} sq = sqrt{s}(ds);",
                  f"const {t} inv = ({c(1.0)} / qas);",
                  f"const {t} t0 = (ok != 0) ? (((-hb) - sq) * inv) : {c(0.0)};",
                  f"const {t} t1 = (ok != 0) ? (((-hb) + sq) * inv) : {c(0.0)};",
                  f"const int hit = (ok && (t1 >= {c(0.0)}) && (t0 <= tmax)) ? 1 : 0;"]
        out = ["t0", "t1", c(0.0)]
    lines += [f"{v4} out;", f"out.v[0] = hit ? {c(1.0)} : {c(0.0)};"]
    lines += [f"out.v[{k + 1}] = {e};" for k, e in enumerate(out)]
    lines.append("return out;")
    return _render(kind, (), w, v4, f"aether_c_{w}_{kind}", params, lines,
                   _deps(w, (3,), (4,)), libm)


def function(kind: str, *, shape=None, shape2=None, width: str = "f64") -> CFunction:
    _check_width(width)
    if kind in _SCALARS:
        if shape is not None or shape2 is not None:
            raise ValueError(f"{kind} takes no shape, got shape={shape!r} shape2={shape2!r}")
        return _scalar_function(kind, width)
    if kind not in kinds():
        raise ValueError(f"unknown kind {kind!r}; known kinds: {', '.join(kinds())}")
    if kind in _RAYS:
        return _ray_function(kind, shape, shape2, width)
    if kind in _QUAT:
        return _quat_function(kind, shape, shape2, width)
    if shape is None:
        raise ValueError(f"{kind} needs a shape")
    shape = tuple(shape)
    shape2 = None if shape2 is None else tuple(shape2)
    _check_shape(shape)
    if shape2 is not None:
        _check_shape(shape2)
        if kind not in ("mat_mm", "vec_outer"):
            raise ValueError(f"{kind} takes no shape2, got shape2={shape2!r}")
    if kind in _VEC_KINDS or kind in _MAT_CWISE:
        return _vector_function(kind, shape, width)
    return _matrix_function(kind, shape, shape2, width)


def emit(names: Iterable[str]) -> str:
    """The closure of ``names``: typedefs, then structs, then functions, each once."""
    order: list[str] = []
    seen: set[str] = set()

    def visit(name: str) -> None:
        if name in seen:
            return
        if name not in _REGISTRY:
            raise KeyError(name)
        seen.add(name)
        for dep in _REGISTRY[name].deps:
            visit(dep)
        order.append(name)

    for name in names:
        visit(name)
    order.sort(key=_RANK.__getitem__)  # stable: dependency order kept within a class
    return "\n".join(_REGISTRY[n].text for n in order)
