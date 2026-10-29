# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
import re, subprocess, shutil, pytest
from aether_dsc import c_core as C

F64_DOT3 = ("static inline aether_c_f64_t aether_c_f64_vec3_dot(aether_c_f64_vec3_t a, aether_c_f64_vec3_t b)\n"
            "{\n    return ((a.v[2] * b.v[2]) + ((a.v[1] * b.v[1]) + (a.v[0] * b.v[0])));\n}\n")

def test_vec3_dot_is_right_nested_with_no_seed():
    f = C.function("vec_dot", shape=(3,), width="f64")
    assert f.name == "aether_c_f64_vec3_dot" and f.text.endswith(F64_DOT3)
    assert f.text.startswith("/* aether-c: vec_dot (3,) f64 aether-c/1 */\n")
    assert f.deps == ("aether_c_f64_t", "aether_c_f64_vec3_t") and f.libm == ()

def test_vec3_norm_is_sqrt_of_right_nested_squares():
    assert "return sqrt(((a.v[2] * a.v[2]) + ((a.v[1] * a.v[1]) + (a.v[0] * a.v[0]))));" in C.function("vec_norm", shape=(3,)).text

def test_vec1_dot_is_one_product():
    assert "return (a.v[0] * b.v[0]);" in C.function("vec_dot", shape=(1,)).text

def test_div_s_multiplies_by_one_reciprocal():
    t = C.function("vec_div_s", shape=(2,)).text
    assert "const aether_c_f64_t r = ((aether_c_f64_t)1.0 / s);" in t and "out.v[1] = (r * a.v[1]);" in t

def test_neg_multiplies_by_minus_one():
    assert "out.v[0] = (a.v[0] * (aether_c_f64_t)-1.0);" in C.function("vec_neg", shape=(1,)).text

def test_min_keeps_operand_order():
    assert "out.v[0] = fmin(a.v[0], b.v[0]);" in C.function("vec_min", shape=(1,)).text

def test_cross_formulas_and_shape_guard():
    t = C.function("vec_cross", shape=(3,)).text
    assert "out.v[0] = ((a.v[1] * b.v[2]) - (a.v[2] * b.v[1]));" in t
    assert "out.v[1] = ((a.v[2] * b.v[0]) - (a.v[0] * b.v[2]));" in t
    assert "out.v[2] = ((a.v[0] * b.v[1]) - (a.v[1] * b.v[0]));" in t
    with pytest.raises(ValueError, match=r"\(4,\)"):
        C.function("vec_cross", shape=(4,))

def test_scalar_formulas():
    assert "return ((aether_c_f64_t)1.0 / sqrt(x));" in C.function("rsqrt").text
    assert "const aether_c_f64_t t = ((x >= lo) || (x != x)) ? x : lo;" in C.function("clip").text
    assert "return ((t <= hi) || (t != t)) ? t : hi;" in C.function("clip").text
    r = C.function("remainder").text
    assert "aether_c_f64_t m = fmod(a, b);" in r and "m = copysign((aether_c_f64_t)0.0, b);" in r
    assert C.function("remainder").libm == ("copysign", "fmod")

def test_f32_never_promotes():
    t = C.function("vec_norm", shape=(3,), width="f32").text
    assert "sqrtf(" in t and "aether_c_f32_t" in t and "double" not in t
    assert C.function("vec_norm", shape=(3,), width="f32").libm == ("sqrtf",)

def test_struct_typedefs_are_guarded():
    s = C.struct((3,), "f64")
    assert s.text == "#ifndef AETHER_C_F64_VEC3_T\n#define AETHER_C_F64_VEC3_T\ntypedef struct { aether_c_f64_t v[3]; } aether_c_f64_vec3_t;\n#endif\n"

MAT_SHAPES = ((2, 2), (3, 3), (4, 4), (2, 3), (3, 2), (1, 3))
MM_PAIRS = (((2, 3), (3, 2)), ((3, 3), (3, 3)), ((2, 2), (2, 2)))
OUTER_PAIRS = (((2,), (3,)), ((3,), (3,)))


def _requests(width):
    """Every (kind, kwargs) the broad tests exercise, derived from kinds()."""
    for kind in C.kinds():
        if kind == "vec_outer":
            for s, s2 in OUTER_PAIRS:
                yield kind, dict(shape=s, shape2=s2, width=width)
        elif kind == "mat_mm":
            for s, s2 in MM_PAIRS:
                yield kind, dict(shape=s, shape2=s2, width=width)
        elif kind.startswith("mat_"):
            for s in MAT_SHAPES:
                yield kind, dict(shape=s, width=width)
        elif kind.startswith("quat_"):
            yield kind, dict(width=width)
        elif kind.startswith("vec_"):
            for s in ((1,), (2,), (3,), (4,), (6,)):
                if kind == "vec_cross" and s != (3,): continue
                yield kind, dict(shape=s, width=width)
        else:
            yield kind, dict(width=width)

def test_texts_are_deterministic_c99_and_c_plus_plus_safe():
    for width in C.WIDTHS:
        for kind, kw in _requests(width):
            a, b = C.function(kind, **kw), C.function(kind, **kw)
            assert a == b and "//" not in a.text and "static inline" in a.text
            assert a.deps[0] == f"aether_c_{width}_t"

def test_emit_orders_the_closure_typedefs_first_each_once():
    names = [C.function("vec_norm", shape=(3,)).name, C.function("vec_dot", shape=(3,)).name]
    text = C.emit(names)
    assert text.index("typedef double aether_c_f64_t;") < text.index("aether_c_f64_vec3_t;") < text.index("aether_c_f64_vec3_norm(")
    assert text.count("typedef struct { aether_c_f64_t v[3]; }") == 1

@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
def test_emitted_closure_compiles_as_strict_c99(tmp_path):
    names = [C.function(k, **kw).name for w in C.WIDTHS for k, kw in _requests(w)]
    src = tmp_path / "t.c"
    src.write_text("#include <math.h>\n" + C.emit(names))
    subprocess.run(["cc", "-std=c99", "-pedantic", "-Wall", "-Wextra", "-Wconversion", "-Wsign-conversion",
                    "-Wdouble-promotion", "-Wno-unused-function", "-Werror", "-fsyntax-only", str(src)], check=True)

def test_mat2x2_sum_is_the_ones_matdot_fold():
    t = C.function("mat_sum", shape=(2, 2)).text
    assert "return ((m.m[1][1] * (aether_c_f64_t)1.0) + ((m.m[1][0] * (aether_c_f64_t)1.0) + ((m.m[0][1] * (aether_c_f64_t)1.0) + (m.m[0][0] * (aether_c_f64_t)1.0))));" in t

def test_mat2x2_mv_row_fold():
    t = C.function("mat_mv", shape=(2, 2)).text
    assert "out.v[0] = ((m.m[0][1] * v.v[1]) + (m.m[0][0] * v.v[0]));" in t

def test_mat1x3_mv():
    assert "out.v[0] = ((m.m[0][2] * v.v[2]) + ((m.m[0][1] * v.v[1]) + (m.m[0][0] * v.v[0])));" in C.function("mat_mv", shape=(1, 3)).text

def test_mat_mm_names_both_shapes_and_folds_over_k():
    f = C.function("mat_mm", shape=(2, 3), shape2=(3, 2))
    assert f.name == "aether_c_f64_mat2x3_mm_mat3x2"
    assert "out.m[0][0] = ((l.m[0][2] * r.m[2][0]) + ((l.m[0][1] * r.m[1][0]) + (l.m[0][0] * r.m[0][0])));" in f.text
    assert set(f.deps) >= {"aether_c_f64_mat2x3_t", "aether_c_f64_mat3x2_t", "aether_c_f64_mat2x2_t"}
    assert f.deps[0] == "aether_c_f64_t"

def test_outer_and_transpose():
    assert "out.m[1][0] = (u.v[1] * v.v[0]);" in C.function("vec_outer", shape=(2,), shape2=(3,)).text
    assert C.function("vec_outer", shape=(2,), shape2=(3,)).name == "aether_c_f64_vec2_outer_vec3"
    assert "out.m[2][0] = m.m[0][2];" in C.function("mat_transpose", shape=(1, 3)).text

def test_mat_cwise_mirror_the_vector_templates():
    assert "out.m[1][0] = (s * a.m[1][0]);" in C.function("mat_scale", shape=(2, 2)).text
    t = C.function("mat_div_s", shape=(2, 2)).text
    assert "const aether_c_f64_t r = ((aether_c_f64_t)1.0 / s);" in t and "out.m[1][1] = (r * a.m[1][1]);" in t
    assert "out.m[0][1] = (m.m[0][1] * (aether_c_f64_t)-1.0);" in C.function("mat_neg", shape=(1, 3)).text
    assert "out.m[0][0] = ((aether_c_f32_t)1.0 / sqrtf(m.m[0][0]));" in C.function("mat_rsqrt", shape=(1, 1), width="f32").text
    assert "out.m[0][0] = fmin(a.m[0][0], b.m[0][0]);" in C.function("mat_min", shape=(1, 3)).text
    assert "out.m[0][2] = fabsf(m.m[0][2]);" in C.function("mat_abs", shape=(1, 3), width="f32").text
    assert "out.m[0][0] = (a.m[0][0] + b.m[0][0]);" in C.function("mat_add", shape=(2, 3)).text

def test_matrix_shape_errors():
    with pytest.raises(ValueError, match=r"\(3,\)"):
        C.function("mat_sum", shape=(3,))
    with pytest.raises(ValueError, match=r"\(2, 3\).*\(2, 3\)"):
        C.function("mat_mm", shape=(2, 3), shape2=(2, 3))
    with pytest.raises(ValueError, match=r"\(2, 2\)"):
        C.function("mat_mm", shape=(2, 2))
    with pytest.raises(ValueError, match=r"\(2,\)"):
        C.function("vec_outer", shape=(2,))

def test_kinds_order_matrix_between_vectors_and_scalars():
    k = C.kinds()
    assert k.index("vec_atan") < k.index("mat_add") < k.index("mat_atan") < k.index("mat_sum") < k.index("rsqrt")
    assert k[-7:] == ("rsqrt", "sign", "clip", "remainder", "ray_tri", "ray_aabb", "ray_sphere")
    assert k[-14] == "vec_outer" and k[-13:-7] == tuple(x for x in k if x.startswith("quat_"))
    assert len(set(k)) == len(k)

def test_unknown_kind_and_name():
    with pytest.raises(ValueError, match="vec_dot"):
        C.function("nope")
    with pytest.raises(KeyError, match="no_such"):
        C.emit(["no_such"])


def test_quat_mul_hamilton_left_to_right():
    t = C.function("quat_mul").text
    assert "out.v[0] = ((((l.v[0] * r.v[0]) - (l.v[1] * r.v[1])) - (l.v[2] * r.v[2])) - (l.v[3] * r.v[3]));" in t
    assert "out.v[1] = ((((l.v[0] * r.v[1]) + (l.v[1] * r.v[0])) + (l.v[2] * r.v[3])) - (l.v[3] * r.v[2]));" in t
    assert "out.v[2] = ((((l.v[0] * r.v[2]) - (l.v[1] * r.v[3])) + (l.v[2] * r.v[0])) + (l.v[3] * r.v[1]));" in t
    assert "out.v[3] = ((((l.v[0] * r.v[3]) + (l.v[1] * r.v[2])) - (l.v[2] * r.v[1])) + (l.v[3] * r.v[0]));" in t

def test_quat_rotate_expands_as_aether_does():
    t = C.function("quat_rotate").text
    assert "const aether_c_f64_t tx = ((aether_c_f64_t)2.0 * ((q.v[2] * v.v[2]) - (q.v[3] * v.v[1])));" in t
    assert "out.v[0] = ((v.v[0] + (q.v[0] * tx)) + ((q.v[2] * tz) - (q.v[3] * ty)));" in t

def test_quat_recip_scales_the_conjugate_by_one_reciprocal():
    t = C.function("quat_recip").text
    assert "const aether_c_f64_t r = ((aether_c_f64_t)1.0 / ((q.v[3] * q.v[3]) + ((q.v[2] * q.v[2]) + ((q.v[1] * q.v[1]) + (q.v[0] * q.v[0])))));" in t
    assert "out.v[0] = (r * q.v[0]);" in t and "out.v[1] = (r * (q.v[1] * (aether_c_f64_t)-1.0));" in t

def test_quat_as_pure_and_as_vec3():
    assert "out.v[0] = (aether_c_f64_t)0.0;" in C.function("quat_as_pure").text
    assert "out.v[2] = q.v[3];" in C.function("quat_as_vec3").text

def test_quat_names_types_deps_and_shape_rejection():
    f = C.function("quat_rotate", width="f32")
    assert f.name == "aether_c_f32_quat_rotate"
    assert f.text.startswith("/* aether-c: quat_rotate (4,) f32 aether-c/1 */\n")
    assert "static inline aether_c_f32_vec3_t aether_c_f32_quat_rotate(aether_c_f32_vec4_t q, aether_c_f32_vec3_t v)" in f.text
    assert f.deps == ("aether_c_f32_t", "aether_c_f32_vec4_t", "aether_c_f32_vec3_t")
    assert C.function("quat_as_pure").deps == ("aether_c_f64_t", "aether_c_f64_vec3_t", "aether_c_f64_vec4_t")
    assert "const aether_c_f32_t r = ((aether_c_f32_t)1.0 /" in C.function("quat_recip", width="f32").text
    with pytest.raises(ValueError):
        C.function("quat_mul", shape=(4,))
    with pytest.raises(ValueError):
        C.function("quat_mul", shape2=(4,))


_LIBM_CALL = re.compile(r"\b(sqrt|fmin|fmax|pow|fabs|exp|log|sin|cos|tan|tanh|asin|acos|atan|fmod|copysign)f?\(")

def test_every_registered_name_audits_deps_libm_and_width_purity():
    for width in C.WIDTHS:
        for kind, kw in _requests(width):
            f = C.function(kind, **kw)
            for dep in f.deps:
                C.emit([dep])  # KeyError if a dep is not registered
            calls = set(re.findall(r"\b((?:sqrt|fmin|fmax|pow|fabs|exp|log|sin|cos|tan|tanh|asin|acos|atan|fmod|copysign)f?)\(", f.text))
            assert set(f.libm) == calls and len(f.libm) == len(set(f.libm)), (f.name, f.libm, calls)
            if width == "f32":
                assert all(c.endswith("f") for c in calls), f.name
                assert not re.search(r"\bdouble\b", f.text), f.name
            else:
                assert not re.search(r"\bfloat\b", f.text), f.name

def test_invalid_shapes_raise_and_register_nothing():
    before = dict(C._REGISTRY)
    for kind, kw in (("vec_add", dict(shape=(0,))), ("vec_add", dict(shape=(-2,))),
                     ("vec_dot", dict(shape=(0,))), ("mat_mv", dict(shape=(0, 3))),
                     ("vec_add", dict(shape=(True,))), ("vec_add", dict(shape=(2.0,))),
                     ("rsqrt", dict(shape=(3,))), ("clip", dict(shape2=(3,))),
                     ("vec_add", dict(shape=(3,), shape2=(3,)))):
        with pytest.raises(ValueError):
            C.function(kind, **kw)
    with pytest.raises(ValueError):
        C.struct((0,), "f32")
    assert C._REGISTRY == before


def test_ray_functions_name_type_and_guard_their_denominators():
    tri = C.function("ray_tri")
    assert tri.name == "aether_c_f64_ray_tri" and tri.deps == ("aether_c_f64_t", "aether_c_f64_vec3_t", "aether_c_f64_vec4_t")
    assert "const aether_c_f64_t dets = (ok != 0) ? det : ((aether_c_f64_t)1.0);" in tri.text
    assert "const aether_c_f64_t inv = (ok != 0) ? (((aether_c_f64_t)1.0) / dets) : ((aether_c_f64_t)0.0);" in tri.text
    assert "out.v[0] = hit ? ((aether_c_f64_t)1.0) : ((aether_c_f64_t)0.0);" in tri.text
    box = C.function("ray_aabb", width="f32")
    assert "copysignf((aether_c_f32_t)1e18, d.v[0])" in box.text and box.libm == ("copysignf",)
    assert "const aether_c_f32_t ds1 = (ok1 != 0) ? d.v[1] : ((aether_c_f32_t)1.0);" in box.text
    sph = C.function("ray_sphere")
    assert "const aether_c_f64_t sq = sqrt(ds);" in sph.text and "const aether_c_f64_t ds = (ok != 0) ? disc : ((aether_c_f64_t)1.0);" in sph.text
    assert sph.libm == ("sqrt",)
    for kind in ("ray_tri", "ray_aabb", "ray_sphere"):
        with pytest.raises(ValueError):
            C.function(kind, shape=(3,))
