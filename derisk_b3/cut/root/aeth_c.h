/* aeth_c.h: scalar math subset of aether in plain C99 (pilot P04, clause C1).
 *
 * Source of the ports: aether at a2efaa7 (banded/detail/Fp32.h, FmaRaw.h, ExactSum.h; random/detail/Philox.h;
 * linalg/Cholesky.h, N = 3, float). Rounding is pinned by the BUILD, not by barriers: the library is built with
 * -ffp-contract=off, no fast-math, no reassociation (flags.txt). A user that compiles this header with other flags
 * gets a different result and no warning; that is the cost of plain C and it is recorded in REPORT.md.
 * The header holds no preprocessor conditional. The only names it does not define are the two arm primitives
 * declared below. */
#pragma once

#include <stdint.h>
#include <string.h>

/* ---- arm-supplied primitives (declared only; see arm_primitives.md) ---- */
float aeth_fmaf(float a, float b, float c);  /* a*b + c with one rounding (IEEE fmaf) */
float aeth_sqrtf(float x);                   /* correctly rounded square root (IEEE sqrt) */

/* ---- types ---- */
typedef struct aeth_band3 { float hi, lo, tail; } aeth_band3;
typedef struct aeth_u4 { uint32_t x, y, z, w; } aeth_u4;
typedef struct aeth_chol3 { float l[9]; int32_t info; } aeth_chol3;

static inline aeth_band3 aeth_mk3(float h, float l, float t)
{
    aeth_band3 r;
    r.hi = h; r.lo = l; r.tail = t;
    return r;
}

/* ---- bit helpers ---- */
static inline int32_t aeth_float_as_int(float f)
{
    int32_t r;
    memcpy(&r, &f, sizeof r);
    return r;
}

static inline float aeth_int_as_float(int32_t i)
{
    float r;
    memcpy(&r, &i, sizeof r);
    return r;
}

/* ---- rounded arithmetic (each is one IEEE operation; contraction is off by build flag) ---- */
static inline float aeth_mulRN(float a, float b) { return a * b; }
static inline float aeth_addRN(float a, float b) { return a + b; }
static inline float aeth_subRN(float a, float b) { return a - b; }

/* ---- error-free transforms ---- */
static inline void aeth_twoSum(float a, float b, float *s, float *e)
{
    float sv = aeth_addRN(a, b);
    float bp = aeth_subRN(sv, a);
    float ap = aeth_subRN(sv, bp);
    float eb = aeth_subRN(b, bp);
    float ea = aeth_subRN(a, ap);
    *s = sv;
    *e = aeth_addRN(ea, eb);
}

static inline void aeth_fast2Sum(float a, float b, float *s, float *e)
{
    float sv = aeth_addRN(a, b);
    float bp = aeth_subRN(sv, a);
    *s = sv;
    *e = aeth_subRN(b, bp);
}

static inline void aeth_twoProd(float a, float b, float *p, float *e)
{
    float pv = aeth_mulRN(a, b);
    *p = pv;
    *e = aeth_fmaf(a, b, -pv);
}

/* ---- special-limb handling ---- */
static inline int aeth_hi_is_special(float x)
{
    return ((uint32_t)aeth_float_as_int(x) & 0x7F800000u) == 0x7F800000u;
}

static inline aeth_band3 aeth_special_carrier(float v)
{
    uint32_t u = (uint32_t)aeth_float_as_int(v);
    int is_nan = (u & 0x7FFFFFFFu) > 0x7F800000u;
    float canon = is_nan ? aeth_int_as_float((int32_t)0x7FC00000u) : v;
    return aeth_mk3(canon, 0.0f, 0.0f);
}

/* ---- compensated sums (order-free exact renormalisation) ---- */
static inline aeth_band3 aeth_exactSum3(float x, float y, float z)
{
    float s, e, rh, rl, m, t;
    aeth_twoSum(y, z, &s, &e);
    aeth_twoSum(x, s, &rh, &rl);
    aeth_twoSum(rl, e, &m, &t);
    if (aeth_hi_is_special(rh))
        return aeth_special_carrier(rh);
    return aeth_mk3(rh, m, t);
}

static inline aeth_band3 aeth_exactSum4(float x, float y, float z, float w)
{
    float s1, e1, s2, e2, s3, e3, u1, v1, u2, v2;
    aeth_twoSum(z, w, &s1, &e1);
    aeth_twoSum(y, s1, &s2, &e2);
    aeth_twoSum(x, s2, &s3, &e3);
    aeth_twoSum(e3, e2, &u1, &v1);
    aeth_twoSum(u1, e1, &u2, &v2);
    if (aeth_hi_is_special(s3))
        return aeth_special_carrier(s3);
    return aeth_mk3(s3, u2, aeth_addRN(v2, v1));
}

/* ---- rounded multiply-add on three-limb operands: a*b + c ---- */
static inline aeth_band3 aeth_fmaRaw(aeth_band3 a, aeth_band3 b, aeth_band3 c)
{
    float p_hi, p_lo, q_hi, q_lo, t_hi, t_lo;
    float sA, eA, sB, eB, main_sum, e_main, mA, emA, mB, emB, resid;
    aeth_twoProd(a.hi, b.hi, &p_hi, &p_lo);
    aeth_twoProd(a.hi, b.lo, &q_hi, &q_lo);
    aeth_twoProd(a.lo, b.hi, &t_hi, &t_lo);
    aeth_twoSum(p_lo, q_hi, &sA, &eA);
    aeth_twoSum(sA, t_hi, &sB, &eB);
    aeth_twoSum(p_hi, c.hi, &main_sum, &e_main);
    aeth_twoSum(e_main, sB, &mA, &emA);
    aeth_twoSum(mA, c.lo, &mB, &emB);
    resid = aeth_addRN(eA, eB);
    resid = resid + emA;
    resid = resid + emB;
    resid = resid + q_lo;
    resid = resid + t_lo;
    resid = aeth_fmaf(a.lo, b.lo, resid);
    resid = aeth_fmaf(a.tail, b.hi, resid);
    resid = aeth_fmaf(b.tail, a.hi, resid);
    resid = resid + c.tail;
    return aeth_mk3(main_sum, mB, resid);
}

/* a*b + c with a one-float addend c */
static inline aeth_band3 aeth_fmaRawS(aeth_band3 a, aeth_band3 b, float c)
{
    float p_hi, p_lo, q_hi, q_lo, t_hi, t_lo;
    float sA, eA, sB, eB, main_sum, e_main, mA, emA, resid;
    aeth_twoProd(a.hi, b.hi, &p_hi, &p_lo);
    aeth_twoProd(a.hi, b.lo, &q_hi, &q_lo);
    aeth_twoProd(a.lo, b.hi, &t_hi, &t_lo);
    aeth_twoSum(p_lo, q_hi, &sA, &eA);
    aeth_twoSum(sA, t_hi, &sB, &eB);
    aeth_twoSum(p_hi, c, &main_sum, &e_main);
    aeth_twoSum(e_main, sB, &mA, &emA);
    resid = aeth_addRN(eA, eB);
    resid = resid + emA;
    resid = resid + q_lo;
    resid = resid + t_lo;
    resid = aeth_fmaf(a.lo, b.lo, resid);
    resid = aeth_fmaf(a.tail, b.hi, resid);
    resid = aeth_fmaf(b.tail, a.hi, resid);
    return aeth_mk3(main_sum, mA, resid);
}

/* ---- counter-based generator: Philox4x32-10 ---- */
static inline uint32_t aeth_mulhilo32(uint32_t a, uint32_t b, uint32_t *hi)
{
    uint64_t product = (uint64_t)a * (uint64_t)b;
    *hi = (uint32_t)(product >> 32);
    return (uint32_t)product;
}

static inline aeth_u4 aeth_philox_round(aeth_u4 c, uint32_t key_x, uint32_t key_y)
{
    uint32_t hi0 = 0, hi1 = 0;
    uint32_t lo0 = aeth_mulhilo32(0xD2511F53u, c.x, &hi0);
    uint32_t lo1 = aeth_mulhilo32(0xCD9E8D57u, c.z, &hi1);
    aeth_u4 r;
    r.x = hi1 ^ c.y ^ key_x;
    r.y = lo1;
    r.z = hi0 ^ c.w ^ key_y;
    r.w = lo0;
    return r;
}

static inline aeth_u4 aeth_philox4x32_10(aeth_u4 c, uint32_t key_x, uint32_t key_y)
{
    int round;
    c = aeth_philox_round(c, key_x, key_y);
    for (round = 1; round < 10; ++round) {
        key_x += 0x9E3779B9u;
        key_y += 0xBB67AE85u;
        c = aeth_philox_round(c, key_x, key_y);
    }
    return c;
}

/* ---- 3x3 Cholesky of the symmetric part of a (row-major), float ---- */
static inline float aeth_sym_at3(const float *a, int i, int j)
{
    return 0.5f * a[i * 3 + j] + 0.5f * a[j * 3 + i];
}

static inline aeth_chol3 aeth_cholesky3(const float *a)
{
    aeth_chol3 f;
    int j, i, k;
    f.info = 0;
    for (i = 0; i < 9; ++i)
        f.l[i] = 0.0f;
    for (j = 0; j < 3; ++j) {
        if (f.info == 0) {
            float d = aeth_sym_at3(a, j, j);
            for (k = 0; k < j; ++k)
                d -= f.l[j * 3 + k] * f.l[j * 3 + k];
            if (d > 0.0f) {
                float ljj = aeth_sqrtf(d);
                f.l[j * 3 + j] = ljj;
                for (i = j + 1; i < 3; ++i) {
                    float s = aeth_sym_at3(a, i, j);
                    for (k = 0; k < j; ++k)
                        s -= f.l[i * 3 + k] * f.l[j * 3 + k];
                    f.l[i * 3 + j] = s / ljj;
                }
            } else {
                f.info = (int32_t)(j + 1);
            }
        }
    }
    return f;
}
