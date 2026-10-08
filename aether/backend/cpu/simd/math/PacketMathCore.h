// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file PacketMathCore.h
 * @brief Lane-generic building blocks shared by the CPU packet math headers
 *        (`PacketExpLog.h`, `PacketSinCos.h`, `PacketArcTrig.h`,
 *        `PacketHyperbolic.h`, `PacketRounding.h`, `PacketMisc.h`).
 *
 * HOST-ONLY. Nothing here is device code and nothing here is reached from
 * `aether/aether.h` or from `aether::math`'s default path; a TU gets packet
 * math only by including `aether/backend/cpu/simd/math/PacketMath.h`, or
 * through the opt-in `AETHER_HOST_VECTOR_MATH` hook (`aether/math/detail/
 * HostVectorMath.h`).
 *
 * ONE ALGORITHM, EVERY WIDTH. Each function body is written ONCE, over a
 * "lane carrier" `D` that is `double` (width 1) or one of the GCC/Clang
 * native vector types the `aether::simd::Packet` specializations already
 * wrap (`__m128d` SSE2 width 2, `__m256d` AVX2 width 4, `__m512d` AVX-512
 * width 8). The arithmetic operators are the compiler's own vector
 * extensions; the few operations that need a named instruction (fused
 * multiply-add, square root, mask test) are the overloads below. Because
 * the source is the same and the per-lane operation sequence is the same,
 * a lane computed at width 4 is BIT-IDENTICAL to the same input computed
 * at width 1 in the same TU (same `-m` flags, `-ffp-contract=off`) — which
 * is what lets the auto-vectoriser hook mix vector calls and scalar
 * remainder calls in one loop without the result depending on where an
 * element landed.
 *
 * Integer work (exponent extraction, `2^n` construction, quadrant bits) is
 * done on the 64-bit lane bit patterns with the magic-number conversion
 * `bits(n + 1.5*2^52) - bits(1.5*2^52) == n` (exact for |n| < 2^51), so no
 * width needs a native int64<->double conversion instruction (AVX2 and
 * SSE2 have none).
 *
 * FLOAT. This port covers `double` lanes only; see `PacketMath.h`.
 *
 * REQUIREMENTS ON THE COMPILE. GCC or Clang (vector extensions). Not
 * `-ffast-math`/`-fassociative-math`: the rounding idiom `(x + M) - M`
 * and the error-free transformations below are exactly the expressions
 * those flags are allowed to rewrite.
 */

#include <cstdint>
#include <cstring>
#include <type_traits>

#if defined(__AVX512F__) || defined(__AVX__) || defined(__SSE2__)
#include <immintrin.h>
#endif

#if !defined(__GNUC__)
#error "aether packet math needs GCC/Clang vector extensions"
#endif

#define AETHER_PM_INLINE __attribute__((always_inline)) inline

// The vector carriers are `__m128d`/`__m256d`/`__m512d`, whose alignment /
// may_alias attributes GCC reports as dropped when they appear as template
// arguments; the attributes are irrelevant to every use below (by-value
// lane carriers), so the diagnostic is silenced for this header family only.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"

namespace aether {
namespace simd {
namespace pm {

// ─────────────────────────────────────────────────────────────────────────
//  Lane carrier traits
// ─────────────────────────────────────────────────────────────────────────

template<class D>
struct Lanes;

template<>
struct Lanes<double> {
    static constexpr std::size_t width = 1;
    using L = std::int64_t;   ///< per-lane mask / bit pattern (all-ones = true)
    using U = std::uint64_t;
};

#if defined(__SSE2__)
template<>
struct Lanes<__m128d> {
    static constexpr std::size_t width = 2;
    using L = decltype(__m128d{} < __m128d{});
    typedef std::uint64_t U __attribute__((vector_size(16)));
};
#endif
#if defined(__AVX__)
template<>
struct Lanes<__m256d> {
    static constexpr std::size_t width = 4;
    using L = decltype(__m256d{} < __m256d{});
    typedef std::uint64_t U __attribute__((vector_size(32)));
};
#endif
#if defined(__AVX512F__)
template<>
struct Lanes<__m512d> {
    static constexpr std::size_t width = 8;
    using L = decltype(__m512d{} < __m512d{});
    typedef std::uint64_t U __attribute__((vector_size(64)));
};
#endif

template<class D>
using LaneMask = typename Lanes<D>::L;

template<class D>
inline constexpr bool kScalar = std::is_same_v<D, double>;

// ─────────────────────────────────────────────────────────────────────────
//  Broadcast, bit casts, comparisons, select
// ─────────────────────────────────────────────────────────────────────────

template<class D>
AETHER_PM_INLINE D bc(double v)
{
    if constexpr (kScalar<D>) {
        return v;
    } else {
        D r = {};
        return r + v; // GCC/Clang vector-scalar broadcast
    }
}

template<class D>
AETHER_PM_INLINE LaneMask<D> bits(D x)
{
    if constexpr (kScalar<D>) {
        return __builtin_bit_cast(std::int64_t, x);
    } else {
        return (LaneMask<D>)x;
    }
}

template<class D>
AETHER_PM_INLINE D fromBits(LaneMask<D> b)
{
    if constexpr (kScalar<D>) {
        return __builtin_bit_cast(double, b);
    } else {
        return (D)b;
    }
}

template<class D>
AETHER_PM_INLINE LaneMask<D> ibc(std::int64_t v)
{
    if constexpr (kScalar<D>) {
        return v;
    } else {
        LaneMask<D> r = {};
        return r + v;
    }
}

/// Logical right shift of every 64-bit lane.
template<class D>
AETHER_PM_INLINE LaneMask<D> srl(LaneMask<D> b, int n)
{
    using U = typename Lanes<D>::U;
    return (LaneMask<D>)((U)b >> n);
}

#define AETHER_PM_CMP(NAME, OP)                                                                    \
    template<class D>                                                                              \
    AETHER_PM_INLINE LaneMask<D> NAME(D a, D b)                                                    \
    {                                                                                              \
        if constexpr (kScalar<D>) {                                                                \
            return -static_cast<std::int64_t>(a OP b);                                             \
        } else {                                                                                   \
            return a OP b;                                                                         \
        }                                                                                          \
    }
AETHER_PM_CMP(lt, <)
AETHER_PM_CMP(le, <=)
AETHER_PM_CMP(gt, >)
AETHER_PM_CMP(ge, >=)
AETHER_PM_CMP(eq, ==)
AETHER_PM_CMP(ne, !=)
#undef AETHER_PM_CMP

template<class D>
AETHER_PM_INLINE LaneMask<D> isnan(D a)
{
    return ne(a, a);
}

/// `m ? a : b` per lane (m lanes are all-ones or all-zero).
template<class D>
AETHER_PM_INLINE D select(LaneMask<D> m, D a, D b)
{
    return fromBits<D>((m & bits(a)) | (~m & bits(b)));
}

/// True when ANY lane of `m` is set.
template<class D>
AETHER_PM_INLINE bool any(LaneMask<D> m)
{
    if constexpr (kScalar<D>) {
        return m != 0;
    }
#if defined(__SSE2__)
    else if constexpr (std::is_same_v<D, __m128d>) {
        return _mm_movemask_pd((__m128d)m) != 0;
    }
#endif
#if defined(__AVX__)
    else if constexpr (std::is_same_v<D, __m256d>) {
        return _mm256_movemask_pd((__m256d)m) != 0;
    }
#endif
#if defined(__AVX512F__)
    else if constexpr (std::is_same_v<D, __m512d>) {
        return _mm512_test_epi64_mask((__m512i)m, (__m512i)m) != 0;
    }
#endif
    else {
        static_assert(sizeof(D) == 0, "unsupported lane carrier");
    }
}

constexpr std::int64_t kSign = std::int64_t(0x8000000000000000ULL);
constexpr std::int64_t kAbs = 0x7FFFFFFFFFFFFFFFLL;

template<class D>
AETHER_PM_INLINE D abs(D x)
{
    return fromBits<D>(bits(x) & kAbs);
}

/// |mag| with the sign bit of `sgn` (IEEE copysign, NaN/zero exact).
template<class D>
AETHER_PM_INLINE D copysign(D mag, D sgn)
{
    return fromBits<D>((bits(mag) & kAbs) | (bits(sgn) & kSign));
}

/// Integer lanes that are non-zero, as an all-ones/all-zero mask.
template<class L>
AETHER_PM_INLINE L inz(L v)
{
    if constexpr (std::is_same_v<L, std::int64_t>) {
        return -static_cast<std::int64_t>(v != 0);
    } else {
        return v != 0;
    }
}

/// Lanes whose sign bit is set (includes -0 and -NaN).
template<class D>
AETHER_PM_INLINE LaneMask<D> signbit(D x)
{
    return inz(bits(x) & kSign);
}

/// Flip the sign of the lanes selected by `m`.
template<class D>
AETHER_PM_INLINE D negateIf(LaneMask<D> m, D x)
{
    return fromBits<D>(bits(x) ^ (m & kSign));
}

template<class D>
AETHER_PM_INLINE LaneMask<D> isinf(D x)
{
    return eq(abs(x), bc<D>(__builtin_inf()));
}

// ─────────────────────────────────────────────────────────────────────────
//  Fused multiply-add, square root
// ─────────────────────────────────────────────────────────────────────────

/// `a*b + c`: ONE rounding where the target has FMA (`__FMA__` / AVX-512),
/// otherwise a multiply then an add. The same choice at every width.
template<class D>
AETHER_PM_INLINE D fma(D a, D b, D c)
{
#if defined(__FMA__) || defined(__AVX512F__)
    if constexpr (kScalar<D>) {
        return __builtin_fma(a, b, c);
    }
#if defined(__SSE2__)
    else if constexpr (std::is_same_v<D, __m128d>) {
        return _mm_fmadd_pd(a, b, c);
    }
#endif
#if defined(__AVX__)
    else if constexpr (std::is_same_v<D, __m256d>) {
        return _mm256_fmadd_pd(a, b, c);
    }
#endif
#if defined(__AVX512F__)
    else if constexpr (std::is_same_v<D, __m512d>) {
        return _mm512_fmadd_pd(a, b, c);
    }
#endif
    else {
        static_assert(sizeof(D) == 0, "unsupported lane carrier");
    }
#else
    return a * b + c;
#endif
}

template<class D>
AETHER_PM_INLINE D sqrt(D x)
{
    if constexpr (kScalar<D>) {
        return __builtin_sqrt(x);
    }
#if defined(__SSE2__)
    else if constexpr (std::is_same_v<D, __m128d>) {
        return _mm_sqrt_pd(x);
    }
#endif
#if defined(__AVX__)
    else if constexpr (std::is_same_v<D, __m256d>) {
        return _mm256_sqrt_pd(x);
    }
#endif
#if defined(__AVX512F__)
    else if constexpr (std::is_same_v<D, __m512d>) {
        return _mm512_sqrt_pd(x);
    }
#endif
    else {
        static_assert(sizeof(D) == 0, "unsupported lane carrier");
    }
}

/// Horner evaluation, coefficients highest degree first.
template<class D, std::size_t N>
AETHER_PM_INLINE D horner(D z, const double (&c)[N])
{
    D p = bc<D>(c[0]);
    for (std::size_t i = 1; i < N; ++i)
        p = fma(p, z, bc<D>(c[i]));
    return p;
}

// ─────────────────────────────────────────────────────────────────────────
//  Error-free transformations (double-double building blocks)
// ─────────────────────────────────────────────────────────────────────────

/// `p + e == a*b` exactly (barring over/underflow). FMA where available,
/// Dekker/Veltkamp splitting otherwise.
template<class D>
AETHER_PM_INLINE void twoProd(D a, D b, D& p, D& e)
{
    p = a * b;
#if defined(__FMA__) || defined(__AVX512F__)
    e = fma(a, b, -p);
#else
    const D c = bc<D>(134217729.0); // 2^27 + 1
    D ta = c * a;
    D ah = ta - (ta - a);
    D al = a - ah;
    D tb = c * b;
    D bh = tb - (tb - b);
    D bl = b - bh;
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
#endif
}

/// `s + e == a + b` exactly (Knuth TwoSum, no ordering precondition).
template<class D>
AETHER_PM_INLINE void twoSum(D a, D b, D& s, D& e)
{
    s = a + b;
    D bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}

/// `s + e == a + b` exactly when |a| >= |b| or a == 0 (Dekker Fast2Sum).
template<class D>
AETHER_PM_INLINE void fastTwoSum(D a, D b, D& s, D& e)
{
    s = a + b;
    e = b - (s - a);
}

/// `c - a*b`, exact whenever that value is representable (the remainder of
/// a correctly rounded quotient or square root): one FMA where the target
/// has it, an exact product split otherwise (never two plain roundings).
template<class D>
AETHER_PM_INLINE D residual(D a, D b, D c)
{
#if defined(__FMA__) || defined(__AVX512F__)
    return fma(-a, b, c);
#else
    D p, e;
    twoProd(a, b, p, e);
    return (c - p) - e;
#endif
}

/// `(qh + ql) = (ah + al) / (bh + bl)` to about 2^-100 relative; qh is the
/// rounded quotient of the high parts, ql its correction.
template<class D>
AETHER_PM_INLINE void ddDiv(D ah, D al, D bh, D bl, D& qh, D& ql)
{
    qh = ah / bh;
    ql = (residual(qh, bh, ah) + (al - qh * bl)) / bh;
}

/// `(ph + pl) = (ah + al) * (bh + bl)` to about 2^-100 relative.
template<class D>
AETHER_PM_INLINE void ddMul(D ah, D al, D bh, D bl, D& ph, D& pl)
{
    D e;
    twoProd(ah, bh, ph, e);
    pl = e + (ah * bl + al * bh);
}

/// `(sh + sl) = sqrt(xh + xl)` for xh > 0 (xh == 0 gives NaN in sl).
template<class D>
AETHER_PM_INLINE void ddSqrt(D xh, D xl, D& sh, D& sl)
{
    sh = sqrt(xh);
    sl = (residual(sh, sh, xh) + xl) / (sh + sh);
}

// ─────────────────────────────────────────────────────────────────────────
//  Integer-valued doubles <-> lane integers, powers of two
// ─────────────────────────────────────────────────────────────────────────

inline constexpr double kRoundMagic = 6755399441055744.0; // 1.5 * 2^52

/// Round to nearest-even integer (current rounding mode), valid for
/// |x| < 2^51; larger |x| are returned unchanged only if already integral
/// — callers bound their arguments first.
template<class D>
AETHER_PM_INLINE D rintSmall(D x)
{
    const D m = bc<D>(kRoundMagic);
    return (x + m) - m;
}

/// Lane integer of an integer-valued double with |n| < 2^51 (exact).
template<class D>
AETHER_PM_INLINE LaneMask<D> toInt(D n)
{
    return bits(n + bc<D>(kRoundMagic)) - bits(bc<D>(kRoundMagic));
}

/// 2^n for integer-valued n in [-1022, 1023] (exact, normal result).
template<class D>
AETHER_PM_INLINE D pow2(D n)
{
    return fromBits<D>(bits(n + bc<D>(kRoundMagic + 1023.0)) << 52);
}

/// Biased exponent field of each lane (as a double in [0, 2047]).
template<class D>
AETHER_PM_INLINE D biasedExponent(D x)
{
    return fromBits<D>((srl<D>(bits(x), 52) & 0x7FF) | bits(bc<D>(kRoundMagic))) - bc<D>(kRoundMagic);
}

/// Round |x| (x >= 0) to the nearest-even integer, any magnitude (values
/// >= 2^52 are already integral and pass through; +inf and NaN pass through).
template<class D>
AETHER_PM_INLINE D rintNonNeg(D ax)
{
    const D two52 = bc<D>(4503599627370496.0);
    return select(lt(ax, two52), (ax + two52) - two52, ax);
}

/// Apply `f` (a scalar `double -> double` callable) to the lanes selected
/// by `m`, keeping `v` on the others. The escape hatch for arguments a
/// vector algorithm does not cover (e.g. huge trig arguments).
template<class D, class F>
AETHER_PM_INLINE D laneFallback(LaneMask<D> m, D x, D v, F f)
{
    if constexpr (kScalar<D>) {
        return m ? f(x) : v;
    } else {
        constexpr std::size_t W = Lanes<D>::width;
        double xb[W], vb[W];
        std::int64_t mb[W];
        std::memcpy(xb, &x, sizeof(D));
        std::memcpy(vb, &v, sizeof(D));
        std::memcpy(mb, &m, sizeof(D));
        for (std::size_t i = 0; i < W; ++i)
            if (mb[i])
                vb[i] = f(xb[i]);
        D r;
        std::memcpy(&r, vb, sizeof(D));
        return r;
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  Packet <-> lane carrier
// ─────────────────────────────────────────────────────────────────────────

template<std::size_t W>
struct CarrierOf;
template<>
struct CarrierOf<1> {
    using type = double;
};
#if defined(__SSE2__)
template<>
struct CarrierOf<2> {
    using type = __m128d;
};
#endif
#if defined(__AVX__)
template<>
struct CarrierOf<4> {
    using type = __m256d;
};
#endif
#if defined(__AVX512F__)
template<>
struct CarrierOf<8> {
    using type = __m512d;
};
#endif

template<std::size_t W>
using Carrier = typename CarrierOf<W>::type;

} // namespace pm
} // namespace simd
} // namespace aether

#pragma GCC diagnostic pop
