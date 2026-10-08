// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file HostVectorMath.h
 * @brief Opt-in auto-vectoriser hook: makes `aether::math::{exp, log, pow,
 *        sin, ...}` on `double` callable from SCALAR host loops that GCC
 *        then vectorises, with the vector lanes computed by the packet math
 *        in `aether/backend/cpu/simd/math/`.
 *
 * ACTIVATION. Only when the TU defines `AETHER_HOST_VECTOR_MATH` before
 * including any aether header (e.g. `-DAETHER_HOST_VECTOR_MATH`), and only
 * for host compiles by GCC on x86-64 that are not nvcc passes. Without the
 * macro this header is never included and every `aether::math` host leg is
 * the `std::` call it always was — same tokens, same code, same bits.
 *
 * MECHANISM. Each routed function `f` is declared once as
 * `extern "C" double aether_hvm_f(double...)` with GCC's
 * `__attribute__((simd))` — the x86-64 vector function ABI contract (the
 * one libmvec uses). It tells the vectoriser that SSE (`b`, 2 lanes), AVX
 * (`c`, 4), AVX2 (`d`, 4) and AVX-512 (`e`, 8) variants, unmasked (`N`) and
 * masked (`M`), exist under the mangled names `_ZGV<isa><mask><lanes><v..>_
 * aether_hvm_f`. This header defines exactly those variants (for the ISAs
 * the TU is compiled for) plus the scalar symbol, each as a hidden-
 * visibility COMDAT so any number of TUs can include it. The bodies are
 * the packet functions themselves (`pm::vpow` on `__m256d`, ...), and the
 * scalar symbol runs the same algorithm at width 1, so an element's value
 * does not depend on whether it was computed in the vector body or in a
 * remainder iteration.
 * Masked variants evaluate every lane (inactive lanes are replaced by 1.0
 * first so no lane takes a slow path).
 *
 * EMISSION. Nothing in the source names a vector variant: the vectoriser
 * creates the call to `_ZGV..._aether_hvm_f` late in optimisation, after GCC
 * has dropped every inline function nothing referenced. Each variant must
 * therefore be kept alive explicitly, or the TU's shared object fails to
 * load on an undefined `_ZGV...` symbol. Marking every variant `used` kept
 * all of them alive in every TU — every function at every width compiled at
 * -O3 even in a TU that calls no math at all, most of such a TU's compile
 * time. Instead, one `used` anchor per function (`hvm_abi::f_keep<D>`, a
 * variable template holding the addresses of `f`'s scalar symbol and
 * variants) is instantiated by the facade's `double` leg `hostvec::f` —
 * itself a template, so only once a call of `aether::math::f` on `double`
 * is instantiated. A TU emits exactly the symbols of the functions it calls
 * (the anchor costs no instruction in the caller), and the emitted code of
 * those symbols and of the calling loops is unchanged.
 *
 * ACCURACY. The routed functions are no longer glibc's, but they are
 * faithfully rounded (error < 1 ULP against the exact result, verified
 * against a 128-bit reference): see the table in
 * `aether/backend/cpu/simd/math/PacketMath.h`. `float` arguments stay on
 * `std::`.
 *
 * CONTRACTION. The packet math and every symbol defined here are compiled
 * with FMA contraction OFF whatever the TU asks for (`#pragma GCC optimize
 * ("fp-contract=off")` around them, push/pop so the TU's own code keeps its
 * setting). The packet math is built from error-free transformations and
 * argument reductions (two-sum, two-product, Cody-Waite and Payne-Hanek legs)
 * that are exact only when each operation rounds where it is written: a TU
 * built with `-ffp-contract=fast` — gcc's default, and a caller's legitimate
 * choice for its own arithmetic — would otherwise fuse them and move sin,
 * cos, tan, atan, asin, acos and atan2 off their faithful results. The
 * attribute only holds for code compiled as part of these out-of-line
 * symbols, which is how a calling loop reaches them (a vector-ABI call, or
 * a call to the scalar symbol: gcc does not inline a function across a
 * differing `optimize` attribute). Code that calls the `pm::` packet
 * functions directly inlines them under its OWN flags. Pinned by
 * tests/test_PacketMath_common.h's `HookUnderContractionMatchesKernels`
 * (pm_hook.cpp is built with `-ffp-contract=fast`).
 *
 * ODR. Every TU in one binary that defines the macro must be compiled for
 * the same `-m`/`-march` target, as for any inline function whose body
 * depends on ISA flags.
 *
 * Routed: exp, exp2, exp10, expm1, log, log2, log10, log1p, pow, sin,
 * cos, tan, tanh, sinh, cosh, asinh, acosh, atanh, atan, atan2, asin,
 * acos, cbrt, hypot, sqrt, rsqrt, floor, ceil, trunc, round, rint, fdim
 * (vector ABI); abs, copysign, fmax, fmin (inline, see `hostvec::fmax`).
 * Not routed (no packet; stay on `std::`): fmod, fma, erf, erfc; the
 * numpy-semantics `remainder` is built on `fmod`. `sincos` reaches the hook
 * through its `sin` and `cos` legs: GCC does not vectorise a call that
 * writes through pointer arguments, so a single two-output symbol would
 * keep the loop scalar.
 */

#if !defined(__GNUC__) || defined(__clang__) || !defined(__x86_64__)
#error "AETHER_HOST_VECTOR_MATH needs GCC targeting x86-64 (vector function ABI)"
#endif

#include <cmath>
#include <immintrin.h>
#include <type_traits>

// Contraction off for the packet math and the symbols below (CONTRACTION
// above). <immintrin.h> comes first so the intrinsics keep their own
// attributes; popped before the inline `hostvec` routes, which must inline
// into the caller's loop.
#pragma GCC push_options
#pragma GCC optimize("fp-contract=off")

#include "aether/backend/cpu/simd/math/PacketArcTrig.h"
#include "aether/backend/cpu/simd/math/PacketExpLog.h"
#include "aether/backend/cpu/simd/math/PacketHyperbolic.h"
#include "aether/backend/cpu/simd/math/PacketMisc.h"
#include "aether/backend/cpu/simd/math/PacketRounding.h"
#include "aether/backend/cpu/simd/math/PacketSinCos.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"

#define AETHER_HVM_ATTR __attribute__((visibility("hidden"))) inline

#if defined(__AVX__)
#define AETHER_HVM_IF_AVX(...) __VA_ARGS__
#else
#define AETHER_HVM_IF_AVX(...)
#endif
#if defined(__AVX2__)
#define AETHER_HVM_IF_AVX2(...) __VA_ARGS__
#else
#define AETHER_HVM_IF_AVX2(...)
#endif
#if defined(__AVX512F__)
#define AETHER_HVM_IF_AVX512(...) __VA_ARGS__
#else
#define AETHER_HVM_IF_AVX512(...)
#endif

// One vector-ABI variant: C++ name NAME_TAG, assembler name SYM.
#define AETHER_HVM_U(NAME, IMPL, TAG, SYM, V)                                                       \
    AETHER_HVM_ATTR V NAME##_##TAG(V x) __asm__(SYM);                                              \
    AETHER_HVM_ATTR V NAME##_##TAG(V x) { return ::aether::simd::pm::IMPL(x); }
#define AETHER_HVM_UM(NAME, IMPL, TAG, SYM, V)                                                      \
    AETHER_HVM_ATTR V NAME##_##TAG(V x, V m) __asm__(SYM);                                         \
    AETHER_HVM_ATTR V NAME##_##TAG(V x, V m)                                                       \
    {                                                                                              \
        namespace pm = ::aether::simd::pm;                                                         \
        return pm::IMPL(pm::select<V>(pm::inz(pm::bits(m)), x, pm::bc<V>(1.0)));                   \
    }
#define AETHER_HVM_UM8(NAME, IMPL, TAG, SYM)                                                        \
    AETHER_HVM_ATTR __m512d NAME##_##TAG(__m512d x, __mmask8 m) __asm__(SYM);                      \
    AETHER_HVM_ATTR __m512d NAME##_##TAG(__m512d x, __mmask8 m)                                    \
    {                                                                                              \
        return ::aether::simd::pm::IMPL(_mm512_mask_blend_pd(m, _mm512_set1_pd(1.0), x));          \
    }
#define AETHER_HVM_B(NAME, IMPL, TAG, SYM, V)                                                       \
    AETHER_HVM_ATTR V NAME##_##TAG(V x, V y) __asm__(SYM);                                         \
    AETHER_HVM_ATTR V NAME##_##TAG(V x, V y) { return ::aether::simd::pm::IMPL(x, y); }
#define AETHER_HVM_BM(NAME, IMPL, TAG, SYM, V)                                                      \
    AETHER_HVM_ATTR V NAME##_##TAG(V x, V y, V m) __asm__(SYM);                                    \
    AETHER_HVM_ATTR V NAME##_##TAG(V x, V y, V m)                                                  \
    {                                                                                              \
        namespace pm = ::aether::simd::pm;                                                         \
        const auto on = pm::inz(pm::bits(m));                                                      \
        return pm::IMPL(pm::select<V>(on, x, pm::bc<V>(1.0)), pm::select<V>(on, y, pm::bc<V>(1.0))); \
    }
#define AETHER_HVM_BM8(NAME, IMPL, TAG, SYM)                                                        \
    AETHER_HVM_ATTR __m512d NAME##_##TAG(__m512d x, __m512d y, __mmask8 m) __asm__(SYM);           \
    AETHER_HVM_ATTR __m512d NAME##_##TAG(__m512d x, __m512d y, __mmask8 m)                         \
    {                                                                                              \
        const __m512d one = _mm512_set1_pd(1.0);                                                   \
        return ::aether::simd::pm::IMPL(_mm512_mask_blend_pd(m, one, x), _mm512_mask_blend_pd(m, one, y)); \
    }

/// The emission anchor of `NAME`'s symbols (EMISSION above): a `used`
/// variable holding the address of the scalar symbol and of every vector
/// variant this TU defines. A variable template, so it exists only once
/// `hostvec::NAME` odr-uses it, i.e. once a facade call of `NAME` on
/// `double` is instantiated.
#define AETHER_HVM_KEEP(NAME)                                                                       \
    struct NAME##_keep_t {                                                                         \
        decltype(&NAME##_s1) s1;                                                                   \
        decltype(&NAME##_b2) b2;                                                                   \
        decltype(&NAME##_b2m) b2m;                                                                 \
        AETHER_HVM_IF_AVX(decltype(&NAME##_c4) c4; decltype(&NAME##_c4m) c4m;)                     \
        AETHER_HVM_IF_AVX2(decltype(&NAME##_d4) d4; decltype(&NAME##_d4m) d4m;)                    \
        AETHER_HVM_IF_AVX512(decltype(&NAME##_e8) e8; decltype(&NAME##_e8m) e8m;)                  \
    };                                                                                             \
    template<class D>                                                                              \
    __attribute__((used, visibility("hidden"))) inline constexpr NAME##_keep_t NAME##_keep{        \
        &NAME##_s1, &NAME##_b2, &NAME##_b2m AETHER_HVM_IF_AVX(, &NAME##_c4, &NAME##_c4m)           \
            AETHER_HVM_IF_AVX2(, &NAME##_d4, &NAME##_d4m) AETHER_HVM_IF_AVX512(, &NAME##_e8, &NAME##_e8m)};

/// Declares `aether_hvm_NAME(double)` as a simd function and defines the
/// scalar symbol and every vector variant the TU's ISA can call.
#define AETHER_HVM_DEFINE_UNARY(NAME, IMPL)                                                         \
    extern "C" {                                                                                   \
    __attribute__((simd, const, nothrow)) double aether_hvm_##NAME(double);                        \
    }                                                                                              \
    namespace aether {                                                                             \
    namespace math {                                                                               \
    namespace detail {                                                                             \
    namespace hvm_abi {                                                                            \
    AETHER_HVM_ATTR double NAME##_s1(double x) __asm__("aether_hvm_" #NAME);                       \
    AETHER_HVM_ATTR double NAME##_s1(double x) { return ::aether::simd::pm::IMPL(x); }             \
    AETHER_HVM_U(NAME, IMPL, b2, "_ZGVbN2v_aether_hvm_" #NAME, __m128d)                            \
    AETHER_HVM_UM(NAME, IMPL, b2m, "_ZGVbM2v_aether_hvm_" #NAME, __m128d)                          \
    AETHER_HVM_IF_AVX(AETHER_HVM_U(NAME, IMPL, c4, "_ZGVcN4v_aether_hvm_" #NAME, __m256d)          \
            AETHER_HVM_UM(NAME, IMPL, c4m, "_ZGVcM4v_aether_hvm_" #NAME, __m256d))                 \
    AETHER_HVM_IF_AVX2(AETHER_HVM_U(NAME, IMPL, d4, "_ZGVdN4v_aether_hvm_" #NAME, __m256d)         \
            AETHER_HVM_UM(NAME, IMPL, d4m, "_ZGVdM4v_aether_hvm_" #NAME, __m256d))                 \
    AETHER_HVM_IF_AVX512(AETHER_HVM_U(NAME, IMPL, e8, "_ZGVeN8v_aether_hvm_" #NAME, __m512d)       \
            AETHER_HVM_UM8(NAME, IMPL, e8m, "_ZGVeM8v_aether_hvm_" #NAME))                         \
    AETHER_HVM_KEEP(NAME)                                                                          \
    }                                                                                              \
    }                                                                                              \
    }                                                                                              \
    }

#define AETHER_HVM_DEFINE_BINARY(NAME, IMPL)                                                        \
    extern "C" {                                                                                   \
    __attribute__((simd, const, nothrow)) double aether_hvm_##NAME(double, double);                \
    }                                                                                              \
    namespace aether {                                                                             \
    namespace math {                                                                               \
    namespace detail {                                                                             \
    namespace hvm_abi {                                                                            \
    AETHER_HVM_ATTR double NAME##_s1(double x, double y) __asm__("aether_hvm_" #NAME);             \
    AETHER_HVM_ATTR double NAME##_s1(double x, double y) { return ::aether::simd::pm::IMPL(x, y); } \
    AETHER_HVM_B(NAME, IMPL, b2, "_ZGVbN2vv_aether_hvm_" #NAME, __m128d)                           \
    AETHER_HVM_BM(NAME, IMPL, b2m, "_ZGVbM2vv_aether_hvm_" #NAME, __m128d)                         \
    AETHER_HVM_IF_AVX(AETHER_HVM_B(NAME, IMPL, c4, "_ZGVcN4vv_aether_hvm_" #NAME, __m256d)         \
            AETHER_HVM_BM(NAME, IMPL, c4m, "_ZGVcM4vv_aether_hvm_" #NAME, __m256d))                \
    AETHER_HVM_IF_AVX2(AETHER_HVM_B(NAME, IMPL, d4, "_ZGVdN4vv_aether_hvm_" #NAME, __m256d)        \
            AETHER_HVM_BM(NAME, IMPL, d4m, "_ZGVdM4vv_aether_hvm_" #NAME, __m256d))                \
    AETHER_HVM_IF_AVX512(AETHER_HVM_B(NAME, IMPL, e8, "_ZGVeN8vv_aether_hvm_" #NAME, __m512d)      \
            AETHER_HVM_BM8(NAME, IMPL, e8m, "_ZGVeM8vv_aether_hvm_" #NAME))                        \
    AETHER_HVM_KEEP(NAME)                                                                          \
    }                                                                                              \
    }                                                                                              \
    }                                                                                              \
    }

AETHER_HVM_DEFINE_UNARY(exp, vexp)
AETHER_HVM_DEFINE_UNARY(exp2, vexp2)
AETHER_HVM_DEFINE_UNARY(exp10, vexp10)
AETHER_HVM_DEFINE_UNARY(log, vlog)
AETHER_HVM_DEFINE_UNARY(log2, vlog2)
AETHER_HVM_DEFINE_UNARY(log10, vlog10)
AETHER_HVM_DEFINE_BINARY(pow, vpow)
AETHER_HVM_DEFINE_UNARY(sin, vsin)
AETHER_HVM_DEFINE_UNARY(cos, vcos)
AETHER_HVM_DEFINE_UNARY(tan, vtan)
AETHER_HVM_DEFINE_UNARY(tanh, vtanh)
AETHER_HVM_DEFINE_UNARY(atan, vatan)
AETHER_HVM_DEFINE_BINARY(atan2, vatan2)
AETHER_HVM_DEFINE_UNARY(asin, vasin)
AETHER_HVM_DEFINE_UNARY(acos, vacos)
AETHER_HVM_DEFINE_UNARY(cbrt, vcbrt)
AETHER_HVM_DEFINE_BINARY(hypot, vhypot)
AETHER_HVM_DEFINE_UNARY(sqrt, sqrt)
AETHER_HVM_DEFINE_UNARY(rsqrt, vrsqrt)
AETHER_HVM_DEFINE_UNARY(floor, vfloor)
AETHER_HVM_DEFINE_UNARY(expm1, vexpm1)
AETHER_HVM_DEFINE_UNARY(log1p, vlog1p)
AETHER_HVM_DEFINE_UNARY(sinh, vsinh)
AETHER_HVM_DEFINE_UNARY(cosh, vcosh)
AETHER_HVM_DEFINE_UNARY(asinh, vasinh)
AETHER_HVM_DEFINE_UNARY(acosh, vacosh)
AETHER_HVM_DEFINE_UNARY(atanh, vatanh)
AETHER_HVM_DEFINE_UNARY(ceil, vceil)
AETHER_HVM_DEFINE_UNARY(trunc, vtrunc)
AETHER_HVM_DEFINE_UNARY(round, vround)
AETHER_HVM_DEFINE_UNARY(rint, vrint)
AETHER_HVM_DEFINE_BINARY(fdim, vfdim)

#undef AETHER_HVM_DEFINE_UNARY
#undef AETHER_HVM_DEFINE_BINARY
#undef AETHER_HVM_KEEP
#undef AETHER_HVM_U
#undef AETHER_HVM_UM
#undef AETHER_HVM_UM8
#undef AETHER_HVM_B
#undef AETHER_HVM_BM
#undef AETHER_HVM_BM8
#undef AETHER_HVM_IF_AVX
#undef AETHER_HVM_IF_AVX2
#undef AETHER_HVM_IF_AVX512
#undef AETHER_HVM_ATTR

#pragma GCC pop_options

namespace aether {
namespace math {
namespace detail {
/// The facade's host legs under `AETHER_HOST_VECTOR_MATH`: `double` goes
/// to the simd-declared symbol, `float` stays on `std::`.
namespace hostvec {
// The `double` legs are templates so that the `(void)&...keep<D>` odr-use
// (no code) instantiates NAME's emission anchor only when a facade call of
// NAME on `double` is itself instantiated (EMISSION above).
#define AETHER_HVM_ROUTE1(NAME, FLOATEXPR)                                                          \
    template<class D>                                                                              \
        requires std::is_same_v<D, double>                                                         \
    inline D NAME(D x)                                                                             \
    {                                                                                              \
        (void)&hvm_abi::NAME##_keep<D>;                                                            \
        return ::aether_hvm_##NAME(x);                                                             \
    }                                                                                              \
    inline float NAME(float x) { return FLOATEXPR; }
#define AETHER_HVM_ROUTE2(NAME)                                                                     \
    template<class D>                                                                              \
        requires std::is_same_v<D, double>                                                         \
    inline D NAME(D a, D b)                                                                        \
    {                                                                                              \
        (void)&hvm_abi::NAME##_keep<D>;                                                            \
        return ::aether_hvm_##NAME(a, b);                                                          \
    }                                                                                              \
    inline float NAME(float a, float b) { return std::NAME(a, b); }
AETHER_HVM_ROUTE1(exp, std::exp(x))
AETHER_HVM_ROUTE1(exp2, std::exp2(x))
AETHER_HVM_ROUTE1(exp10, std::pow(10.0f, x))
AETHER_HVM_ROUTE1(log, std::log(x))
AETHER_HVM_ROUTE1(log2, std::log2(x))
AETHER_HVM_ROUTE1(log10, std::log10(x))
AETHER_HVM_ROUTE2(pow)
AETHER_HVM_ROUTE1(sin, std::sin(x))
AETHER_HVM_ROUTE1(cos, std::cos(x))
AETHER_HVM_ROUTE1(tan, std::tan(x))
AETHER_HVM_ROUTE1(tanh, std::tanh(x))
AETHER_HVM_ROUTE1(atan, std::atan(x))
AETHER_HVM_ROUTE2(atan2)
AETHER_HVM_ROUTE1(asin, std::asin(x))
AETHER_HVM_ROUTE1(acos, std::acos(x))
AETHER_HVM_ROUTE1(cbrt, std::cbrt(x))
AETHER_HVM_ROUTE2(hypot)
AETHER_HVM_ROUTE1(sqrt, std::sqrt(x))
AETHER_HVM_ROUTE1(rsqrt, 1.0f / std::sqrt(x))
AETHER_HVM_ROUTE1(floor, std::floor(x))
AETHER_HVM_ROUTE1(expm1, std::expm1(x))
AETHER_HVM_ROUTE1(log1p, std::log1p(x))
AETHER_HVM_ROUTE1(sinh, std::sinh(x))
AETHER_HVM_ROUTE1(cosh, std::cosh(x))
AETHER_HVM_ROUTE1(asinh, std::asinh(x))
AETHER_HVM_ROUTE1(acosh, std::acosh(x))
AETHER_HVM_ROUTE1(atanh, std::atanh(x))
AETHER_HVM_ROUTE1(ceil, std::ceil(x))
AETHER_HVM_ROUTE1(trunc, std::trunc(x))
AETHER_HVM_ROUTE1(round, std::round(x))
AETHER_HVM_ROUTE1(rint, std::rint(x))
AETHER_HVM_ROUTE2(fdim)
// abs: a sign-bit mask GCC already vectorises inline, so it stays inline
// (the packet `abs`, bit for bit what std::fabs gives).
inline double abs(double x) { return ::aether::simd::pm::abs(x); }
inline float abs(float x) { return std::abs(x); }
// copysign: the same kind of sign-bit mask, inline for the same reason.
inline double copysign(double a, double b) { return ::aether::simd::pm::copysign(a, b); }
inline float copysign(float a, float b) { return std::copysign(a, b); }
// fmax/fmin: a library call the vectoriser cannot widen (glibc has no vector
// variant), so they would keep any loop that uses them scalar. Written as the
// compare-select glibc's out-of-line x86-64 fmax/fmin compute (the SECOND
// operand on a tie; a NaN operand yields the other), which GCC vectorises as
// a compare + blend. Same results as std:: bit for bit, except NaN payloads
// and the sign of a +0/-0 tie, which C leaves unspecified (GCC's own inline
// expansion of std::fmax already returns the first operand there).
inline double fmax(double a, double b) { return ((a > b) | (b != b)) ? a : b; }
inline double fmin(double a, double b) { return ((a < b) | (b != b)) ? a : b; }
inline float fmax(float a, float b) { return std::fmax(a, b); }
inline float fmin(float a, float b) { return std::fmin(a, b); }
#undef AETHER_HVM_ROUTE1
#undef AETHER_HVM_ROUTE2
} // namespace hostvec
} // namespace detail
} // namespace math
} // namespace aether

#pragma GCC diagnostic pop
