// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Packet.h
 * @brief `aether::simd::Packet<DataT, Width>` — SIMD packet type wrapping
 *        platform intrinsics for CPU vectorization.
 *
 * Full specializations for:
 *
 *   | DataT  | Width | Register  | ISA        |
 *   |--------|-------|-----------|------------|
 *   | double | 8     | `__m512d` | AVX-512    |
 *   | float  | 16    | `__m512`  | AVX-512    |
 *   | double | 4     | `__m256d` | AVX2       |
 *   | float  | 8     | `__m256`  | AVX2       |
 *   | double | 2     | `__m128d` | SSE2       |
 *   | float  | 4     | `__m128`  | SSE2       |
 *   | any    | 1     | scalar    | (fallback) |
 *
 * Member surface, kept deliberately narrow: `zero`/`broadcast`/`load`/
 * `maskLoad`, `store`/`maskStore`, arithmetic (`+`,`-`,`*`,`/`, unary `-`)
 * + `fmadd`, comparison-to-mask (`cmpGe`/`cmpLt`/`cmpGt`), and `select`.
 * `abs`/`sqrt`/`max`/`min`/`fmsub`/`fnmadd`/`copysign`/`pow`, the
 * compound-assignment operators, and any `Native*` aliases are added
 * alongside whichever expression node first needs them.
 *
 * `select` (a lane-wise `mask ? a : b` blend) exists because the eagle
 * port needed it (`eagle::cpu::Host`'s `selectMasked` workaround, built
 * built entirely over the public `store`/`maskStore`/`load` surface because
 * this header omitted it). Each specialization below gets a hidden-friend
 * `select(MaskT, Packet, Packet)`, exactly like `fmadd`: an ISA blend
 * intrinsic where the hardware has one (AVX-512 `mask_blend`, AVX2
 * `blendv`), a vectorised bitwise composition (`(mask & a) | (~mask & b)`)
 * where it does not (plain SSE2, which predates `blendv`), and the bare
 * ternary for the Width==1 scalar case. Hidden friend, not a free function
 * at namespace scope, on purpose: an unqualified `select(mask, a, b)` call
 * is found only via ADL on the `aether::simd::Packet`/`PacketMask`
 * argument types, which is what lets it coexist with POSIX `::%select`
 * (`<sys/select.h>`, 5 parameters) in the same translation unit — see
 * `tests/test_SimdSelect.cpp`'s RED-before-GREEN comment for the exact
 * diagnostic an unqualified call produced before this existed.
 */

#include <cstdint>

#include "aether/backend/cpu/simd/PacketMask.h"
#include "aether/backend/cpu/simd/PacketTraits.h"

namespace aether {
namespace simd {

// ═══════════════════════════════════════════════════════════════════════
//  Primary template — never instantiated directly; only the
//  specializations below are used.
// ═══════════════════════════════════════════════════════════════════════

template<class DataT, std::size_t Width>
struct Packet;

// ═══════════════════════════════════════════════════════════════════════
//  Width == 1  (scalar fallback — zero overhead)
// ═══════════════════════════════════════════════════════════════════════

template<class DataT>
struct Packet<DataT, 1> {
    using MaskT = PacketMask<DataT, 1>;
    static constexpr std::size_t width = 1;

    DataT val_;

    Packet() = default;
    explicit Packet(DataT v)
        : val_(v)
    {
    }

    static Packet broadcast(DataT v) { return Packet{ v }; }
    static Packet zero() { return Packet{ DataT{ 0 } }; }

    static Packet load(const DataT* p) { return Packet{ *p }; }
    static void store(DataT* p, Packet a) { *p = a.val_; }
    static Packet maskLoad(const DataT* p, MaskT m) { return m.mask_ ? Packet{ *p } : zero(); }
    static void maskStore(DataT* p, MaskT m, Packet a)
    {
        if (m.mask_)
            *p = a.val_;
    }

    friend Packet operator+(Packet a, Packet b) { return Packet{ a.val_ + b.val_ }; }
    friend Packet operator-(Packet a, Packet b) { return Packet{ a.val_ - b.val_ }; }
    friend Packet operator*(Packet a, Packet b) { return Packet{ a.val_ * b.val_ }; }
    friend Packet operator/(Packet a, Packet b) { return Packet{ a.val_ / b.val_ }; }
    Packet operator-() const { return Packet{ -val_ }; }

    friend Packet fmadd(Packet a, Packet b, Packet c) { return Packet{ a.val_ * b.val_ + c.val_ }; }

    /** @brief `mask ? a : b`, lane-wise — here just the C++ ternary (Width==1). */
    friend Packet select(MaskT m, Packet a, Packet b) { return m.mask_ ? a : b; }

    friend MaskT cmpGe(Packet a, Packet b) { return MaskT{ a.val_ >= b.val_ }; }
    friend MaskT cmpLt(Packet a, Packet b) { return MaskT{ a.val_ < b.val_ }; }
    friend MaskT cmpGt(Packet a, Packet b) { return MaskT{ a.val_ > b.val_ }; }
};

// ═══════════════════════════════════════════════════════════════════════
//  AVX-512  double×8, float×16
// ═══════════════════════════════════════════════════════════════════════

#if defined(__AVX512F__)

template<>
struct Packet<double, 8> {
    using MaskT = PacketMask<double, 8>;
    static constexpr std::size_t width = 8;

    __m512d reg_;

    Packet() = default;
    explicit Packet(__m512d r)
        : reg_(r)
    {
    }

    static Packet broadcast(double v) { return Packet{ _mm512_set1_pd(v) }; }
    static Packet zero() { return Packet{ _mm512_setzero_pd() }; }

    static Packet load(const double* p) { return Packet{ _mm512_loadu_pd(p) }; }
    static void store(double* p, Packet a) { _mm512_storeu_pd(p, a.reg_); }
    static Packet maskLoad(const double* p, MaskT m) { return Packet{ _mm512_maskz_loadu_pd(m.mask_, p) }; }
    static void maskStore(double* p, MaskT m, Packet a) { _mm512_mask_storeu_pd(p, m.mask_, a.reg_); }

    friend Packet operator+(Packet a, Packet b) { return Packet{ _mm512_add_pd(a.reg_, b.reg_) }; }
    friend Packet operator-(Packet a, Packet b) { return Packet{ _mm512_sub_pd(a.reg_, b.reg_) }; }
    friend Packet operator*(Packet a, Packet b) { return Packet{ _mm512_mul_pd(a.reg_, b.reg_) }; }
    friend Packet operator/(Packet a, Packet b) { return Packet{ _mm512_div_pd(a.reg_, b.reg_) }; }
    Packet operator-() const { return Packet{ _mm512_sub_pd(_mm512_setzero_pd(), reg_) }; }

    friend Packet fmadd(Packet a, Packet b, Packet c) { return Packet{ _mm512_fmadd_pd(a.reg_, b.reg_, c.reg_) }; }

    /** @brief `mask ? a : b`, lane-wise — AVX-512 has a dedicated blend
     *  instruction driven straight off the `__mmask8` (no vector mask to
     *  build first, unlike AVX2/SSE2 below). `_mm512_mask_blend_pd(k, x, y)`
     *  takes `y` where bit `k` is set and `x` otherwise, so `x`/`y` are
     *  passed as `b`/`a` (swapped from this function's own argument order)
     *  to land `a` on the SET (mask-true) lanes. */
    friend Packet select(MaskT m, Packet a, Packet b) { return Packet{ _mm512_mask_blend_pd(m.mask_, b.reg_, a.reg_) }; }

    friend MaskT cmpGe(Packet a, Packet b) { return MaskT{ _mm512_cmp_pd_mask(a.reg_, b.reg_, _CMP_GE_OQ) }; }
    friend MaskT cmpLt(Packet a, Packet b) { return MaskT{ _mm512_cmp_pd_mask(a.reg_, b.reg_, _CMP_LT_OQ) }; }
    friend MaskT cmpGt(Packet a, Packet b) { return MaskT{ _mm512_cmp_pd_mask(a.reg_, b.reg_, _CMP_GT_OQ) }; }
};

template<>
struct Packet<float, 16> {
    using MaskT = PacketMask<float, 16>;
    static constexpr std::size_t width = 16;

    __m512 reg_;

    Packet() = default;
    explicit Packet(__m512 r)
        : reg_(r)
    {
    }

    static Packet broadcast(float v) { return Packet{ _mm512_set1_ps(v) }; }
    static Packet zero() { return Packet{ _mm512_setzero_ps() }; }

    static Packet load(const float* p) { return Packet{ _mm512_loadu_ps(p) }; }
    static void store(float* p, Packet a) { _mm512_storeu_ps(p, a.reg_); }
    static Packet maskLoad(const float* p, MaskT m) { return Packet{ _mm512_maskz_loadu_ps(m.mask_, p) }; }
    static void maskStore(float* p, MaskT m, Packet a) { _mm512_mask_storeu_ps(p, m.mask_, a.reg_); }

    friend Packet operator+(Packet a, Packet b) { return Packet{ _mm512_add_ps(a.reg_, b.reg_) }; }
    friend Packet operator-(Packet a, Packet b) { return Packet{ _mm512_sub_ps(a.reg_, b.reg_) }; }
    friend Packet operator*(Packet a, Packet b) { return Packet{ _mm512_mul_ps(a.reg_, b.reg_) }; }
    friend Packet operator/(Packet a, Packet b) { return Packet{ _mm512_div_ps(a.reg_, b.reg_) }; }
    Packet operator-() const { return Packet{ _mm512_sub_ps(_mm512_setzero_ps(), reg_) }; }

    friend Packet fmadd(Packet a, Packet b, Packet c) { return Packet{ _mm512_fmadd_ps(a.reg_, b.reg_, c.reg_) }; }

    /** @brief `mask ? a : b`, lane-wise — see the `double×8` specialization's note. */
    friend Packet select(MaskT m, Packet a, Packet b) { return Packet{ _mm512_mask_blend_ps(m.mask_, b.reg_, a.reg_) }; }

    friend MaskT cmpGe(Packet a, Packet b) { return MaskT{ _mm512_cmp_ps_mask(a.reg_, b.reg_, _CMP_GE_OQ) }; }
    friend MaskT cmpLt(Packet a, Packet b) { return MaskT{ _mm512_cmp_ps_mask(a.reg_, b.reg_, _CMP_LT_OQ) }; }
    friend MaskT cmpGt(Packet a, Packet b) { return MaskT{ _mm512_cmp_ps_mask(a.reg_, b.reg_, _CMP_GT_OQ) }; }
};

#endif // __AVX512F__

// ═══════════════════════════════════════════════════════════════════════
//  AVX2  double×4, float×8
// ═══════════════════════════════════════════════════════════════════════

#if defined(__AVX2__) || defined(__AVX__)

template<>
struct Packet<double, 4> {
    using MaskT = PacketMask<double, 4>;
    static constexpr std::size_t width = 4;

    __m256d reg_;

    Packet() = default;
    explicit Packet(__m256d r)
        : reg_(r)
    {
    }

    static Packet broadcast(double v) { return Packet{ _mm256_set1_pd(v) }; }
    static Packet zero() { return Packet{ _mm256_setzero_pd() }; }

    static Packet load(const double* p) { return Packet{ _mm256_loadu_pd(p) }; }
    static void store(double* p, Packet a) { _mm256_storeu_pd(p, a.reg_); }

    static Packet maskLoad(const double* p, MaskT m)
    {
        alignas(32) std::int64_t bits[4];
        for (std::size_t k = 0; k < 4; ++k)
            bits[k] = m.lane(k) ? std::int64_t(-1) : std::int64_t(0);
        __m256i imask = _mm256_load_si256(reinterpret_cast<const __m256i*>(bits));
        return Packet{ _mm256_maskload_pd(p, imask) };
    }

    static void maskStore(double* p, MaskT m, Packet a)
    {
        alignas(32) std::int64_t bits[4];
        for (std::size_t k = 0; k < 4; ++k)
            bits[k] = m.lane(k) ? std::int64_t(-1) : std::int64_t(0);
        __m256i imask = _mm256_load_si256(reinterpret_cast<const __m256i*>(bits));
        _mm256_maskstore_pd(p, imask, a.reg_);
    }

    friend Packet operator+(Packet a, Packet b) { return Packet{ _mm256_add_pd(a.reg_, b.reg_) }; }
    friend Packet operator-(Packet a, Packet b) { return Packet{ _mm256_sub_pd(a.reg_, b.reg_) }; }
    friend Packet operator*(Packet a, Packet b) { return Packet{ _mm256_mul_pd(a.reg_, b.reg_) }; }
    friend Packet operator/(Packet a, Packet b) { return Packet{ _mm256_div_pd(a.reg_, b.reg_) }; }
    Packet operator-() const { return Packet{ _mm256_sub_pd(_mm256_setzero_pd(), reg_) }; }

#if defined(__FMA__)
    friend Packet fmadd(Packet a, Packet b, Packet c) { return Packet{ _mm256_fmadd_pd(a.reg_, b.reg_, c.reg_) }; }
#else
    friend Packet fmadd(Packet a, Packet b, Packet c) { return a * b + c; }
#endif

    /** @brief `mask ? a : b`, lane-wise. `PacketMask`'s storage here is a
     *  plain integer bitmask (no AVX-512 `__mmask`), so lane `k` is first
     *  broadcast to a full 64-bit -1/0 pattern (the sign-bit convention
     *  `_mm256_blendv_pd` reads) via the same per-lane `bits[]` build
     *  `maskLoad`/`maskStore` above already use, then `_mm256_blendv_pd`
     *  (AVX2's blend instruction) picks `a`'s lane where that pattern is
     *  all-ones. */
    friend Packet select(MaskT m, Packet a, Packet b)
    {
        alignas(32) std::int64_t bits[4];
        for (std::size_t k = 0; k < 4; ++k)
            bits[k] = m.lane(k) ? std::int64_t(-1) : std::int64_t(0);
        __m256i imask = _mm256_load_si256(reinterpret_cast<const __m256i*>(bits));
        return Packet{ _mm256_blendv_pd(b.reg_, a.reg_, _mm256_castsi256_pd(imask)) };
    }

    friend MaskT cmpGe(Packet a, Packet b)
    {
        return MaskT{ static_cast<unsigned>(_mm256_movemask_pd(_mm256_cmp_pd(a.reg_, b.reg_, _CMP_GE_OQ))) };
    }
    friend MaskT cmpLt(Packet a, Packet b)
    {
        return MaskT{ static_cast<unsigned>(_mm256_movemask_pd(_mm256_cmp_pd(a.reg_, b.reg_, _CMP_LT_OQ))) };
    }
    friend MaskT cmpGt(Packet a, Packet b)
    {
        return MaskT{ static_cast<unsigned>(_mm256_movemask_pd(_mm256_cmp_pd(a.reg_, b.reg_, _CMP_GT_OQ))) };
    }
};

template<>
struct Packet<float, 8> {
    using MaskT = PacketMask<float, 8>;
    static constexpr std::size_t width = 8;

    __m256 reg_;

    Packet() = default;
    explicit Packet(__m256 r)
        : reg_(r)
    {
    }

    static Packet broadcast(float v) { return Packet{ _mm256_set1_ps(v) }; }
    static Packet zero() { return Packet{ _mm256_setzero_ps() }; }

    static Packet load(const float* p) { return Packet{ _mm256_loadu_ps(p) }; }
    static void store(float* p, Packet a) { _mm256_storeu_ps(p, a.reg_); }

    static Packet maskLoad(const float* p, MaskT m)
    {
        alignas(32) std::int32_t bits[8];
        for (std::size_t k = 0; k < 8; ++k)
            bits[k] = m.lane(k) ? std::int32_t(-1) : std::int32_t(0);
        __m256i imask = _mm256_load_si256(reinterpret_cast<const __m256i*>(bits));
        return Packet{ _mm256_maskload_ps(p, imask) };
    }

    static void maskStore(float* p, MaskT m, Packet a)
    {
        alignas(32) std::int32_t bits[8];
        for (std::size_t k = 0; k < 8; ++k)
            bits[k] = m.lane(k) ? std::int32_t(-1) : std::int32_t(0);
        __m256i imask = _mm256_load_si256(reinterpret_cast<const __m256i*>(bits));
        _mm256_maskstore_ps(p, imask, a.reg_);
    }

    friend Packet operator+(Packet a, Packet b) { return Packet{ _mm256_add_ps(a.reg_, b.reg_) }; }
    friend Packet operator-(Packet a, Packet b) { return Packet{ _mm256_sub_ps(a.reg_, b.reg_) }; }
    friend Packet operator*(Packet a, Packet b) { return Packet{ _mm256_mul_ps(a.reg_, b.reg_) }; }
    friend Packet operator/(Packet a, Packet b) { return Packet{ _mm256_div_ps(a.reg_, b.reg_) }; }
    Packet operator-() const { return Packet{ _mm256_sub_ps(_mm256_setzero_ps(), reg_) }; }

#if defined(__FMA__)
    friend Packet fmadd(Packet a, Packet b, Packet c) { return Packet{ _mm256_fmadd_ps(a.reg_, b.reg_, c.reg_) }; }
#else
    friend Packet fmadd(Packet a, Packet b, Packet c) { return a * b + c; }
#endif

    /** @brief `mask ? a : b`, lane-wise — see the `double×4` specialization's note. */
    friend Packet select(MaskT m, Packet a, Packet b)
    {
        alignas(32) std::int32_t bits[8];
        for (std::size_t k = 0; k < 8; ++k)
            bits[k] = m.lane(k) ? std::int32_t(-1) : std::int32_t(0);
        __m256i imask = _mm256_load_si256(reinterpret_cast<const __m256i*>(bits));
        return Packet{ _mm256_blendv_ps(b.reg_, a.reg_, _mm256_castsi256_ps(imask)) };
    }

    friend MaskT cmpGe(Packet a, Packet b)
    {
        return MaskT{ static_cast<unsigned>(_mm256_movemask_ps(_mm256_cmp_ps(a.reg_, b.reg_, _CMP_GE_OQ))) };
    }
    friend MaskT cmpLt(Packet a, Packet b)
    {
        return MaskT{ static_cast<unsigned>(_mm256_movemask_ps(_mm256_cmp_ps(a.reg_, b.reg_, _CMP_LT_OQ))) };
    }
    friend MaskT cmpGt(Packet a, Packet b)
    {
        return MaskT{ static_cast<unsigned>(_mm256_movemask_ps(_mm256_cmp_ps(a.reg_, b.reg_, _CMP_GT_OQ))) };
    }
};

#endif // __AVX2__ || __AVX__

// ═══════════════════════════════════════════════════════════════════════
//  SSE2  double×2, float×4
// ═══════════════════════════════════════════════════════════════════════

#if defined(__SSE2__) && !defined(__AVX2__) && !defined(__AVX__)

namespace detail {
inline unsigned sse2MaskFromCmp128d(__m128d cmp) { return static_cast<unsigned>(_mm_movemask_pd(cmp)); }
inline unsigned sse2MaskFromCmp128(__m128 cmp) { return static_cast<unsigned>(_mm_movemask_ps(cmp)); }
} // namespace detail

template<>
struct Packet<double, 2> {
    using MaskT = PacketMask<double, 2>;
    static constexpr std::size_t width = 2;

    __m128d reg_;

    Packet() = default;
    explicit Packet(__m128d r)
        : reg_(r)
    {
    }

    static Packet broadcast(double v) { return Packet{ _mm_set1_pd(v) }; }
    static Packet zero() { return Packet{ _mm_setzero_pd() }; }

    static Packet load(const double* p) { return Packet{ _mm_loadu_pd(p) }; }
    static void store(double* p, Packet a) { _mm_storeu_pd(p, a.reg_); }

    static Packet maskLoad(const double* p, MaskT m)
    {
        alignas(16) double tmp[2] = { 0.0, 0.0 };
        if (m.lane(0))
            tmp[0] = p[0];
        if (m.lane(1))
            tmp[1] = p[1];
        return Packet{ _mm_load_pd(tmp) };
    }

    static void maskStore(double* p, MaskT m, Packet a)
    {
        alignas(16) double tmp[2];
        _mm_store_pd(tmp, a.reg_);
        if (m.lane(0))
            p[0] = tmp[0];
        if (m.lane(1))
            p[1] = tmp[1];
    }

    friend Packet operator+(Packet a, Packet b) { return Packet{ _mm_add_pd(a.reg_, b.reg_) }; }
    friend Packet operator-(Packet a, Packet b) { return Packet{ _mm_sub_pd(a.reg_, b.reg_) }; }
    friend Packet operator*(Packet a, Packet b) { return Packet{ _mm_mul_pd(a.reg_, b.reg_) }; }
    friend Packet operator/(Packet a, Packet b) { return Packet{ _mm_div_pd(a.reg_, b.reg_) }; }
    Packet operator-() const { return Packet{ _mm_sub_pd(_mm_setzero_pd(), reg_) }; }

    friend Packet fmadd(Packet a, Packet b, Packet c) { return a * b + c; }

    /** @brief `mask ? a : b`, lane-wise. Plain SSE2 predates `blendv`
     *  (SSE4.1), so this composes the classic bitwise blend from three
     *  SSE2-only ops instead: `(mask & a) | (~mask & b)`, where `mask` is
     *  the per-lane all-ones/all-zero pattern `maskLoad`/`maskStore` above
     *  already build the same way. Still a genuine 2-wide vector op, not a
     *  per-lane scalar loop — the "scalar fallback" `fmadd` takes right
     *  above it (`a*b+c`) is the same idea: compose from ops the ISA
     *  already has rather than reach for one it doesn't. */
    friend Packet select(MaskT m, Packet a, Packet b)
    {
        alignas(16) std::int64_t bits[2] = {
            m.lane(0) ? std::int64_t(-1) : std::int64_t(0),
            m.lane(1) ? std::int64_t(-1) : std::int64_t(0),
        };
        __m128d maskReg = _mm_castsi128_pd(_mm_load_si128(reinterpret_cast<const __m128i*>(bits)));
        return Packet{ _mm_or_pd(_mm_and_pd(maskReg, a.reg_), _mm_andnot_pd(maskReg, b.reg_)) };
    }

    friend MaskT cmpGe(Packet a, Packet b) { return MaskT{ detail::sse2MaskFromCmp128d(_mm_cmpge_pd(a.reg_, b.reg_)) }; }
    friend MaskT cmpLt(Packet a, Packet b) { return MaskT{ detail::sse2MaskFromCmp128d(_mm_cmplt_pd(a.reg_, b.reg_)) }; }
    friend MaskT cmpGt(Packet a, Packet b) { return MaskT{ detail::sse2MaskFromCmp128d(_mm_cmpgt_pd(a.reg_, b.reg_)) }; }
};

template<>
struct Packet<float, 4> {
    using MaskT = PacketMask<float, 4>;
    static constexpr std::size_t width = 4;

    __m128 reg_;

    Packet() = default;
    explicit Packet(__m128 r)
        : reg_(r)
    {
    }

    static Packet broadcast(float v) { return Packet{ _mm_set1_ps(v) }; }
    static Packet zero() { return Packet{ _mm_setzero_ps() }; }

    static Packet load(const float* p) { return Packet{ _mm_loadu_ps(p) }; }
    static void store(float* p, Packet a) { _mm_storeu_ps(p, a.reg_); }

    static Packet maskLoad(const float* p, MaskT m)
    {
        alignas(16) float tmp[4] = { 0.f, 0.f, 0.f, 0.f };
        for (std::size_t k = 0; k < 4; ++k)
            if (m.lane(k))
                tmp[k] = p[k];
        return Packet{ _mm_load_ps(tmp) };
    }

    static void maskStore(float* p, MaskT m, Packet a)
    {
        alignas(16) float tmp[4];
        _mm_store_ps(tmp, a.reg_);
        for (std::size_t k = 0; k < 4; ++k)
            if (m.lane(k))
                p[k] = tmp[k];
    }

    friend Packet operator+(Packet a, Packet b) { return Packet{ _mm_add_ps(a.reg_, b.reg_) }; }
    friend Packet operator-(Packet a, Packet b) { return Packet{ _mm_sub_ps(a.reg_, b.reg_) }; }
    friend Packet operator*(Packet a, Packet b) { return Packet{ _mm_mul_ps(a.reg_, b.reg_) }; }
    friend Packet operator/(Packet a, Packet b) { return Packet{ _mm_div_ps(a.reg_, b.reg_) }; }
    Packet operator-() const { return Packet{ _mm_sub_ps(_mm_setzero_ps(), reg_) }; }

    friend Packet fmadd(Packet a, Packet b, Packet c) { return a * b + c; }

    /** @brief `mask ? a : b`, lane-wise — see the SSE2 `double×2` specialization's note. */
    friend Packet select(MaskT m, Packet a, Packet b)
    {
        alignas(16) std::int32_t bits[4];
        for (std::size_t k = 0; k < 4; ++k)
            bits[k] = m.lane(k) ? std::int32_t(-1) : std::int32_t(0);
        __m128 maskReg = _mm_castsi128_ps(_mm_load_si128(reinterpret_cast<const __m128i*>(bits)));
        return Packet{ _mm_or_ps(_mm_and_ps(maskReg, a.reg_), _mm_andnot_ps(maskReg, b.reg_)) };
    }

    friend MaskT cmpGe(Packet a, Packet b) { return MaskT{ detail::sse2MaskFromCmp128(_mm_cmpge_ps(a.reg_, b.reg_)) }; }
    friend MaskT cmpLt(Packet a, Packet b) { return MaskT{ detail::sse2MaskFromCmp128(_mm_cmplt_ps(a.reg_, b.reg_)) }; }
    friend MaskT cmpGt(Packet a, Packet b) { return MaskT{ detail::sse2MaskFromCmp128(_mm_cmpgt_ps(a.reg_, b.reg_)) }; }
};

#endif // __SSE2__ only

} // namespace simd
} // namespace aether
