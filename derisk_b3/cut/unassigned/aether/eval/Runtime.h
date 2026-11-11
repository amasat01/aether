// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Runtime.h
 * @brief `aether::eval::runtimeEval`: the dual-mode (`AETHER_DEVICEHOST`)
 *        runtime evaluator over `RuntimeView`s.
 *
 * OP COVERAGE — ELEMENTWISE ONLY:
 *   - `Assign`   — `out = a`, covering BOTH "copy" (same dtype) and "cast"
 *                  (double<->float): the loop body always writes
 *                  `static_cast<TOut>(a)`, which is a no-op cast when
 *                  `TOut == TIn` — one code path, two named operations,
 *                  since they are the SAME operation with `TOut` possibly
 *                  differing.
 *   - `Add`/`Sub`     — `out = a ± b` (`a`±`b`).
 *   - `Scale`         — `out = s * a` (`s·a`) — `s` is a HOST scalar `T`
 *                        broadcast over every element, matching
 *                        `aether/expr/Operations.h`'s own `s*e`/`e*s`
 *                        convention (`CWiseScale` — a single scalar, not a
 *                        per-sample array); this evaluator does not invent a
 *                        second "scalar" convention.
 *   - `AddScaled`/`SubScaled` — `out = a ± s*b` (`a±s·b`).
 *
 * Geometric/quat/matrix operations stay static-only: the fully-static
 * `View`/`Item` + `expr/` ET machinery remains the perf path. This
 * evaluator's role is validation/fallback/interop coverage — it is not a
 * perf claim.
 *
 * ADDRESSING: every operand's element at flat (row-major, over `out`'s OWN
 * `rank`/`extents`) linear index `linear` is located through THAT operand's
 * OWN `strides` (`a`, `b` and `out` may each be non-contiguous/differently-
 * strided views of the same LOGICAL shape; only the LOGICAL shape, not the
 * physical layout, is required to agree). Bounds come from `extents`
 * (`out.rank`/`out.extents` is the canonical shape every operand is
 * validated against pre-launch).
 *
 * DRIVERS: the per-element core (`detail::evalElement`) is ONE
 * `AETHER_DEVICEHOST()` function, compiled for both arms from the SAME
 * source, so it never needs a later retrofit for device compilability. The
 * host driver (`detail::dispatch`, CPU path) calls it in a serial loop over
 * the flat element domain; the CUDA driver launches ONE generic grid-stride
 * kernel (`detail::evalKernel`) over the descriptor that calls the SAME
 * core per thread — a single kernel handles every `RuntimeOp` via the
 * op-tag switch inside `evalElement`, rather than one kernel per operation.
 *
 * ERRORS: `runtimeEval` itself validates shapes/dtypes and THROWS
 * `aether::Error` PRE-LAUNCH (host-only — mirrors every other aether
 * host-side failure); `detail::evalElement` (the device-legal core) asserts
 * NOTHING — by the time it runs, the host entry point has already validated
 * everything it depends on (heap/recursion/exception/virtual-free are all
 * off-limits on device).
 */

#include <cstddef>
#include <cstdint>

#include "aether/backend/cuda/Launch.h"
#include "aether/device/Device.h"
#include "aether/dtype/DType.h"
#include "aether/err/Error.h"
#include "aether/index/Offset.h"
#include "aether/layout/detail/Carray.h"
#include "aether/macros.h"
#include "aether/view/RuntimeView.h"

namespace aether {
namespace eval {

/** @brief Runtime-evaluator op tags — see the file docstring. */
enum class RuntimeOp : unsigned char {
    Assign,
    Add,
    Sub,
    Scale,
    AddScaled,
    SubScaled,
};

namespace detail {

/**
 * @brief One element of the generic strided elementwise loop — THE
 *        device-legal core both drivers below call. Decomposes
 *        `linear` (row-major over `out.rank`/`out.extents`) into a
 *        multi-index, addresses `a`/`b`/`out` through EACH OPERAND'S OWN
 *        `strides`, and applies `op`.
 *
 * `b`'s pointer/strides are simply never DEREFERENCED for the ops that do
 * not use it (`Assign`/`Scale`) — callers pass a default `RuntimeView{}`
 * for `b` then (arithmetic on its strides is harmless: no read of `b.data`
 * ever happens on those paths).
 */
template<class TIn, class TOut>
AETHER_DEVICEHOST() inline void evalElement(
    const RuntimeView& a, const RuntimeView& b, const RuntimeView& out, RuntimeOp op, TIn scalar, offset_t linear)
{
    aether::detail::Carray<offset_t, RuntimeRankCap> multiIdx{};
    offset_t remaining = linear;
    for (std::size_t k = out.rank; k-- > 0;) {
        const offset_t e = out.extents[k];
        multiIdx[k] = remaining % e;
        remaining /= e;
    }

    offset_t offA = 0, offB = 0, offOut = 0;
    for (std::size_t k = 0; k < out.rank; ++k) {
        offA += multiIdx[k] * a.strides[k];
        offB += multiIdx[k] * b.strides[k];
        offOut += multiIdx[k] * out.strides[k];
    }

    const TIn* aPtr = static_cast<const TIn*>(a.data);
    const TIn* bPtr = static_cast<const TIn*>(b.data);
    TOut* outPtr = static_cast<TOut*>(out.data);

    switch (op) {
    case RuntimeOp::Assign:
        outPtr[offOut] = static_cast<TOut>(aPtr[offA]);
        return;
    case RuntimeOp::Add:
        outPtr[offOut] = static_cast<TOut>(aPtr[offA] + bPtr[offB]);
        return;
    case RuntimeOp::Sub:
        outPtr[offOut] = static_cast<TOut>(aPtr[offA] - bPtr[offB]);
        return;
    case RuntimeOp::Scale:
        outPtr[offOut] = static_cast<TOut>(scalar * aPtr[offA]);
        return;
    case RuntimeOp::AddScaled:
        outPtr[offOut] = static_cast<TOut>(aPtr[offA] + scalar * bPtr[offB]);
        return;
    case RuntimeOp::SubScaled:
        outPtr[offOut] = static_cast<TOut>(aPtr[offA] - scalar * bPtr[offB]);
        return;
    }
}

/**
 * @brief `AETHER_DEVICE_COMPILER` as a DEPENDENT constant.
 *
 * `static_assert(AETHER_DEVICE_COMPILER, ...)` written directly in a
 * template body is a NON-dependent condition: the compiler evaluates it
 * while merely PARSING the template, so a host TU that only *includes* this
 * header would be rejected. Routing the same constant through a variable
 * template makes it dependent, so the diagnostic fires exactly when the
 * launch path is INSTANTIATED — including aether from a host TU stays free.
 */
template<class...>
inline constexpr bool device_compiler_v = (AETHER_DEVICE_COMPILER != 0);

#if defined(AETHER_HAS_CUDA) && AETHER_DEVICE_COMPILER
/** @brief The ONE generic grid-stride CUDA kernel — every `RuntimeOp`
 *         routes through here via `evalElement`'s op-tag switch.
 *
 * A kernel DEFINITION is nvcc grammar, so it exists only in a
 * device-compiled TU. Nothing host-visible depends on it — `dispatch` below
 * is the only caller and it is fenced identically.
 */
template<class TIn, class TOut>
AETHER_KERNEL() void evalKernel(RuntimeView a, RuntimeView b, RuntimeView out, RuntimeOp op, TIn scalar, offset_t total)
{
    for (offset_t linear = blockIdx.x * blockDim.x + threadIdx.x; linear < total; linear += gridDim.x * blockDim.x)
        evalElement<TIn, TOut>(a, b, out, op, scalar, linear);
}
#endif

/** @brief Product of `v.extents[0..v.rank)` at FULL `std::size_t` width —
 *         mirrors `extents::extentWide()`'s "the guard's input stays wide"
 *         convention (`aether/index/Offset.h`). Host-only. */
inline std::size_t flatSizeWide(const RuntimeView& v)
{
    // Empty product == 1 (a rank-0/scalar descriptor has exactly ONE
    // element) — mirrors `RuntimeView::size()`/`View::size()`'s convention.
    std::size_t total = 1;
    for (std::size_t k = 0; k < v.rank; ++k)
        total *= static_cast<std::size_t>(v.extents[k]);
    return total;
}

inline bool sameShape(const RuntimeView& x, const RuntimeView& y)
{
    if (x.rank != y.rank)
        return false;
    for (std::size_t k = 0; k < x.rank; ++k) {
        if (x.extents[k] != y.extents[k])
            return false;
    }
    return true;
}

inline bool isFloat32(const DType& dt) { return dt.code() == static_cast<std::uint8_t>(kDLFloat) && dt.bits() == 32 && dt.lanes() == 1; }
inline bool isFloat64(const DType& dt) { return dt.code() == static_cast<std::uint8_t>(kDLFloat) && dt.bits() == 64 && dt.lanes() == 1; }

/** @brief Host entry point for one resolved `<TIn, TOut>` pair: validates
 *         the flat element count against the addressable cap, then
 *         either launches `evalKernel` (CUDA device targets) or runs the
 *         SAME `evalElement` core in a serial host loop. */
#if !defined(AETHER_HAS_CUDA) || AETHER_DEVICE_COMPILER
template<class TIn, class TOut>
void dispatch(RuntimeOp op, const RuntimeView& a, const RuntimeView& b, const RuntimeView& out, double scalar)
{
    const std::size_t total = flatSizeWide(out);
    if (total == 0)
        return;
    if (!offset_fits(total)) {
        err::fail("aether::eval::runtimeEval", err::device_label(out.device), total,
            "flat element count exceeds the offset_t addressable maximum");
    }
    const TIn s = static_cast<TIn>(scalar);

#ifdef AETHER_HAS_CUDA
    if (out.device.is_cuda()) {
        const cuda::LaunchConfig cfg = cuda::launchConfig(total);
        evalKernel<TIn, TOut><<<cfg.blocks, cfg.threads>>>(a, b, out, op, s, static_cast<offset_t>(total));
        cuda::checkLastLaunch("aether::eval::runtimeEval");
        return;
    }
#endif
    for (offset_t linear = 0; linear < static_cast<offset_t>(total); ++linear)
        evalElement<TIn, TOut>(a, b, out, op, s, linear);
}
#endif // !AETHER_HAS_CUDA || AETHER_DEVICE_COMPILER

} // namespace detail

/**
 * @brief Elementwise runtime evaluator entry point  — host-callable.
 *        Validates `a`/`b`/`out` (shape + dtype) and throws
 *        `aether::Error` on mismatch BEFORE dispatching; the
 *        dispatch itself either runs on the host or launches
 *        `detail::evalKernel` on `out.device`.
 *
 * `b` is read only by `Add`/`Sub`/`AddScaled`/`SubScaled` — pass a
 * default-constructed `RuntimeView{}` for the unary ops (`Assign`/`Scale`).
 * `scalar` is read only by `Scale`/`AddScaled`/`SubScaled` (ignored, but
 * harmless, otherwise).
 *
 * `Assign` is the ONE op allowed to see `a.dtype != out.dtype` (that is
 * precisely what makes it also serve as a `double<->float` cast); every
 * other op requires every operand it actually reads to share `out`'s
 * dtype.
 *
 * @throws aether::Error  on a shape mismatch (`a`/`b` vs `out`), an
 *         unsupported dtype (v1: only `double`/`float`), a `dtype`
 *         disagreement not covered by the `Assign` cast allowance, or (via
 *         `detail::dispatch`) a flat element count too large to address.
 */
#if !defined(AETHER_HAS_CUDA) || AETHER_DEVICE_COMPILER
inline void runtimeEval(RuntimeOp op, const RuntimeView& a, const RuntimeView& b, const RuntimeView& out, double scalar = 0.0)
{
    if (!detail::sameShape(a, out)) {
        err::fail("aether::eval::runtimeEval", err::device_label(out.device), 0, "shape mismatch: a vs out");
    }
    const bool usesB
        = (op == RuntimeOp::Add || op == RuntimeOp::Sub || op == RuntimeOp::AddScaled || op == RuntimeOp::SubScaled);
    if (usesB && !detail::sameShape(b, out)) {
        err::fail("aether::eval::runtimeEval", err::device_label(out.device), 0, "shape mismatch: b vs out");
    }

    const bool aF32 = detail::isFloat32(a.dtype), aF64 = detail::isFloat64(a.dtype);
    const bool outF32 = detail::isFloat32(out.dtype), outF64 = detail::isFloat64(out.dtype);
    if (!(aF32 || aF64)) {
        err::fail("aether::eval::runtimeEval", err::device_label(out.device), 0,
            "unsupported input dtype (v1: double/float only)");
    }
    if (!(outF32 || outF64)) {
        err::fail("aether::eval::runtimeEval", err::device_label(out.device), 0,
            "unsupported output dtype (v1: double/float only)");
    }

    if (op == RuntimeOp::Assign) {
        if (aF64 && outF64) {
            detail::dispatch<double, double>(op, a, b, out, scalar);
        } else if (aF32 && outF32) {
            detail::dispatch<float, float>(op, a, b, out, scalar);
        } else if (aF64 && outF32) {
            detail::dispatch<double, float>(op, a, b, out, scalar);
        } else {
            detail::dispatch<float, double>(op, a, b, out, scalar);
        }
        return;
    }

    if (!(a.dtype == out.dtype)) {
        err::fail("aether::eval::runtimeEval", err::device_label(out.device), 0,
            "dtype mismatch: a vs out (only Assign performs a cast)");
    }
    if (usesB && !(b.dtype == out.dtype)) {
        err::fail("aether::eval::runtimeEval", err::device_label(out.device), 0, "dtype mismatch: b vs out");
    }

    if (outF64) {
        detail::dispatch<double, double>(op, a, b, out, scalar);
    } else {
        detail::dispatch<float, float>(op, a, b, out, scalar);
    }
}
#else
/**
 * @brief The LAUNCH-PATH REFUSAL overload, present ONLY in a CUDA-mode
 *        translation unit compiled by the HOST compiler.
 *
 * `runtimeEval` is the one aether entry point that can reach a `<<<>>>`
 * launch, and `<<<>>>` is nvcc grammar. Two things had to hold at once:
 *
 *   1. INCLUDING `aether/eval/Runtime.h` from a `.cpp` must stay FREE. This
 *      rules out putting the `static_assert` inside `detail::dispatch`:
 *      `runtimeEval` is a NON-template `inline` function that odr-uses all
 *      four `dispatch<TIn,TOut>` specialisations, so g++ instantiates every
 *      one of them at include time and the assert fires on a TU that never
 *      called anything (measured).
 *   2. The host copy must never merely BEHAVE differently (e.g. throw, or
 *      silently run the serial loop over device pointers). `dispatch` and
 *      `runtimeEval` are COMDAT-weak; the linker keeps whichever copy comes
 *      first on the link line, so a divergent host copy could hijack the
 *      real launch for every `.cu` caller in the same binary — exactly the
 *      class of defect `tests/odr/check_comdat_link_order.sh` exists to
 *      catch. Hence the real definitions are ABSENT here rather than
 *      different, and this template emits no symbol at all.
 *
 * The empty trailing pack makes the condition DEPENDENT, so the assert is
 * checked when a caller instantiates it — never on inclusion.
 */
template<class... Never>
void runtimeEval(RuntimeOp op, const RuntimeView& a, const RuntimeView& b, const RuntimeView& out, double scalar = 0.0)
{
    static_cast<void>(op), static_cast<void>(a), static_cast<void>(b), static_cast<void>(out),
        static_cast<void>(scalar);
    static_assert(detail::device_compiler_v<Never...>,
        "aether::eval::runtimeEval: kernel launches need nvcc. This translation "
        "unit is compiled by the host compiler (no __CUDACC__), so it cannot "
        "dispatch to a CUDA device; move the call into a .cu translation unit. "
        "Do NOT define AETHER_CPP_MODE for this source to get around it: that "
        "flips the member set for one TU only and is an ODR violation.");
}
#endif // !AETHER_HAS_CUDA || AETHER_DEVICE_COMPILER

} // namespace eval
} // namespace aether
