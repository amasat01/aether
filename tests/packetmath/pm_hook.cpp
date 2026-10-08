// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// The AETHER_HOST_VECTOR_MATH auto-vectoriser hook, exercised the way a
// consumer uses it: plain scalar loops over aether::math::<fn>, compiled with
// the macro defined (tests/CMakeLists.txt adds it for this TU only, together
// with -ffp-contract=fast: the hook's symbols must stay contraction-free
// whatever the consumer's TU asks for, see HostVectorMath.h). The facade instantiations here carry the hook's ABI
// tag, so they cannot merge with the default ones in the other TUs.

#if !defined(AETHER_HOST_VECTOR_MATH)
#error "pm_hook.cpp must be compiled with -DAETHER_HOST_VECTOR_MATH"
#endif

#include "aether/math/math.h"
#include "tests/packetmath/PacketMathKernels.h"

namespace aether_pm_test {

void hookPowLoop(const double* a, const double* b, double* out, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        out[i] = aether::math::pow(a[i], b[i]);
}

void hookPowLoopConditional(const double* a, const double* b, double* out, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        out[i] = a[i] > 1.0 ? aether::math::pow(a[i], b[i]) : a[i];
}

void hookSinExpLoop(const double* a, double* out, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        out[i] = aether::math::sin(a[i]) + aether::math::exp(a[i]);
}

void hookFmaxFminLoop(const double* a, const double* b, double* mx, double* mn, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i) {
        mx[i] = aether::math::fmax(a[i], b[i]);
        mn[i] = aether::math::fmin(a[i], b[i]);
    }
}

void hookRootFloorAbsLoop(const double* x, double* sq, double* rs, double* fl, double* ab, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i) {
        sq[i] = aether::math::sqrt(x[i]);
        rs[i] = aether::math::rsqrt(x[i]);
        fl[i] = aether::math::floor(x[i]);
        ab[i] = aether::math::abs(x[i]);
    }
}

// One scalar loop per newly routed function (each must vectorise on its own).
#define AETHER_PM_HOOK_LOOP1(NAME)                                                                  \
    case kHook_##NAME:                                                                             \
        for (std::size_t i = 0; i < n; ++i)                                                        \
            out[i] = aether::math::NAME(x[i]);                                                     \
        break;
#define AETHER_PM_HOOK_LOOP2(NAME)                                                                  \
    case kHook_##NAME:                                                                             \
        for (std::size_t i = 0; i < n; ++i)                                                        \
            out[i] = aether::math::NAME(x[i], y[i]);                                               \
        break;

void hookParityLoop(int fn, const double* x, const double* y, double* out, std::size_t n)
{
    switch (fn) {
        AETHER_PM_HOOK_LOOP1(expm1)
        AETHER_PM_HOOK_LOOP1(log1p)
        AETHER_PM_HOOK_LOOP1(sinh)
        AETHER_PM_HOOK_LOOP1(cosh)
        AETHER_PM_HOOK_LOOP1(asinh)
        AETHER_PM_HOOK_LOOP1(acosh)
        AETHER_PM_HOOK_LOOP1(atanh)
        AETHER_PM_HOOK_LOOP1(ceil)
        AETHER_PM_HOOK_LOOP1(trunc)
        AETHER_PM_HOOK_LOOP1(round)
        AETHER_PM_HOOK_LOOP1(rint)
        AETHER_PM_HOOK_LOOP2(fdim)
        AETHER_PM_HOOK_LOOP2(copysign)
    default: break;
    }
}
#undef AETHER_PM_HOOK_LOOP1
#undef AETHER_PM_HOOK_LOOP2

double hookParityScalar(int fn, double x, double y)
{
    switch (fn) {
    case kHook_expm1: return ::aether_hvm_expm1(x);
    case kHook_log1p: return ::aether_hvm_log1p(x);
    case kHook_sinh: return ::aether_hvm_sinh(x);
    case kHook_cosh: return ::aether_hvm_cosh(x);
    case kHook_asinh: return ::aether_hvm_asinh(x);
    case kHook_acosh: return ::aether_hvm_acosh(x);
    case kHook_atanh: return ::aether_hvm_atanh(x);
    case kHook_ceil: return ::aether_hvm_ceil(x);
    case kHook_trunc: return ::aether_hvm_trunc(x);
    case kHook_round: return ::aether_hvm_round(x);
    case kHook_rint: return ::aether_hvm_rint(x);
    case kHook_fdim: return ::aether_hvm_fdim(x, y);
    case kHook_copysign: return ::aether::simd::pm::copysign(x, y);
    default: return x;
    }
}

// Every routed function, one plain loop each, in this -ffp-contract=fast TU.
#define AETHER_PM_ALL1(ENUM, NAME)                                                                  \
    case ENUM:                                                                                     \
        for (std::size_t i = 0; i < n; ++i)                                                        \
            out[i] = aether::math::NAME(x[i]);                                                     \
        return true;
#define AETHER_PM_ALL2(ENUM, NAME)                                                                  \
    case ENUM:                                                                                     \
        for (std::size_t i = 0; i < n; ++i)                                                        \
            out[i] = aether::math::NAME(x[i], y[i]);                                               \
        return true;

bool hookUnaryLoop(int fn, const double* x, double* out, std::size_t n)
{
    switch (fn) {
        AETHER_PM_ALL1(kExp, exp)
        AETHER_PM_ALL1(kExp2, exp2)
        AETHER_PM_ALL1(kExp10, exp10)
        AETHER_PM_ALL1(kExpm1, expm1)
        AETHER_PM_ALL1(kLog, log)
        AETHER_PM_ALL1(kLog2, log2)
        AETHER_PM_ALL1(kLog10, log10)
        AETHER_PM_ALL1(kLog1p, log1p)
        AETHER_PM_ALL1(kSin, sin)
        AETHER_PM_ALL1(kCos, cos)
        AETHER_PM_ALL1(kTan, tan)
        AETHER_PM_ALL1(kAtan, atan)
        AETHER_PM_ALL1(kAsin, asin)
        AETHER_PM_ALL1(kAcos, acos)
        AETHER_PM_ALL1(kSinh, sinh)
        AETHER_PM_ALL1(kCosh, cosh)
        AETHER_PM_ALL1(kTanh, tanh)
        AETHER_PM_ALL1(kAsinh, asinh)
        AETHER_PM_ALL1(kAcosh, acosh)
        AETHER_PM_ALL1(kAtanh, atanh)
        AETHER_PM_ALL1(kFloor, floor)
        AETHER_PM_ALL1(kCeil, ceil)
        AETHER_PM_ALL1(kTrunc, trunc)
        AETHER_PM_ALL1(kRound, round)
        AETHER_PM_ALL1(kRint, rint)
        AETHER_PM_ALL1(kAbs, abs)
        AETHER_PM_ALL1(kSqrt, sqrt)
        AETHER_PM_ALL1(kRsqrt, rsqrt)
        AETHER_PM_ALL1(kCbrt, cbrt)
    default: return false; // sincos legs: reached through sin and cos
    }
}

bool hookBinaryLoop(int fn, const double* x, const double* y, double* out, std::size_t n)
{
    switch (fn) {
        AETHER_PM_ALL2(kPow, pow)
        AETHER_PM_ALL2(kAtan2, atan2)
        AETHER_PM_ALL2(kHypot, hypot)
        AETHER_PM_ALL2(kFdim, fdim)
        AETHER_PM_ALL2(kCopysign, copysign)
    default: return false; // fmax/fmin: inline selects, no arithmetic
    }
}
#undef AETHER_PM_ALL1
#undef AETHER_PM_ALL2

double hookPowScalar(double a, double b) { return ::aether_hvm_pow(a, b); }
double hookSinScalar(double a) { return ::aether_hvm_sin(a); }
double hookExpScalar(double a) { return ::aether_hvm_exp(a); }
double hookSqrtScalar(double a) { return ::aether_hvm_sqrt(a); }
double hookRsqrtScalar(double a) { return ::aether_hvm_rsqrt(a); }
double hookFloorScalar(double a) { return ::aether_hvm_floor(a); }

} // namespace aether_pm_test
