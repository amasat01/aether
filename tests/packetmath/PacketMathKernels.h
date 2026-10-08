// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Interface between the packet-math accuracy tests (test_PacketMath.{cpp,cu},
// which hold the corpus, the std:: reference and the ULP checks) and the
// per-ISA kernel TUs in this directory, each compiled with its own -m flags:
//
//   pm_kernels_w1_w4.cpp  target default flags (AVX2+FMA): widths 1 and 4
//   pm_kernels_w8.cpp     + -mavx512f: width 8 (runtime-gated on the CPU)
//   pm_kernels_w2.cpp     -march=x86-64 (SSE2, no FMA): width 2
//   pm_hook.cpp           -DAETHER_HOST_VECTOR_MATH: the auto-vectoriser hook
//
// Only plain-C++ types cross this boundary, so no ISA-specific inline code is
// shared between TUs built for different targets.

#include <cstddef>

namespace aether_pm_test {

enum Unary : int {
    kExp,
    kExp2,
    kExp10,
    kExpm1,
    kLog,
    kLog2,
    kLog10,
    kLog1p,
    kSin,
    kCos,
    kSinCosSin,
    kSinCosCos,
    kTan,
    kAtan,
    kAsin,
    kAcos,
    kSinh,
    kCosh,
    kTanh,
    kAsinh,
    kAcosh,
    kAtanh,
    kFloor,
    kCeil,
    kTrunc,
    kRound,
    kRint,
    kAbs,
    kSqrt,
    kRsqrt,
    kCbrt,
    kNumUnary
};

enum Binary : int { kPow, kAtan2, kHypot, kFmax, kFmin, kFdim, kCopysign, kNumBinary };

struct Kernels {
    std::size_t width;
    bool (*available)();
    void (*unary)(int fn, const double* x, double* out, std::size_t n);
    void (*binary)(int fn, const double* x, const double* y, double* out, std::size_t n);
};

const Kernels& kernelsW1();
const Kernels& kernelsW2();
const Kernels& kernelsW4();
const Kernels& kernelsW8();

// pm_hook.cpp: loops over aether::math::<fn> in a TU built with the hook.
void hookPowLoop(const double* a, const double* b, double* out, std::size_t n);
void hookPowLoopConditional(const double* a, const double* b, double* out, std::size_t n);
void hookSinExpLoop(const double* a, double* out, std::size_t n);
void hookFmaxFminLoop(const double* a, const double* b, double* mx, double* mn, std::size_t n);
void hookRootFloorAbsLoop(const double* x, double* sq, double* rs, double* fl, double* ab, std::size_t n);
// The functions routed to the hook beyond the original transcendental set.
enum HookParity : int {
    kHook_expm1,
    kHook_log1p,
    kHook_sinh,
    kHook_cosh,
    kHook_asinh,
    kHook_acosh,
    kHook_atanh,
    kHook_ceil,
    kHook_trunc,
    kHook_round,
    kHook_rint,
    kHook_fdim,
    kHook_copysign,
    kNumHookParity
};
void hookParityLoop(int fn, const double* x, const double* y, double* out, std::size_t n);
// Every routed function as a facade loop (Unary / Binary ids); false when
// the id has no loop of its own (sincos legs, fmax/fmin).
bool hookUnaryLoop(int fn, const double* x, double* out, std::size_t n);
bool hookBinaryLoop(int fn, const double* x, const double* y, double* out, std::size_t n);
// The scalar symbols the vectorised loops fall back to (one call per element).
double hookParityScalar(int fn, double x, double y);
double hookPowScalar(double a, double b);
double hookSinScalar(double a);
double hookExpScalar(double a);
double hookSqrtScalar(double a);
double hookRsqrtScalar(double a);
double hookFloorScalar(double a);

} // namespace aether_pm_test
