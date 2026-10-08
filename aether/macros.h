// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file macros.h
 * @brief AETHER_-prefixed device/host qualification macros.
 *
 * Canonical macro surface for aether: every qualifier is spelled
 * `AETHER_<NAME>`, so an aether TU can be included alongside another
 * header that defines the unprefixed `KERNEL`/`DEVICE`/… names unguarded,
 * without a redefinition collision.
 *
 * One consumer-facing macro lives outside this file: `AETHER_CHECK_CUDA`
 * (`backend/cuda/Launch.h`), which throws `aether::Error` on a failing CUDA
 * runtime call. Listed here so it shows up alongside the rest of the macro
 * surface even though it is defined elsewhere.
 *
 * ★ TWO AXES ★ The build mode (`AETHER_HAS_CUDA` xor
 * `AETHER_CPP_MODE`) and the compiler (`AETHER_DEVICE_COMPILER`, below) are
 * INDEPENDENT. A CUDA-mode build may compile some of its translation units
 * with `g++`; those see the same types with the same layout and may
 * allocate / upload / download / view / index, and only the kernel-defining
 * and kernel-launching syntax is withheld from them. Do NOT reach for
 * `AETHER_CPP_MODE` per source to make such a TU compile: that flips the
 * MEMBER SET for one TU and is an ODR violation no linker will diagnose.
 */

/**
 * @brief Mark a function as device-only (`__device__`).
 *
 * In `AETHER_CPP_MODE` builds this expands to nothing.
 */
/**
 * @brief Mark a function as host-only (`__host__`).
 *
 * Expands to nothing in `AETHER_CPP_MODE` builds, and also when a CUDA-mode
 * build compiles a translation unit with the HOST compiler — `g++` has
 * no spelling for the execution-space attributes, and the raw `__host__`
 * this used to expand to was exactly what broke eagle's `.cpp` tests.
 */
/**
 * @brief Mark a function as callable from both device and host
 *        (`__device__ __host__`).
 *
 * In `AETHER_CPP_MODE` builds this expands to nothing.
 */
/**
 * @brief Prevent inlining of a function (`__noinline__` on CUDA,
 *        compiler-specific attribute on CPU).
 */
/**
 * @brief Mark a function as a CUDA kernel entry point (`__global__`).
 *
 * NOT DEFINED in an `AETHER_CPP_MODE` build (kernels are replaced by plain
 * functions via the build system), and NOT DEFINED either when a CUDA-mode
 * build compiles a translation unit with the HOST compiler — a kernel
 * definition there must be a loud compile error, never a silent host
 * function. See `AETHER_DEVICE_COMPILER` below.
 */
/**
 * @brief Qualify a kernel parameter as grid-constant read-only
 *        (`const __grid_constant__` on Volta+, plain `const` otherwise).
 *
 * Expands to nothing in a host-compiler TU of a CUDA-mode build: it
 * only ever qualifies a KERNEL parameter, and such a TU has no kernels.
 */
/**
 * @brief Context-aware device/host qualifier.
 *
 * Expands to `__device__` when compiling device code (`__CUDA_ARCH__`
 * defined) and to `__host__` otherwise. In `AETHER_CPP_MODE` builds this
 * expands to nothing.
 */
/**
 * @brief Declare a `__shared__` variable in a CUDA kernel.
 *
 * Expands to nothing when `__CUDA_ARCH__` is not defined, and likewise in a
 * host-compiler TU of a CUDA-mode build — shared memory only exists
 * inside a kernel, which such a TU cannot define.
 */
/**
 * @brief Force-inline a function (`__forceinline__` on device, `inline` on
 *        host).
 */
/**
 * @brief `1` when the current translation unit is being compiled by a CUDA
 *        device compiler (`__CUDACC__` defined), `0` otherwise. ALWAYS
 *        defined, in every build mode, so it may be used as a plain
 *        `#if AETHER_DEVICE_COMPILER` selector and inside `static_assert`.
 *
 * ★ THE TWO AXES ★  aether separates two independent questions:
 *
 *   - BUILD MODE (`AETHER_HAS_CUDA` xor `AETHER_CPP_MODE`, baked into the
 *     installed package by CMake) decides the MEMBER SET of every type and
 *     which runtime backend exists. It is a property of the BUILD and is
 *     therefore identical for every TU linked into one binary — which is
 *     what makes the One Definition Rule hold.
 *   - COMPILER (this macro) decides only what SYNTAX may appear in THIS TU.
 *     A translation unit compiled by the host compiler (`g++`) inside a
 *     CUDA-mode build sees the SAME types with the SAME layout, and may
 *     allocate / upload / download / view / index exactly as a `.cu` TU
 *     does; it merely cannot DEFINE a kernel or write a `<<<>>>` launch,
 *     because those are nvcc grammar.
 *
 * Consumers must NOT define `AETHER_CPP_MODE` per source to make a host TU
 * compile: that flips the member set for that TU only and is an ODR
 * violation the linker will not diagnose.
 * Simply compile the TU with `g++` — that is what this macro is for.
 *
 * @note Distinct from `AETHER_DEVICE_COMPILE` (no trailing `R`), which is
 *       defined only during nvcc's DEVICE pass (`__CUDA_ARCH__`). nvcc's
 *       HOST pass has `AETHER_DEVICE_COMPILER == 1` but no
 *       `AETHER_DEVICE_COMPILE` — the host pass still accepts `<<<>>>` and
 *       `__global__`, which is precisely the distinction that matters here.
 */
#if defined(__CUDACC__)
#define AETHER_DEVICE_COMPILER 1
#else
#define AETHER_DEVICE_COMPILER 0
#endif

#ifndef AETHER_CPP_MODE
#if AETHER_DEVICE_COMPILER
#define AETHER_DEVICE(...) __device__
#define AETHER_HOST(...) __host__
#define AETHER_DEVICEANDHOST(...) __device__ __host__
#define AETHER_NOINLINE(...) __noinline__
#define AETHER_KERNEL(...) __global__
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
#define AETHER_GRID_CONSTANT(...) const __grid_constant__
#else
#define AETHER_GRID_CONSTANT(...) const
#endif
#if defined(__CUDA_ARCH__)
#define AETHER_DEVICEHOST(...) AETHER_DEVICE()
#define AETHER_SHARED(...) __shared__
#else
#define AETHER_DEVICEHOST(...) AETHER_HOST()
#define AETHER_SHARED(...)
#endif
#else
// CUDA BUILD MODE, HOST COMPILER. The execution-space attributes are
// nvcc grammar; `g++` has no spelling for them, and they carry no meaning
// for a TU that emits host code only. They therefore expand to NOTHING —
// which leaves every declaration, signature and data member byte-identical
// to the nvcc-compiled view of the same header (attributes are not part of
// the type), so the layout-identity gate (tests/test_HostTuLayout.cu +
// tests/hosttu/host_tu_probe.cpp) can hold.
//
// `AETHER_KERNEL` is deliberately NOT DEFINED here: a kernel definition in a
// host TU must be a LOUD compile error ("AETHER_KERNEL was not declared"),
// never a silently host-compiled function that would then fail to link (or,
// worse, link against the nvcc-compiled twin and run on the wrong side).
#define AETHER_DEVICE(...)
#define AETHER_HOST(...)
#define AETHER_DEVICEANDHOST(...)
#define AETHER_DEVICEHOST(...)
#define AETHER_SHARED(...)
#define AETHER_GRID_CONSTANT(...)
#ifdef _WIN32
#define AETHER_NOINLINE(...) __declspec(noinline)
#else
#define AETHER_NOINLINE(...) __attribute__((noinline))
#endif
#endif
#else
#define AETHER_DEVICE(...)
#define AETHER_HOST(...)
#define AETHER_DEVICEANDHOST(...)
#define AETHER_DEVICEHOST(...)
#ifdef _WIN32
#define AETHER_NOINLINE(...) __declspec(noinline)
#else
#define AETHER_NOINLINE(...) __attribute__((noinline))
#endif
#endif
#if defined(__CUDA_ARCH__)
#define AETHER_FORCEINLINE(...) __forceinline__
#else
#define AETHER_FORCEINLINE(...) inline
#endif

/**
 * @brief Defined (with no value) while compiling for the device
 *        (`__CUDA_ARCH__` is defined).
 *
 * A differential-compile gate: lets a header branch on "am I being
 * compiled as device code right now" without depending on `AETHER_CPP_MODE`,
 * which only says whether CUDA is enabled for the *build*, not which pass of
 * a given TU is active.
 */
#if defined(__CUDA_ARCH__)
#define AETHER_DEVICE_COMPILE
#endif

/**
 * @brief Defined (with no value) when the translation unit is being compiled
 *        with `__cplusplus >= 202302L` (C++23).
 *
 * Host-only use: device code is capped at C++20 (nvcc's device-code
 * standard ceiling), so this gate must never be tested from a
 * `AETHER_DEVICE_COMPILE` path.
 */
#if defined(__cplusplus) && (__cplusplus >= 202302L)
#define AETHER_HAS_CXX23
#endif

/**
 * @brief Bit-copy `n` bytes from `src` to `dst`.
 *
 * Expands to `__builtin_memcpy` — the GCC/Clang compiler builtin every call
 * site used before this macro existed — everywhere EXCEPT a translation unit
 * parsed by NVRTC (`__CUDACC_RTC__`, defined only by NVRTC's own EDG front
 * end, never by nvcc or a host compiler): NVRTC's EDG does not recognise
 * `__builtin_memcpy`, but its device runtime resolves a builtin `memcpy` with
 * no header needed (measured). A non-NVRTC build is therefore
 * TOKEN-IDENTICAL before and after a call site switches to this macro — the
 * preprocessor emits the exact same `__builtin_memcpy(...)` text it always
 * did — so nvcc's PTX and g++'s object code cannot change.
 */
#ifdef __CUDACC_RTC__
#define AETHER_BITCOPY(dst, src, n) memcpy((dst), (src), (n))
#else
#define AETHER_BITCOPY(dst, src, n) __builtin_memcpy((dst), (src), (n))
#endif

/**
 * @brief Pin a host floating-point value as ROUNDED: no FMA contraction may
 *        reach across it.
 *
 * gcc and clang contract `a*b + c` into one fused multiply-add by default
 * (`-ffp-contract=fast`), across statements and across inlined calls, and
 * `#pragma STDC FP_CONTRACT OFF` is accepted and ignored by gcc. That is
 * harmless for ordinary arithmetic and fatal for an error-free
 * transformation: a two-sum residual is exact only if the sum and the
 * operands it is computed from are the rounded values the derivation names.
 * After `AETHER_FP_BARRIER(x)`, `x` holds a rounded value the optimiser
 * cannot see through, so a product computed before it can no longer be
 * fused into a sum after it.
 *
 * `x` must be a modifiable lvalue of type `float` or `double`; it is named
 * twice, so pass no expression with side effects. Host: an empty asm with
 * the value as an in/out register operand, zero instructions. Device
 * (`__CUDA_ARCH__`): expands to nothing — the host compiler's contraction is
 * what it guards; device code pins its roundings with the `__d*_rn`
 * intrinsics or inline PTX instead. An unknown host compiler gets a
 * `volatile` round trip rather than nothing.
 */
#if defined(__CUDA_ARCH__) || defined(__CUDACC_RTC__)
#define AETHER_FP_BARRIER(x) ((void)0)
#elif defined(__GNUC__) || defined(__clang__)
#if defined(__x86_64__) || defined(__i386__)
#define AETHER_FP_BARRIER(x) __asm__("" : "+x"(x))
#elif defined(__aarch64__) || defined(__arm__)
#define AETHER_FP_BARRIER(x) __asm__("" : "+w"(x))
#else
#define AETHER_FP_BARRIER(x) __asm__("" : "+r"(x))
#endif
#else
#define AETHER_FP_BARRIER(x)                                                   \
    do {                                                                       \
        volatile auto aetherFpBarrier_ = (x);                                  \
        (x)                            = aetherFpBarrier_;                     \
    } while (0)
#endif
