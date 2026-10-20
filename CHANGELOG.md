# Changelog

## 0.2.4 (C++ library 0.2.1, `aether-dsc` 0.2.4)

A plain-C reference core: `aether_dsc.c_core` is a pure-Python generator of C99
functions (vectors, matrices, quaternions and four scalar formulas, at both
`double` and `float` widths) that reproduce aether's expression-template
arithmetic bit for bit. The contract is the fold order: dot, norm and sum
products are right-nested with the highest index outermost and no zero seed,
division by a scalar multiplies by one reciprocal, and `neg` is `x * -1`. The
`CCore` test suite pins every generated function `memcmp`-equal to the
templates, and CI gates the committed corpus headers as strict C99 and with
`tools/c_core/mint_golden.py --check`.

It exists so that a consumer such as hawk can inline the functions it needs
into standalone C that matches the templates bit for bit. The C++ library is
unchanged.

## 0.2.3 (C++ library 0.2.1, `aether-dsc` 0.2.3)

Host vector math (`AETHER_HOST_VECTOR_MATH`): no masked vector variants on the
GCC releases that miscompile masked calls. GCC 13.1-13.3 and 14.1-14.2
if-convert a conditional call of a function with masked (inbranch) vector
variants using the inverted mask, so a loop such as
`r[i] = c[i] > lim ? a[i] : aether::math::tanh(b[i]) + 0.1` evaluated the lanes
that should be inactive and returned wrong results. The bug is fixed upstream
by GCC commit abbfe1e2616d ("middle-end: Fix ifcvt predicate generation for
masked function calls"), released in 13.4 and 14.3.

The header now applies a rule automatically: `AETHER_HVM_MASKED` is 0 on
exactly those compilers (GCC, not clang, 13 before 13.4 and 14 before 14.3) and
1 everywhere else. Under 0 every routed function is declared
`simd("notinbranch")` and no masked `_ZGV?M?v_` variant is defined or kept
alive; GCC cannot if-convert a call that has no masked clone, so only loops
with a conditional aether math call run scalar there. Unconditional math calls
keep their unmasked vector variants and every other loop vectorises as before.
Define `-DAETHER_HVM_MASKED=0` or `=1` to override the rule (for tests, or for
a distribution compiler that carries the backport). With the macro at 1 the
generated code is byte-identical to 0.2.2. CI (`host-gcc`, GCC 12, 13, 14) runs
`tests/packetmath/check_hvm_masked.sh`, which checks the results, the
vectorisation of the unaffected loops and the symbols; on 13 and 14 a further
step requires the miscompile to reproduce with `AETHER_HVM_MASKED=1`.

## 0.2.2 (C++ library 0.2.1, `aether-dsc` 0.2.2)

`orBool` on the device is a one-byte store (`ST.U8`) instead of a 32-bit word
`atomicOr`, which read and wrote the three bytes around the slot and ran out of
bounds on a one-byte allocation. The operation is an idempotent set-to-1, so a
plain byte store is correct and race-free; the invariant is documented at the
function.

Host SIMD `packetExp` and `packetLog` do their exponent-field arithmetic in
unsigned lanes: the signed form overflowed (undefined behaviour) for large
arguments. Results are bit-identical.

A sanitizer gate keeps memory and undefined-behaviour defects out: a CI job runs
the CPU suite under AddressSanitizer and UndefinedBehaviorSanitizer, and
`make sanitize-gate` runs the CUDA suite under compute-sanitizer (memcheck,
racecheck, synccheck, initcheck) and valgrind. Planted canaries must trip every
tool, so a gate that stops instrumenting goes red (`tests/sanitize/README.md`).

## 0.2.1 (C++ library 0.2.1, `aether-dsc` 0.2.1)

Host SIMD `packetExp` and `packetLog` use table-driven algorithms: a 128-entry
2^(i/128) table (exp) or (1/c, log c) table (log), a short polynomial, no
division, the table read by a gather (`PacketExpLog.h`, tables in
`PacketExpLogTables.h`). The scheme follows ARM's optimized-routines
(MIT OR Apache-2.0 WITH LLVM-exception; see `NOTICE`). Lanes outside the fast
path (exp: |x| >= 700 or NaN; log: x <= 0, subnormal, inf, NaN, |x - 1| < 1/16)
and `log` without a hardware FMA keep the double-double scheme, so special
values, subnormal results and the pow/exp2/exp10/expm1/log2/log10/log1p
family are unchanged. Accuracy stays inside the faithful bar; the packet-math
corpora gain the exp overflow/underflow/subnormal edges and log near 1 and
below the normal range.

`aether-dsc` 0.2.1 (2026-10-09): `payload()` is safe under free-threaded CPython: concurrent first calls
return one `Payload` object (the cache fill is now under a lock). README gains a
Threading section.

## 0.2.0 (first public release)

`AETHER_CUDA_ARCHS` picks the CUDA architectures when `CMAKE_CUDA_ARCHITECTURES`
is not given: `native` (the default; with no GPU, PTX for the oldest
architecture nvcc compiles, at least `sm_60`), `all` (every architecture from
Pascal to Blackwell that nvcc compiles), or one architecture or a list; the
Makefile's `cudaarch` passes it through. aether-dsc finds CCCL under CUDA 13's
`include/cccl` layout, prefers the CUDA root of the NVRTC library actually
loaded, and checks against the oldest architecture NVRTC compiles.

Host code that depends on exact rounding is now safe in a TU built with FMA
contraction (`-ffp-contract=fast`, gcc's and clang's default). The
`AETHER_HOST_VECTOR_MATH` symbols and the packet math they run are compiled
with contraction off whatever the TU asks for (`#pragma GCC optimize` with
push/pop in `aether/math/detail/HostVectorMath.h`; the TU's own arithmetic
keeps its setting); before, a contracting TU moved `sin`, `cos`, `tan`,
`atan`, `asin`, `acos` and `atan2` off their faithful results on some inputs.
New `AETHER_FP_BARRIER(x)` (`aether/macros.h`) pins a host value as rounded so
no product can be fused across it; `aether::accum::atomic`'s two-sum residual
and compensated sum use it, so the compensation stays exact when a term is a
product. Pinned by `ContractionGuard.*` and
`PacketMathHostVectorHook.HookUnderContractionMatchesKernels` (the hook TU is
now built with `-ffp-contract=fast`); both fail with the guards removed.

Adds vector math for the CPU packet layer
(`aether/backend/cpu/simd/math/PacketMath.h`, host-only, opt-in include):
`exp`/`exp2`/`exp10`/`expm1`, `log`/`log2`/`log10`/`log1p`, `pow` (with C's
special cases), `sin`/`cos`/`sincos`/`tan`, `atan`/`atan2`/`asin`/`acos`, the
hyperbolic and inverse hyperbolic functions, `cbrt`, `hypot`, `sqrt`,
`rsqrt`, the rounding family, `abs`, `fmax`/`fmin`/`fdim`/`copysign` and
`fma`, on `aether::simd::Packet<double, W>` for W = 1, 2 (SSE2), 4 (AVX2),
8 (AVX-512). Every function is faithfully rounded (error < 1 ULP against the
exact result, every width; exact for the rounding and selection functions),
verified against a 128-bit mpmath reference over more than 10^6
inputs per function (2^20 seeded random inputs plus special values and
domain edges): `make faithful-gate` regenerates that corpus with
`tools/packet_math_ref/gen_golden.py`, checks it against committed md5 sums
and scores every width; the unit suite scores a committed hard-case subset.

Adds the opt-in `AETHER_HOST_VECTOR_MATH` macro: in a GCC x86-64 host TU it
routes the `double` math functions of `aether::math` to x86-64
vector-ABI functions built on that packet math, so GCC can vectorise scalar
loops that call them. Without the macro nothing changes: the default
`aether::math` host path compiles to the same object code as before.

Adds the numpy-parity elementwise functions to `aether::math`, on host and
device: `expm1`, `log1p` (`explog.h`), `sinh`, `cosh`, `asinh`, `acosh`,
`atanh` (new `aether/math/hyperbolic.h`), `rint` and `remainder` (`round.h`;
`remainder` has numpy's semantics, the sign of the divisor, not C's IEEE
remainder), `clip`, `isnan`, `isinf` (`math.h`), and `erf`, `erfc` (new
`aether/math/special.h`). Under `AETHER_HOST_VECTOR_MATH` the `double` host
legs of `expm1`, `log1p`, the hyperbolics, `ceil`, `trunc`, `round`, `rint`
and `fdim` route to the packet math, and `copysign` to an inline sign-bit
mask; `fmod`, `remainder`, `fma`, `clip`, `erf` and `erfc` stay scalar on
the host.

Makes the device FP64 `aether::math::sin`/`cos`/`sincos` total: Inf and NaN
give NaN and arguments of any size up to `DBL_MAX` are reduced exactly (full-
range Payne-Hanek, still inline with no stack), within 2 ULP of glibc.

Removes the unused consumer aliases `Vec3f`, `Mat66d`, `Vec4dView`,
`Vec6dView`, `Vec3fView`, `Mat66dView` and `Mat36dView` from `aether/aether.h`
(spell them as `Vec<float, 3>`, `MatView<double, 6, 6>`, ...), and the opt-in
`aether/unprefixed.h` short macro spellings (use the `AETHER_*` names).

aether is the RAPTOR family's numerics core: header-only C++23/CUDA typed
tensor views, expression templates, and a dual-mode evaluator that runs
the same code on the GPU or on CPU threads.

Adds the generic DLPack buffer layer, `aether::interop::BufferView`: a
zero-copy view over a DLPack producer (either generation), a raw
`__cuda_array_interface__`, or a raw `__array_interface__`. Access is
reported exactly as the producer declared it (`ReadWrite`, `ReadOnly`, or
`Unknown` for legacy DLPack, never upgraded to writable on its own),
checked against a `Requirements` object that names the first failed check,
and a type-erased keep-alive holds the producer's buffer alive across an
import/export chain. A `BufferView` re-exports as DLPack under the 1.0
stream contract. This layer is host-only by design; eagle's plugin adds
the CUDA-aware stream ordering on top of it.

Device types are now tracked as metadata in pure C++ builds: a CPP_MODE
build (no CUDA toolchain on the machine) can still record and compare a
buffer's declared device type without needing the CUDA headers that type
would otherwise pull in.

Licensed under Apache-2.0, with a documentation site (install,
quickstart, executed C++ tutorials, examples, and the API reference).

This is aether's first public release.
