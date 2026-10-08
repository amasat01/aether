// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// tools/bandwidth_probe.cu — SASS-level bandwidth audit probe for
// `View::as_readonly()` and `DeviceBundle`/`BundleIndex` +
// `backend/cuda/bundle/{LoadStore,Assign}.h`. Standalone TU, compiled
// directly by `tools/bandwidth_audit.sh` via bare `nvcc` — never wired into
// tests/CMakeLists.txt's glob (mirrors `tools/volatile_probe.cu`'s
// placement/rationale: a probe compiled only by its own shell script, never
// executed, never linked into a gtest binary).
//
// Three kernels, three SASS assertions (`tools/bandwidth_audit.sh`):
//
//   - `bandwidthProbeBundleKernel` — an ordinary (`layout_right`,
//     contiguous, 16-byte-alignable) `Vec3dView` axpy3 via `BundleIndex<2>`/
//     `DeviceBundle`. Must contain `LDG.E.128` and `STG.E.128` (the
//     vectorized fast path's payoff, both load and store sides).
//   - `bandwidthProbeReadOnlyKernel` — a scalar read through `View::
//     as_readonly()`. Must contain a `.CI` (cache-invariant / read-only
//     data cache) load (the read-only path's payoff).
//   - `bandwidthProbeFallbackKernel` — the SAME bundle axpy shape as the
//     first kernel, but over a `layout_stride` view on BOTH the read and
//     write side. MUST CONTAIN NEITHER `LDG.E.128` NOR `STG.E.128` anywhere
//     in its SASS.
//
// WHY `layout_stride`, not a genuinely misaligned raw pointer, for the
// fallback probe: `backend/cuda/bundle/LoadStore.h`'s vectorization guard
// is TWO conditions ANDed together — `Layout == layout_right` (a
// COMPILE-TIME fact) and 16-byte pointer alignment (a RUNTIME fact, since
// the base address is a kernel argument unknown until launch). A genuinely
// misaligned pointer therefore can NEVER be "red-proof by construction" at
// the SASS level: ptxas cannot prove misalignment from pure pointer
// arithmetic on an opaque runtime value, so it retains BOTH the vectorized
// and scalar branches, predicated (`@P0`/`@!P0`) — confirmed empirically
// (a `buf+1`-offset `__shared__` probe still emits a predicated
// `LDS.U.128`). `layout_stride` is the OTHER half of the same guard, and it
// IS decidable at compile time (`if constexpr`) — a `layout_stride` view's
// SASS is GUARANTEED scalar-only for EVERY possible runtime pointer value,
// which is exactly what a MUST-NOT probe requires. Genuine runtime
// misalignment ("misaligned raw-pointer wrap -> fallback path, values
// still exact") is instead a VALUE-correctness property, gated in
// `tests/test_Bundle.cu`/`test_Bundle.cpp` via bit-identity against
// the scalar reference — not a SASS-mnemonic-presence property.

#include <aether/aether.h>

AETHER_KERNEL()
void bandwidthProbeBundleKernel(aether::Vec3dView a, aether::Vec3dView b, double s, aether::Vec3dView out)
{
    constexpr std::size_t W = 2;
    const aether::BundleIndex<W> bi = aether::BundleIndex<W>::make(threadIdx.x, blockIdx.x, blockDim.x);
    if (bi.base() >= a.samples())
        return;
    auto expr = a + s * b;
    if (bi.base() + W <= a.samples()) {
        out[bi] = expr;
    } else {
        aether::bundleAssign(out, expr, bi, bi.mask(a.samples()));
    }
}

AETHER_KERNEL()
void bandwidthProbeReadOnlyKernel(aether::Vec3dView a, aether::Vec3dView out)
{
    const aether::SampleIndex i = aether::SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);
    if (i.global() >= a.samples())
        return;
    auto aro = a.as_readonly();
    out[i] = aro[i].get();
}

using StrideExt = aether::extents<3, aether::dyn>;
using StrideMap = aether::layout_stride::mapping<StrideExt>;

AETHER_KERNEL()
void bandwidthProbeFallbackKernel(double* rawA, double* rawB, double s, double* rawOut, aether::offset_t n)
{
    constexpr std::size_t W = 2;
    // stride-2 in the sample mode (component stride stays natural: n*2 per
    // component) — deliberately non-contiguous, so `Layout == layout_right`
    // is FALSE at compile time and the vectorized fast path never even
    // instantiates (see file docstring).
    const aether::detail::Carray<std::size_t, 2> strides{ static_cast<std::size_t>(n) * 2, 2 };
    const StrideMap map(StrideExt{ static_cast<std::size_t>(n) }, strides);
    aether::View<double, StrideExt, aether::layout_stride> a(rawA, map, aether::Device(kDLCUDA));
    aether::View<double, StrideExt, aether::layout_stride> b(rawB, map, aether::Device(kDLCUDA));
    aether::View<double, StrideExt, aether::layout_stride> out(rawOut, map, aether::Device(kDLCUDA));

    const aether::BundleIndex<W> bi = aether::BundleIndex<W>::make(threadIdx.x, blockIdx.x, blockDim.x);
    if (bi.base() + W > n)
        return;
    out[bi] = a + s * b;
}
