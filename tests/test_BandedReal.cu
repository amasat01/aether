// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_BandedReal.cu
 * @brief `BandedReal` (CUDA mode): the shared host battery
 *        (`test_BandedReal_common.h`) plus device-only checks beyond it:
 *        the `aether::math` facade routes on the device pass (`#if
 *        defined(AETHER_DEVICE_COMPILE)` selects a different macro
 *        expansion, so a missing route fails to compile rather than
 *        computing a wrong number); the value type's comparison,
 *        classification and codec-bridge code compute identically on both
 *        arms (measured word for word over an enumerated corpus, since
 *        "same source" is an assumption about ptxas, not a fact); and the
 *        chain is FP64-free in SASS, checked by
 *        `tests/sass/check_band_fp64free.sh` against `bandChainKernel`.
 *
 * `bandChainKernel` and `bandFp64InjectedKernel` are fixed-name audit
 * subjects: do not rename, do not dead-strip. The second is the positive
 * control (the identical body plus one deliberate FP64 operation) the
 * scanner must find.
 *
 * `tests/odr/check_comdat_link_order.sh` measures link order directly, and
 * the banded host arms carry FP-barrier guards
 * (`aether/banded/detail/Fp32.h`) against COMDAT link-order hazards.
 */

#include "test_BandedReal_common.h"

#include <cstdint>
#include <vector>

namespace aether_tests {
namespace BandedRealTest {

using aether::SampleIndex;

constexpr int kBlock = 128;

// =========================================================================
//  Device probes
// =========================================================================

/// @brief What one device thread reports about one PAIR of values.
struct BrProbe {
    std::uint32_t absHi, absLo, absTail; ///< aether::math::abs(a)
    std::uint32_t maxHi, maxLo, maxTail; ///< aether::math::fmax(a, b)
    std::uint32_t minHi, minLo, minTail; ///< aether::math::fmin(a, b)
    std::uint32_t csHi, csLo, csTail;    ///< aether::math::copysign(a, b)
    std::uint32_t divHi, divLo, divTail; ///< a / b (the namespace-scope operator)
    std::uint32_t cmp;                   ///< the six relations, packed
    std::uint32_t cls;                   ///< isNan | isInf<<1 | isZero<<2 (of a)
    std::uint64_t packWord;              ///< the pack terminal's stored word
};

AETHER_DEVICEHOST() AETHER_FORCEINLINE() std::uint32_t brPackCmp(
    BandedReal a, BandedReal b)
{
    return (a < b ? 1u : 0u) | (a <= b ? 2u : 0u) | (a == b ? 4u : 0u)
        | (a > b ? 8u : 0u) | (a >= b ? 16u : 0u) | (a != b ? 32u : 0u);
}

AETHER_DEVICEHOST() AETHER_FORCEINLINE() std::uint32_t brBits(float f)
{
    return static_cast<std::uint32_t>(bd::floatAsInt(f));
}

/// @brief Every facade entry point, every comparison, on the device.
AETHER_KERNEL() void bandedRealFacadeKernel(const BandedReal* __restrict__ a,
    const BandedReal* __restrict__ b, BrProbe* __restrict__ out, int n)
{
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n)
        return;
    const BandedReal ra = a[i];
    const BandedReal rb = b[i];

    const Band va = aether::math::abs(ra);
    const Band vm = aether::math::fmax(ra, rb);
    const Band vn = aether::math::fmin(ra, rb);
    const Band vc = aether::math::copysign(ra, rb);
    const Band vd = ra / rb;

    BrProbe r;
    r.absHi = brBits(va.hi);  r.absLo = brBits(va.lo);  r.absTail = brBits(va.tail);
    r.maxHi = brBits(vm.hi);  r.maxLo = brBits(vm.lo);  r.maxTail = brBits(vm.tail);
    r.minHi = brBits(vn.hi);  r.minLo = brBits(vn.lo);  r.minTail = brBits(vn.tail);
    r.csHi  = brBits(vc.hi);  r.csLo  = brBits(vc.lo);  r.csTail  = brBits(vc.tail);
    r.divHi = brBits(vd.hi);  r.divLo = brBits(vd.lo);  r.divTail = brBits(vd.tail);
    r.cmp   = brPackCmp(ra, rb);
    r.cls   = (ra.isNan() ? 1u : 0u) | (ra.isInf() ? 2u : 0u)
        | (ra.isZero() ? 4u : 0u);
    /* the PACK terminal, exercised on the device: a Band chain landing back in
     * the storage word */
    r.packWord = BandedReal(vm).toBits();
    out[i]     = r;
}

/// @brief `std::numeric_limits<BandedReal>` read from a DEVICE body. The whole
/// point of the `static constexpr` member + literal-read accessor shape
/// (`aether/banded/BandedLimits.h`) is that no host-constexpr call appears here
/// and `--expt-relaxed-constexpr` is never needed; if that shape were lost this
/// kernel would stop compiling.
AETHER_KERNEL() void bandedRealLimitsKernel(std::uint64_t* __restrict__ out)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;
    out[0] = std::numeric_limits<BandedReal>::max().toBits();
    out[1] = std::numeric_limits<BandedReal>::lowest().toBits();
    out[2] = std::numeric_limits<BandedReal>::min().toBits();
    out[3] = std::numeric_limits<BandedReal>::epsilon().toBits();
    out[4] = std::numeric_limits<BandedReal>::infinity().toBits();
    out[5] = std::numeric_limits<BandedReal>::quiet_NaN().toBits();
    out[6] = std::numeric_limits<BandedReal>::round_error().toBits();
    out[7] = std::numeric_limits<BandedReal>::denorm_min().toBits();
}

/**
 * @brief The FP64-free audit subject. Fixed name; do not rename.
 *
 * An adaptive-step controller's shape in the banded carrier: a capped step
 * `copysign(min(|x|, |y|), x)` and the error-ratio fold around it, with
 * storage loads at the front and a storage store at the back so the codec
 * boundary crossings are part of what is measured.
 *
 * The op inventory `tests/sass/check_band_fp64free.sh`'s budget is derived
 * from this list, not from a measurement:
 *   3 abs, 1 min, 1 max, 1 copysign, 1 div, 1 mul, 1 add, 1 sub  = 10 ops
 *   2 unpacks (the two loads) + 1 pack (the store)               =  3 crossings
 * Change the body and the derivation in that script must move with it.
 */
AETHER_KERNEL() void bandChainKernel(BandedReal* __restrict__ out,
    const BandedReal* __restrict__ a, const BandedReal* __restrict__ b, int n)
{
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n)
        return;
    const Band x = static_cast<Band>(a[i]);
    const Band y = static_cast<Band>(b[i]);

    const Band ax  = aether::math::abs(x);
    const Band ay  = aether::math::abs(y);
    const Band cap = aether::math::copysign(aether::math::fmin(ax, ay), x);
    const Band r   = x / y;
    const Band s   = aether::math::fmax(aether::math::abs(r), ay);
    const Band t   = (s * cap) + r;
    out[i]         = t - cap;
}

/**
 * @brief ★ THE POSITIVE CONTROL. Fixed name; do not rename.
 *
 * The identical body with ONE deliberate FP64 operation. `poison` is a KERNEL
 * ARGUMENT, so the multiply cannot be constant-folded away. The audit is
 * REQUIRED to find FP64 here; without that leg, "0 FP64 hits" on the kernel
 * above is equally consistent with a working scan and with a symbol lookup that
 * quietly matched nothing.
 */
AETHER_KERNEL() void bandFp64InjectedKernel(BandedReal* __restrict__ out,
    const BandedReal* __restrict__ a, const BandedReal* __restrict__ b, int n,
    double poison)
{
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n)
        return;
    const Band x0      = static_cast<Band>(a[i]);
    const float spoilt = static_cast<float>(poison * 2.0 + 1.0);
    const Band x       = Band{ x0.hi * spoilt, x0.lo, x0.tail };
    const Band y       = static_cast<Band>(b[i]);

    const Band ax  = aether::math::abs(x);
    const Band ay  = aether::math::abs(y);
    const Band cap = aether::math::copysign(aether::math::fmin(ax, ay), x);
    const Band r   = x / y;
    const Band s   = aether::math::fmax(aether::math::abs(r), ay);
    const Band t   = (s * cap) + r;
    out[i]         = t - cap;
}

// =========================================================================
//  Device rows
// =========================================================================

/// @brief The pair corpus, encoded, with the reserved code points appended so
/// the device arm meets them too.
inline void buildDevicePairs(
    std::vector<BandedReal>& av, std::vector<BandedReal>& bv)
{
    const auto& c = pairCorpus();
    for (std::size_t i = 0; i + 1 < c.size(); i++) {
        av.push_back(BandedReal::fromDouble(c[i]));
        bv.push_back(BandedReal::fromDouble(c[i + 1]));
    }
    const BandedReal specials[] = {
        BandedReal::fromBits(0ull),
        BandedReal::fromBits(0x8000000000000000ull),
        NL::infinity(),
        NL::neg_infinity(),
        NL::quiet_NaN(),
        NL::max(),
        NL::lowest(),
        NL::min(),
        NL::epsilon(),
        BandedReal::fromDouble(1.0),
        BandedReal::fromDouble(-1.0),
    };
    const int ns = static_cast<int>(sizeof(specials) / sizeof(specials[0]));
    for (int i = 0; i < ns; i++)
        for (int j = 0; j < ns; j++) {
            av.push_back(specials[i]);
            bv.push_back(specials[j]);
        }
}

TEST_F(BandedRealCert, TheFacadeAndTheComparisonsAreBitIdenticalOnDevice)
{
    std::vector<BandedReal> av, bv;
    buildDevicePairs(av, bv);
    const int n = static_cast<int>(av.size());
    ASSERT_GT(n, 900);

    BandedReal *dA = nullptr, *dB = nullptr;
    BrProbe* dOut = nullptr;
    ASSERT_EQ(cudaMalloc(&dA, sizeof(BandedReal) * n), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dB, sizeof(BandedReal) * n), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dOut, sizeof(BrProbe) * n), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dA, av.data(), sizeof(BandedReal) * n, cudaMemcpyHostToDevice),
        cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dB, bv.data(), sizeof(BandedReal) * n, cudaMemcpyHostToDevice),
        cudaSuccess);

    bandedRealFacadeKernel<<<(n + kBlock - 1) / kBlock, kBlock>>>(dA, dB, dOut, n);
    aether::cuda::checkLastLaunch("bandedRealFacadeKernel");

    std::vector<BrProbe> got(static_cast<std::size_t>(n));
    ASSERT_EQ(cudaMemcpy(got.data(), dOut, sizeof(BrProbe) * n, cudaMemcpyDeviceToHost),
        cudaSuccess);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dOut);

    std::size_t bad = 0, cmpBad = 0, clsBad = 0, packBad = 0;
    for (int i = 0; i < n; i++) {
        const BandedReal ra = av[static_cast<std::size_t>(i)];
        const BandedReal rb = bv[static_cast<std::size_t>(i)];
        const Band ha       = aether::math::abs(ra);
        const Band hm       = aether::math::fmax(ra, rb);
        const Band hn       = aether::math::fmin(ra, rb);
        const Band hc       = aether::math::copysign(ra, rb);
        const Band hd       = ra / rb;
        const BrProbe& g    = got[static_cast<std::size_t>(i)];
        if (g.absHi != fbits(ha.hi) || g.absLo != fbits(ha.lo)
            || g.absTail != fbits(ha.tail))
            bad++;
        if (g.maxHi != fbits(hm.hi) || g.maxLo != fbits(hm.lo)
            || g.maxTail != fbits(hm.tail))
            bad++;
        if (g.minHi != fbits(hn.hi) || g.minLo != fbits(hn.lo)
            || g.minTail != fbits(hn.tail))
            bad++;
        if (g.csHi != fbits(hc.hi) || g.csLo != fbits(hc.lo)
            || g.csTail != fbits(hc.tail))
            bad++;
        if (g.divHi != fbits(hd.hi) || g.divLo != fbits(hd.lo)
            || g.divTail != fbits(hd.tail))
            bad++;
        if (g.cmp != brPackCmp(ra, rb))
            cmpBad++;
        const std::uint32_t hcls = (ra.isNan() ? 1u : 0u) | (ra.isInf() ? 2u : 0u)
            | (ra.isZero() ? 4u : 0u);
        if (g.cls != hcls)
            clsBad++;
        if (g.packWord != BandedReal(hm).toBits())
            packBad++;
    }
    std::printf("[BandedReal/device] ArmDiff over %d pairs: %zu facade, %zu "
                "comparison, %zu classification, %zu pack mismatches\n",
        n, bad, cmpBad, clsBad, packBad);
    EXPECT_EQ(bad, 0u) << "the device arm of an aether::math banded entry point "
                          "disagrees with the host arm bit for bit";
    EXPECT_EQ(cmpBad, 0u);
    EXPECT_EQ(clsBad, 0u);
    EXPECT_EQ(packBad, 0u);
}

TEST_F(BandedRealCert, LimitsAreLiteralReadsInsideADeviceBody)
{
    std::uint64_t* dOut = nullptr;
    ASSERT_EQ(cudaMalloc(&dOut, sizeof(std::uint64_t) * 8), cudaSuccess);
    bandedRealLimitsKernel<<<1, 32>>>(dOut);
    aether::cuda::checkLastLaunch("bandedRealLimitsKernel");
    std::uint64_t got[8] = { 0 };
    ASSERT_EQ(cudaMemcpy(got, dOut, sizeof(got), cudaMemcpyDeviceToHost), cudaSuccess);
    cudaFree(dOut);

    EXPECT_EQ(got[0], NL::max().toBits());
    EXPECT_EQ(got[1], NL::lowest().toBits());
    EXPECT_EQ(got[2], NL::min().toBits());
    EXPECT_EQ(got[3], NL::epsilon().toBits());
    EXPECT_EQ(got[4], NL::infinity().toBits());
    EXPECT_EQ(got[5], NL::quiet_NaN().toBits());
    EXPECT_EQ(got[6], NL::round_error().toBits());
    EXPECT_EQ(got[7], NL::denorm_min().toBits());
}

TEST_F(BandedRealCert, TheAuditedChainKernelAgreesWithTheHostAndItsControlDoesNot)
{
    /* This row exists for two reasons at once. It is the INSTANTIATION the SASS
     * audit needs (a header-only type emits nothing to scan until something
     * instantiates it), and it is the KNOWN-ANSWER check that the audited body
     * actually computes the chain rather than something the optimiser folded
     * away — an audit of a kernel that computes nothing is an audit of nothing.
     * The injected control is LAUNCHED too, so its presence in the binary is
     * not an accident of the linker. */
    const int n = 256;
    std::vector<BandedReal> av(n), bv(n);
    for (int i = 0; i < n; i++) {
        av[static_cast<std::size_t>(i)]
            = BandedReal::fromDouble(std::ldexp(1.0 + 0.01 * double(i), i % 41 - 20));
        bv[static_cast<std::size_t>(i)]
            = BandedReal::fromDouble(std::ldexp(2.0 - 0.003 * double(i), 7 - i % 13));
    }
    BandedReal *dA = nullptr, *dB = nullptr, *dOut = nullptr, *dInj = nullptr;
    ASSERT_EQ(cudaMalloc(&dA, sizeof(BandedReal) * n), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dB, sizeof(BandedReal) * n), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dOut, sizeof(BandedReal) * n), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dInj, sizeof(BandedReal) * n), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dA, av.data(), sizeof(BandedReal) * n, cudaMemcpyHostToDevice),
        cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dB, bv.data(), sizeof(BandedReal) * n, cudaMemcpyHostToDevice),
        cudaSuccess);

    bandChainKernel<<<(n + kBlock - 1) / kBlock, kBlock>>>(dOut, dA, dB, n);
    aether::cuda::checkLastLaunch("bandChainKernel");
    bandFp64InjectedKernel<<<(n + kBlock - 1) / kBlock, kBlock>>>(dInj, dA, dB, n, 0.0);
    aether::cuda::checkLastLaunch("bandFp64InjectedKernel");

    std::vector<BandedReal> got(static_cast<std::size_t>(n));
    std::vector<BandedReal> inj(static_cast<std::size_t>(n));
    ASSERT_EQ(cudaMemcpy(got.data(), dOut, sizeof(BandedReal) * n, cudaMemcpyDeviceToHost),
        cudaSuccess);
    ASSERT_EQ(cudaMemcpy(inj.data(), dInj, sizeof(BandedReal) * n, cudaMemcpyDeviceToHost),
        cudaSuccess);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dOut);
    cudaFree(dInj);

    std::size_t bad = 0, trivial = 0, injDiff = 0;
    for (int i = 0; i < n; i++) {
        const Band x   = static_cast<Band>(av[static_cast<std::size_t>(i)]);
        const Band y   = static_cast<Band>(bv[static_cast<std::size_t>(i)]);
        const Band ax  = aether::math::abs(x);
        const Band ay  = aether::math::abs(y);
        const Band cap = aether::math::copysign(aether::math::fmin(ax, ay), x);
        const Band r   = x / y;
        const Band s   = aether::math::fmax(aether::math::abs(r), ay);
        const Band t   = (s * cap) + r;
        const BandedReal want = t - cap;
        if (got[static_cast<std::size_t>(i)].toBits() != want.toBits())
            bad++;
        if (got[static_cast<std::size_t>(i)].isZero())
            trivial++;
        /* poison = 0 makes the injected kernel's scale factor exactly 1.0f, so
         * the two agree numerically. The control's job is to carry an FP64
         * INSTRUCTION, not a different answer. */
        if (inj[static_cast<std::size_t>(i)].toBits() != want.toBits())
            injDiff++;
    }
    std::printf("[BandedReal/device] audited chain over %d samples: %zu "
                "mismatches, %zu trivially-zero results, %zu injected-arm "
                "differences\n",
        n, bad, trivial, injDiff);
    EXPECT_EQ(bad, 0u);
    EXPECT_EQ(injDiff, 0u);
    EXPECT_LT(trivial, static_cast<std::size_t>(n) / 4)
        << "most of the audited chain's outputs are zero, so the SASS audit is "
           "scanning a kernel the optimiser may have hollowed out";
}

} // namespace BandedRealTest
} // namespace aether_tests
