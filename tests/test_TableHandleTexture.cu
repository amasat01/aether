// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Device texture-carrier read path for `aether::TableHandle`. The test
// name keeps the `TableView` suite prefix (`test_TableView.cu`'s own row
// 14, `DeviceTextureCarrierReadsMatchThePlainCarrier`, is deferred to
// here) even though the body lives in this file, not test_TableView.cu:
// the mechanism under test is `TableHandle`'s texture carrier, not
// `TableView`.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace {

using aether::Chunk;
using aether::Device;
using aether::Plain;
using aether::SampleIndex;
using aether::TableHandle;
using aether::Texture;
using aether::TextureBinding;

/** @brief Element count shared by both rows below. */
constexpr std::size_t kN = 257; // several full blocks + a genuine partial tail

/** @brief Upload `host` (size `n`) to a fresh device `Chunk` of `double`. */
Chunk uploadDoubles(const std::vector<double>& host)
{
    const std::size_t bytes = host.size() * sizeof(double);
    Chunk stage = Chunk::allocate(Device(kDLCUDAHost), bytes);
    std::memcpy(stage.data(), host.data(), bytes);
    Chunk device = Chunk::allocate(Device(kDLCUDA), bytes);
    aether::copy(device, stage);
    return device;
}

// ---------------------------------------------------------------------------
// Row 14 — TableView.DeviceTextureCarrierReadsMatchThePlainCarrier.
// ---------------------------------------------------------------------------

/** @brief Read the same block once through the plain (`__ldg`-routed)
 *  carrier and once through the texture carrier, writing both results out
 *  for a HOST-side bit comparison — the claim is bit-identity between the
 *  two carriers, so the comparison itself must not launder any difference
 *  through device-side arithmetic. */
__global__ void dualCarrierReadKernel(
    TableHandle<double, Plain> plain, TableHandle<double, Texture> tex, double* outPlain, double* outTex, int n)
{
    const SampleIndex i = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);
    if (i.global() >= static_cast<aether::offset_t>(n))
        return;
    outPlain[i.global()] = plain[i.global()];
    outTex[i.global()] = tex[i.global()];
}

TEST(TableView, DeviceTextureCarrierReadsMatchThePlainCarrier)
{
    std::vector<double> host(kN);
    for (std::size_t i = 0; i < kN; ++i)
        host[i] = static_cast<double>(i) * 2.0 - 3.0;

    Chunk device = uploadDoubles(host);

    const auto v = aether::make_view<double, aether::dyn>(device, kN);
    TableHandle<double, Plain> plain(v);

    TextureBinding<double> binding = TextureBinding<double>::bind(device, kN);
    ASSERT_TRUE(binding.valid());
    TableHandle<double, Texture> tex(v.data(), binding.handle(), 0);

    double* dOutPlain = nullptr;
    double* dOutTex = nullptr;
    ASSERT_EQ(cudaMalloc(&dOutPlain, kN * sizeof(double)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dOutTex, kN * sizeof(double)), cudaSuccess);

    const auto cfg = aether::cuda::launchConfig(static_cast<aether::offset_t>(kN), 128);
    dualCarrierReadKernel<<<cfg.blocks, cfg.threads>>>(plain, tex, dOutPlain, dOutTex, static_cast<int>(kN));
    aether::cuda::checkLastLaunch("dualCarrierReadKernel");
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<double> hOutPlain(kN, 0.0), hOutTex(kN, 0.0);
    ASSERT_EQ(cudaMemcpy(hOutPlain.data(), dOutPlain, kN * sizeof(double), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(hOutTex.data(), dOutTex, kN * sizeof(double), cudaMemcpyDeviceToHost), cudaSuccess);
    cudaFree(dOutPlain);
    cudaFree(dOutTex);

    // Non-vacuity: both carriers must have actually read the uploaded ramp,
    // not two independently-zeroed buffers comparing equal to each other.
    std::size_t nonZero = 0;
    for (std::size_t i = 0; i < kN; ++i)
        if (hOutPlain[i] != 0.0)
            ++nonZero;
    ASSERT_GT(nonZero, kN / 2) << "the plain carrier produced a degenerate read stream — did the "
                                  "kernel run?";

    for (std::size_t i = 0; i < kN; ++i) {
        std::uint64_t bp = 0, bt = 0;
        std::memcpy(&bp, &hOutPlain[i], sizeof(bp));
        std::memcpy(&bt, &hOutTex[i], sizeof(bt));
        EXPECT_EQ(bt, bp) << "index " << i << ": texture carrier bits 0x" << std::hex << bt
                           << " != plain carrier bits 0x" << bp << std::dec;
        EXPECT_EQ(hOutTex[i], host[i]) << "index " << i << ": texture carrier did not reproduce the "
                                                            "uploaded value";
    }
}

// ---------------------------------------------------------------------------
// The claim: the `int2` texel fetch is a pure bit round-trip for `double`
// — no rounding, no reinterpretation loss — proven over an enumerated
// corpus of raw IEEE bit patterns (both zeros, both infinities, a
// canonical NaN, subnormals, and a representative binade sweep), not
// merely a ramp of "ordinary" values: the ramp in the row-14 test above
// could pass even if the fetch quietly mishandled a special value the ramp
// never produces.
// ---------------------------------------------------------------------------

__global__ void textureRoundTripKernel(TableHandle<double, Texture> tex, double* out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = tex[static_cast<aether::offset_t>(i)];
}

namespace {
std::uint64_t dbits(double x)
{
    std::uint64_t b = 0;
    std::memcpy(&b, &x, sizeof(b));
    return b;
}
double fromBits(std::uint64_t b)
{
    double x = 0.0;
    std::memcpy(&x, &b, sizeof(x));
    return x;
}
} // namespace

TEST(TextureSoftDoubleTest, DeviceRoundTrip_IsBitExact)
{
    std::vector<std::uint64_t> corpusBits = {
        0x0000000000000000ull, // +0
        0x8000000000000000ull, // -0
        0x7FF0000000000000ull, // +Inf
        0xFFF0000000000000ull, // -Inf
        0x7FF8000000000000ull, // canonical qNaN
        0x0000000000000001ull, // smallest subnormal
        0x800FFFFFFFFFFFFFull, // largest-magnitude negative subnormal
        0x3FF0000000000000ull, // 1.0
        0xBFF0000000000000ull, // -1.0
        0x4024000000000000ull, // 10.0
        0x3FE0000000000000ull, // 0.5
        0x7FEFFFFFFFFFFFFFull, // largest finite
        0xFFEFFFFFFFFFFFFFull, // largest finite negative
    };
    for (int e = -100; e <= 100; e += 7)
        for (std::uint64_t mant : { 0x0000000000000ull, 0xFFFFFFFFFFFFFull, 0x5555555555555ull })
            for (bool neg : { false, true }) {
                const std::uint64_t u = (neg ? (1ull << 63) : 0ull)
                    | (static_cast<std::uint64_t>(e + 1023) << 52) | mant;
                corpusBits.push_back(u);
            }
    ASSERT_GT(corpusBits.size(), 100u);

    std::vector<double> host(corpusBits.size());
    for (std::size_t i = 0; i < corpusBits.size(); ++i)
        host[i] = fromBits(corpusBits[i]);

    Chunk device = uploadDoubles(host);
    const auto v = aether::make_view<double, aether::dyn>(device, host.size());
    TextureBinding<double> binding = TextureBinding<double>::bind(device, host.size());
    ASSERT_TRUE(binding.valid());
    TableHandle<double, Texture> tex(v.data(), binding.handle(), 0);

    double* dOut = nullptr;
    const std::size_t n = host.size();
    ASSERT_EQ(cudaMalloc(&dOut, n * sizeof(double)), cudaSuccess);
    textureRoundTripKernel<<<static_cast<unsigned>((n + 63) / 64), 64>>>(tex, dOut, static_cast<int>(n));
    aether::cuda::checkLastLaunch("textureRoundTripKernel");
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<double> hOut(n, 0.0);
    ASSERT_EQ(cudaMemcpy(hOut.data(), dOut, n * sizeof(double), cudaMemcpyDeviceToHost), cudaSuccess);
    cudaFree(dOut);

    for (std::size_t i = 0; i < n; ++i) {
        SCOPED_TRACE(testing::Message() << "row " << i << " input bits 0x" << std::hex << corpusBits[i]);
        EXPECT_EQ(dbits(hOut[i]), corpusBits[i]) << "the int2 texture round-trip did not reproduce the "
                                                     "input bit pattern exactly";
    }
}

} // namespace
} // namespace aether_tests
