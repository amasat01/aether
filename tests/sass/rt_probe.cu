// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Register probe for aether::rt (aether/expr/Rt.h): each rolled twin next to
// its static form on the long shapes where the static form is costly. Standalone
// TU, compiled directly, never globbed into tests/CMakeLists.txt, never run:
//
//   nvcc -std=c++20 -O3 -arch=sm_61 -Xptxas -v -I<aether> -c rt_probe.cu -o /dev/null
//
// Expected: each rt* kernel uses fewer registers than its static* sibling and
// spills 0 bytes (sm_61, nvcc 12.6: dot 144 = 17 vs 32, matVec 16x144 = 20 vs
// 255 + spill).

#include <aether/aether.h>

namespace {

using Vec144 = aether::View<double, aether::extents<144, aether::dyn>, aether::layout_right>;
using Vec16  = aether::View<double, aether::extents<16, aether::dyn>, aether::layout_right>;
using Mat    = aether::View<double, aether::extents<16, 144, aether::dyn>, aether::layout_right>;
using Out    = aether::View<double, aether::extents<aether::dyn>, aether::layout_right>;

AETHER_DEVICE() aether::SampleIndex sample()
{
    return aether::SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);
}

} // namespace

AETHER_KERNEL()
void staticDot144(Vec144 a, Vec144 b, Out out)
{
    const aether::SampleIndex i = sample();
    out(i.global())             = a[i].get().dot(b[i].get());
}

AETHER_KERNEL()
void rtDot144(Vec144 a, Vec144 b, Out out)
{
    const aether::SampleIndex i = sample();
    out(i.global())             = aether::rt::dot(a[i], b[i]);
}

AETHER_KERNEL()
void staticMatVec16x144(Mat m, Vec144 v, Vec16 out)
{
    const aether::SampleIndex i = sample();
    out[i]                      = m * v;
}

AETHER_KERNEL()
void rtMatVec16x144(Mat m, Vec144 v, Vec16 out)
{
    const aether::SampleIndex i = sample();
    aether::rt::matVec(out[i], m[i], v[i]);
}
