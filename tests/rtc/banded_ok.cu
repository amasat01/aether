// NVRTC row source: the banded carrier's device chain, nothing host-only.
#include <aether/banded/banded.h>
#include <aether/banded/BandedRealOps.h>

using aether::banded::BandedReal;

extern "C" __global__ void bandRtcChain(BandedReal* out, const BandedReal* a, const BandedReal* b, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const BandedReal x = a[i];
    const BandedReal y = b[i];
    BandedReal r = (x + y) * x / y;
    r = aether::math::sqrt(r) + aether::math::exp(x) - aether::math::log(y);
    r = r + aether::math::sin(x) + aether::math::atan2(x, y);
    out[i] = r;
}

// The lazy chain states (Chain.h): an unrestored product-sum restored once at the store.
extern "C" __global__ void bandRtcLazyChain(BandedReal* out, const BandedReal* a, const BandedReal* b, int n)
{
    namespace bd = aether::banded::detail;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const auto s = (a[i].band() * b[i].band()) * a[i].band() + b[i].band();
    out[i] = BandedReal(s);
}
