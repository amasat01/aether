// NVRTC row RED arm: the same chain plus one host-only symbol. It must FAIL to compile.
#include <aether/banded/banded.h>
#include <aether/banded/BandedRealOps.h>

using aether::banded::BandedReal;

extern "C" __global__ void bandRtcHostSymbol(BandedReal* out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    out[i] = BandedReal::fromDouble(1.5);
}
