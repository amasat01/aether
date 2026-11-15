//! The device arms of the helpers.
/// Device spelling of the atomic add.
pub const ARM_ACCUM_ADD: &str = r####"    atomicAdd(&slot, static_cast<E>(term));"####;
/// Device spelling of the compensated atomic add.
pub const ARM_ACCUM_ADD_COMPENSATED: &str = r####"    const E old = atomicAdd(&slot, t);
    const E s = old + t;
    atomicAdd(&compSlot, aether::accum::twoSumErrAtomic(old, t, s));"####;
/// Device spelling of the atomic or.
pub const ARM_ACCUM_OR: &str = r####"    atomicOr(reinterpret_cast<unsigned int*>(&slot), static_cast<unsigned int>(bits));"####;
/// Device spelling of the finished-count commit.
pub const ARM_FINISH_COUNT: &str =
    r####"    atomicAdd(reinterpret_cast<unsigned int*>(cell), 1u);"####;
/// Device spelling of the mix-count commit.
pub const ARM_MIX_COUNT: &str = r####"    const unsigned active = __activemask();
    unsigned below;
    asm("mov.u32 %0, %%lanemask_lt;" : "=r"(below));
    const bool leader = (active & below) == 0u;
    for (int j = 0; j < K; ++j) {
        const unsigned picked = __ballot_sync(active, k == j);
        if (leader && picked != 0u)
            atomicAdd(reinterpret_cast<unsigned int*>(cells + j),
                      static_cast<unsigned int>(__popc(picked)));
    }"####;
/// Device spelling of the spacing helper.
pub const ARM_SPACING: &str = r####"    const T up = ::nextafter(a, a * T(2) + T(1)) - a;"####;
/// The tile helper of the device text.
pub const TILED: &str = r####"
template <class T, std::size_t W>
struct TiledLane {
    using element_type = T;
    using element_extents = aether::extents<W>;
    T* base;
    @QUAL@ T& at(std::size_t k) const { return base[k * 32]; }
    /** the lane as an aether expression (a stride-0 sample mode: every sample reads this lane's elements) */
    @QUAL@ auto expr() const
    {
        using Ext = aether::extents<W, aether::dyn>;
        using V = aether::View<T, Ext, aether::layout_stride>;
        return V(base, typename V::mapping_type(Ext(1), {std::size_t{32}, std::size_t{0}}), aether::Device{});
    }
};

template <class T, std::size_t W, class ViewT>
@QUAL@ TiledLane<T, W> tiled(const ViewT& plane, long long lane)
{
    const std::size_t i = static_cast<std::size_t>(lane);
    return TiledLane<T, W>{plane.data() + (i / 32 * W) * 32 + i % 32};
}
"####;
/// The device warp form of the row sum.
pub const SAMPLE_SUM_WARP: &str = r####"template <class E>
__device__ __forceinline__ void ssPin_(E& x)
{
    // A rounded term: a product in the body is never fused into a combine.
    if constexpr (sizeof(E) == 4) asm("" : "+f"(x));
    else asm("" : "+d"(x));
}

template <int M, int OP, class E>
__device__ __forceinline__ void ssRound_(E* h, E* c, int off, bool up)
{
    // One butterfly round over a lane's M columns: it keeps one contiguous
    // half (the upper one when `up`), adds its partner's (lane ^ off) copy
    // of that half, and ships the other half to the partner.
    constexpr int LO = M / 2, UP = M - LO;
#pragma unroll
    for (int j = 0; j < UP; ++j) {
        const E sh = up ? (j < LO ? h[j] : ssIdentity_<OP, E>()) : h[LO + j];
        const E sc = up ? (j < LO ? c[j] : E(0)) : c[LO + j];
        const E rh = __shfl_xor_sync(0xffffffffu, sh, off);
        const E rc = __shfl_xor_sync(0xffffffffu, sc, off);
        if (up) pairCombine_<OP>(h[LO + j], c[LO + j], rh, rc);
        else if (j < LO) pairCombine_<OP>(h[j], c[j], rh, rc);
    }
#pragma unroll
    for (int j = 0; j < UP; ++j) {
        h[j] = up ? h[LO + j] : (j < LO ? h[j] : ssIdentity_<OP, E>());
        c[j] = up ? c[LO + j] : (j < LO ? c[j] : E(0));
    }
}

template <int N, int OP = 0, class E, class Store>
__device__ __forceinline__ void sampleSumWarp_(E (&h)[N], Store&& store)
{
    // The warp's 32 lanes' terms summed column by column in five rounds
    // (lane ^ 16, 8, 4, 2, 1); each lane then stores the sums of the
    // contiguous run of columns it holds. Every lane of the warp calls it.
    E c[N];
#pragma unroll
    for (int k = 0; k < N; ++k) {
        ssPin_(h[k]);
        c[k] = E(0);
    }
    unsigned lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));
    constexpr int M2 = N - N / 2, M3 = M2 - M2 / 2, M4 = M3 - M3 / 2;
    constexpr int M5 = M4 - M4 / 2, M6 = M5 - M5 / 2;
    ssRound_<N, OP>(h, c, 16, (lane & 16u) != 0u);
    ssRound_<M2, OP>(h, c, 8, (lane & 8u) != 0u);
    ssRound_<M3, OP>(h, c, 4, (lane & 4u) != 0u);
    ssRound_<M4, OP>(h, c, 2, (lane & 2u) != 0u);
    ssRound_<M5, OP>(h, c, 1, (lane & 1u) != 0u);
    // the columns a lane holds: the rounds' split points are M / 2 of the
    // padded counts above, so a lane may hold fewer than M (or none)
    const int halves[5] = {N / 2, M2 / 2, M3 / 2, M4 / 2, M5 / 2};
    int lo = 0, m = N;
#pragma unroll
    for (int r = 0; r < 5; ++r) {
        if (lane & (16u >> r)) {
            lo += halves[r];
            m -= halves[r];
        } else if (m > halves[r]) {
            m = halves[r];
        }
    }
#pragma unroll
    for (int j = 0; j < M6; ++j)
        if (j < m) store(lo + j, h[j], c[j]);
}
"####;
/// The header a banded unit with a row sum includes.
pub const BANDED_SUM_INCLUDE: &str = r####"#include <aether/banded/Accum.h>
#include <eagle/cuda/WarpShuffle.h>
"####;
/// The banded pair combine of the row sum.
pub const BANDED_SUM: &str = r####"
template <int OP = 0>
@QUAL@ void pairCombine_(aether::banded::BandAccum& s, aether::banded::BandAccum&,
                             aether::banded::BandAccum s2, aether::banded::BandAccum)
{
    s.combine(s2);
}

@QUAL@ void ssPin_(aether::banded::BandAccum&) {}

// hawk.max / hawk.min: the fold is exact, so a partial is a bare Band and two partials merge with the certified fmax / fmin.
template <class E>
@QUAL@ E ssInf_(float sign) { return E(sign * __builtin_inff(), 0.0f, 0.0f); }

@QUAL@ aether::banded::Band ssExtreme_(int op, aether::banded::Band a, aether::banded::Band b)
{
    return op == 1 ? aether::banded::detail::fmax(a, b) : aether::banded::detail::fmin(a, b);
}

template <int OP = 0>
@QUAL@ void pairCombine_(aether::banded::Band& s, aether::banded::Band&, aether::banded::Band s2, aether::banded::Band)
{
    s = ssExtreme_(OP, s, s2);
}

@QUAL@ void ssPin_(aether::banded::Band&) {}

@QUAL@ Real ssOut_(aether::banded::Band s) { return Real(s); }

@QUAL@ Real ssOut_(aether::banded::BandAccum s) { return s.toBandedReal(); }

@QUAL@ Real ssComp_(aether::banded::BandAccum) { return Real{}; }

__device__ __forceinline__ aether::banded::BandAccum ssShfl_(aether::banded::BandAccum v, int off)
{
    return eagle::cuda::shflXor(0xffffffffu, v, off);
}

__device__ __forceinline__ aether::banded::Band ssShfl_(aether::banded::Band v, int off)
{
    return eagle::cuda::shflXor(0xffffffffu, v, off);
}
"####;
/// The banded output helpers that replace the pinning one.
pub const BANDED_SS_HELPERS: &str = r####"template <class E>
@QUAL@ E ssOut_(E s) { return s; }

template <class E>
@QUAL@ E ssComp_(E c) { return c; }

template <class E>
__device__ __forceinline__ E ssShfl_(E v, int off)
{
    return __shfl_xor_sync(0xffffffffu, v, off);
}

template <class E>
__device__ __forceinline__ void ssPin_"####;
/// The warp-sum helper.
pub const WARP_SUM: &str = r####"
template <class T>
@QUAL@ T warp_sum(T v)
{
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}
"####;
