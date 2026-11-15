//! Fixed C++ text of the host lane features: the interchanged lane loops, the lane split's entries and the mask helpers.
/// The mask helpers of the host text.
pub const HOST_MASK_HELPERS: &str = r####"#if (defined(__AVX512F__) && defined(__GNUC__) && !defined(__clang__)) || defined(__aarch64__)
#define HAWK_HOST_LANE_GUARDS 1
#else
#define HAWK_HOST_LANE_GUARDS 0
#endif

namespace hawk_abi {

// Samples per block of the lane-guarded loop (one zmm of guard bytes).
inline constexpr aether::offset_t kHostLaneBlock = 64;

static inline aether::offset_t lane_block_end(aether::offset_t lo, aether::offset_t hi)
{
    return hi - lo < kHostLaneBlock ? hi : lo + kHostLaneBlock;
}

struct HostLaneMaskRef {
    HAWK_LANE_T lane;
    bool eval() const { return lane != 0.0; }
};

// A block's guards, one float64 lane per sample of [base, base + kHostLaneBlock).
struct HostLaneMask {
    const HAWK_LANE_T* lanes;
    aether::offset_t base;
    HostLaneMaskRef operator[](const aether::SampleIndex& at) const
    {
        return HostLaneMaskRef{ lanes[at.global() - base] };
    }
};

struct HostMaskRef {
    const unsigned char* at;
    bool eval() const { return *at != 0; }
};

struct HostMask {
    const unsigned char* data;
    HostMaskRef operator[](const aether::SampleIndex& at) const
    {
        return HostMaskRef{ data + at.global() };
    }
    // Widen the guards of samples [lo, hi) into out[0, hi - lo).
    HostLaneMask lanes(HAWK_LANE_T* out, aether::offset_t lo, aether::offset_t hi) const
    {
        for (aether::offset_t s = lo; s < hi; ++s)
            out[s - lo] = data[s] != 0 ? 1.0 : 0.0;
        return HostLaneMask{ out, lo };
    }
};

static inline HostMask host_mask(const eagle::plugin::ScalarHandle& m)
{
    return HostMask{ static_cast<const unsigned char*>(m.data) };
}

#if HAWK_HOST_LANE_GUARDS
struct HostLaneFinishRef {
    HAWK_LANE_T* lane;
    void operator=(bool v) const { *lane = v ? 1.0 : 0.0; }
};

// The finish epilogue's mask writes inside a lane-guarded block land in the
// sample's own guard lane, so the block's loop stores no byte and calls nothing.
struct HostLaneFinish {
    HAWK_LANE_T* lanes;
    aether::offset_t base;
    HostLaneFinishRef operator[](const aether::SampleIndex& at) const
    {
        return HostLaneFinishRef{ lanes + (at.global() - base) };
    }
};

// The finish epilogue's count inside a lane-guarded block: a plain increment
// of the block's own tally, which finish_lanes adds to the counter at once.
// (A float64 tally keeps the loop's narrowest type the float64 lane; it
// counts at most kHostLaneBlock, exactly.)
struct HostLaneTally {
    HAWK_LANE_T* finished;
};

static inline void finish_count(const HostLaneTally& tally) { *tally.finished += 1.0; }

// Commit a block's guard lanes back to the mask bytes of samples [lo, hi) (a
// set lane is a set byte), then its tally to the counter in ONE relaxed add:
// the same total as one increment per newly finished sample.
template <class ViewT>
static inline void finish_lanes(const HostMask& m, const HAWK_LANE_T* lanes,
                                aether::offset_t lo, aether::offset_t hi,
                                HAWK_LANE_T tally, const ViewT& counter)
{
    unsigned char* bytes = const_cast<unsigned char*>(m.data);
    for (aether::offset_t s = lo; s < hi; ++s) {
        if (lanes[s - lo] != 0.0)
            bytes[s] = 1;
    }
    const auto finished = static_cast<std::uint32_t>(tally);
    if (finished != 0) {
        auto* cell = const_cast<std::uint32_t*>(
            reinterpret_cast<const std::uint32_t*>(counter.data()));
        std::atomic_ref<std::uint32_t>(*cell).fetch_add(finished, std::memory_order_relaxed);
    }
}
#endif

}  // namespace hawk_abi
"####;
/// The statement that tells the compiler loop iterations are independent.
pub const INDEPENDENT_ITERATIONS: &str = r####"#if defined(__clang__)
#pragma clang loop vectorize(assume_safety)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif"####;
/// The helpers of the interchanged lane loops.
pub const INTERCHANGE_HELPERS: &str = r####"#if defined(__aarch64__)
#include <algorithm>
#endif
namespace hawk_abi {

template <aether::offset_t Tile, class T>
static inline void lane_put(T* lanes, aether::offset_t t, const T& v)
{
    lanes[t] = v;
}

template <aether::offset_t Tile, class T, std::size_t... Es>
static inline void lane_put(T* lanes, aether::offset_t t,
                            const aether::Item<T, Es...>& v)
{
    for (std::size_t c = 0; c < (std::size_t{1} * ... * Es); ++c)
        lanes[static_cast<aether::offset_t>(c) * Tile + t] = v.data()[c];
}

// A finished lane keeps its old value: `on ? v : old`, one select per row.
template <aether::offset_t Tile, class T>
static inline void lane_keep(T* lanes, aether::offset_t t, bool on, const T& v)
{
    lanes[t] = blend(on, v, lanes[t]);
}

template <aether::offset_t Tile, class T, std::size_t... Es>
static inline void lane_keep(T* lanes, aether::offset_t t, bool on,
                             const aether::Item<T, Es...>& v)
{
    for (std::size_t c = 0; c < (std::size_t{1} * ... * Es); ++c) {
        T& slot = lanes[static_cast<aether::offset_t>(c) * Tile + t];
        slot = blend(on, v.data()[c], slot);
    }
}

template <class V>
struct LaneOf {
    using type = V;
    static V get(const V* lanes, aether::offset_t, aether::offset_t t)
    {
        return lanes[t];
    }
};

template <class T, std::size_t... Es>
struct LaneOf<aether::Item<T, Es...>> {
    using type = T;
    static aether::Item<T, Es...> get(const T* lanes, aether::offset_t tile,
                                      aether::offset_t t)
    {
        aether::Item<T, Es...> v;
        for (std::size_t c = 0; c < (std::size_t{1} * ... * Es); ++c)
            v.data()[c] = lanes[static_cast<aether::offset_t>(c) * tile + t];
        return v;
    }
};

template <class V, aether::offset_t Tile>
static inline V lane_get(const typename LaneOf<V>::type* lanes, aether::offset_t t)
{
    return LaneOf<V>::get(lanes, Tile, t);
}

// The lanes one chunk of the interchanged loop advances together: one vector
// register of float64 (8 under AVX-512, 4 under AVX, 2 otherwise), so a
// chunk's carried values stay in registers across its trips.
#if defined(__AVX512F__)
inline constexpr aether::offset_t kHostChunk = 8;
#elif defined(__AVX__)
inline constexpr aether::offset_t kHostChunk = 4;
#else
inline constexpr aether::offset_t kHostChunk = 2;
#endif
// Chunks' worth of vector registers the target wants on top of the step's own
// (chunk_vectors): four on aarch64, whose cores issue four NEON FMAs a cycle.
#if defined(__aarch64__)
inline constexpr aether::offset_t kHostChunkVectors = 4;
#else
inline constexpr aether::offset_t kHostChunkVectors = 1;
#endif

}  // namespace hawk_abi
"####;
/// The helper that writes one row of lanes.
pub const LANE_ROW_HELPER: &str = r####"namespace hawk_abi {

template <class T, std::size_t W, aether::offset_t Tile>
struct TileLane {
    using element_type = T;
    using element_extents = aether::extents<W>;
    T* base;
    T& at(std::size_t k) const { return base[k]; }
    /** the row as an aether expression (a stride-0 sample mode: every sample reads this lane's row) */
    auto expr() const
    {
        using Ext = aether::extents<W, aether::dyn>;
        using V = aether::View<T, Ext, aether::layout_stride>;
        return V(base, typename V::mapping_type(Ext(1), {std::size_t{1}, std::size_t{0}}), aether::Device{});
    }
};

template <class T, std::size_t W, aether::offset_t Tile>
static inline TileLane<T, W, Tile> tile_lane(T* rows, aether::offset_t t)
{
    return TileLane<T, W, Tile>{rows + static_cast<std::size_t>(t) * W};
}

}  // namespace hawk_abi
"####;
/// The helpers of the loop entries.
pub const LOOP_HELPERS: &str = r####"namespace hawk_abi {

#if defined(__AVX512F__)
inline constexpr aether::offset_t kHostVectorBytes = 64;
#elif defined(__AVX__)
inline constexpr aether::offset_t kHostVectorBytes = 32;
#else
inline constexpr aether::offset_t kHostVectorBytes = 16;
#endif
inline constexpr aether::offset_t kHostLoopRegisters = 8;
inline constexpr aether::offset_t kHostLoopTile =
    kHostLoopRegisters * kHostVectorBytes / static_cast<aether::offset_t>(sizeof(HAWK_LANE_T));

}  // namespace hawk_abi
"####;
/// The prelude of the loop entries.
pub const LOOP_PRELUDE: &str = r####"#ifndef HAWK_HOST_INTERCHANGE
#define HAWK_HOST_INTERCHANGE 1
#endif
// hawk host loop interchange: tile = hawk_abi::kHostLoopTile
"####;
/// The statement that keeps the compiler from unrolling a copy-in loop.
pub const COPY_IN_NO_UNROLL: &str = r####"#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC unroll 1
#endif"####;
/// The statement that sets the trip loop's iteration hint.
pub const TRIP_ITERATIONS: &str = r####"#if defined(__clang__) && defined(__aarch64__)
#pragma clang loop vectorize(assume_safety) interleave_count(4)
#elif defined(__clang__)
#pragma clang loop vectorize(assume_safety)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif"####;
/// The run entry's text.
pub const RUN_ENTRY: &str = r####"extern "C" const std::int64_t @NAME@_host_run_k = @STOP@;
extern "C" std::int64_t @NAME@_host_run_rounds = 0;
extern "C" void @NAME@_host(void* const* params,
    std::int64_t base,
    std::int64_t count,
    std::int64_t nSamples);
extern "C" void @NAME@_host_run(void* const* params,
    std::int64_t base,
    std::int64_t count,
    std::int64_t nSamples)
{
    void* hawk_own[@LEN@];
    for (std::size_t hawk_s = 0; hawk_s < @LEN@; ++hawk_s)
        hawk_own[hawk_s] = params[hawk_s];
    eagle::plugin::ScalarHandle hawk_word = *static_cast<const eagle::plugin::ScalarHandle*>(params[@WORD@]);
    std::int64_t hawk_left = static_cast<std::int64_t>(
        static_cast<const Int*>(hawk_word.data)[0]);
    Int hawk_k = 0;
    hawk_word.data = &hawk_k;
    hawk_own[@WORD@] = &hawk_word;
    const unsigned char* hawk_done = static_cast<const unsigned char*>(
        static_cast<const eagle::plugin::ScalarHandle*>(params[@DONE@])->data);
    std::int64_t hawk_rounds = 0;
    while (hawk_left > 0) {
        bool hawk_live = false;
        for (std::int64_t hawk_s = base; hawk_s < base + count; ++hawk_s) {
            if (hawk_done[hawk_s] == 0) {
                hawk_live = true;
                break;
            }
        }
        if (!hawk_live) break;
        hawk_k = static_cast<Int>(hawk_left < @STOP@ ? hawk_left : @STOP@);
        @NAME@_host(hawk_own, base, count, nSamples);
        hawk_left -= static_cast<std::int64_t>(hawk_k);
        ++hawk_rounds;
    }
    std::atomic_ref<std::int64_t> hawk_most(@NAME@_host_run_rounds);
    std::int64_t hawk_seen = hawk_most.load(std::memory_order_relaxed);
    while (hawk_seen < hawk_rounds
           && !hawk_most.compare_exchange_weak(hawk_seen, hawk_rounds,
                                               std::memory_order_relaxed)) {
    }
}"####;
