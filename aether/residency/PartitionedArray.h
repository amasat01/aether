// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file PartitionedArray.h
 * @brief `aether::PartitionedArray<T,Es...>`: single-node
 *        multi-GPU pull-in — a host pinned chunk plus one per-device `Chunk`
 *        per rank, laid out per `aether/residency/Partition.h`'s PADDED
 *        block pitch (`padded_block_samples`), with `scatter()`/`gather()`
 *        moving real data H<->D through `aether::copy`/`copyAsync`
 *        (`aether/chunk/Copy.h`) and `deviceView(r)` exposing rank `r`'s own
 *        block as a `View` with `realCount(r)` samples but the block's
 *        TRUE physical (`padded`) leading-mode pitch — `Partition.h`'s own
 *        "padded stride, real extent" shape, `layout_stride` when
 *        `innerSize > 1` and short (see `deviceView`'s own docstring for
 *        why a compact `layout_right` view would silently mismatch the
 *        physical layout `scatter`/`gather` actually write).
 *
 * Rationale: every primitive this type needs already
 * exists — `Chunk` per-device allocation (`aether/chunk/Chunk.h`),
 * `PartitionSpec`/`padded_block_samples` (`aether/residency/Partition.h`),
 * legal-pair `copy`/`copyAsync` (`aether/chunk/Copy.h`) — this file is the
 * minimal glue wiring them into a single owning type, mirroring
 * `ReplicaSet`'s own "minimal, not a framework" scope (`Replica.h`).
 *
 * LAYOUT (v1, rank-generic over `Es...`, matching `Partition.h`/`make_view`'s
 * own "caller spells `dyn` explicitly as the trailing extent" convention):
 * `Es...` is `extents<Es...>`'s FULL argument list — every leading mode
 * static, the LAST mode `aether::dyn` (the SAMPLE mode `PartitionSpec`
 * slices). `n` = the logical (unpadded) total sample count; `innerSize` =
 * the product of the leading static extents (1 for a plain rank-1 array).
 *
 * The HOST chunk is allocated COMPACT at `n` samples (`kDLCUDAHost` when
 * this build has a CUDA backend, `kDLCPU` otherwise — pinned memory is a
 * CUDA-only concept; `AETHER_CPP_MODE`'s single-CPU-device degenerate case
 * needs a HOST-ADDRESSABLE chunk on BOTH ends of the copy,
 * which `kDLCPU<->kDLCPU` already is, per `Copy.h`'s legal-pair matrix).
 * Each DEVICE chunk is allocated at the UNIFORM `padded` pitch (one
 * `Device` per rank, `devices.size() == parts` (an explicit Device list,
 * no auto-discovery — mirrors `ReplicaSet`) — every rank's storage is the
 * SAME byte size regardless of whether its own real occupancy is short (the
 * trailing block only, per `padded_block_samples`'s own closed form).
 *
 * scatter()/gather() move ONLY the `innerSize` per-block STRIPS that hold
 * REAL data (never the padding tail): for each leading-mode "row" `c` (a
 * `layout_right` array of shape `(Es..., dyn)` stores component `c`'s `n`
 * (host) or `padded` (device) samples CONTIGUOUSLY — moving block `r`
 * copies `innerSize` such row-strips, `real(r)` elements each, host stride
 * `n`, device stride `padded` — genuinely a pitched (2D) transfer, done
 * here as `innerSize` separate `aether::copy`/`copyAsync` calls over
 * `Chunk::borrow()`-wrapped sub-ranges (no new copy primitive; reuses the
 * SAME legal-pair-checked machinery `Copy.h` already provides). For a
 * rank-1 array (`Es...` is just `dyn`, `innerSize == 1`) this collapses to
 * exactly one contiguous copy per rank, the common case.
 *
 * ⚠ CORRECTNESS NOTE: `Chunk::allocate(Device(kDLCUDA, id), ...)` did not
 * select `id` before `cudaMalloc` previously — every CUDA allocation
 * silently landed on whatever device was CURRENT (invisible on a
 * single-GPU box, where `id` is always 0). `aether/chunk/Chunk.h`'s
 * `kDLCUDA` branch now scopes a `cudaSetDevice(id)`/restore around the
 * allocation — the fix this type's whole "one Chunk per physical device"
 * premise depends on. The fix is behavior-preserving for every EXISTING
 * id-0 test (confirmed by inspection: `cudaGetDevice` reads back 0,
 * `cudaSetDevice(0)` is a no-op, restored to 0 again); the actual 2-GPU
 * exercise still needs real multi-GPU hardware to confirm.
 */

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "aether/chunk/Chunk.h"
#include "aether/chunk/Copy.h"
#include "aether/device/Device.h"
#include "aether/err/Error.h"
#include "aether/layout/Extents.h"
#include "aether/layout/detail/Carray.h"
#include "aether/residency/Partition.h"
#include "aether/residency/Transport.h" // moveTo()/exchange() are transport<Transport>-constrained
#include "aether/view/MakeView.h" // make_view() split out of view/View.h
#include "aether/view/View.h"

namespace aether {

/**
 * @brief Single-node multi-GPU partitioned array: one
 *        `Chunk` per device in `devices` (v1: an explicit list), holding
 *        rank `r`'s `PartitionSpec`-derived, PADDED-pitch block of a logical
 *        `n`-sample array whose canonical (compact) copy lives in a single
 *        host pinned `Chunk`.
 */
template<class T, std::size_t... Es>
class PartitionedArray {
    using Extents = extents<Es...>;

    static_assert(Extents::Rank >= 1, "PartitionedArray: Extents must have rank >= 1");
    static_assert(Extents::RankDynamic == 1, "PartitionedArray: exactly one dynamic (SAMPLE) mode is supported");
    static_assert(Extents::static_extent(Extents::Rank - 1) == dyn,
        "PartitionedArray: the trailing mode must be the dynamic SAMPLE mode (mirrors Partition.h)");

public:
    /** @brief The host canonical (COMPACT, `n` samples) view type — `hostView()`'s return type. */
    using ViewT = View<T, Extents, layout_right>;
    /** @brief `deviceView(r)`'s return type — see its own docstring for why
     *         this is `layout_stride`, not `layout_right`. */
    using DeviceViewT = View<T, Extents, layout_stride>;

    /**
     * @brief Allocate the host pinned chunk (`n` samples, compact) and one
     *        device `Chunk` per entry of `devices` (`padded_block_samples(n,
     *        devices.size(), pad_to)` samples each, uniform).
     *
     * @throws aether::Error  if `devices.empty()` (via `padded_block_samples`'s
     *         own `parts must be > 0` check) or `pad_to == 0`, or on any
     *         underlying `Chunk::allocate` failure (unsupported device kind
     *         for this build included — e.g. a `kDLCUDA` device in an
     *         `AETHER_CPP_MODE` build).
     */
    PartitionedArray(std::vector<Device> devices, std::size_t n, std::size_t pad_to = 32)
        : devices_(std::move(devices))
        , n_(n)
        , innerSize_(computeInnerSize_())
        , padded_(padded_block_samples(n, devices_.size(), pad_to))
        , hostChunk_(Chunk::allocate(hostDevice_(), innerSize_ * n_ * sizeof(T)))
    {
        deviceChunks_.reserve(devices_.size());
        for (const Device& d : devices_)
            deviceChunks_.push_back(Chunk::allocate(d, innerSize_ * padded_ * sizeof(T)));
    }

    /** @brief Number of ranks (== `devices.size()` at construction). */
    std::size_t parts() const { return devices_.size(); }
    /** @brief Logical (unpadded) total sample count. */
    std::size_t samples() const { return n_; }
    /** @brief The uniform per-rank PADDED pitch (samples), `Partition.h`'s closed form. */
    std::size_t padded() const { return padded_; }
    /** @brief Rank `r`'s REAL (valid) sample count — SHORT only for a trailing block. */
    std::size_t realCount(std::size_t r) const
    {
        const std::size_t offset = r * padded_;
        return (offset < n_) ? std::min(padded_, n_ - offset) : std::size_t{ 0 };
    }
    /** @brief Rank `r`'s `Device`. */
    Device device(std::size_t r) const { return devices_[r]; }

    /** @brief A writable view over the host-resident CANONICAL (compact,
     *         `n` samples) copy — fill it before `scatter()`, read it after
     *         `gather()`. */
    ViewT hostView() { return make_view<T, Es...>(hostChunk_, n_); }

    /**
     * @brief A writable view over rank `r`'s own device-resident block —
     *        `realCount(r)` samples (never the padding tail; SHORT only for
     *        a trailing block).
     *
     * NOT a compact `layout_right` view at `realCount(r)`: `deviceChunks_[r]`
     * is physically laid out at the UNIFORM `padded_` pitch (every leading
     * mode's row-strip is `padded_` elements apart — what `scatter()`/
     * `gather()` actually write, `copyRows_`'s own `dstPitchElems`/
     * `srcPitchElems`). For `innerSize_ == 1` (a plain rank-1 array) a
     * compact-at-`real` view and a padded-stride-at-`real` view address
     * IDENTICAL bytes (there is no leading mode to mis-stride), so the two
     * shapes only diverge for `innerSize_ > 1` WHEN a block is genuinely
     * short (`real < padded_`) — exactly the case
     * `test_PartitionedArray.cpp`'s rank-2 (`innerSize_ == 3`) degenerate-
     * path test exists to catch (it did: a
     * first cut here returned a compact `make_view<T,Es...>(deviceChunks_[r],
     * realCount(r))`, which silently read component 1's data from the WRONG
     * offset whenever `real != padded_`).
     */
    DeviceViewT deviceView(std::size_t r)
    {
        // The strides layout_right WOULD assign at the chunk's TRUE physical
        // capacity (padded_) — reuses Partition.h's own stride-derivation
        // helper (`detail::partitionStridesOf`) rather than re-deriving the
        // layout_right formula here.
        ViewT capacity                                          = make_view<T, Es...>(deviceChunks_[r], padded_);
        const detail::Carray<std::size_t, Extents::Rank> strides = detail::partitionStridesOf(capacity);
        Extents realExt(realCount(r));
        typename layout_stride::mapping<Extents> map(realExt, strides);
        return DeviceViewT(capacity.data(), map, capacity.device());
    }

    /** @brief Blocking H->each device's block: every rank's
     *         `realCount(r)` real samples, never the padding tail. */
    void scatter()
    {
        for (std::size_t r = 0; r < devices_.size(); ++r) {
            const std::size_t real = realCount(r);
            if (real == 0)
                continue;
            copyRows_(deviceChunks_[r], padded_, 0, hostChunk_, n_, r * padded_, real);
        }
    }

    /** @brief Blocking each device's block -> H. */
    void gather()
    {
        for (std::size_t r = 0; r < devices_.size(); ++r) {
            const std::size_t real = realCount(r);
            if (real == 0)
                continue;
            copyRows_(hostChunk_, n_, r * padded_, deviceChunks_[r], padded_, 0, real);
        }
    }

#ifdef AETHER_HAS_CUDA
    /**
     * @brief Asynchronous H->each device's block, one `Stream` per rank —
     *        the host chunk is `kDLCUDAHost` and every device chunk is
     *        `kDLCUDA`, `Copy.h`'s CUDAHost<->CUDA async-legal pair, so this
     *        is ALWAYS available in a CUDA-backend build). `streams.size()`
     *        must equal `parts()`; the caller owns synchronization (mirrors
     *        `Chunk::copyAsync`/`ReplicaSet::broadcast(src, stream)`).
     */
    void scatter(const std::vector<Stream>& streams)
    {
        for (std::size_t r = 0; r < devices_.size(); ++r) {
            const std::size_t real = realCount(r);
            if (real == 0)
                continue;
            copyRowsAsync_(deviceChunks_[r], padded_, 0, hostChunk_, n_, r * padded_, real, streams[r]);
        }
    }

    /** @brief Asynchronous each device's block -> H, one `Stream` per rank. */
    void gather(const std::vector<Stream>& streams)
    {
        for (std::size_t r = 0; r < devices_.size(); ++r) {
            const std::size_t real = realCount(r);
            if (real == 0)
                continue;
            copyRowsAsync_(hostChunk_, n_, r * padded_, deviceChunks_[r], padded_, 0, real, streams[r]);
        }
    }
#endif

    /**
     * @brief D2D: relocate rank `r`'s ENTIRE device-resident block (the
     *        full `padded_`-pitch storage, real data plus any padding
     *        tail) onto `device`, through `transport` — never touches
     *        `hostChunk_`. Blocking.
     *        `r`'s `Device` becomes `device`; the OLD chunk is freed once
     *        the move completes.
     *
     * `Transport` need only satisfy `aether::transport` (constrained here
     * via `requires`, mirroring `Replica<T,Transport>`'s own constraint) —
     * `StreamTransport`'s `route(a, b)` decides SAME/P2P/STAGED/HOST per
     * `aether/residency/PeerAccess.h`; in an `AETHER_CPP_MODE` build every
     * device is `kDLCPU`, so this compiles and moves data through the
     * ordinary CPU<->CPU `aether::copy` path (`peerRoute` always HOST).
     *
     * @throws aether::Error  if `r >= parts()`.
     */
    template<transport Transport>
    void moveTo(std::size_t r, Device device, Transport& t)
    {
        if (r >= devices_.size())
            err::fail("PartitionedArray::moveTo", "n/a", 0, "rank " + std::to_string(r) + " >= parts()=" + std::to_string(devices_.size()));
        Chunk moved = Chunk::allocate(device, innerSize_ * padded_ * sizeof(T));
        t.copy(moved, deviceChunks_[r]);
        deviceChunks_[r] = std::move(moved);
        devices_[r]      = device;
    }

#ifdef AETHER_HAS_CUDA
    /** @brief Asynchronous counterpart of `moveTo` (CUDA builds only) — the
     *         caller owns synchronization (`t.wait()`/`t.fence()` or a
     *         plain `cudaStreamSynchronize(stream)`) before reading rank
     *         `r`'s relocated block. */
    template<transport Transport>
    void moveTo(std::size_t r, Device device, Transport& t, Stream stream)
    {
        if (r >= devices_.size())
            err::fail("PartitionedArray::moveTo", "n/a", 0, "rank " + std::to_string(r) + " >= parts()=" + std::to_string(devices_.size()));
        Chunk moved = Chunk::allocate(device, innerSize_ * padded_ * sizeof(T));
        t.copyAsync(moved, deviceChunks_[r], stream);
        deviceChunks_[r] = std::move(moved);
        devices_[r]      = device;
    }
#endif

    /**
     * @brief D2D: swap ranks `ra`/`rb`'s device-resident block CONTENTS
     *        (the full `padded_`-pitch storage) through `transport` —
     *        each rank's OWN `Device` is UNCHANGED (`devices_[ra]`/`devices_[rb]` do not
     *        move); only the bytes resident there do. Blocking. A no-op
     *        when `ra == rb`.
     *
     * @throws aether::Error  if `ra >= parts()` or `rb >= parts()`.
     */
    template<transport Transport>
    void exchange(std::size_t ra, std::size_t rb, Transport& t)
    {
        if (ra >= devices_.size() || rb >= devices_.size()) {
            err::fail("PartitionedArray::exchange", "n/a", 0,
                "rank " + std::to_string(ra) + "/" + std::to_string(rb) + " >= parts()=" + std::to_string(devices_.size()));
        }
        if (ra == rb)
            return;
        const std::size_t bytes = innerSize_ * padded_ * sizeof(T);
        Chunk aOnB = Chunk::allocate(devices_[rb], bytes); // will hold what WAS on ra, resident on rb's device
        Chunk bOnA = Chunk::allocate(devices_[ra], bytes); // will hold what WAS on rb, resident on ra's device
        t.copy(aOnB, deviceChunks_[ra]);
        t.copy(bOnA, deviceChunks_[rb]);
        deviceChunks_[ra] = std::move(bOnA);
        deviceChunks_[rb] = std::move(aOnB);
    }

#ifdef AETHER_HAS_CUDA
    /** @brief Asynchronous counterpart of `exchange` (CUDA builds only) —
     *         both legs issued on the SAME `stream`, so the driver's own
     *         same-stream FIFO ordering keeps the two temporaries valid;
     *         the caller owns synchronization before reading either rank. */
    template<transport Transport>
    void exchange(std::size_t ra, std::size_t rb, Transport& t, Stream stream)
    {
        if (ra >= devices_.size() || rb >= devices_.size()) {
            err::fail("PartitionedArray::exchange", "n/a", 0,
                "rank " + std::to_string(ra) + "/" + std::to_string(rb) + " >= parts()=" + std::to_string(devices_.size()));
        }
        if (ra == rb)
            return;
        const std::size_t bytes = innerSize_ * padded_ * sizeof(T);
        Chunk aOnB = Chunk::allocate(devices_[rb], bytes);
        Chunk bOnA = Chunk::allocate(devices_[ra], bytes);
        t.copyAsync(aOnB, deviceChunks_[ra], stream);
        t.copyAsync(bOnA, deviceChunks_[rb], stream);
        deviceChunks_[ra] = std::move(bOnA);
        deviceChunks_[rb] = std::move(aOnB);
    }
#endif

private:
    /** @brief `kDLCUDAHost` when this build has a CUDA backend; `kDLCPU`
     *         otherwise (pinned memory is a CUDA-only concept — the
     *         `AETHER_CPP_MODE` degenerate path needs a plain
     *         host-addressable chunk instead). */
    static Device hostDevice_()
    {
#ifdef AETHER_HAS_CUDA
        return Device(kDLCUDAHost);
#else
        return Device(kDLCPU);
#endif
    }

    /** @brief Product of the LEADING static extents (`Es...` minus the
     *         trailing `dyn` SAMPLE mode) — `1` for a plain rank-1 array. */
    static constexpr std::size_t computeInnerSize_()
    {
        // `if constexpr` discards the loop branch entirely for the rank-1
        // case (sizeof...(Es) == 1, just the trailing dyn SAMPLE mode, no
        // leading extents at all) — avoids instantiating a `0 < 0` loop
        // bound nvcc otherwise (correctly, if noisily) flags as a
        // statically-known-false comparison for that degenerate case.
        if constexpr (sizeof...(Es) <= 1) {
            return 1;
        } else {
            constexpr std::size_t vals[]  = { Es... };
            constexpr std::size_t leading = sizeof...(Es) - 1;
            std::size_t inner             = 1;
            for (std::size_t i = 0; i < leading; ++i)
                inner *= vals[i];
            return inner;
        }
    }

    /**
     * @brief Blocking pitched copy: `innerSize_` row-strips of
     *        `rowElems` elements each, `dst`'s rows `dstPitchElems` apart
     *        starting at `dstRowOffsetElems`, `src`'s rows `srcPitchElems`
     *        apart starting at `srcRowOffsetElems` — reuses `aether::copy`'s
     *        own legal-pair-checked machinery per row via
     *        `Chunk::borrow()`, rather than a new copy primitive.
     */
    void copyRows_(Chunk& dst, std::size_t dstPitchElems, std::size_t dstRowOffsetElems, const Chunk& src,
        std::size_t srcPitchElems, std::size_t srcRowOffsetElems, std::size_t rowElems)
    {
        for (std::size_t c = 0; c < innerSize_; ++c) {
            const std::size_t dOff = c * dstPitchElems + dstRowOffsetElems;
            const std::size_t sOff = c * srcPitchElems + srcRowOffsetElems;
            Chunk dstRow = Chunk::borrow(dst.data() + dOff * sizeof(T), rowElems * sizeof(T), dst.device());
            Chunk srcRow = Chunk::borrow(src.data() + sOff * sizeof(T), rowElems * sizeof(T), src.device());
            copy(dstRow, srcRow);
        }
    }

#ifdef AETHER_HAS_CUDA
    /** @brief Asynchronous counterpart of `copyRows_` — same row split, `aether::copyAsync` per row on `stream`. */
    void copyRowsAsync_(Chunk& dst, std::size_t dstPitchElems, std::size_t dstRowOffsetElems, const Chunk& src,
        std::size_t srcPitchElems, std::size_t srcRowOffsetElems, std::size_t rowElems, Stream stream)
    {
        for (std::size_t c = 0; c < innerSize_; ++c) {
            const std::size_t dOff = c * dstPitchElems + dstRowOffsetElems;
            const std::size_t sOff = c * srcPitchElems + srcRowOffsetElems;
            Chunk dstRow = Chunk::borrow(dst.data() + dOff * sizeof(T), rowElems * sizeof(T), dst.device());
            Chunk srcRow = Chunk::borrow(src.data() + sOff * sizeof(T), rowElems * sizeof(T), src.device());
            copyAsync(dstRow, srcRow, stream);
        }
    }
#endif

    std::vector<Device> devices_;
    std::size_t n_;
    std::size_t innerSize_;
    std::size_t padded_;
    Chunk hostChunk_;
    std::vector<Chunk> deviceChunks_;
};

} // namespace aether
