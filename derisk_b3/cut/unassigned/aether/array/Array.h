// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Array.h
 * @brief `aether::Array`: the GPU-resident, host-mirrored container you
 *        allocate once and read/write via `hostView()`/`upload()` —
 *        aether's owning convenience per the symmetry rule.
 *
 *        `Array<T, Es...>` appends the dynamic sample mode
 *        (`extents<Es..., dyn>`), e.g. `Array<double,3>` is a `Vec3dArray`.
 *
 * CUDA mode: a pinned host `Chunk` (`kDLCUDAHost`) plus a device `Chunk`
 * (`kDLCUDA`). `AETHER_CPP_MODE`: a single `kDLCPU` chunk — `hostView()`/
 * `hostPacked()`/`hostSpare()` and their `device*` counterparts all view
 * the same chunk, and `upload()`/`download()` are no-ops. Move-only
 * (inherited from `Chunk`'s own move-only ownership) — rule of zero.
 *
 * `upload`/`download`: a bare call is blocking, a `Stream` argument makes
 * it async — no separate `*Async` names.
 *
 * Capacity != size: `samples()` is the current logical count
 * (vector-`size()`-like); `capacity()` is how many samples the backing
 * `Chunk`(s) actually hold room for — always a multiple of
 * `kCapacityAlignment` (the same rounding rule `aether::padded_block_samples`
 * uses for `PartitionSpec`'s default `pad_to`, reused here via
 * `padded_block_samples(n, 1, kCapacityAlignment)` so the two constants
 * never drift apart). `reserve`/`resize`/`push_back` grow `capacity()`
 * (host-side realloc + pitched copy, host-throw on allocation failure);
 * `clear()` drops `samples()` to 0 without releasing `capacity()`. Growth
 * within capacity never moves data or changes any view's stride.
 *
 * Because the pitch between consecutive components of a multi-component
 * Array (e.g. `Array<double,3>`) is `capacity()`, not `samples()`, a
 * `layout_right` view built naively over `samples()` samples would silently
 * mis-stride the moment `capacity() != samples()` — `deviceView()`'s note
 * in `aether/residency/PartitionedArray.h` names this exact trap for its
 * own padded per-device blocks. `hostView()`/`deviceView()` therefore
 * return a `layout_stride` view whose stride is derived from a
 * `capacity()`-sized `layout_right` mapping
 * (`aether::detail::partitionStridesOf`, `aether/residency/Partition.h` —
 * the same technique `PartitionedArray::deviceView()` already uses for its
 * own padded blocks) and whose shape (dynamic sample extent) is
 * `samples()` — shape from size, strides from capacity. `hostPacked()`/
 * `devicePacked()` are the compact `layout_right` view, valid only when
 * `capacity() == samples()` (host-throw otherwise). `hostSpare()`/
 * `deviceSpare()` are the writable `[samples(), capacity())` tail view a
 * device append kernel (`AppendCounter`) targets; `commit(count)`
 * (host-side, bounds-checked against spare capacity) is how the host
 * adopts what a kernel wrote there into `samples()`.
 */

#include <cstddef>

#include "aether/chunk/Chunk.h"
#include "aether/chunk/Copy.h"
#include "aether/device/Device.h"
#include "aether/err/Error.h"
#include "aether/layout/Extents.h"
#include "aether/layout/Layout.h"
#include "aether/macros.h"
#include "aether/residency/Partition.h"
#include "aether/view/Item.h"
#include "aether/view/MakeView.h" // make_view() split out of view/View.h
#include "aether/view/View.h"

namespace aether {

/**
 * @brief The capacity alignment quantum every `Array`'s reserved capacity
 *        is rounded up to. Deliberately the same numeric value as
 *        `aether::PartitionSpec::pad_to`'s own default (`aether/residency/
 *        Partition.h`) — reused via `padded_block_samples(n, 1,
 *        kCapacityAlignment)` rather than re-implemented, so an `Array`'s
 *        own outer-mode pitch and a partition block's default pitch share
 *        one rounding rule end to end (never two constants to keep in sync).
 */
inline constexpr std::size_t kCapacityAlignment = 32;

template<class T, std::size_t... Es>
class Array {
public:
    /** @brief `extents<Es..., dyn>` — the symmetry rule. */
    using Extents = extents<Es..., dyn>;
    /** @brief `hostView()`/`deviceView()`'s return type: pitch =
     *         `capacity()` (quantised), shape = `samples()`. */
    using ViewT = View<T, Extents, layout_stride>;
    using ConstViewT = View<const T, Extents, layout_stride>;
    /** @brief `hostPacked()`/`devicePacked()`'s return type: the original
     *         compact accessor — valid only when `capacity() == samples()`. */
    using PackedViewT = View<T, Extents, layout_right>;
    using ConstPackedViewT = View<const T, Extents, layout_right>;

    /** @brief Allocate storage for `n_samples` items of shape `Es...`.
     *         `capacity()` is `n_samples` rounded up to `kCapacityAlignment`
     *         — unconditional, even for a fresh Array: growth later must
     *         never re-stride an Array that never grew. */
    explicit Array(std::size_t n_samples)
        : samples_(n_samples)
        , capacity_(quantizedCapacity_(n_samples))
#ifdef AETHER_HAS_CUDA
        , hostChunk_(Chunk::allocate(Device(kDLCUDAHost), bytes_(capacity_)))
        , deviceChunk_(Chunk::allocate(Device(kDLCUDA), bytes_(capacity_)))
#else
        , hostChunk_(Chunk::allocate(Device(kDLCPU), bytes_(capacity_)))
#endif
    {
    }

    /** @brief Number of samples (the batch mode's current extent — vector-
     *         `size()`-like; unaffected by any reserved headroom). */
    std::size_t samples() const { return samples_; }
    /** @brief Total element count — `(Es * ...) * samples()`. */
    std::size_t size() const { return elementCount_(samples_); }
    /** @brief How many samples the backing `Chunk`(s) actually hold room
     *         for — always a multiple of `kCapacityAlignment`. */
    std::size_t capacity() const { return capacity_; }

    /** @brief Grow `capacity()` to at least `n` (quantised up to
     *         `kCapacityAlignment`) — a no-op when `n <= capacity()`.
     *         Otherwise reallocates the backing `Chunk`(s) and copies the
     *         existing `samples()` real elements, pitch-aware (component c's
     *         `samples()` elements move from offset `c*capacity()` to
     *         `c*newCapacity`), preserving `samples()`. `data()` pointer
     *         identity is not preserved across a realloc (host-throw on
     *         allocation failure — from `Chunk::allocate`). */
    void reserve(std::size_t n)
    {
        if (n <= capacity_)
            return;
        reallocate_(quantizedCapacity_(n));
    }

    /** @brief Set `samples()` to `n` — no reallocation (pointer/stride
     *         stable) while `n <= capacity()`; otherwise `reserve(n)` first
     *         (realloc + pitched copy of the old `samples()` real
     *         elements), then adopts `n` as the new `samples()`. Shrinking
     *         (`n < samples()`) never reallocates and never releases
     *         `capacity()`. */
    void resize(std::size_t n)
    {
        if (n > capacity_)
            reserve(n);
        samples_ = n;
    }

    /** @brief Append one `Item<T,Es...>`, host-side, amortised-doubling
     *         `reserve()` when `samples() == capacity()` (a fresh,
     *         zero-capacity Array grows to `kCapacityAlignment`). Writes
     *         through the host chunk only — mirrors `upload()`/`download()`'s
     *         explicit-transfer contract; `upload()` afterwards to reach the
     *         device copy in CUDA mode. */
    void push_back(const Item<T, Es...>& item)
    {
        if (samples_ == capacity_)
            reserve(capacity_ == 0 ? kCapacityAlignment : capacity_ * 2);
        T* base = reinterpret_cast<T*>(hostChunk_.data());
        for (std::size_t c = 0; c < itemSize_; ++c)
            base[c * capacity_ + samples_] = item.data()[c];
        ++samples_;
    }

    /** @brief `samples() <- 0`; `capacity()` is kept (no deallocation). */
    void clear() { samples_ = 0; }

    /** @brief A writable view over the host-resident copy — shape
     *         `samples()`, pitch `capacity()` (shape from size, strides
     *         from capacity). Sample extent is `0` for a `samples() == 0`
     *         Array — any per-sample expression-template assignment over
     *         it is a no-op by construction (nothing to iterate). */
    ViewT hostView() { return stridedView_(hostChunk_, capacity_, 0, samples_); }
    /** @brief A read-only view over the host-resident copy (see `hostView()`). */
    ConstViewT hostView() const { return stridedView_(hostChunk_, capacity_, 0, samples_); }

    /** @brief The compact `layout_right` accessor, available only when
     *         `capacity() == samples()` (a full, non-reserving Array).
     * @throws aether::Error  if `capacity() != samples()`. */
    PackedViewT hostPacked()
    {
        requirePacked_();
        return make_view<T, Es..., dyn>(hostChunk_, samples_);
    }
    /** @overload hostPacked() const */
    ConstPackedViewT hostPacked() const
    {
        requirePacked_();
        return make_view<const T, Es..., dyn>(hostChunk_, samples_);
    }

    /** @brief A writable view over the reserved tail `[samples(),
     *         capacity())` of the host-resident copy — the target a caller
     *         (or, on the device side, `AppendCounter`) writes new samples
     *         into before `commit()`ing them. */
    ViewT hostSpare() { return stridedView_(hostChunk_, capacity_, samples_, capacity_ - samples_); }
    /** @overload hostSpare() const */
    ConstViewT hostSpare() const { return stridedView_(hostChunk_, capacity_, samples_, capacity_ - samples_); }

#ifdef AETHER_HAS_CUDA
    /** @brief A writable view over the device-resident copy (see `hostView()`). */
    ViewT deviceView() { return stridedView_(deviceChunk_, capacity_, 0, samples_); }
    /** @brief A read-only view over the device-resident copy. */
    ConstViewT deviceView() const { return stridedView_(deviceChunk_, capacity_, 0, samples_); }

    /** @brief The device-resident counterpart of `hostPacked()`. */
    PackedViewT devicePacked()
    {
        requirePacked_();
        return make_view<T, Es..., dyn>(deviceChunk_, samples_);
    }
    /** @overload devicePacked() const */
    ConstPackedViewT devicePacked() const
    {
        requirePacked_();
        return make_view<const T, Es..., dyn>(deviceChunk_, samples_);
    }

    /** @brief The device-resident counterpart of `hostSpare()` — what a
     *         device append kernel actually writes into. */
    ViewT deviceSpare() { return stridedView_(deviceChunk_, capacity_, samples_, capacity_ - samples_); }
    /** @overload deviceSpare() const */
    ConstViewT deviceSpare() const { return stridedView_(deviceChunk_, capacity_, samples_, capacity_ - samples_); }
#else
    /** @brief `AETHER_CPP_MODE`: the same single chunk as `hostView()`. */
    ViewT deviceView() { return hostView(); }
    ConstViewT deviceView() const { return hostView(); }
    PackedViewT devicePacked() { return hostPacked(); }
    ConstPackedViewT devicePacked() const { return hostPacked(); }
    ViewT deviceSpare() { return hostSpare(); }
    ConstViewT deviceSpare() const { return hostSpare(); }
#endif

    /** @brief Adopt `count` newly-written spare samples — `samples() <-
     *         samples() + count` (host-side; bounds-checked against spare
     *         capacity, host-throw on overflow). The device-side atomic
     *         append counter that produces `count` is not here. */
    void commit(std::size_t count)
    {
        if (count > capacity_ - samples_) {
            err::fail("Array::commit", err::device_label(hostChunk_.device()), bytes_(count),
                "count=" + std::to_string(count) + " exceeds spare capacity ("
                    + std::to_string(capacity_ - samples_) + " = capacity()=" + std::to_string(capacity_)
                    + " - samples()=" + std::to_string(samples_) + ")");
        }
        samples_ += count;
    }

    /** @brief Blocking host -> device copy. No-op in `AETHER_CPP_MODE`. */
    void upload()
    {
#ifdef AETHER_HAS_CUDA
        copy(deviceChunk_, hostChunk_);
#endif
    }
    /** @brief Blocking device -> host copy. No-op in `AETHER_CPP_MODE`. */
    void download()
    {
#ifdef AETHER_HAS_CUDA
        copy(hostChunk_, deviceChunk_);
#endif
    }

#ifdef AETHER_HAS_CUDA
    /** @brief Asynchronous host -> device copy on `stream`. */
    void upload(Stream stream) { copyAsync(deviceChunk_, hostChunk_, stream); }
    /** @brief Asynchronous device -> host copy on `stream`. */
    void download(Stream stream) { copyAsync(hostChunk_, deviceChunk_, stream); }
#endif

private:
    static constexpr std::size_t itemSize_ = (Es * ... * std::size_t{ 1 });

    static std::size_t elementCount_(std::size_t n) { return itemSize_ * n; }
    static std::size_t bytes_(std::size_t n) { return elementCount_(n) * sizeof(T); }
    /** @brief `n` rounded up to `kCapacityAlignment`, via
     *         `aether::padded_block_samples`'s own `parts=1` closed form
     *         (`aether/residency/Partition.h`) — reused, not re-derived. */
    static std::size_t quantizedCapacity_(std::size_t n) { return padded_block_samples(n, 1, kCapacityAlignment); }

    /** @brief `layout_stride` view over `[start, start+count)` samples, with
     *         per-mode strides read off a `capacity`-sized `layout_right`
     *         mapping (`aether::detail::partitionStridesOf`) — the same
     *         technique `PartitionedArray::deviceView()` uses for its own
     *         padded-pitch blocks (see that file's docstring for why a
     *         compact `layout_right` view at `count` would silently
     *         mis-stride whenever `capacity != count`). */
    static ViewT stridedView_(Chunk& chunk, std::size_t capacity, std::size_t start, std::size_t count)
    {
        PackedViewT capacityView = make_view<T, Es..., dyn>(chunk, capacity);
        const auto strides       = detail::partitionStridesOf(capacityView);
        Extents ext(count);
        typename layout_stride::mapping<Extents> map(ext, strides);
        return ViewT(capacityView.data() + start, map, capacityView.device());
    }
    /** @overload */
    static ConstViewT stridedView_(const Chunk& chunk, std::size_t capacity, std::size_t start, std::size_t count)
    {
        ConstPackedViewT capacityView = make_view<const T, Es..., dyn>(chunk, capacity);
        const auto strides           = detail::partitionStridesOf(capacityView);
        Extents ext(count);
        typename layout_stride::mapping<Extents> map(ext, strides);
        return ConstViewT(capacityView.data() + start, map, capacityView.device());
    }

    /** @brief Host-throw guard shared by `hostPacked()`/`devicePacked()`
     *         (and their `const` overloads): `packed()` is only meaningful
     *         when there is no reserved headroom to mis-stride against. */
    void requirePacked_() const
    {
        if (capacity_ != samples_) {
            err::fail("Array::packed", "n/a", bytes_(samples_),
                "capacity()=" + std::to_string(capacity_) + " != samples()=" + std::to_string(samples_)
                    + ": packed() requires an exactly-full Array (no reserved headroom)");
        }
    }

    /** @brief Pitched realloc: allocate `newCapacity`-sized `Chunk`(s) and
     *         copy the existing `samples()` real elements component-by-
     *         component (old pitch `capacity_`, new pitch `newCapacity`) —
     *         mirrors `PartitionedArray::copyRows_`'s row-strip technique. */
    void reallocate_(std::size_t newCapacity)
    {
        const std::size_t newBytes = bytes_(newCapacity);
#ifdef AETHER_HAS_CUDA
        Chunk newHost   = Chunk::allocate(Device(kDLCUDAHost), newBytes);
        Chunk newDevice = Chunk::allocate(Device(kDLCUDA), newBytes);
        copyRowsPitch_(newHost, newCapacity, hostChunk_, capacity_, samples_);
        copyRowsPitch_(newDevice, newCapacity, deviceChunk_, capacity_, samples_);
        hostChunk_   = std::move(newHost);
        deviceChunk_ = std::move(newDevice);
#else
        Chunk newHost = Chunk::allocate(Device(kDLCPU), newBytes);
        copyRowsPitch_(newHost, newCapacity, hostChunk_, capacity_, samples_);
        hostChunk_ = std::move(newHost);
#endif
        capacity_ = newCapacity;
    }

    /** @brief `itemSize_` component row-strips of `count` elements each,
     *         `src` (pitch `srcPitch`) -> `dst` (pitch `dstPitch`) — reuses
     *         `aether::copy`'s legal-pair-checked machinery per row via
     *         `Chunk::borrow()`, exactly `PartitionedArray::copyRows_`'s
     *         technique (no new copy primitive). A no-op per row when
     *         `count == 0`. */
    static void copyRowsPitch_(
        Chunk& dst, std::size_t dstPitch, const Chunk& src, std::size_t srcPitch, std::size_t count)
    {
        for (std::size_t c = 0; c < itemSize_; ++c) {
            const std::size_t dOff = c * dstPitch;
            const std::size_t sOff = c * srcPitch;
            Chunk dstRow = Chunk::borrow(dst.data() + dOff * sizeof(T), count * sizeof(T), dst.device());
            Chunk srcRow = Chunk::borrow(src.data() + sOff * sizeof(T), count * sizeof(T), src.device());
            copy(dstRow, srcRow);
        }
    }

    std::size_t samples_;
    std::size_t capacity_;
    Chunk hostChunk_;
#ifdef AETHER_HAS_CUDA
    Chunk deviceChunk_;
#endif
};

} // namespace aether
