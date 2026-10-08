// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file AppendCounter.h
 * @brief `aether::accum::AppendCounter`: a device-side atomic counter that
 *        hands out slots in an `Array`'s reserved (`spare()`) capacity, for
 *        the host to commit as the new size.
 */

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

#include "aether/array/Array.h"
#include "aether/chunk/Chunk.h"
#include "aether/chunk/Copy.h"
#include "aether/device/Device.h"
#include "aether/err/DeviceFlag.h"
#include "aether/err/Error.h"
#include "aether/index/Offset.h"
#include "aether/macros.h"

namespace aether {
namespace accum {

/** @brief The "no slot" sentinel `tryAppend()` returns on overflow — the
 *         same bit pattern as `aether::offset_max`, an `offset_t` value no
 *         real capacity reaches. */
inline constexpr offset_t npos = offset_max;

/**
 * @brief Non-owning, `AETHER_DEVICEHOST()`-safe view of one `AppendCounter`
 *        — what a kernel takes by value.
 */
class AppendCounterView {
public:
    AppendCounterView() = default;

    AETHER_DEVICEHOST() constexpr AppendCounterView(std::uint32_t* counter, std::size_t capacity, DeviceFlagView overflow)
        : counter_(counter)
        , capacity_(capacity)
        , overflow_(overflow)
    {
    }

    /**
     * @brief Claim the next slot. Device: `atomicAdd(counter, 1u)`. Host
     *        (incl. `AETHER_CPP_MODE`, where this same code path runs as
     *        plain host code): a relaxed `std::atomic<uint32_t>::%fetch_add`.
     *
     * @return A slot `< capacity` (write it), or `aether::accum::npos` when
     *         the draw would overflow — capacity is UNCHANGED in that case
     *         (the counter itself may keep climbing past `capacity` across
     *         many overflowing threads; that raw value is what
     *         `AppendCounter::rawCount()` reports back, clamped by
     *         `commit()`), and the overflow flag is set. NEVER returns a
     *         slot `>= capacity`.
     */
    AETHER_DEVICEHOST() offset_t tryAppend() const
    {
#if defined(__CUDA_ARCH__)
        const std::uint32_t slot = atomicAdd(counter_, 1u);
#else
        auto* a                  = reinterpret_cast<std::atomic<std::uint32_t>*>(counter_);
        const std::uint32_t slot = a->fetch_add(1u, std::memory_order_relaxed);
#endif
        if (static_cast<std::size_t>(slot) >= capacity_) {
            overflow_.set();
            return npos;
        }
        return static_cast<offset_t>(slot);
    }

private:
    std::uint32_t* counter_ = nullptr;
    std::size_t capacity_   = 0;
    DeviceFlagView overflow_{};
};

/**
 * @brief Owning device-side append counter over a declared `capacity`. One
 *        `uint32_t` atomic slot counter (a host-pinned `Chunk`, plus a
 *        device `Chunk` when built with CUDA) and one `DeviceFlag` for
 *        overflow.
 */
class AppendCounter {
public:
    explicit AppendCounter(std::size_t capacity)
        : capacity_(capacity)
#ifdef AETHER_HAS_CUDA
        , counterHost_(Chunk::allocate(Device(kDLCUDAHost), sizeof(std::uint32_t)))
        , counterDevice_(Chunk::allocate(Device(kDLCUDA), sizeof(std::uint32_t)))
#else
        , counterHost_(Chunk::allocate(Device(kDLCPU), sizeof(std::uint32_t)))
#endif
    {
        reset();
    }

    /** @brief Declared capacity — the bound `tryAppend()` enforces. */
    [[nodiscard]] std::size_t capacity() const { return capacity_; }

    /** @brief The view a kernel (or host code, in `AETHER_CPP_MODE`) takes by value. */
    [[nodiscard]] AppendCounterView deviceView() const
    {
#ifdef AETHER_HAS_CUDA
        return AppendCounterView(
            reinterpret_cast<std::uint32_t*>(counterDevice_.data()), capacity_, overflow_.deviceView());
#else
        return AppendCounterView(
            reinterpret_cast<std::uint32_t*>(counterHost_.data()), capacity_, overflow_.deviceView());
#endif
    }

    /**
     * @brief Non-destructive host-side read of the RAW counter (may exceed
     *        `capacity()` — every overflowing thread still incremented it
     *        before discovering the overflow). Downloads first (a no-op in
     *        `AETHER_CPP_MODE`).
     */
    [[nodiscard]] std::size_t rawCount() const
    {
        download_();
        std::uint32_t v = 0;
        std::memcpy(&v, counterHost_.data(), sizeof(v));
        return static_cast<std::size_t>(v);
    }

    /** @brief Non-destructive peek at the overflow flag (does not clear it). */
    [[nodiscard]] bool overflowed() const { return overflow_.isSet(); }

    /**
     * @brief Throw if the overflow flag is set, and clear it. Call this
     *        after `commit()`, which never throws, so a rejected overflow
     *        does not discard the samples that did fit.
     * @throws aether::Error  if the flag was set.
     */
    void checkOverflow(const std::string& context = "AppendCounter::checkOverflow") { overflow_.checkAndClear(context); }

    /** @brief Reset the raw counter to `0` for a fresh step. Does NOT touch the overflow flag. */
    void reset()
    {
        const std::uint32_t z = 0;
        std::memcpy(counterHost_.data(), &z, sizeof(z));
#ifdef AETHER_HAS_CUDA
        copy(counterDevice_, counterHost_);
#endif
    }

    /**
     * @brief Read the raw count back, clamp it to `capacity()`, adopt that
     *        many newly-written spare samples into `arr` via
     *        `arr.commit(...)`, then reset this counter for the next step.
     *        Never throws; pair with `overflowed()`/`checkOverflow()` to
     *        surface an overflow.
     *
     * @return The clamped count actually committed.
     */
    template<class T, std::size_t... Es>
    std::size_t commit(Array<T, Es...>& arr)
    {
        const std::size_t raw     = rawCount();
        const std::size_t clamped = std::min(raw, capacity_);
        arr.commit(clamped);
        reset();
        return clamped;
    }

private:
    void download_() const
    {
#ifdef AETHER_HAS_CUDA
        copy(const_cast<Chunk&>(counterHost_), counterDevice_);
#endif
    }

    std::size_t capacity_;
    Chunk counterHost_;
#ifdef AETHER_HAS_CUDA
    Chunk counterDevice_;
#endif
    DeviceFlag overflow_;
};

} // namespace accum
} // namespace aether
