// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file DeviceFlag.h
 * @brief `aether::DeviceFlag`: the device-side half of aether's error
 *        contract.
 *
 * The rule: host code fails loud with exceptions; device code cannot throw,
 * so it sets a flag instead. `aether/err/Error.h` is explicitly the
 * HOST-ONLY half (`std::runtime_error`, unusable from device code);
 * `DeviceFlag` is the device-visible counterpart: a sticky `uint32_t` a
 * kernel sets with `atomicOr` (host arm: a relaxed `std::atomic` store —
 * the same monotone-OR-is-a-store argument `aether::accum::atomic::orBool_`
 * already makes, reused here rather than re-derived) and the host reads
 * back, clears, and turns into a thrown `aether::Error` via
 * `checkAndClear()`.
 *
 * "Sticky": once set, a flag stays set until the host explicitly clears it
 * (`clear()` or `checkAndClear()`) — never auto-clears on read (`isSet()`),
 * so a caller may poll a live flag from multiple places between clears.
 *
 * First consumer: `aether::accum::AppendCounter` sets this on a capacity
 * overflow; every future device error path is invited to reuse this same
 * type rather than growing a second one.
 */

#include <atomic>
#include <cstdint>
#include <cstring>
#include <string>

#include "aether/chunk/Chunk.h"
#include "aether/chunk/Copy.h"
#include "aether/device/Device.h"
#include "aether/err/Error.h"
#include "aether/macros.h"

namespace aether {

/**
 * @brief Non-owning, `AETHER_DEVICEHOST()`-safe view onto one `DeviceFlag`'s
 *        sticky word — what a kernel actually takes by value (mirrors
 *        `View`'s own "no separate handle type" convention, `aether/view/
 *        View.h`).
 */
class DeviceFlagView {
public:
    DeviceFlagView() = default;

    AETHER_DEVICEHOST() constexpr explicit DeviceFlagView(std::uint32_t* ptr)
        : ptr_(ptr)
    {
    }

    /**
     * @brief Set the flag. Device: `atomicOr(ptr, 1u)`. Host (including a
     *        `AETHER_CPP_MODE` build, where this same code path runs as
     *        plain host code): a relaxed `std::atomic<uint32_t>` store of
     *        `1u` — monotone-OR-with-true degenerates to a store, exactly
     *        the argument `orBool_` already makes (`aether/accum/atomic.h`).
     */
    AETHER_DEVICEHOST() void set() const
    {
#if defined(__CUDA_ARCH__)
        atomicOr(ptr_, 1u);
#else
        auto* a = reinterpret_cast<std::atomic<std::uint32_t>*>(ptr_);
        a->store(1u, std::memory_order_relaxed);
#endif
    }

private:
    std::uint32_t* ptr_ = nullptr;
};

/**
 * @brief Owning sticky device-side flag. One `uint32_t` slot, held
 *        in a pinned host `Chunk` plus a device `Chunk` in CUDA mode (the
 *        same two-chunk shape `Array` uses), a single `kDLCPU` chunk in
 *        `AETHER_CPP_MODE`.
 */
class DeviceFlag {
public:
    DeviceFlag()
#ifdef AETHER_HAS_CUDA
        : hostChunk_(Chunk::allocate(Device(kDLCUDAHost), sizeof(std::uint32_t)))
        , deviceChunk_(Chunk::allocate(Device(kDLCUDA), sizeof(std::uint32_t)))
#else
        : hostChunk_(Chunk::allocate(Device(kDLCPU), sizeof(std::uint32_t)))
#endif
    {
        clear();
    }

    /** @brief The view a kernel (or host code, in `AETHER_CPP_MODE`) takes by value. */
    DeviceFlagView deviceView() const
    {
#ifdef AETHER_HAS_CUDA
        return DeviceFlagView(reinterpret_cast<std::uint32_t*>(deviceChunk_.data()));
#else
        return DeviceFlagView(reinterpret_cast<std::uint32_t*>(hostChunk_.data()));
#endif
    }

    /**
     * @brief Non-destructive host-side read: is the flag currently set?
     *        Downloads the device word first (a no-op in `AETHER_CPP_MODE`,
     *        where the "device" write already landed in this same chunk).
     */
    bool isSet() const
    {
        download_();
        std::uint32_t v = 0;
        std::memcpy(&v, hostChunk_.data(), sizeof(v));
        return v != 0;
    }

    /** @brief Reset the flag to clear, on both host and device copies. */
    void clear()
    {
        const std::uint32_t z = 0;
        std::memcpy(hostChunk_.data(), &z, sizeof(z));
#ifdef AETHER_HAS_CUDA
        copy(deviceChunk_, hostChunk_);
#endif
    }

    /**
     * @brief The host-throw half of the device-flag pattern. Downloads
     *        and checks the flag; if set, clears it (so a caller that
     *        catches the exception may keep using the flag for the next
     *        step) and throws `aether::Error` naming `context`.
     * @throws aether::Error  if the flag was set.
     */
    void checkAndClear(const std::string& context)
    {
        if (isSet()) {
            clear();
            err::fail(context, "DeviceFlag", sizeof(std::uint32_t), "device flag was set");
        }
    }

private:
    void download_() const
    {
#ifdef AETHER_HAS_CUDA
        copy(const_cast<Chunk&>(hostChunk_), deviceChunk_);
#endif
    }

    Chunk hostChunk_;
#ifdef AETHER_HAS_CUDA
    Chunk deviceChunk_;
#endif
};

} // namespace aether
