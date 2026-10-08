// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Device.h
 * @brief `Device`: a thin, `AETHER_DEVICEHOST()`-safe wrapper around
 *        DLPack's `DLDevice`, plus the `backend_enabled<DLDeviceType>`
 *        compile-time backend gate.
 */

#include <cstdint>

#include "aether/dtype/dlpack.h"
#include "aether/macros.h"

namespace aether {

/**
 * @brief Thin wrapper around `DLDevice` (device_type/device_id) — constexpr
 *        constructible and safe to use in `AETHER_DEVICEHOST()` code.
 */
struct Device {
    /** @brief The wrapped DLPack device descriptor. */
    DLDevice raw{ kDLCPU, 0 };

    // No AETHER_DEVICEHOST() here: a `= default` special member on its
    // first declaration is automatically host+device eligible under nvcc,
    // which warns (#20012-D) that an explicit annotation is redundant.
    constexpr Device() = default;

    /** @brief Construct from a device kind and (optional) device index. */
    AETHER_DEVICEHOST() constexpr Device(DLDeviceType type, std::int32_t id = 0)
        : raw{ type, id }
    {
    }

    /** @brief Construct by wrapping an existing `DLDevice` verbatim. */
    AETHER_DEVICEHOST() constexpr explicit Device(DLDevice dl)
        : raw(dl)
    {
    }

    /** @brief The DLPack device kind. */
    AETHER_DEVICEHOST() constexpr DLDeviceType type() const { return raw.device_type; }
    /** @brief The device index (0 for CPU / pinned / managed memory). */
    AETHER_DEVICEHOST() constexpr std::int32_t id() const { return raw.device_id; }

    /** @brief `true` for `kDLCPU`. */
    AETHER_DEVICEHOST() constexpr bool is_cpu() const { return raw.device_type == kDLCPU; }
    /** @brief `true` for `kDLCUDA` (device memory). */
    AETHER_DEVICEHOST() constexpr bool is_cuda() const { return raw.device_type == kDLCUDA; }
    /** @brief `true` for `kDLCUDAHost` (pinned host memory). */
    AETHER_DEVICEHOST() constexpr bool is_cuda_host() const { return raw.device_type == kDLCUDAHost; }
};

AETHER_DEVICEHOST() constexpr bool operator==(const Device& a, const Device& b)
{
    return a.raw.device_type == b.raw.device_type && a.raw.device_id == b.raw.device_id;
}

AETHER_DEVICEHOST() constexpr bool operator!=(const Device& a, const Device& b)
{
    return !(a == b);
}

/**
 * @brief Compile-time backend gate: is `Type` usable in THIS build?
 *
 * `kDLCPU` is always enabled. `kDLCUDA`/`kDLCUDAHost` are enabled only when
 * `AETHER_HAS_CUDA` is defined (root `CMakeLists.txt` defines it exactly
 * when the build is configured in CUDA mode, i.e. NOT `AETHER_CPP_MODE`).
 * Every other `DLDeviceType` is disabled.
 */
template<DLDeviceType Type>
inline constexpr bool backend_enabled = (Type == kDLCPU);

#ifdef AETHER_HAS_CUDA
template<>
inline constexpr bool backend_enabled<kDLCUDA> = true;
template<>
inline constexpr bool backend_enabled<kDLCUDAHost> = true;
#endif

} // namespace aether
