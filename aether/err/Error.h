// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Error.h
 * @brief `aether::Error` and message-building helpers.
 *
 * Every host-side failure in aether throws `aether::Error` with a message
 * naming the operation, the device(s) involved, and the size.
 *
 * Host-only header: uses `std::string`/`std::runtime_error`; never include
 * this from a device-compiled translation unit.
 */

#include <stdexcept>
#include <string>

#include "aether/device/Device.h"
#include "aether/macros.h"

#ifdef AETHER_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace aether {

/** @brief aether's exception type. */
class Error : public std::runtime_error {
public:
    explicit Error(const std::string& message)
        : std::runtime_error(message)
    {
    }
};

namespace err {

/** @brief Human-readable name for a `DLDeviceType`, for error messages. */
inline std::string device_type_name(DLDeviceType type)
{
    switch (type) {
    case kDLCPU: return "CPU";
    case kDLCUDA: return "CUDA";
    case kDLCUDAHost: return "CUDAHost";
    default: return "Device(type=" + std::to_string(static_cast<int>(type)) + ")";
    }
}

/** @brief Human-readable `"Type[id]"` label for a `Device`, for error messages. */
inline std::string device_label(const Device& device)
{
    return device_type_name(device.type()) + "[" + std::to_string(device.id()) + "]";
}

/**
 * @brief Build and throw an `aether::Error` naming `operation`, `where`
 *        (typically `device_label(...)`, or `"A -> B"` for a copy path),
 *        and `bytes`. `detail` adds optional extra context (e.g. an
 *        underlying CUDA status string).
 */
[[noreturn]] inline void fail(const std::string& operation, const std::string& where,
    std::size_t bytes, const std::string& detail = "")
{
    std::string msg
        = operation + " failed on " + where + ", size=" + std::to_string(bytes) + " bytes";
    if (!detail.empty())
        msg += " (" + detail + ")";
    throw Error(msg);
}

#ifdef AETHER_HAS_CUDA
/**
 * @brief Check a CUDA runtime API call's status; throw `aether::Error`
 *        naming `operation`/`where`/`bytes` on failure.
 */
inline void checkCuda(cudaError_t status, const std::string& operation,
    const std::string& where, std::size_t bytes)
{
    if (status != cudaSuccess)
        fail(operation, where, bytes, cudaGetErrorString(status));
}
#endif

} // namespace err
} // namespace aether
