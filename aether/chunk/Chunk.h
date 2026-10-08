// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Chunk.h
 * @brief `Chunk`: a move-only byte allocation on ONE `Device` —
 *        OWNING by default, or a NON-OWNING wrap of foreign memory.
 *
 * Host-only management type: allocation and deallocation are always
 * host-side calls (`cudaMalloc`, `::%operator new`, …) even when the memory
 * itself lives on a CUDA device — there is no device-side `Chunk` API.
 * `Chunk` carries no element type or shape; those live in layers built on
 * top (dtype/layout/view).
 *
 * `Chunk::borrow()` wraps caller-owned memory (e.g. a `std::vector<T>`'s backing store, or any
 * foreign allocation) as a `Chunk` whose deleter is a no-op — `owns()`
 * reports `false`. Every other `Chunk` API (`data()`/`size()`/`device()`/
 * `alignment()`, move semantics) is unchanged and works identically on a
 * borrowed chunk; only destruction (and move-assignment's "free the old
 * chunk first" step) differs — a borrowed chunk's backing memory outlives
 * it, by construction, and this type never touches it at teardown.
 */

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <new>
#include <utility>

#include "aether/device/Device.h"
#include "aether/err/Error.h"

#ifdef AETHER_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace aether {

/**
 * @brief Move-only owning byte allocation on a single `Device`.
 */
class Chunk {
public:
    /** @brief Default-constructed empty chunk: nullptr, size 0, kDLCPU[0], alignment 0. */
    Chunk() = default;

    Chunk(const Chunk&) = delete;
    Chunk& operator=(const Chunk&) = delete;

    /** @brief Move constructor: leaves `other` an empty chunk. */
    Chunk(Chunk&& other) noexcept
        : data_{ std::exchange(other.data_, nullptr) }
        , size_{ std::exchange(other.size_, std::size_t{ 0 }) }
        , alignment_{ std::exchange(other.alignment_, std::size_t{ 0 }) }
        , device_{ std::exchange(other.device_, Device{}) }
        , owns_{ std::exchange(other.owns_, true) }
    {
    }

    /** @brief Move assignment: frees this chunk's own memory first (a no-op
     *         when `!owns()`), leaves `other` empty. */
    Chunk& operator=(Chunk&& other) noexcept
    {
        if (this == &other)
            return *this;
        free_();
        data_      = std::exchange(other.data_, nullptr);
        size_      = std::exchange(other.size_, std::size_t{ 0 });
        alignment_ = std::exchange(other.alignment_, std::size_t{ 0 });
        device_    = std::exchange(other.device_, Device{});
        owns_      = std::exchange(other.owns_, true);
        return *this;
    }

    /** @brief Frees the backing allocation. Never throws. */
    ~Chunk() { free_(); }

    /**
     * @brief Allocate `bytes` bytes on `device`, aligned to `alignment`
     *        (default 256 — the CUDA contract alignment).
     *
     * `bytes == 0` ALWAYS returns a valid empty chunk (`data() == nullptr`),
     * for any device kind whatsoever, WITHOUT validating that kind against
     * this build's backend — no allocator call is made at all. Validation
     * (and the real allocation) only happens once `bytes > 0`: requesting a
     * `kDLCUDA`/`kDLCUDAHost` chunk in a build with no CUDA backend
     * (`AETHER_CPP_MODE`, or CUDA simply not compiled in) throws, and so
     * does any device kind this library does not implement at all
     * (`kDLOpenCL`, …).
     *
     * @throws aether::Error  on an unsupported device kind (for this build,
     *         or altogether) with `bytes > 0`, a `kDLCUDA` alignment
     *         request finer than the 256-byte `cudaMalloc` contract, or an
     *         allocation failure reported by the underlying allocator.
     */
    static Chunk allocate(Device device, std::size_t bytes, std::size_t alignment = 256)
    {
        Chunk chunk;
        chunk.device_    = device;
        chunk.alignment_ = alignment;

        if (bytes == 0)
            return chunk;
        chunk.size_ = bytes;

        switch (device.type()) {
        case kDLCPU:
            try {
                chunk.data_ = static_cast<std::byte*>(
                    ::operator new(bytes, std::align_val_t{ alignment }));
            } catch (const std::bad_alloc& e) {
                err::fail("Chunk::allocate", err::device_label(device), bytes, e.what());
            }
            break;
#ifdef AETHER_HAS_CUDA
        case kDLCUDAHost: {
            void* ptr = nullptr;
            err::checkCuda(cudaMallocHost(&ptr, bytes), "Chunk::allocate",
                err::device_label(device), bytes);
            chunk.data_ = static_cast<std::byte*>(ptr);
            break;
        }
        case kDLCUDA: {
            if (alignment > 256) {
                err::fail("Chunk::allocate", err::device_label(device), bytes,
                    "kDLCUDA alignment > 256 is finer than the cudaMalloc contract");
            }
            // `cudaMalloc` allocates on
            // whatever device is CURRENT in this thread's CUDA context, NOT
            // on `device.id()` — before this fix, EVERY `Chunk::allocate`
            // silently landed on device 0 regardless of the requested id
            // (invisible on a single-GPU box, and every prior gate here ran
            // with the default `Device(kDLCUDA)` == id 0, so it never
            // tripped). `PartitionedArray`'s whole premise (one Chunk per
            // physical device, `aether/residency/PartitionedArray.h`) needs
            // this to be genuinely per-id. Scoped: select `device.id()`,
            // allocate, restore the caller's prior current device — this
            // function must not leak a global context change.
            int priorDevice = 0;
            err::checkCuda(cudaGetDevice(&priorDevice), "Chunk::allocate",
                err::device_label(device), bytes);
            err::checkCuda(cudaSetDevice(device.id()), "Chunk::allocate",
                err::device_label(device), bytes);
            void* ptr        = nullptr;
            cudaError_t allocSt = cudaMalloc(&ptr, bytes);
            err::checkCuda(cudaSetDevice(priorDevice), "Chunk::allocate",
                err::device_label(device), bytes); // restore FIRST, even on alloc failure below
            err::checkCuda(allocSt, "Chunk::allocate", err::device_label(device), bytes);
            // The CUDA contract guarantees cudaMalloc returns 256-byte
            // aligned memory; assert it rather than throw — a violation
            // here is an environment/toolchain bug, not a caller error.
            assert((reinterpret_cast<std::uintptr_t>(ptr) % 256) == 0
                && "cudaMalloc returned memory that is not 256-byte aligned");
            chunk.data_ = static_cast<std::byte*>(ptr);
            break;
        }
#else
        case kDLCUDAHost:
        case kDLCUDA:
            err::fail("Chunk::allocate", err::device_label(device), bytes,
                "this build has no CUDA backend (AETHER_CPP_MODE / no CUDA toolchain)");
            break;
#endif
        default:
            err::fail("Chunk::allocate", err::device_label(device), bytes, "unsupported device kind");
        }

        return chunk;
    }

    /**
     * @brief Wrap `bytes` bytes of CALLER-OWNED memory at `ptr` as a
     *        NON-OWNING `Chunk` — the deleter is a no-op, so this
     *        chunk's destruction (and a move-assignment's "free the old
     *        chunk first" step) never touches `ptr`. `ptr` must outlive
     *        every use of the returned chunk (and every view built over
     *        it) — the caller's responsibility, exactly as for the raw-
     *        pointer `make_view` overload this pairs with
     *        (`aether/view/View.h`).
     *
     * Unlike `allocate()`, `bytes == 0` does NOT bypass validation here —
     * there is no allocator call to skip in the first place — but a null
     * `ptr` is only rejected when `bytes > 0` (mirrors `make_view`'s own
     * "null pointer with a non-zero span" check), so `borrow(nullptr, 0,
     * dev)` is a valid empty, non-owning chunk.
     *
     * @throws aether::Error  if `ptr == nullptr` and `bytes > 0`.
     */
    static Chunk borrow(void* ptr, std::size_t bytes, Device device, std::size_t alignment = 256)
    {
        if (ptr == nullptr && bytes > 0) {
            err::fail("Chunk::borrow", err::device_label(device), bytes, "null pointer with a non-zero byte count");
        }
        Chunk chunk;
        chunk.data_      = static_cast<std::byte*>(ptr);
        chunk.size_      = bytes;
        chunk.alignment_ = alignment;
        chunk.device_    = device;
        chunk.owns_      = false;
        return chunk;
    }

    /** @brief Raw byte pointer (`nullptr` for an empty/zero-byte chunk). */
    std::byte* data() const { return data_; }
    /** @brief Allocation size in bytes. */
    std::size_t size() const { return size_; }
    /** @brief Device this chunk's memory lives on. */
    Device device() const { return device_; }
    /** @brief Alignment this chunk was allocated with. */
    std::size_t alignment() const { return alignment_; }
    /** @brief `true` for a chunk from `allocate()` (frees on destruction);
     *         `false` for a chunk from `borrow()` (never frees). */
    bool owns() const { return owns_; }

private:
    void free_() noexcept
    {
        if (!owns_ || data_ == nullptr)
            return;
        switch (device_.type()) {
        case kDLCPU:
            ::operator delete(data_, std::align_val_t{ alignment_ });
            break;
#ifdef AETHER_HAS_CUDA
        case kDLCUDAHost: {
            [[maybe_unused]] cudaError_t st = cudaFreeHost(data_);
            break;
        }
        case kDLCUDA: {
            [[maybe_unused]] cudaError_t st = cudaFree(data_);
            break;
        }
#endif
        default:
            break;
        }
        data_ = nullptr;
    }

    std::byte* data_       = nullptr;
    std::size_t size_      = 0;
    std::size_t alignment_ = 0;
    Device device_{};
    /** @brief `true` for every chunk from `allocate()`; `false` for `borrow()`. */
    bool owns_ = true;
};

} // namespace aether
