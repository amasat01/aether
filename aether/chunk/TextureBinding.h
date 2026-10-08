// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file TextureBinding.h
 * @brief `aether::TextureBinding<T>`: a move-only RAII
 *        `cudaTextureObject_t` lifetime object bound to a `Chunk`'s
 *        device memory.
 *
 * Host-only management type (mirrors `Chunk`'s own host-only-management
 * docstring convention). `TextureBinding` OWNS the `cudaTextureObject_t`
 * descriptor only — never the underlying memory (that stays the `Chunk`'s
 * job, exactly as `View` never owns the `Chunk` it is built over).
 *
 * ## The rule this class enforces
 *
 * MOVE-ASSIGNMENT MUST RELEASE THE DESTINATION'S TEXTURE OBJECT BEFORE
 * ADOPTING THE SOURCE'S. Skipping that release does not merely leak a
 * handle: a live `cudaTextureObject_t` is a descriptor that keeps pointing
 * at the device address range it was bound to, so once that range is freed
 * and the allocator hands the block to the next request, the orphaned
 * handle silently serves the NEW occupant's bytes, with no error of any
 * kind. Measured end-to-end by `tests/test_TextureBinding.cu` (recycled
 * handle values observed with `rc=cudaSuccess`).
 *
 * ## Liveness, and why `cudaDestroyTextureObject` cannot be the probe
 *
 * `cudaDestroyTextureObject` called a second time on an already-destroyed
 * (or never-created) handle reports `cudaSuccess` on this driver — a test
 * built on it would report "leaked" for every handle, fixed or not. The
 * liveness instrument the tests use instead is
 * `cudaGetTextureObjectResourceDesc`: `cudaSuccess` on a live object,
 * `cudaErrorInvalidValue` on a destroyed or never-created one.
 *
 * ## `AETHER_CPP_MODE`
 *
 * The class still exists and is still move-only, but `bind()` is a no-op
 * (`handle() == 0`, `valid() == false` always), a build-mode compile-time
 * degrade rather than a runtime branch, so no dead CUDA-API call site
 * exists in a CPP_MODE binary at all.
 */

#include <utility>

#include "aether/chunk/Chunk.h"
#include "aether/dtype/Fetch.h"
#include "aether/err/Error.h"
#include "aether/macros.h"

#ifdef AETHER_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace aether {

/**
 * @brief Move-only RAII `cudaTextureObject_t` bound over `T`-typed elements
 *        of a `Chunk`'s device memory.
 */
template<class T>
class TextureBinding {
public:
    /** @brief Default-constructed, unbound (`handle() == 0`, `valid() ==
     *  false`). */
    TextureBinding() = default;

    TextureBinding(const TextureBinding&) = delete;
    TextureBinding& operator=(const TextureBinding&) = delete;

    /** @brief Move constructor: leaves `other` unbound. */
    TextureBinding(TextureBinding&& other) noexcept
        : tex_{ std::exchange(other.tex_, texture_handle_t{ 0 }) }
    {
    }

    /**
     * @brief Move assignment: destroys THIS binding's own texture object
     *        FIRST (a no-op when `!valid()`), THEN adopts `other`'s handle,
     *        leaving `other` unbound. See the file docstring for why the
     *        release must come before the adoption.
     */
    TextureBinding& operator=(TextureBinding&& other) noexcept
    {
        if (this == &other)
            return *this;
        destroy_();
        tex_ = std::exchange(other.tex_, texture_handle_t{ 0 });
        return *this;
    }

    ~TextureBinding() { destroy_(); }

    /**
     * @brief Bind a texture object over `chunk`'s device memory, `count`
     *        elements of `T` (`count * sizeof(T)` bytes, must not exceed
     *        `chunk.size()`).
     *
     * `count == 0` always returns a valid, UNBOUND binding (`valid() ==
     * false`) with no CUDA call made at all — mirrors `Chunk::allocate`'s
     * own "size 0 is always legal, no allocator call" convention.
     *
     * @throws aether::Error  in a CUDA-enabled build: `chunk.device().type()
     *         != kDLCUDA` (a texture object can only be bound to device
     *         memory), `count * sizeof(T)` exceeds `chunk.size()`, or a CUDA
     *         binding failure. Never throws in `AETHER_CPP_MODE` — `count`
     *         and `chunk` are ignored there and the result is always
     *         unbound (see file docstring).
     */
    static TextureBinding bind(const Chunk& chunk, std::size_t count)
    {
        TextureBinding tb;
#ifdef AETHER_HAS_CUDA
        if (count == 0)
            return tb;
        if (chunk.device().type() != kDLCUDA) {
            err::fail("TextureBinding::bind", err::device_label(chunk.device()), chunk.size(),
                "a texture object can only be bound to kDLCUDA device memory");
        }
        const std::size_t bytes = count * sizeof(T);
        if (bytes > chunk.size()) {
            err::fail("TextureBinding::bind", err::device_label(chunk.device()), chunk.size(),
                "requested texture span (" + std::to_string(bytes)
                    + " bytes) exceeds the chunk's own allocation");
        }

        cudaResourceDesc resDesc{};
        resDesc.resType = cudaResourceTypeLinear;
        resDesc.res.linear.devPtr = chunk.data();
        resDesc.res.linear.desc = cudaCreateChannelDesc<typename dtype::Fetch<T>::type>();
        resDesc.res.linear.sizeInBytes = bytes;

        cudaTextureDesc texDesc{};
        texDesc.readMode = cudaReadModeElementType;

        cudaTextureObject_t tex = 0;
        err::checkCuda(cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr),
            "TextureBinding::bind", err::device_label(chunk.device()), bytes);
        tb.tex_ = tex;
#else
        (void)chunk;
        (void)count;
#endif
        return tb;
    }

    /** @brief The bound `cudaTextureObject_t`-compatible handle (`0` when
     *  unbound, or always in `AETHER_CPP_MODE`). Pass to
     *  `TableHandle<T, Texture>`'s constructor to build a device-readable
     *  handle over this binding. */
    texture_handle_t handle() const { return tex_; }

    /** @brief `true` iff a live texture object is bound (always `false` in
     *  `AETHER_CPP_MODE`). */
    bool valid() const { return tex_ != 0; }

private:
    void destroy_() noexcept
    {
#ifdef AETHER_HAS_CUDA
        if (tex_ != 0) {
            [[maybe_unused]] cudaError_t st
                = cudaDestroyTextureObject(static_cast<cudaTextureObject_t>(tex_));
            tex_ = 0;
        }
#endif
    }

    texture_handle_t tex_ = 0;
};

} // namespace aether
