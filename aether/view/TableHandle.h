// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file TableHandle.h
 * @brief `aether::TableHandle<T, Carrier>`: a bounded, read-only
 *        table-read handle, `__ldg`-routed (`Carrier = Plain`, the
 *        default) or `tex1Dfetch`-routed (`Carrier = Texture`).
 *
 * A flat, offset-indexed read with no shape to check it against. The
 * plain carrier (still the default and still the whole of
 * `TableHandle<T>`'s original surface) never touches a
 * `cudaTextureObject_t`; the texture carrier lands alongside it as a
 * sibling partial specialization, not a rewrite: `TableHandle<T>` /
 * `TableHandle<T, Plain>` are unchanged in every respect (same type, same
 * ABI, same tests).
 *
 * "Compile-time-boundless": unlike `aether::View`, a `TableHandle` carries
 * no extents/rank/mapping at all — `operator[](offset_t)` is a raw flat
 * index with no shape to check it against. The plain carrier is built
 * from a `ScalarView<T>` (which does carry a shape, at construction time
 * only) so the caller supplies the underlying storage the ordinary way;
 * the shape is then discarded — `TableHandle` remembers only where the
 * data starts.
 *
 * The plain carrier reuses the ReadOnly view policy machinery
 * (`aether/view/View.h`) verbatim rather than re-deriving the `__ldg`
 * branch: internally this is just a `View<T, extents<dyn>, layout_right,
 * false, true>` (a read-only `ScalarView<T>`), and `operator[]` forwards
 * straight to that view's own `operator()`.
 *
 * ## The texture carrier
 *
 * `TableHandle<T, Texture>` is built from a plain pointer plus a
 * `texture_handle_t` (typically `TextureBinding<T>::%handle()`,
 * `aether/chunk/TextureBinding.h` — the lifetime object that owns the
 * `cudaTextureObject_t` this handle merely views, mirroring the plain
 * carrier's own View-wrapping convention: `TableHandle` never owns
 * anything). Deliberately not constructed from a `TextureBinding<T>`
 * directly — that type is host-only (it drags in `Chunk.h`/`err/Error.h`),
 * and this header must stay reachable from `aether/device.h` (the
 * NVRTC-parseable umbrella); a plain `texture_handle_t` keeps the texture
 * carrier's declaration exactly as device-safe as the plain one.
 *
 * Reads route through `dtype::Fetch<T>` (`aether/dtype/Fetch.h`):
 * `tex1Dfetch<Fetch<T>::%type>` under `__CUDA_ARCH__`, decoded per
 * `needsConversion` (only `double` needs it, via `__hiloint2double`); a
 * plain pointer load otherwise (the host pass of a real CUDA build, or an
 * `AETHER_CPP_MODE` build, where the texture carrier still exists but
 * never touches a `cudaTextureObject_t` at all).
 *
 * `operator[]` returns `const T`, never a bare `T` — a fetched (computed)
 * value must not be assignable at the type surface (matches the plain
 * carrier's own read-only contract; see `tests/test_TextureBinding.cu`'s
 * `TextureWriteInvariant.Anchor`). There is no non-const overload on the
 * texture carrier at all: texture memory is read-only, so no write
 * spelling exists to guard against silently discarding —
 * `tests/compile_fail/check_texture_write_rejected.sh` pins that
 * `h[i] = v` does not compile for this carrier.
 */

#include "aether/dtype/Fetch.h"
#include "aether/index/Offset.h"
#include "aether/layout/Extents.h"
#include "aether/layout/Layout.h"
#include "aether/macros.h"
#include "aether/view/View.h"

namespace aether {

/** @brief Carrier tag selecting `TableHandle`'s plain, `__ldg`-routed read
 *  path — the default. */
struct Plain { };

/** @brief Carrier tag selecting `TableHandle`'s texture-object read path —
 *  see the file docstring. */
struct Texture { };

template<class T, class Carrier = Plain>
class TableHandle;

/**
 * @brief Bounded, read-only, non-texture table-read handle over `T` data.
 *        `operator[](offset_t)` is `__ldg`-routed on device (via the
 *        wrapped read-only `View`'s own policy) and a plain load on host.
 */
template<class T>
class TableHandle<T, Plain> {
public:
    using element_type = T;
    /** @brief The wrapped read-only batched-scalar view type. */
    using view_type = View<T, extents<dyn>, layout_right, false, true>;

    // No AETHER_DEVICEHOST() here — see aether::View's default ctor note
    // (a `= default` special member is automatically host+device eligible
    // under nvcc; an explicit annotation on it warns #20012-D as redundant).
    constexpr TableHandle() = default;

    /**
     * @brief Wrap a batched scalar view for bounded, read-only,
     *        `__ldg`-routed table reads. The source view need not already
     *        be read-only-qualified — `as_readonly()` is applied here,
     *        exactly once (mirrors that method's own single-entry-point
     *        convention: calling it twice is a `static_assert` error, so a
     *        `TableHandle` is never built from an already read-only view by
     *        this ctor; pass the plain `View<T, extents<dyn>>` you already
     *        have).
     */
    AETHER_DEVICEHOST() constexpr explicit TableHandle(const View<T, extents<dyn>>& v)
        : view_(v.as_readonly())
    {
    }

    /**
     * @brief Flat, offset-indexed, compile-time-boundless read: no rank or
     *        extent is checked against `idx`. Device-compiled reads route
     *        through `__ldg` (the wrapped view's ReadOnly `operator()`
     *        path); host is a plain load.
     */
    AETHER_DEVICEHOST() constexpr T operator[](offset_t idx) const { return view_(idx); }

    /** @brief The underlying (read-only-qualified) pointer. */
    AETHER_DEVICEHOST() constexpr T* data() const { return view_.data(); }

private:
    view_type view_{};
};

/**
 * @brief Bounded, read-only, texture-carrier table-read handle over `T`
 *        data. `operator[](offset_t)` is `tex1Dfetch`-routed on device
 *        (via `dtype::Fetch<T>`) and a plain pointer load otherwise (host
 *        pass of a real CUDA build, or an `AETHER_CPP_MODE` build — see
 *        the file docstring).
 *
 * Does not own the `cudaTextureObject_t` it wraps — `TextureBinding<T>`
 * (`aether/chunk/TextureBinding.h`, host-only) owns that lifetime; this
 * handle is a cheap, trivially-copyable, kernel-crossable view of it,
 * exactly like the plain carrier is a view over a `Chunk`-backed `View`.
 */
template<class T>
class TableHandle<T, Texture> {
public:
    using element_type = T;

    /** @brief Default-constructed: unbound (`valid() == false`), `data()`
     *  and `handle()` both null/zero. */
    constexpr TableHandle() = default;

    /**
     * @brief Wrap `ptr` (the same device memory `tex` is bound over) and
     *        `tex` (typically `TextureBinding<T>::%handle()`) for bounded,
     *        texture-routed table reads at `offset`.
     *
     * `ptr` is required even though the device-compiled read path never
     * dereferences it directly: it is the plain-load fallback for every
     * pass that is not `__CUDA_ARCH__` (the host pass of a real CUDA build,
     * where a texture object cannot be fetched from at all; every
     * `AETHER_CPP_MODE` read, which never has a texture object to fetch
     * from in the first place) — see the file docstring.
     */
    AETHER_DEVICEHOST() constexpr TableHandle(const T* ptr, texture_handle_t tex, offset_t offset = 0)
        : ptr_(ptr)
        , tex_(tex)
        , offset_(offset)
    {
    }

    /**
     * @brief Flat, offset-indexed, compile-time-boundless read (see the
     *        plain carrier's own `operator[]` docstring for the shared
     *        "no rank/extent check" contract). Returns `const T`, never a
     *        bare `T` — see the file docstring. There is deliberately no
     *        non-const overload anywhere on this specialization: texture
     *        memory is read-only, and unlike the plain carrier (whose
     *        non-const spelling has a real target, the wrapped `View`)
     *        there is no write path here for a guard to even stand in
     *        front of.
     */
    AETHER_DEVICEHOST() constexpr const T operator[](offset_t idx) const
    {
#if defined(__CUDA_ARCH__) && !defined(AETHER_CPP_MODE)
        using FetchT = typename dtype::Fetch<T>::type;
        const FetchT r = tex1Dfetch<FetchT>(tex_, static_cast<int>(idx + offset_));
        if constexpr (dtype::Fetch<T>::needsConversion) {
            // Only `double` (FetchT == int2) takes this branch today — see
            // dtype/Fetch.h. `x` = low word, `y` = high word.
            return __hiloint2double(r.y, r.x);
        } else {
            return r;
        }
#else
        return ptr_[idx + offset_];
#endif
    }

    /** @brief The wrapped texture-object handle (`0` when unbound, or
     *  always in `AETHER_CPP_MODE`). */
    AETHER_DEVICEHOST() constexpr texture_handle_t handle() const { return tex_; }

    /** @brief `true` iff a live texture object is wrapped (always `false`
     *  in `AETHER_CPP_MODE`). */
    AETHER_DEVICEHOST() constexpr bool valid() const { return tex_ != 0; }

    /** @brief The underlying plain pointer (the host-pass / CPP_MODE
     *  fallback's own read target). */
    AETHER_DEVICEHOST() constexpr const T* data() const { return ptr_; }

private:
    const T* ptr_ = nullptr;
    texture_handle_t tex_ = 0;
    offset_t offset_ = 0;
};

} // namespace aether
