// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file RuntimeView.h
 * @brief `aether::RuntimeView`: rank/extents/strides/dtype/device held in
 *        a fixed-capacity runtime descriptor — the DLPack landing type
 *        (`aether/interop/DLPack.h`) and the operand type of the dual-mode
 *        runtime evaluator (`aether/eval/Runtime.h`).
 *
 * The runtime arm exists for validation/fallback/interop coverage, not as
 * a perf claim (the static `View`/`Item` stay the fast path).
 * `RuntimeView` is the descriptor that arm operates over:
 *   - `void* data` — untyped, so the same descriptor shape covers every
 *     supported dtype without a template parameter (a runtime type, by
 *     construction, cannot carry a compile-time one);
 *   - `DType dtype` / `Device device` — the existing wrappers
 *     (`aether/dtype/DType.h`, `aether/device/Device.h`), already
 *     `AETHER_DEVICEHOST()`-safe and trivially copyable;
 *   - `rank` (a runtime count, `std::size_t` — mirrors `extents::Rank`'s
 *     own "compile-time mode count, not an offset" convention, just
 *     resolved at runtime here) plus fixed-capacity `extents`/`strides`
 *     arrays, rank cap 6 (`RuntimeRankCap`) — a `Carray<offset_t, 6>` each
 *     (element-unit offsets are the narrow hot-chain type everywhere else
 *     in aether; the runtime evaluator's own address arithmetic reuses
 *     the same width rather than inventing a second convention).
 *   - No `byte_offset` member: DLPack's `byte_offset` is folded into
 *     `data` at the interop boundary (`aether/interop/DLPack.h` rejects
 *     any nonzero one rather than carrying it through), so by
 *     construction every `RuntimeView` this library ever produces already
 *     has an implicit `byte_offset == 0`.
 *
 * The whole struct is a plain aggregate of trivially-copyable fields — no
 * user-declared special members, no heap, no virtuals — so it is itself
 * trivially copyable (kernel-passable by value) even though a couple of
 * its member functions (`as<>`, `RuntimeView::as()`'s host-only
 * validation path) are host-only and throw: a class's special members
 * determine triviality, not its ordinary member function templates, and
 * nvcc never instantiates an uncalled member-function template just
 * because the enclosing type crosses into device code — the exact same
 * mix `aether/view/View.h` used to establish before its own device-safety
 * split (host-only `make_view` alongside `AETHER_DEVICEHOST()` accessors).
 *
 * `as<>`'s out-of-line definition, and the `detail::dynModeIndices`/
 * `buildExtentsFromRuntime{,Impl}`/`dtypeLabel` helpers only it calls,
 * live in `aether/view/MakeRuntimeView.h` (the `<string>`/`err::`-using
 * content). `as<>` stays declared here (a declaration needs neither) so
 * the struct's shape is unchanged; only a call to `as<>` needs
 * `MakeRuntimeView.h`'s definition in the TU. `fromView()` below never
 * used either and stays here unmodified — this header (declaration-only
 * for `as<>`) is standalone under NVRTC.
 */

#include <cstddef>

#include "aether/device/Device.h"
#include "aether/dtype/DType.h"
#include "aether/index/Offset.h"
#include "aether/layout/Extents.h"
#include "aether/layout/Layout.h"
#include "aether/layout/detail/Carray.h"
#include "aether/macros.h"
#include "aether/view/View.h"

namespace aether {

/** @brief Rank cap for `RuntimeView`: the fixed-capacity descriptor
 *         carries at most this many modes. */
inline constexpr std::size_t RuntimeRankCap = 6;

/**
 * @brief Fixed-capacity, trivially-copyable runtime tensor descriptor.
 *        See the file docstring for the field rationale.
 */
struct RuntimeView {
    /** @brief Untyped data pointer (never owning — mirrors `View`). */
    void* data = nullptr;
    /** @brief The element dtype (DLPack-native, `aether/dtype/DType.h`). */
    DType dtype{};
    /** @brief The device this descriptor's memory lives on. */
    Device device{};
    /** @brief Number of modes (`<= RuntimeRankCap`). A runtime count, not
     *         an offset — stays `std::size_t` (mirrors `extents::Rank`). */
    std::size_t rank = 0;
    /** @brief Per-mode extents, element units, `offset_t`. Only indices
     *         `[0, rank)` are meaningful. */
    detail::Carray<offset_t, RuntimeRankCap> extents{};
    /** @brief Per-mode strides, element units (not bytes — matches DLPack's
     *         own convention) — may be non-contiguous. Only indices
     *         `[0, rank)` are meaningful. */
    detail::Carray<offset_t, RuntimeRankCap> strides{};

    /**
     * @brief Product of `extents[0..rank)` — the logical element count.
     *        Device-legal: callable from the runtime evaluator kernel.
     *        By construction every `RuntimeView` this library produces
     *        (`fromView` below, `aether/interop/DLPack.h`'s `fromDLPack`)
     *        has already been guarded, so this narrow product does not
     *        itself re-derive that guard — see those call sites for the
     *        wide check.
     */
    AETHER_DEVICEHOST() constexpr offset_t size() const
    {
        // Empty product == 1 (a rank-0/scalar descriptor has exactly one
        // element) — mirrors `View::size()`'s own convention exactly.
        offset_t total = 1;
        for (std::size_t i = 0; i < rank; ++i)
            total *= extents[i];
        return total;
    }

    /**
     * @brief Checked promotion onto a static fast-path view: throws
     *        `aether::Error` unless dtype, rank, every static extent of
     *        `StaticViewT`, the stride pattern (row-major/`layout_right`
     *        only — the one static layout aether's static `View` actually
     *        uses), and the device (a known, backend-enabled kind) all
     *        match; otherwise returns the static view, zero-copy (same
     *        `data` pointer, no allocation, no data movement).
     *
     * Host-only (throws); never called from device code, so it carries no
     * `AETHER_DEVICEHOST()`.
     *
     * Guarded rather than split into a separate header: a split is
     * impossible here without changing `RuntimeView`'s public shape from
     * a member `rt.as<T>` to a free function, breaking every existing
     * call site. An entirely unannotated function — even a bodiless
     * declaration, never mind a call — is rejected outright by NVRTC's
     * JIT mode ("host functions are not allowed in JIT mode").
     *
     * Guards on `__CUDACC_RTC__` rather than `__CUDA_ARCH__`:
     * `__CUDA_ARCH__` is also defined during nvcc's own device-compilation
     * pass of an ordinary `.cu` file (not just NVRTC) — and that pass
     * still fully parses the whole translation unit, including host-only
     * call sites elsewhere in the same `.cu` file (guarding on
     * `__CUDA_ARCH__` broke existing host-side `.as<>` calls in nvcc's
     * device pass — a real build regression, not an NVRTC-only one).
     * `__CUDACC_RTC__` is the macro that actually distinguishes "compiled
     * by NVRTC" from "compiled by nvcc" (host or device pass alike) — NVRTC
     * defines it, nvcc never does — so guarding on it removes `as<>`
     * only from NVRTC's view, leaving every nvcc-compiled `.cu`/`.cpp` TU
     * (host and device pass) exactly as it always was.
     */
#if !defined(__CUDACC_RTC__)
    template<class StaticViewT>
    StaticViewT as() const;
#endif
};

// `fromView()` lives in `aether/view/MakeRuntimeView.h` — every existing
// call site is host-side, and being an unannotated free function (like
// `make_view`), it is rejected outright by NVRTC's JIT mode even
// bodiless/uncalled, exactly like `TableView.h`'s `makeTableView()`.
// Unlike `as<>` above, a plain free function can be split cleanly (no
// member declared here at all, matching the `make_view`/`reshaped`
// pattern) rather than guarded — this header is standalone under NVRTC.
// Include `aether/view/MakeRuntimeView.h` (or `aether/aether.h`, which
// does) wherever `fromView()` is called.

} // namespace aether
