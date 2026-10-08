// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file DLPackLegacy.h
 * @brief `aether::interop::toDLPackLegacy`: export a `RuntimeView` as a
 *        heap-allocated pre-1.0 `DLManagedTensor*`. Import of the legacy
 *        struct already existed
 *        (`aether::interop::fromDLPack(DLManagedTensor*)` in `DLPack.h`);
 *        this file adds the export direction.
 *
 * Deliberately a THIN sibling of `aether::interop::toDLPack(const
 * RuntimeView&)` (`DLPack.h`) — same validation, same ownership model
 * (heap-allocated tensor + shape/strides bookkeeping, freed together by one
 * `deleter` that also deletes `self`, mirroring `DLPack.h`'s own
 * `detail::deleteExported`), same `RuntimeView`-typed and static-`View`-typed
 * overloads. The ONLY two differences are the target struct generation
 * (`DLManagedTensor`, no `version`/`flags` fields, `dl_tensor` FIRST — see
 * `aether/dtype/dlpack.h`'s own "Legacy DLManagedTensor" doc comment) and the
 * `bool` dtype policy below.
 *
 * ★ kDLBool POLICY: the pre-1.0 `DLDataTypeCode` enum predates `kDLBool` —
 * other independently-vendored pre-1.0 subsets (e.g. eagle's
 * `plugin/dlpack.h`) carry no `kDLBool` value at all (`kDLInt=0,
 * kDLUInt=1, kDLFloat=2, kDLBfloat=4`). A `bool` `RuntimeView` is
 * therefore exported here as `kDLUInt`/bits=8 instead of `kDLBool`/bits=8
 * — the closest code a legacy consumer can actually interpret —
 * documented POLICY, not a re-derivation of `dtype_of<bool>()` (which
 * correctly returns `kDLBool` for the VERSIONED export `toDLPack` still
 * uses).
 */

#include <cstdint>
#include <string>
#include <vector>

#include "aether/dtype/DType.h"
#include "aether/dtype/dlpack.h"
#include "aether/err/Error.h"
#include "aether/interop/DLPack.h"
#include "aether/view/MakeRuntimeView.h" // fromView() split out of view/RuntimeView.h
#include "aether/view/RuntimeView.h"
#include "aether/view/View.h"

namespace aether {
namespace interop {

namespace detail {

/** @brief The legacy `bool` policy — see the file docstring. Every other
 *         dtype passes through unchanged (identical to what `toDLPack`
 *         exports for the same `RuntimeView`). */
inline DLDataType legacyDType(const DType& dt)
{
    if (dt.code() == static_cast<std::uint8_t>(kDLBool) && dt.bits() == 8 && dt.lanes() == 1) {
        return DLDataType{ static_cast<std::uint8_t>(kDLUInt), 8, 1 };
    }
    return dt.raw;
}

/** @brief Heap-owned `shape`/`strides` storage for a LEGACY exported tensor
 *         — same rationale as `DLPack.h`'s own `ExportCtx` (a `DLTensor`'s
 *         `shape`/`strides` are raw pointers that must outlive this call's
 *         stack frame). A SEPARATE type from `DLPack.h`'s `ExportCtx` only
 *         because the two live in the same `detail` namespace and each
 *         needs its own `deleteExported` overload keyed to its own tensor
 *         generation — the field shapes are otherwise identical. */
struct LegacyExportCtx {
    std::vector<std::int64_t> shape;
    std::vector<std::int64_t> strides;
};

inline void deleteLegacyExported(DLManagedTensor* self)
{
    delete static_cast<LegacyExportCtx*>(self->manager_ctx);
    delete self;
}

} // namespace detail

/**
 * @brief Export a `RuntimeView` as a heap-allocated pre-1.0
 *        `DLManagedTensor*`. The caller (or whatever framework it hands the
 *        pointer to) owns the result and must eventually call its `deleter`
 *        — identical ownership contract to `aether::interop::toDLPack`.
 *
 * @throws aether::Error  same rejection set as `toDLPack` (unsupported rank/
 *         dtype/device for this build) — defensive, since every
 *         `RuntimeView` this library itself produces already satisfies all
 *         three.
 */
inline DLManagedTensor* toDLPackLegacy(const RuntimeView& view)
{
    if (view.rank > RuntimeRankCap) {
        err::fail("aether::interop::toDLPackLegacy", err::device_label(view.device), 0,
            "rank " + std::to_string(view.rank) + " exceeds RuntimeRankCap");
    }
    if (!detail::dtypeIsSupported(view.dtype.raw)) {
        err::fail("aether::interop::toDLPackLegacy", err::device_label(view.device), 0, "unsupported dtype");
    }
    if (!detail::deviceIsKnown(view.device.type())) {
        err::fail("aether::interop::toDLPackLegacy", err::device_label(view.device), 0,
            "unknown/unsupported device kind for this build");
    }

    auto* ctx = new detail::LegacyExportCtx();
    ctx->shape.resize(view.rank);
    ctx->strides.resize(view.rank);
    for (std::size_t i = 0; i < view.rank; ++i) {
        ctx->shape[i] = static_cast<std::int64_t>(view.extents[i]);
        ctx->strides[i] = static_cast<std::int64_t>(view.strides[i]);
    }

    auto* tensor = new DLManagedTensor{};
    tensor->manager_ctx = ctx;
    tensor->deleter = &detail::deleteLegacyExported;
    tensor->dl_tensor.data = view.data;
    tensor->dl_tensor.device = view.device.raw;
    tensor->dl_tensor.ndim = static_cast<std::int32_t>(view.rank);
    tensor->dl_tensor.dtype = detail::legacyDType(view.dtype);
    tensor->dl_tensor.shape = ctx->shape.data();
    tensor->dl_tensor.strides = ctx->strides.data();
    tensor->dl_tensor.byte_offset = 0;
    return tensor;
}

/** @brief `toDLPackLegacy(view)` overload for a static `View` — converts via
 *         `aether::fromView` first (host-only, always succeeds), mirroring
 *         `aether::interop::toDLPack`'s own static-`View` overload exactly. */
template<class T, class Extents, class Layout, bool Volatile>
DLManagedTensor* toDLPackLegacy(const View<T, Extents, Layout, Volatile>& v)
{
    return toDLPackLegacy(fromView(v));
}

} // namespace interop
} // namespace aether
