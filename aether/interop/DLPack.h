// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file DLPack.h
 * @brief `aether::interop::fromDLPack`/`toDLPack`: DLPack <-> `RuntimeView`
 *        conversion — C-ABI level. HOST-ONLY (heap, exceptions — no
 *        `AETHER_DEVICEHOST()` anywhere in this file; nothing here is ever
 *        called from device code).
 *
 * `fromDLPack` accepts BOTH DLPack struct generations (`DLManagedTensor`,
 * the legacy pre-1.0 shape still in wide circulation, and
 * `DLManagedTensorVersioned`, "the current standard DLPack exchange data
 * structure" per `dtype/dlpack.h`'s own vendored comment) and returns a
 * `RuntimeView` (zero-copy — `RuntimeView::data` aliases the producer's own
 * memory) PLUS a `DLPackOwner` RAII token that calls the producer's deleter
 * exactly once, on destruction or explicit `release()`.
 *
 * STREAM-SEMANTICS CONTRACT (DLPack 1.0): DLPack itself defines NO implicit
 * ordering around the deleter — it is a plain host-side call. A consumer
 * that issued asynchronous device work against the imported memory (e.g.
 * launched a kernel reading through the `RuntimeView`) is responsible for
 * ordering that work relative to the deleter call itself (synchronize the
 * stream, or otherwise guarantee completion) BEFORE letting the
 * `DLPackOwner` run its deleter — exactly the same caller responsibility
 * `aether::Array::upload()/download()`'s blocking-by-default convention
 * already carries elsewhere in this library. `fromDLPack`/`DLPackOwner`
 * add no stream synchronization of their own.
 *
 * REJECTIONS (all a HOST throw of `aether::Error`): unsupported
 * dtype code (only the `dtype_of<T>()`-supported set — `double`/`float`/
 * `bool`/`int64`/`int32`/`uint64`/`uint32`/`uint8`, lanes==1 — passes; `bool`
 * maps to `kDLBool`), rank > `RuntimeRankCap` (6),
 * `byte_offset != 0`, an unknown/unsupported-for-this-build device type. A
 * REJECTED import still calls the producer's deleter EXACTLY ONCE (the
 * import path takes ownership the moment it is handed the pointer; refusing
 * it must not leak) before rethrowing.
 */

#include <cstddef>
#include <cstdint>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "aether/device/Device.h"
#include "aether/dtype/DType.h"
#include "aether/dtype/dlpack.h"
#include "aether/err/Error.h"
#include "aether/index/Offset.h"
#include "aether/view/MakeRuntimeView.h" // fromView() split out of view/RuntimeView.h
#include "aether/view/RuntimeView.h"
#include "aether/view/View.h"

namespace aether {
namespace interop {

/**
 * @brief RAII ownership token for an imported DLPack tensor: calls the
 *        producer's `deleter` exactly once. Move-only (mirrors `Chunk`'s
 *        ownership discipline) — the `RuntimeView` a `DLPackImport` carries
 *        remains valid only as long as its `DLPackOwner` is alive (or until
 *        `release()` is called).
 */
class DLPackOwner {
public:
    DLPackOwner() = default;

    static DLPackOwner fromLegacy(DLManagedTensor* t)
    {
        DLPackOwner o;
        o.legacy_ = t;
        return o;
    }
    static DLPackOwner fromVersioned(DLManagedTensorVersioned* t)
    {
        DLPackOwner o;
        o.versioned_ = t;
        return o;
    }

    DLPackOwner(const DLPackOwner&) = delete;
    DLPackOwner& operator=(const DLPackOwner&) = delete;

    DLPackOwner(DLPackOwner&& other) noexcept
        : legacy_(std::exchange(other.legacy_, nullptr))
        , versioned_(std::exchange(other.versioned_, nullptr))
    {
    }
    DLPackOwner& operator=(DLPackOwner&& other) noexcept
    {
        if (this == &other)
            return *this;
        release();
        legacy_ = std::exchange(other.legacy_, nullptr);
        versioned_ = std::exchange(other.versioned_, nullptr);
        return *this;
    }

    ~DLPackOwner() { release(); }

    /** @brief Invoke the producer's deleter now — idempotent (a no-op once
     *         already released, or on a default-constructed token). See the
     *         file docstring's stream-semantics contract: the CALLER must
     *         order any outstanding async device work before this runs. */
    void release()
    {
        if (legacy_ != nullptr) {
            DLManagedTensor* t = std::exchange(legacy_, nullptr);
            if (t->deleter != nullptr)
                t->deleter(t);
        }
        if (versioned_ != nullptr) {
            DLManagedTensorVersioned* t = std::exchange(versioned_, nullptr);
            if (t->deleter != nullptr)
                t->deleter(t);
        }
    }

    /** @brief `true` if this token still owns a live tensor (has not been
     *         released, moved-from, or default-constructed). */
    bool owns() const { return legacy_ != nullptr || versioned_ != nullptr; }

private:
    DLManagedTensor* legacy_ = nullptr;
    DLManagedTensorVersioned* versioned_ = nullptr;
};

/** @brief Result of a DLPack import: the zero-copy `RuntimeView` plus
 *         the `DLPackOwner` whose lifetime must dominate the view's use. */
struct DLPackImport {
    RuntimeView view;
    DLPackOwner owner;
};

namespace detail {

inline bool dtypeIsSupported(DLDataType dt)
{
    if (dt.lanes != 1)
        return false;
    switch (dt.code) {
    case kDLFloat:
        return dt.bits == 64 || dt.bits == 32;
    case kDLBool:
        return dt.bits == 8;
    case kDLInt:
        return dt.bits == 64 || dt.bits == 32;
    case kDLUInt:
        return dt.bits == 64 || dt.bits == 32 || dt.bits == 8;
    default:
        return false;
    }
}

/**
 * @brief `true` for a device type an import/export of THIS build accepts.
 *
 * A CUDA build accepts host memory and the two CUDA kinds it can operate on
 * (`kDLCUDA`, `kDLCUDAHost`). A pure C++ build (`AETHER_CPP_MODE`, no
 * `AETHER_HAS_CUDA`) accepts every device type DLPack defines as METADATA: the
 * record is imported and exported unchanged, device memory is never
 * dereferenced, and only an operation that needs host access refuses it
 * (`hostAccessible`). That lets a pure C++ consumer hold a CUDA buffer that a
 * separately loaded device backend operates on. A code DLPack does not define
 * is refused in both builds.
 */
inline bool deviceIsKnown(DLDeviceType type)
{
    switch (type) {
    case kDLCPU:
        return true;
#ifdef AETHER_HAS_CUDA
    case kDLCUDA:
    case kDLCUDAHost:
        return true;
#else
    case kDLCUDA:
    case kDLCUDAHost:
    case kDLOpenCL:
    case kDLVulkan:
    case kDLMetal:
    case kDLVPI:
    case kDLROCM:
    case kDLROCMHost:
    case kDLExtDev:
    case kDLCUDAManaged:
    case kDLOneAPI:
    case kDLWebGPU:
    case kDLHexagon:
    case kDLMAIA:
        return true;
#endif
    default:
        return false;
    }
}

} // namespace detail

/**
 * @brief `true` when the host may dereference memory of device type `type`:
 *        `kDLCPU`, and the page-locked host kinds (`kDLCUDAHost`,
 *        `kDLROCMHost`) whose address is a host address. Device, managed
 *        and every other kind are records only on the host.
 */
inline bool hostAccessible(DLDeviceType type)
{
    return type == kDLCPU || type == kDLCUDAHost || type == kDLROCMHost;
}

namespace detail {

/**
 * @brief Convert one `DLTensor` (the shared payload of both DLPack struct
 *        generations) into a `RuntimeView` — the validating core BOTH
 *        `fromDLPack` overloads share. Throws `aether::Error` on any
 *        rejection; never touches ownership (the caller wraps this in
 *        a try/catch that calls the producer's deleter on the throw path —
 *        see the two `fromDLPack` overloads below).
 */
inline RuntimeView runtimeViewFromDLTensor(const DLTensor& t)
{
    if (t.ndim < 0 || static_cast<std::size_t>(t.ndim) > RuntimeRankCap) {
        err::fail("aether::interop::fromDLPack", "DLPack", 0,
            "rank " + std::to_string(t.ndim) + " exceeds RuntimeView's cap of "
                + std::to_string(RuntimeRankCap));
    }
    if (t.byte_offset != 0) {
        err::fail("aether::interop::fromDLPack", "DLPack", 0,
            "byte_offset=" + std::to_string(t.byte_offset) + " != 0 is not supported (v1)");
    }
    if (!dtypeIsSupported(t.dtype)) {
        err::fail("aether::interop::fromDLPack", "DLPack", 0,
            "unsupported dtype (code=" + std::to_string(static_cast<unsigned>(t.dtype.code))
                + ", bits=" + std::to_string(static_cast<unsigned>(t.dtype.bits))
                + ", lanes=" + std::to_string(static_cast<unsigned>(t.dtype.lanes)) + ")");
    }
    if (!deviceIsKnown(t.device.device_type)) {
        err::fail("aether::interop::fromDLPack", "DLPack", 0,
            "unknown/unsupported device kind (type="
                + std::to_string(static_cast<int>(t.device.device_type)) + ") for this build");
    }

    const std::size_t rank = static_cast<std::size_t>(t.ndim);

    // Every individual extent/stride must itself be representable in
    // offset_t — checked BEFORE narrowing, mirroring make_view's own
    // "guard first, narrow second" discipline.
    for (std::size_t i = 0; i < rank; ++i) {
        if (t.shape[i] < 0 || !offset_fits(static_cast<std::size_t>(t.shape[i]))) {
            err::fail("aether::interop::fromDLPack", "DLPack", 0,
                "shape[" + std::to_string(i) + "]=" + std::to_string(t.shape[i])
                    + " is negative or exceeds offset_t's addressable maximum");
        }
        if (t.strides != nullptr) {
            if (t.strides[i] < 0 || !offset_fits(static_cast<std::size_t>(t.strides[i]))) {
                err::fail("aether::interop::fromDLPack", "DLPack", 0,
                    "strides[" + std::to_string(i) + "]=" + std::to_string(t.strides[i])
                        + " is negative or exceeds offset_t's addressable maximum");
            }
        }
    }

    RuntimeView rt;
    rt.data = t.data;
    rt.dtype = DType(t.dtype);
    rt.device = Device(t.device);
    rt.rank = rank;
    for (std::size_t i = 0; i < rank; ++i)
        rt.extents[i] = static_cast<offset_t>(t.shape[i]);

    if (t.strides != nullptr) {
        for (std::size_t i = 0; i < rank; ++i)
            rt.strides[i] = static_cast<offset_t>(t.strides[i]);
    } else {
        // NULL strides means "tensor is compact and row-majored" (DLPack's
        // own contract, dtype/dlpack.h's DLTensor::strides doc comment).
        offset_t acc = 1;
        for (std::size_t i = rank; i-- > 0;) {
            rt.strides[i] = acc;
            acc *= rt.extents[i];
        }
    }

    // Final WIDE required-span check (mdspan's own strided formula,
    // mirroring aether::layout_stride::mapping::required_span_size()):
    // catches an overall address range that overflows even when every
    // individual extent/stride above fit on its own.
    std::size_t span = 1;
    bool anyZero = false;
    for (std::size_t i = 0; i < rank; ++i) {
        if (rt.extents[i] == 0) {
            anyZero = true;
            break;
        }
    }
    if (!anyZero) {
        for (std::size_t i = 0; i < rank; ++i)
            span += (static_cast<std::size_t>(rt.extents[i]) - 1) * static_cast<std::size_t>(rt.strides[i]);
    } else {
        span = 0;
    }
    if (!offset_fits(span)) {
        err::fail("aether::interop::fromDLPack", "DLPack", 0,
            "required span " + std::to_string(span) + " elements exceeds offset_t's addressable maximum");
    }

    return rt;
}

} // namespace detail

/** @brief Import a LEGACY `DLManagedTensor*`. See the file docstring
 *         for ownership/stream-semantics and the rejection list. */
inline DLPackImport fromDLPack(DLManagedTensor* managed)
{
    if (managed == nullptr)
        err::fail("aether::interop::fromDLPack", "DLPack", 0, "null DLManagedTensor*");
    try {
        RuntimeView rt = detail::runtimeViewFromDLTensor(managed->dl_tensor);
        return DLPackImport{ rt, DLPackOwner::fromLegacy(managed) };
    } catch (...) {
        if (managed->deleter != nullptr)
            managed->deleter(managed);
        throw;
    }
}

/** @brief Import a `DLManagedTensorVersioned*` ("the current standard
 *         DLPack exchange data structure"). Same contract as the legacy
 *         overload above. */
inline DLPackImport fromDLPack(DLManagedTensorVersioned* managed)
{
    if (managed == nullptr)
        err::fail("aether::interop::fromDLPack", "DLPack", 0, "null DLManagedTensorVersioned*");
    try {
        RuntimeView rt = detail::runtimeViewFromDLTensor(managed->dl_tensor);
        return DLPackImport{ rt, DLPackOwner::fromVersioned(managed) };
    } catch (...) {
        if (managed->deleter != nullptr)
            managed->deleter(managed);
        throw;
    }
}

namespace detail {

/** @brief Heap-owned `shape`/`strides` storage for an EXPORTED tensor —
 *         `DLTensor::shape`/`strides` are raw `int64_t*`, so something must
 *         own that storage for as long as the exported tensor lives; the
 *         exported `DLManagedTensorVersioned::manager_ctx` points at one of
 *         these, and its `deleter` frees both this and the tensor itself. */
struct ExportCtx {
    std::vector<std::int64_t> shape;
    std::vector<std::int64_t> strides;
};

inline void deleteExported(DLManagedTensorVersioned* self)
{
    delete static_cast<ExportCtx*>(self->manager_ctx);
    delete self;
}

} // namespace detail

/**
 * @brief Export a `RuntimeView` as a heap-allocated `DLManagedTensorVersioned*`
 *        the caller (or whatever framework it hands the pointer to)
 *        owns the result and must eventually call its `deleter`. `strides`
 *        are exported VERBATIM from `view.strides` (element units — SoA and
 *        any other stride pattern this library can represent round-trips
 *        exactly, never forced back to a compact/contiguous shape).
 *
 * @throws aether::Error  if `view`'s dtype or device is not one this
 *         library recognizes (defensive — every `RuntimeView` this library
 *         itself produces already satisfies both, via `fromView`/`fromDLPack`).
 */
inline DLManagedTensorVersioned* toDLPack(const RuntimeView& view)
{
    if (view.rank > RuntimeRankCap) {
        err::fail("aether::interop::toDLPack", err::device_label(view.device), 0,
            "rank " + std::to_string(view.rank) + " exceeds RuntimeRankCap");
    }
    if (!detail::dtypeIsSupported(view.dtype.raw)) {
        err::fail("aether::interop::toDLPack", err::device_label(view.device), 0, "unsupported dtype");
    }
    if (!detail::deviceIsKnown(view.device.type())) {
        err::fail("aether::interop::toDLPack", err::device_label(view.device), 0,
            "unknown/unsupported device kind for this build");
    }

    auto* ctx = new detail::ExportCtx();
    ctx->shape.resize(view.rank);
    ctx->strides.resize(view.rank);
    for (std::size_t i = 0; i < view.rank; ++i) {
        ctx->shape[i] = static_cast<std::int64_t>(view.extents[i]);
        ctx->strides[i] = static_cast<std::int64_t>(view.strides[i]);
    }

    auto* tensor = new DLManagedTensorVersioned{};
    tensor->version = DLPackVersion{ static_cast<std::uint32_t>(DLPACK_MAJOR_VERSION),
        static_cast<std::uint32_t>(DLPACK_MINOR_VERSION) };
    tensor->manager_ctx = ctx;
    tensor->deleter = &detail::deleteExported;
    tensor->flags = 0;
    tensor->dl_tensor.data = view.data;
    tensor->dl_tensor.device = view.device.raw;
    tensor->dl_tensor.ndim = static_cast<std::int32_t>(view.rank);
    tensor->dl_tensor.dtype = view.dtype.raw;
    tensor->dl_tensor.shape = ctx->shape.data();
    tensor->dl_tensor.strides = ctx->strides.data();
    tensor->dl_tensor.byte_offset = 0;
    return tensor;
}

/** @brief `toDLPack(view)` overload for a static `View` — converts via
 *         `aether::fromView` first (host-only, always succeeds). */
template<class T, class Extents, class Layout, bool Volatile>
DLManagedTensorVersioned* toDLPack(const View<T, Extents, Layout, Volatile>& v)
{
    return toDLPack(fromView(v));
}

} // namespace interop
} // namespace aether
