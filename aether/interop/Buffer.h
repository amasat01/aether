// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Buffer.h
 * @brief The generic buffer layer: `aether::interop::BufferView` — a
 *        zero-copy `RuntimeView` plus the metadata every consumer of a
 *        foreign array needs (access, owner, producer, keep-alive) —
 *        imported from DLPack (both generations) or from raw
 *        `__cuda_array_interface__`/`__array_interface__` fields, validated
 *        against `Requirements`, and re-exported as DLPack with the access
 *        flag carried. HOST-ONLY: no CUDA runtime call anywhere in this
 *        file (stream ordering is the CUDA-aware layer's job, see eagle's
 *        `plugin/interop.h`).
 *
 * BUFFER MODEL. Two independent axes: ownership (the allocating library's
 * name, or `"external"` when borrowed) and access (`Access`). The layer
 * REPORTS access exactly as the producer declares it:
 *
 *   | producer form                           | `Access`      |
 *   |-----------------------------------------|---------------|
 *   | versioned DLPack, read-only flag set    | `ReadOnly`    |
 *   | versioned DLPack, flag clear            | `ReadWrite`   |
 *   | legacy (pre-1.0) DLPack                 | `Unknown`     |
 *   | array interface, `data[1]` true         | `ReadOnly`    |
 *   | array interface, `data[1]` false        | `ReadWrite`   |
 *
 * It never upgrades `Unknown` to writable by itself; a caller may assert
 * writability (`assumeWritable`) and that assertion is recorded on the view.
 * Policy belongs to the consumer: `Requirements::writable` is the helper a
 * consumer that writes through the buffer sets; a read-only consumer leaves
 * it off and accepts any access.
 *
 * LIFETIME. Every view carries a type-erased keep-alive
 * (`std::shared_ptr<void>`). For a DLPack import it owns the producer's
 * managed tensor and calls its deleter exactly once, when the last copy of
 * the keep-alive dies. A DLPack export stores a copy of the same keep-alive
 * in the exported tensor's context, so a chain import -> export -> import ->
 * ... keeps the FIRST producer alive until the last view anywhere in the
 * chain is gone.
 *
 * STREAMS. DLPack's deleter is a plain host call with no implicit ordering:
 * a consumer that queued device work against the memory orders that work
 * before releasing the last keep-alive.
 */

#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "aether/device/Device.h"
#include "aether/dtype/DType.h"
#include "aether/dtype/dlpack.h"
#include "aether/err/Error.h"
#include "aether/index/Offset.h"
#include "aether/interop/DLPack.h"
#include "aether/interop/DLPackLegacy.h"
#include "aether/view/RuntimeView.h"

namespace aether {
namespace interop {

/** @brief How a buffer may be accessed, as its producer declared it. */
enum class Access {
    /** @brief The producer allows writes through the buffer. */
    ReadWrite,
    /** @brief The producer declared the buffer read-only. */
    ReadOnly,
    /** @brief The producer's protocol carries no access flag (legacy DLPack). */
    Unknown,
};

/** @brief The shared spelling of an `Access` value: `"read-write"`,
 *         `"read-only"` or `"unknown"`. */
inline const char* accessName(Access a)
{
    switch (a) {
    case Access::ReadWrite:
        return "read-write";
    case Access::ReadOnly:
        return "read-only";
    case Access::Unknown:
        return "unknown";
    }
    return "unknown";
}

/** @brief The owner name of borrowed memory. */
inline constexpr const char* kExternalOwner = "external";

/**
 * @brief A zero-copy buffer plus its access, ownership and lifetime record.
 *
 * `view.data` already includes the producer's `byte_offset`. Copies share
 * the keep-alive, so any copy keeps the producer alive.
 */
struct BufferView {
    /** @brief The zero-copy descriptor (element-unit strides). */
    RuntimeView view{};
    /** @brief Access as the producer declared it. */
    Access access = Access::Unknown;
    /** @brief `true` once a caller asserted writability via `assumeWritable`. */
    bool assumedWritable = false;
    /** @brief The allocating library's name, or `kExternalOwner`. */
    std::string owner = kExternalOwner;
    /** @brief The producer's type name (e.g. `"numpy.ndarray"`); empty when owned. */
    std::string producer;
    /** @brief Keeps the producer's memory alive while any copy of this view lives. */
    std::shared_ptr<void> keepAlive;

    /** @brief `true` when the memory belongs to a library of this family. */
    bool owned() const { return owner != kExternalOwner; }

    /** @brief `true` when writes are allowed: declared read-write, or unknown
     *         with an explicit writability assertion. */
    bool writable() const
    {
        return access == Access::ReadWrite || (access == Access::Unknown && assumedWritable);
    }
};

/** @brief Record the caller's assertion that `buffer` may be written. The
 *         reported `access` is unchanged; `writable()` honours the assertion
 *         only for `Access::Unknown` (a declared read-only buffer stays
 *         read-only). */
inline void assumeWritable(BufferView& buffer) { buffer.assumedWritable = true; }

/** @brief A view of memory a library of this family allocated (`owner` is
 *         that library's name, `producer` stays empty). */
inline BufferView ownedBuffer(
    std::string owner, const RuntimeView& view, Access access, std::shared_ptr<void> keepAlive = {})
{
    BufferView b;
    b.view = view;
    b.access = access;
    b.owner = std::move(owner);
    b.keepAlive = std::move(keepAlive);
    return b;
}

namespace detail {

/** @brief The validating core of both DLPack imports, with `byte_offset`
 *         folded into the data pointer. */
inline RuntimeView bufferViewFromDLTensor(const DLTensor& t)
{
    DLTensor folded = t;
    folded.data = static_cast<void*>(static_cast<char*>(t.data) + t.byte_offset);
    folded.byte_offset = 0;
    return runtimeViewFromDLTensor(folded);
}

} // namespace detail

/**
 * @brief Import a versioned DLPack tensor. Access follows its read-only
 *        flag. Ownership of `managed` passes to the returned view's
 *        keep-alive; on refusal the deleter runs once before the throw.
 *
 * @throws aether::Error  null tensor, a newer major DLPack version, or any
 *         rejection of `fromDLPack` other than a non-zero `byte_offset`.
 */
inline BufferView importDLPack(DLManagedTensorVersioned* managed, std::string producer = "dlpack")
{
    if (managed == nullptr)
        throw Error("aether::interop::importDLPack: null DLManagedTensorVersioned*");
    std::shared_ptr<void> keep(static_cast<void*>(managed), [](void* p) {
        auto* t = static_cast<DLManagedTensorVersioned*>(p);
        if (t->deleter != nullptr)
            t->deleter(t);
    });
    if (managed->version.major > static_cast<std::uint32_t>(DLPACK_MAJOR_VERSION)) {
        throw Error("aether::interop::importDLPack: DLPack major version "
            + std::to_string(managed->version.major) + " is newer than the supported "
            + std::to_string(DLPACK_MAJOR_VERSION));
    }
    BufferView b;
    b.view = detail::bufferViewFromDLTensor(managed->dl_tensor);
    b.access = (managed->flags & DLPACK_FLAG_BITMASK_READ_ONLY) != 0 ? Access::ReadOnly : Access::ReadWrite;
    b.producer = std::move(producer);
    b.keepAlive = std::move(keep);
    return b;
}

/** @brief Import a legacy (pre-1.0) DLPack tensor. Its protocol carries no
 *         access flag, so `access` is `Access::Unknown`. Same ownership and
 *         refusal contract as the versioned overload. */
inline BufferView importDLPack(DLManagedTensor* managed, std::string producer = "dlpack")
{
    if (managed == nullptr)
        throw Error("aether::interop::importDLPack: null DLManagedTensor*");
    std::shared_ptr<void> keep(static_cast<void*>(managed), [](void* p) {
        auto* t = static_cast<DLManagedTensor*>(p);
        if (t->deleter != nullptr)
            t->deleter(t);
    });
    BufferView b;
    b.view = detail::bufferViewFromDLTensor(managed->dl_tensor);
    b.access = Access::Unknown;
    b.producer = std::move(producer);
    b.keepAlive = std::move(keep);
    return b;
}

/**
 * @brief The raw fields of a `__cuda_array_interface__` (device) or
 *        `__array_interface__` (host) dictionary, so a language binding can
 *        build a `BufferView` without DLPack.
 */
struct ArrayInterface {
    /** @brief `data[0]`: the buffer address. */
    void* data = nullptr;
    /** @brief `data[1]`: the producer's read-only flag. */
    bool readOnly = false;
    /** @brief `shape`, in elements. */
    std::vector<std::int64_t> shape;
    /** @brief `strides` in BYTES, or empty for C-contiguous (the protocol's `None`). */
    std::vector<std::int64_t> strides;
    /** @brief `typestr`, e.g. `"<f8"`, `"|u1"`, `"|b1"`. */
    std::string typestr;
    /** @brief Where the memory lives (`kDLCUDA` + ordinal for the device protocol, `kDLCPU` for the host one). */
    DLDevice device{ kDLCPU, 0 };
    /** @brief The producer's type name. */
    std::string producer;
    /** @brief Whatever keeps the producer alive (e.g. a reference to the object). */
    std::shared_ptr<void> keepAlive;
};

namespace detail {

/** @brief Parse an array-interface `typestr` into a DLPack dtype. */
inline DLDataType dtypeFromTypestr(const std::string& ts)
{
    if (ts.size() < 3)
        throw Error("aether::interop::fromArrayInterface: typestr '" + ts + "' is malformed");
    const char order = ts[0];
    const bool little = order == '<' || order == '|' || (order == '=' && std::endian::native == std::endian::little);
    if (!little)
        throw Error("aether::interop::fromArrayInterface: typestr '" + ts + "' is not little-endian");
    const char kind = ts[1];
    int bytes = 0;
    try {
        bytes = std::stoi(ts.substr(2));
    } catch (...) {
        throw Error("aether::interop::fromArrayInterface: typestr '" + ts + "' is malformed");
    }
    const auto bits = static_cast<std::uint8_t>(bytes * 8);
    switch (kind) {
    case 'f':
        return DLDataType{ static_cast<std::uint8_t>(kDLFloat), bits, 1 };
    case 'i':
        return DLDataType{ static_cast<std::uint8_t>(kDLInt), bits, 1 };
    case 'u':
        return DLDataType{ static_cast<std::uint8_t>(kDLUInt), bits, 1 };
    case 'b':
        return DLDataType{ static_cast<std::uint8_t>(kDLBool), bits, 1 };
    default:
        throw Error("aether::interop::fromArrayInterface: typestr '" + ts + "' has an unsupported kind");
    }
}

} // namespace detail

/**
 * @brief Build a `BufferView` from raw array-interface fields. Byte strides
 *        are converted to element strides (they must be multiples of the
 *        item size); access follows `readOnly`. The resulting record is
 *        validated exactly like a DLPack import.
 *
 * @throws aether::Error  malformed or unsupported `typestr`, byte strides
 *         that are not a multiple of the item size, or any `fromDLPack`
 *         rejection.
 */
inline BufferView fromArrayInterface(const ArrayInterface& ai)
{
    const DLDataType dt = detail::dtypeFromTypestr(ai.typestr);
    const std::int64_t item = static_cast<std::int64_t>(dt.bits) / 8;
    if (!ai.strides.empty() && ai.strides.size() != ai.shape.size()) {
        throw Error("aether::interop::fromArrayInterface: strides has "
            + std::to_string(ai.strides.size()) + " entries for a rank-"
            + std::to_string(ai.shape.size()) + " shape");
    }
    std::vector<std::int64_t> shape = ai.shape;
    std::vector<std::int64_t> strides;
    for (std::size_t i = 0; i < ai.strides.size(); ++i) {
        if (ai.strides[i] % item != 0) {
            throw Error("aether::interop::fromArrayInterface: byte stride " + std::to_string(ai.strides[i])
                + " is not a multiple of the item size " + std::to_string(item));
        }
        strides.push_back(ai.strides[i] / item);
    }
    DLTensor t{};
    t.data = ai.data;
    t.device = ai.device;
    t.ndim = static_cast<std::int32_t>(shape.size());
    t.dtype = dt;
    t.shape = shape.data();
    t.strides = strides.empty() ? nullptr : strides.data();
    t.byte_offset = 0;
    BufferView b;
    b.view = detail::runtimeViewFromDLTensor(t);
    b.access = ai.readOnly ? Access::ReadOnly : Access::ReadWrite;
    b.producer = ai.producer;
    b.keepAlive = ai.keepAlive;
    return b;
}

/**
 * @brief What a consumer needs from a buffer. Every unset field is not
 *        checked. `writable` is the consumer's policy switch: set it for a
 *        buffer the consumer writes through.
 */
struct Requirements {
    /** @brief Required element dtype. */
    std::optional<DType> dtype;
    /** @brief Required element count (product of the shape). */
    std::optional<std::int64_t> count;
    /** @brief Required shape, in elements. */
    std::optional<std::vector<std::int64_t>> shape;
    /** @brief Require C-contiguous storage with a unit innermost stride. */
    bool contiguous = false;
    /** @brief Required byte alignment of the data pointer (0 = unchecked). */
    std::size_t alignment = 0;
    /** @brief Required device type (e.g. `kDLCUDA`, `kDLCPU`). */
    std::optional<DLDeviceType> deviceType;
    /** @brief Required device ordinal. */
    std::optional<std::int32_t> deviceId;
    /** @brief Require writes to be allowed (`BufferView::writable()`). */
    bool writable = false;
};

/** @brief A dtype's readable name (`"float64"`, `"int32"`, `"bool"`, ...). */
inline std::string dtypeName(const DType& dt)
{
    const unsigned bits = dt.bits();
    if (dt.lanes() == 1) {
        switch (dt.code()) {
        case kDLFloat:
            return "float" + std::to_string(bits);
        case kDLInt:
            return "int" + std::to_string(bits);
        case kDLUInt:
            return "uint" + std::to_string(bits);
        case kDLBool:
            return "bool";
        default:
            break;
        }
    }
    return "dtype(code=" + std::to_string(static_cast<unsigned>(dt.code())) + ", bits=" + std::to_string(bits)
        + ", lanes=" + std::to_string(static_cast<unsigned>(dt.lanes())) + ")";
}

/** @brief A device type's readable name (`"cpu"`, `"cuda"`, ...). */
inline std::string deviceTypeName(DLDeviceType t)
{
    switch (t) {
    case kDLCPU:
        return "cpu";
    case kDLCUDA:
        return "cuda";
    case kDLCUDAHost:
        return "cuda_host";
    case kDLCUDAManaged:
        return "cuda_managed";
    default:
        return "device_type " + std::to_string(static_cast<int>(t));
    }
}

namespace detail {

inline std::string tupleText(const std::int64_t* v, std::size_t n)
{
    std::string s = "(";
    for (std::size_t i = 0; i < n; ++i) {
        if (i != 0)
            s += ", ";
        s += std::to_string(v[i]);
    }
    if (n == 1)
        s += ",";
    return s + ")";
}

inline bool isCContiguous(const RuntimeView& v)
{
    offset_t expected = 1;
    for (std::size_t i = v.rank; i-- > 0;) {
        if (v.extents[i] == 0)
            return true;
        if (v.extents[i] != 1 && v.strides[i] != expected)
            return false;
        expected *= v.extents[i];
    }
    return true;
}

} // namespace detail

/**
 * @brief Every refusal `requirements` raises against `buffer`, one message
 *        per failed check. Each message names the check and both values,
 *        in the form `"<check>: required <want>, got <have>"`. Empty when
 *        the buffer satisfies every requirement.
 */
inline std::vector<std::string> refusals(const BufferView& buffer, const Requirements& req)
{
    std::vector<std::string> out;
    const RuntimeView& v = buffer.view;
    std::vector<std::int64_t> shape(v.rank);
    std::vector<std::int64_t> strides(v.rank);
    std::int64_t count = 1;
    for (std::size_t i = 0; i < v.rank; ++i) {
        shape[i] = static_cast<std::int64_t>(v.extents[i]);
        strides[i] = static_cast<std::int64_t>(v.strides[i]);
        count *= shape[i];
    }
    if (req.dtype && !(req.dtype->raw.code == v.dtype.raw.code && req.dtype->raw.bits == v.dtype.raw.bits
            && req.dtype->raw.lanes == v.dtype.raw.lanes)) {
        out.push_back("dtype: required " + dtypeName(*req.dtype) + ", got " + dtypeName(v.dtype));
    }
    if (req.count && *req.count != count) {
        out.push_back(
            "count: required " + std::to_string(*req.count) + " elements, got " + std::to_string(count));
    }
    if (req.shape && *req.shape != shape) {
        out.push_back("shape: required " + detail::tupleText(req.shape->data(), req.shape->size()) + ", got "
            + detail::tupleText(shape.data(), shape.size()));
    }
    if (req.contiguous && !detail::isCContiguous(v)) {
        out.push_back("stride: required C-contiguous with unit stride, got strides "
            + detail::tupleText(strides.data(), strides.size()) + " for shape "
            + detail::tupleText(shape.data(), shape.size()));
    }
    if (req.alignment != 0) {
        const auto addr = reinterpret_cast<std::uintptr_t>(v.data);
        if (addr % req.alignment != 0) {
            out.push_back("alignment: required " + std::to_string(req.alignment) + "-byte aligned, got address "
                + std::to_string(addr) + " (" + std::to_string(addr % req.alignment) + " bytes past a boundary)");
        }
    }
    if (req.deviceType && *req.deviceType != v.device.type()) {
        out.push_back("device: required " + deviceTypeName(*req.deviceType) + ", got "
            + deviceTypeName(v.device.type()));
    }
    if (req.deviceId && *req.deviceId != v.device.id()) {
        out.push_back("device id: required " + std::to_string(*req.deviceId) + ", got "
            + std::to_string(v.device.id()));
    }
    if (req.writable && !buffer.writable()) {
        std::string got = accessName(buffer.access);
        if (buffer.access == Access::Unknown)
            got += " (assert writability explicitly to accept it)";
        out.push_back("access: required read-write, got " + got);
    }
    return out;
}

/** @brief Throw one `aether::Error` carrying every refusal (joined by
 *         `"; "`) when `buffer` fails `requirements`; a no-op otherwise. */
inline void require(const BufferView& buffer, const Requirements& req)
{
    const std::vector<std::string> r = refusals(buffer, req);
    if (r.empty())
        return;
    std::string msg = "aether::interop::require: ";
    for (std::size_t i = 0; i < r.size(); ++i) {
        if (i != 0)
            msg += "; ";
        msg += r[i];
    }
    throw Error(msg);
}

/** @brief Refuse, naming `op`, an operation that reads or writes `buffer`'s
 *         memory from the host when that memory is not host-addressable
 *         (`hostAccessible`). Importing, inspecting and re-exporting a record
 *         never needs this; dereferencing its data does. */
inline void requireHostAccess(const BufferView& buffer, const char* op)
{
    if (hostAccessible(buffer.view.device.type()))
        return;
    throw Error(std::string(op) + ": device: host access needs host memory, got "
        + deviceTypeName(buffer.view.device.type()) + " (a record only on the host)");
}

namespace detail {

/** @brief Context of a `BufferView` export: shape/strides storage plus a
 *         share of the view's keep-alive. */
struct BufferExportCtx {
    std::vector<std::int64_t> shape;
    std::vector<std::int64_t> strides;
    std::shared_ptr<void> keepAlive;
};

inline BufferExportCtx* makeExportCtx(const BufferView& b)
{
    auto* ctx = new BufferExportCtx();
    ctx->shape.resize(b.view.rank);
    ctx->strides.resize(b.view.rank);
    for (std::size_t i = 0; i < b.view.rank; ++i) {
        ctx->shape[i] = static_cast<std::int64_t>(b.view.extents[i]);
        ctx->strides[i] = static_cast<std::int64_t>(b.view.strides[i]);
    }
    ctx->keepAlive = b.keepAlive;
    return ctx;
}

inline void fillTensor(DLTensor& t, const BufferView& b, BufferExportCtx* ctx, DLDataType dtype)
{
    t.data = b.view.data;
    t.device = b.view.device.raw;
    t.ndim = static_cast<std::int32_t>(b.view.rank);
    t.dtype = dtype;
    t.shape = ctx->shape.data();
    t.strides = ctx->strides.data();
    t.byte_offset = 0;
}

inline void checkExportable(const BufferView& b, const char* op)
{
    if (b.view.rank > RuntimeRankCap)
        throw Error(std::string(op) + ": rank " + std::to_string(b.view.rank) + " exceeds RuntimeRankCap");
    if (!dtypeIsSupported(b.view.dtype.raw))
        throw Error(std::string(op) + ": unsupported dtype " + dtypeName(b.view.dtype));
    if (!deviceIsKnown(b.view.device.type()))
        throw Error(std::string(op) + ": unsupported device " + deviceTypeName(b.view.device.type()));
}

} // namespace detail

/**
 * @brief Export `buffer` as a heap-allocated versioned DLPack tensor,
 *        zero-copy. The read-only flag is set unless the buffer is writable
 *        (`BufferView::writable()`), so an `Unknown` buffer exports as
 *        read-only unless writability was asserted. The tensor's context
 *        holds a share of the keep-alive: the producer stays alive until
 *        the consumer calls the tensor's deleter AND every other view is
 *        gone. The caller (or the framework it hands the tensor to) owns
 *        the result and calls its deleter once.
 */
inline DLManagedTensorVersioned* exportDLPack(const BufferView& buffer)
{
    detail::checkExportable(buffer, "aether::interop::exportDLPack");
    auto* ctx = detail::makeExportCtx(buffer);
    auto* tensor = new DLManagedTensorVersioned{};
    tensor->version = DLPackVersion{ static_cast<std::uint32_t>(DLPACK_MAJOR_VERSION),
        static_cast<std::uint32_t>(DLPACK_MINOR_VERSION) };
    tensor->manager_ctx = ctx;
    tensor->deleter = [](DLManagedTensorVersioned* self) {
        delete static_cast<detail::BufferExportCtx*>(self->manager_ctx);
        delete self;
    };
    tensor->flags = buffer.writable() ? 0 : DLPACK_FLAG_BITMASK_READ_ONLY;
    detail::fillTensor(tensor->dl_tensor, buffer, ctx, buffer.view.dtype.raw);
    return tensor;
}

/**
 * @brief Export `buffer` as a heap-allocated legacy (pre-1.0) DLPack tensor,
 *        zero-copy. The legacy struct has no access flag, so a buffer
 *        declared read-only is REFUSED rather than exported as if it were
 *        writable (the same rule numpy applies); an `Unknown` buffer exports
 *        unchanged (its consumer sees it as unknown again). `bool` follows
 *        `toDLPackLegacy`'s `kDLUInt` policy. Same keep-alive contract as
 *        `exportDLPack`.
 *
 * @throws aether::Error  a read-only buffer, or an unsupported rank, dtype or device.
 */
inline DLManagedTensor* exportDLPackLegacy(const BufferView& buffer)
{
    detail::checkExportable(buffer, "aether::interop::exportDLPackLegacy");
    if (buffer.access == Access::ReadOnly) {
        throw Error("aether::interop::exportDLPackLegacy: access: legacy DLPack cannot carry the read-only "
                    "flag, got read-only (export the versioned struct)");
    }
    auto* ctx = detail::makeExportCtx(buffer);
    auto* tensor = new DLManagedTensor{};
    tensor->manager_ctx = ctx;
    tensor->deleter = [](DLManagedTensor* self) {
        delete static_cast<detail::BufferExportCtx*>(self->manager_ctx);
        delete self;
    };
    detail::fillTensor(tensor->dl_tensor, buffer, ctx, detail::legacyDType(buffer.view.dtype));
    return tensor;
}

} // namespace interop
} // namespace aether
