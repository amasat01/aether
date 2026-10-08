// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// tools/dlpack_preone_probe.cpp — de-risk probe for eagle's pre-1.0
// DLPack adapter: what exactly must a bidirectional aether<->eagle
// DLPack adapter translate, given aether's `toDLPack`
// (aether/interop/DLPack.h) emits DLPack v1.0's `DLManagedTensorVersioned`
// while eagle's vendored, hand-written, pre-1.0 subset
// (eagle/plugin/dlpack.h, read-only exemplar) only knows the legacy,
// unversioned `DLManagedTensor`?
//
// METHOD: both headers define SAME-NAMED C structs/enums
// (`DLTensor`/`DLManagedTensor`/`DLDevice`/`DLDataType`/`kDL*`) in the
// global namespace — they cannot be `#include`d together directly (a
// redefinition error). eagle's copy is included here wrapped in its own
// namespace (`eagle_abi`) — legal C++ (the `extern "C"` block only affects
// the linkage of any declared functions, not the namespace-scoping of the
// typedefs eagle/plugin/dlpack.h actually contains) — so both copies are
// simultaneously visible and directly comparable via `sizeof`/`offsetof`/
// `static_assert`, with no source modification to the read-only eagle
// header. Host-only, no GPU, no CUDA context — pure compile-time ABI
// comparison; the driver script also runs the resulting binary, which does
// nothing but print the findings already proven at compile time.
//
// Standalone TU, compiled+run directly by tools/dlpack_preone_audit.sh
// (mirrors tools/volatile_probe.cu's placement/rationale) — never wired
// into tests/CMakeLists.txt's glob (a de-risk probe, not a permanent
// regression test).

#include <cstddef>
#include <cstdio>

// aether's own vendored DLPack ABI (verbatim upstream v1.0) — global
// namespace, exactly as aether/interop/DLPack.h itself sees it.
#include <aether/dtype/dlpack.h>

// eagle's vendored PRE-1.0 subset (READ-ONLY exemplar — NOT modified),
// wrapped in its own namespace so its same-named typedefs don't collide
// with aether's global-namespace copy above.
namespace eagle_abi {
#include "eagle/plugin/dlpack.h"
} // namespace eagle_abi

namespace {

// ---------------------------------------------------------------------
// Sub-structs BOTH headers define with the SAME field set — the INNER
// tensor payload every real producer/consumer actually reads/writes.
// FINDING (see the file docstring): these are ABI-IDENTICAL.
// ---------------------------------------------------------------------

static_assert(sizeof(eagle_abi::DLDevice) == sizeof(::DLDevice), "DLDevice size mismatch");
static_assert(offsetof(eagle_abi::DLDevice, device_type) == offsetof(::DLDevice, device_type), "DLDevice.device_type offset mismatch");
static_assert(offsetof(eagle_abi::DLDevice, device_id) == offsetof(::DLDevice, device_id), "DLDevice.device_id offset mismatch");

static_assert(sizeof(eagle_abi::DLDataType) == sizeof(::DLDataType), "DLDataType size mismatch");
static_assert(offsetof(eagle_abi::DLDataType, code) == offsetof(::DLDataType, code), "DLDataType.code offset mismatch");
static_assert(offsetof(eagle_abi::DLDataType, bits) == offsetof(::DLDataType, bits), "DLDataType.bits offset mismatch");
static_assert(offsetof(eagle_abi::DLDataType, lanes) == offsetof(::DLDataType, lanes), "DLDataType.lanes offset mismatch");

static_assert(sizeof(eagle_abi::DLTensor) == sizeof(::DLTensor), "DLTensor size mismatch");
static_assert(offsetof(eagle_abi::DLTensor, data) == offsetof(::DLTensor, data), "DLTensor.data offset mismatch");
static_assert(offsetof(eagle_abi::DLTensor, device) == offsetof(::DLTensor, device), "DLTensor.device offset mismatch");
static_assert(offsetof(eagle_abi::DLTensor, ndim) == offsetof(::DLTensor, ndim), "DLTensor.ndim offset mismatch");
static_assert(offsetof(eagle_abi::DLTensor, dtype) == offsetof(::DLTensor, dtype), "DLTensor.dtype offset mismatch");
static_assert(offsetof(eagle_abi::DLTensor, shape) == offsetof(::DLTensor, shape), "DLTensor.shape offset mismatch");
static_assert(offsetof(eagle_abi::DLTensor, strides) == offsetof(::DLTensor, strides), "DLTensor.strides offset mismatch");
static_assert(offsetof(eagle_abi::DLTensor, byte_offset) == offsetof(::DLTensor, byte_offset), "DLTensor.byte_offset offset mismatch");

// ---------------------------------------------------------------------
// eagle's ONLY envelope (`DLManagedTensor`, unversioned/"legacy") vs
// aether's TWO envelopes. aether's vendored header (a verbatim upstream
// v1.0 copy) still carries the LEGACY `DLManagedTensor` for backward
// compat, alongside the NEW `DLManagedTensorVersioned` — but
// `aether::interop::toDLPack()` (aether/interop/DLPack.h) only EVER
// PRODUCES `DLManagedTensorVersioned*`.
//
// FINDING 1 (the reassuring half): eagle's `DLManagedTensor` and aether's
// OWN (already-vendored, currently UNUSED by toDLPack) legacy
// `::DLManagedTensor` are ABI-IDENTICAL — same 3 fields, same order:
// `dl_tensor`, `manager_ctx`, `deleter(SelfPtr*)`. A `reinterpret_cast`
// between the two (crossing the plugin boundary via an opaque `void*`,
// DLPack's own intended interop contract) is layout-safe.
// ---------------------------------------------------------------------

static_assert(sizeof(eagle_abi::DLManagedTensor) == sizeof(::DLManagedTensor), "legacy DLManagedTensor size mismatch (eagle vs aether's OWN vendored legacy struct)");
static_assert(offsetof(eagle_abi::DLManagedTensor, dl_tensor) == offsetof(::DLManagedTensor, dl_tensor), "legacy DLManagedTensor.dl_tensor offset mismatch");
static_assert(offsetof(eagle_abi::DLManagedTensor, manager_ctx) == offsetof(::DLManagedTensor, manager_ctx), "legacy DLManagedTensor.manager_ctx offset mismatch");
static_assert(offsetof(eagle_abi::DLManagedTensor, deleter) == offsetof(::DLManagedTensor, deleter), "legacy DLManagedTensor.deleter offset mismatch");

// FINDING 2 (the real risk): eagle's `DLManagedTensor` (3 fields,
// `dl_tensor` FIRST) is NOT layout-compatible with what `toDLPack()`
// ACTUALLY hands back, `::DLManagedTensorVersioned` (5 fields — a NEW
// leading `version`, a NEW `flags`, `dl_tensor` LAST). A caller that
// `reinterpret_cast`s aether's real toDLPack() output straight to
// `eagle::DLManagedTensor*` would read `version.{major,minor}` as if it
// were the first 8 bytes of a `DLTensor` (i.e. garbage `data`/`device`
// fields) — silent memory corruption, not a compile error. Proven here by
// the sizes/offsets genuinely differing (the assertion below is the
// NEGATIVE control: it must hold, i.e. the two envelopes are NOT the same
// size, confirming they are not accidentally compatible).
static_assert(sizeof(::DLManagedTensorVersioned) != sizeof(eagle_abi::DLManagedTensor),
    "unexpected: DLManagedTensorVersioned and eagle's DLManagedTensor came out the SAME size — re-verify FINDING 2 by hand");

// ---------------------------------------------------------------------
// Device-kind / dtype-code enumerator coverage (numeric VALUES that
// overlap agree; eagle's sets are a narrow subset — this is a coverage
// gap, not a layout mismatch).
// ---------------------------------------------------------------------

static_assert(static_cast<int>(eagle_abi::kDLCPU) == static_cast<int>(::kDLCPU), "kDLCPU value mismatch");
static_assert(static_cast<int>(eagle_abi::kDLCUDA) == static_cast<int>(::kDLCUDA), "kDLCUDA value mismatch");
static_assert(static_cast<int>(eagle_abi::kDLCUDAHost) == static_cast<int>(::kDLCUDAHost), "kDLCUDAHost value mismatch");
static_assert(static_cast<int>(eagle_abi::kDLCUDAManaged) == static_cast<int>(::kDLCUDAManaged), "kDLCUDAManaged value mismatch");
// eagle's DLDeviceType has NO kDLOpenCL/kDLVulkan/kDLMetal/kDLVPI/kDLROCM/
// kDLROCMHost/kDLExtDev/kDLOneAPI/kDLWebGPU/kDLHexagon/kDLMAIA entries —
// not exercised by aether today (Device.h's backend_enabled only lights up
// kDLCPU/kDLCUDA/kDLCUDAHost), so currently moot; would matter the day
// aether grows another device kind eagle must also consume.

static_assert(static_cast<int>(eagle_abi::kDLInt) == static_cast<int>(::kDLInt), "kDLInt value mismatch");
static_assert(static_cast<int>(eagle_abi::kDLUInt) == static_cast<int>(::kDLUInt), "kDLUInt value mismatch");
static_assert(static_cast<int>(eagle_abi::kDLFloat) == static_cast<int>(::kDLFloat), "kDLFloat value mismatch");
static_assert(static_cast<int>(eagle_abi::kDLBfloat) == static_cast<int>(::kDLBfloat), "kDLBfloat value mismatch");
// FINDING 3 ("missing kDLBool"):
// eagle_abi::kDLBool DOES NOT EXIST — eagle/plugin/dlpack.h's
// DLDataTypeCode enum is exactly {kDLInt=0, kDLUInt=1, kDLFloat=2,
// kDLBfloat=4}, no kDLOpaqueHandle(3)/kDLComplex(5)/kDLBool(6). aether's
// `dtype_of<bool>()` (aether/dtype/DType.h) maps to `kDLBool` (code=6,
// bits=8) — a bool-dtype tensor crossing into an eagle-based consumer
// carries a numeric `dtype.code == 6` (the byte value survives the wire
// unharmed, `DLDataType::code` is `uint8_t` in BOTH headers) that eagle's
// own code has NO SYMBOL for and, in practice, no type-dispatch case for.
// (Left as a comment, not a static_assert, so this probe TU still
// COMPILES — referencing a nonexistent enumerator would defeat the "this
// TU proves the REST of the layout" purpose; confirmed by direct source
// reading of eagle/plugin/dlpack.h, unambiguous — a 4-entry enum with no
// 5th/6th value.)

} // namespace

int main()
{
    std::printf("PASS: DLTensor/DLDevice/DLDataType are ABI-identical between eagle's pre-1.0 header and aether's vendored copy (sizeof(DLTensor)=%zu)\n",
        sizeof(::DLTensor));
    std::printf("PASS: eagle's DLManagedTensor (unversioned) is ABI-identical to aether's OWN vendored legacy ::DLManagedTensor (sizeof=%zu)\n",
        sizeof(::DLManagedTensor));
    std::printf("FINDING: ::DLManagedTensorVersioned (what aether::interop::toDLPack() actually returns, sizeof=%zu) is "
                 "NOT layout-compatible with eagle's DLManagedTensor (sizeof=%zu) — a real translating adapter is needed "
                 "for THAT envelope, but can target aether's own already-vendored legacy struct as the eagle-compatible shape.\n",
        sizeof(::DLManagedTensorVersioned), sizeof(eagle_abi::DLManagedTensor));
    std::printf("FINDING: eagle's DLDataTypeCode has no kDLBool (aether dtype_of<bool>() emits code=6, byte value survives "
                 "the wire, but eagle has no symbol/dispatch case for it).\n");
    return 0;
}
