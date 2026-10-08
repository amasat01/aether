// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file device.h
 * @brief The device-safe (NVRTC-parseable) subset of the aether
 *        umbrella — register-resident + batched-view expression-template
 *        arithmetic, with every HOST-ONLY header (anything that reaches
 *        `<string>`/`aether/err/Error.h`/allocation) either split out or
 *        excluded.
 *
 * `aether/aether.h` = `device.h` + the host-only surface (allocation,
 * the `Make*` factories / checked-promotion, the error contract,
 * residency, DLPack interop, pretty-printing, …). A consumer that only
 * needs device-compilable arithmetic (a future codegen-ISA JIT path)
 * `#include`s `<aether/device.h>` directly and pays for
 * none of the host machinery — this is what makes an aether umbrella
 * NVRTC-parseable at all (see `tools/nvrtc/README.md` and
 * `tools/nvrtc/audit_device_umbrella.sh`, which gate exactly this claim,
 * and `tests/compile_fail/check_device_umbrella_hostfree.sh`, which gates
 * that the excluded host factories stay unreachable from here).
 *
 * Included, in dependency order:
 *  - the 9 headers found standalone-PASS under NVRTC with no source
 *    change: `view/Item.h`, `expr/Expression.h`,
 *    `expr/Operations.h` (which itself pulls in `expr/nodes/Arithmetic.h`
 *    — `+`/`-`/`s*e`), `expr/Reduce.h`, `index/{BundleIndex,Offset,
 *    SampleIndex}.h`, `layout/{Extents,Layout}.h`;
 *  - `device/Device.h`, `dtype/DType.h`, `dtype/WorkingType.h` — standalone
 *    PASS once `dtype/dlpack.h`'s plain-C `<stdint.h>`/`<stddef.h>` became
 *    `<cstdint>`/`<cstddef>`;
 *  - `view/View.h`, `view/Reshape.h`, `view/TableHandle.h`,
 *    `view/TableView.h`, `view/WorkView.h` — standalone PASS AFTER
 *    their `<string>`/`err::`-using host factory functions were moved out
 *    to `view/{MakeView,MakeReshape,MakeTableView}.h` (which stay OUT of
 *    this umbrella — see the compile-fail gate above). `view/TableView.h`'s
 *    `makeTableView()` was found RED only once `View.h`'s own `<string>`
 *    issue no longer masked it (probed: `TableView.h(93)`, an
 *    UNANNOTATED free function — NVRTC's JIT mode rejects one outright,
 *    even bodiless/uncalled, unlike nvcc's normal device pass); the same
 *    "split, don't guard" call as `make_view`/`reshaped` applies
 *    cleanly (a free function, unlike `RuntimeView::as<>()` below).
 *  - `view/RuntimeView.h` — standalone PASS AFTER
 *    `as<>()`'s out-of-line DEFINITION (plus the helpers only it calls)
 *    moved to `view/MakeRuntimeView.h`, AND (unlike every other split above)
 *    GUARDED `as<>()`'s DECLARATION itself with
 *    `#if !defined(__CUDACC_RTC__)` in place — probed: even
 *    the bare declaration, no body, no call site, was RED
 *    (`RuntimeView.h(127)`, same unannotated-function JIT-mode rejection
 *    as `TableView.h` above). A split was not possible here without
 *    turning `rt.as<T>()` from a member call into a free function,
 *    breaking every existing call site — this is the
 *    one `#if !defined(__CUDACC_RTC__)` guard this umbrella adds (an
 *    "unless a split is impossible" carve-out). NOTE the macro:
 *    `__CUDA_ARCH__` is ALSO defined during nvcc's own device-compilation
 *    pass of an ordinary `.cu` file, and that pass fully parses the whole
 *    TU including host-only call sites elsewhere in it — guarding on it
 *    broke `tests/test_DLPackRoundtrip.cu`'s and `tests/test_RuntimeView
 *    .cu`'s existing HOST-side `.as<>()` calls (a real build regression,
 *    found by actually building `build_cuda_release`, not by NVRTC alone).
 *    `__CUDACC_RTC__` is NVRTC-only (nvcc never defines it, either pass —
 *    verified by probe), so it removes `as<>()` ONLY from NVRTC's view;
 *    the out-of-line definition in `MakeRuntimeView.h` carries the
 *    identical guard so nvcc's own passes still see a consistent class
 *    shape.
 *  - `expr/Assign.h` — the out-of-line `Item`/`View` assignment machinery
 *    (`RecursiveAssign`, `SampleRef`, `Item`'s expression ctor) a
 *    materializing `Item c = a + b` needs. Standalone-PASS once `View.h`
 *    stopped dragging in `Error.h` (it never itself included
 *    `err/Error.h`, see `aether/expr/Assign.h`'s own note).
 *
 * Excluded, with the first NVRTC error (probed on this tree,
 * `compute_61`, `tools/nvrtc/README.md`'s canonical option list):
 *  - DEVIATION: `aether/banded/banded.h` — RED, `aether/banded/Band.h`'s `#include <cstring>`
 *    `catastrophic error: cannot open source file "cstring"` (not one of
 *    `tools/nvrtc/shim_include`'s shimmed names). `aether/banded/` is
 *    out of scope here, and extending the shim set is out of scope too —
 *    Band IS device code by construction, this is an environment gap, not
 *    a design one; left out, listed as RED.
 *  - DEVIATION: `aether/view/Tile.h` — RED, unconditionally
 *    `#include`s `aether/backend/cpu/PacketItem.h` ->
 *    `aether/backend/cpu/simd/PacketMask.h`'s non-AVX512 default constructor
 *    `error: A function without execution space annotations
 *    (__host__/__device__/__global__) is considered a host function, and
 *    host functions are not allowed in JIT mode.` — nvcc's normal offline
 *    compile pass tolerates an un-annotated host-only free function
 *    reached from a device TU; NVRTC's JIT mode does not. This is a real
 *    `Tile.h` defect (its `PacketTile`/packet-`slot()` half needs the same
 *    host/device split `View.h` just got — the `#include
 *    "aether/backend/cpu/PacketItem.h"` in `Tile.h` is unconditional,
 *    no `#if !defined(__CUDACC_RTC__)` guard); left out, reported as a
 *    candidate defect for later.
 */

#include "aether/macros.h"

#include "aether/device/Device.h"
#include "aether/dtype/DType.h"
#include "aether/dtype/Fetch.h"
#include "aether/dtype/WorkingType.h"
#include "aether/expr/Assign.h"
#include "aether/expr/Expression.h"
#include "aether/expr/Operations.h"
#include "aether/expr/Reduce.h"
#include "aether/index/BundleIndex.h"
#include "aether/index/Offset.h"
#include "aether/index/SampleIndex.h"
#include "aether/layout/Extents.h"
#include "aether/layout/Layout.h"
#include "aether/view/Item.h"
#include "aether/view/Reshape.h"
#include "aether/view/RuntimeView.h"
#include "aether/view/TableHandle.h"
#include "aether/view/TableView.h"
#include "aether/view/View.h"
#include "aether/view/WorkView.h"
