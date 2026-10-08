# NVRTC options for aether

**Rule**: NVRTC ships NO C++ standard library (its docs admit only builtin
`std::move`/`std::forward`/`std::initializer_list`); pointing `-I` at the
REAL host libstdc++ does not work either — those headers need GNU compiler
builtins (`__SIZE_TYPE__` etc.) NVRTC's frontend never defines. Use the CUDA
toolkit's own `libcu++` (`cuda/std/...`) instead — NVRTC-safe by construction.

## Canonical option list (derive every path at runtime, never hard-code)
```
--gpu-architecture=<arch>
--std=c++20
--include-path=<repo root>                                 # aether/aether.h
--include-path=<repo>/tools/nvrtc/shim_include              # cstddef -> cuda/std/*
--include-path=<CUDA_ROOT>/targets/x86_64-linux/include     # libcu++
--pre-include=<repo>/tools/nvrtc/shim_include/__aether_nvrtc_prelude.h
```
`CUDA_ROOT` = `dirname $(command -v nvcc)/..` (see `nvrtc_parse_audit.sh`).
## Diagnosis (each step verified by running NVRTC, not by reading)
1. `<cstddef>` etc. unresolved: NVRTC has no stdlib, no `-I` default; real
   libstdc++ via `-I` then hits `__SIZE_TYPE__` undefined — no NVRTC option
   restores GNU builtin-macro emulation.
2. `shim_include/` re-exports `cuda/std/{cstddef,cstdint,cmath,concepts,
   array,type_traits,utility,mdspan,cassert}` into `::std`; `cstdint` also
   patches `SIZE_MAX`/`PTRDIFF_MAX`, undefined by this libcu++'s own NVRTC
   branch (needed by `aether/layout/Extents.h`).
3. `BoundedTrig.h` calls `__builtin_memcpy`, unrecognized by NVRTC's
   frontend; the prelude defines it (libcu++ avoids this itself, using
   `__builtin_bit_cast` instead — the real long-term aether-side fix).
## `aether/device.h`, the device-safe umbrella (closes the gap below)
`#include <aether/aether.h>` (the FULL umbrella) is still, and stays,
RED — `accum/AccumPlane.h` etc. unconditionally `#include <algorithm>`,
no NVRTC guard, by design (host-only surface). But materializing
`Item c = a + b;` (not `auto c = a+b;`) is no longer blocked: this split
`view/View.h`'s host-only `make_view()` factories out to `view/MakeView.h`
(same for `reshaped()` -> `view/MakeReshape.h`, `RuntimeView::as<>()`'s
definition -> `view/MakeRuntimeView.h`, `makeTableView()` -> `view/
MakeTableView.h`) and fixed `dtype/dlpack.h`'s plain-C `<stdint.h>`/
`<stddef.h>` to `<cstdint>`/`<cstddef>` (this shim's `cstdint`/`cstddef`
serve those, not the C names) — so `expr/Assign.h -> view/View.h ->
err/Error.h` no longer fires. `aether/device.h` (new) is the resulting
device-safe umbrella: the 9 originally-passing headers plus
`view/{View,Reshape,RuntimeView,TableHandle,TableView,WorkView}.h`,
`dtype/{DType,WorkingType}.h`, `device/Device.h`, `expr/Assign.h` — see
that header's own docstring for the exact include list, and for two
NEW root causes the original diagnosis above never encountered:

- A bodiless, UNANNOTATED function/method DECLARATION (no
  `__host__`/`__device__`/`__global__`) is rejected outright by NVRTC's
  JIT mode, even uncalled — stricter than nvcc's normal offline device
  pass, which tolerates one fine as long as nothing calls it from device
  code. Hit `RuntimeView::as<>()` (a class member — split impossible
  without breaking `rt.as<T>()` call sites, so GUARDED
  `#if !defined(__CUDA_ARCH__)` instead, both declaration and its
  out-of-line definition) and `TableView.h`'s/`RuntimeView.h`'s free
  functions `makeTableView()`/`fromView()` (both plain functions, so
  SPLIT cleanly like `make_view`, no guard needed).
- `aether/banded/banded.h` (`aether/banded/Band.h`'s `#include <cstring>`,
  unshimmed) and `aether/view/Tile.h` (unconditionally pulls in
  `aether/backend/cpu/PacketItem.h`, itself full of the unannotated-
  function pattern above) are BOTH still RED and excluded from
  `device.h` — see `device.h`'s own docstring for the exact error lines;
  both are deliberate exclusions from the candidate list, not silently
  dropped.

## Gate: `tools/nvrtc/audit_device_umbrella.sh [gpu-arch]`
Two subjects (`tools/nvrtc/device_umbrella_probe.cpp`), same canonical
option list as above, derived the same way: `device` (`#include
<aether/device.h>` + a materializing `Item c = a + b` — MUST PASS) and
`control` (`#include <aether/aether.h>` — MUST stay RED, the non-vacuity
proof this audit can still fail). Exit code is 0 iff both hold.

## Manual steps
None — every path above derives from the active toolchain (nvcc -> `CUDA_ROOT` -> libcu++); nothing outside this repo needed changing.
