// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file aether.h
 * @brief Umbrella header for the aether library.
 *
 * Start here: aether is the vector-algebra / GPU-memory core the family's
 * examples build on (`Array`, `View`, `Item`, expression templates, CUDA
 * launch helpers). Most code just does `#include <aether/aether.h>` and
 * uses `Array`/`View`/`Item` — see `aether/device.h`'s own docstring if you
 * only need the device-safe (NVRTC-parseable) half.
 *
 * Includes macros, dtype/device/chunk/err, layout/view/index/array (the
 * cornerstone: `extents`/layouts, `View`, `Item`, `SampleIndex`/
 * `PacketIndex`, `reshaped`, `Array`) plus the consumer alias surface
 * below, expr/ (the CRTP `Expression` base + `aether_expression` concept,
 * the `Sum`/`CWiseScale` nodes, `+`/`-`/`s*e`/`e*s`/`e/s`, and the
 * `RecursiveAssign` static evaluator behind `View::operator[]`/`Item`'s
 * expression ctor — `Item`/`View` are expression leaves), `backend/cpu/`
 * (the SIMD packet layer — `Packet`/`PacketMask`/`PacketItem`,
 * `packetGet`/`packetAssign`/`packetCapture`/`packetFor`/`packetFlatFor`/
 * `packetEval`/`packetEvalParallel`) and `backend/cuda/Launch.h` (CUDA
 * launch helpers — a no-op include outside CUDA-mode builds).
 *
 * Deliberately does not include `aether/dtype/Format.h` — that header is
 * host-only pretty-printing kept separate precisely so the umbrella stays
 * device-safe (see Format.h); include it explicitly when you need
 * `to_string()`/`operator<<`.
 *
 * Adds the geometric/reduction/quaternion/slicing member-function surface
 * on `Expression` itself (`cross`/`unitVector`, `dot`/`norm`/
 * `squaredNorm`/`cubedNorm`/`rNorm`/`rSquaredNorm`/`rCubedNorm`/
 * `maxNorm`/`sum` + free `isFinite`, `quatMul`/`quatConj`/
 * `quatReciprocal`/`quatRotate`/`asPureQuaternion`/`asBack3DVector`,
 * `segment`/`head`/`tail`) backed by `expr/Reduce.h` + `expr/nodes/
 * {Geometric,Quaternion,Structural}.h`; `math/math.h` (device-legal
 * scalar dispatch — `abs`/`floor`/`fma`/`fmax`/`fmin`/`min`/`pow`/
 * `sqrt`/`isfinite`/`sin`/`cos`/`sincos`, the last three backed by
 * `math/detail/BoundedTrig.h`'s zero-stack FP64 device path) and
 * `expr/nodes/Product.h` (`MatVec`/`MatMat`/`Outer`/`CWiseMul` +
 * `Expression::{row,col,block,transpose,cwiseMul,matDot}()`, the fused
 * multi-load assignment specializations, and `View::as_volatile()` in
 * `view/View.h`).
 *
 * The runtime arm: `view/RuntimeView.h` (the fixed-capacity runtime
 * descriptor + checked `RuntimeView::as<>` promotion + `fromView()`),
 * `eval/Runtime.h` (the dual-mode elementwise runtime evaluator,
 * `eval::runtimeEval`) and `interop/DLPack.h` (`interop::fromDLPack`/
 * `toDLPack`, the DLPack <-> `RuntimeView` C-ABI bridge). Geometric/
 * quat/matrix stay static-only (the perf path). Deliberately does not
 * include `interop/Mdspan.h` — same reasoning as `dtype/Format.h` above
 * (host-only, and on this workspace's pinned toolchain also
 * environment-gated inert, see that header's own docstring); include it
 * explicitly when you need the `std::mdspan` adapter.
 *
 * Max-bandwidth device access idioms: `View::as_readonly()` (`__ldg`-
 * routed device loads, `view/View.h`); `BundleIndex<W>`
 * (`index/BundleIndex.h`), `DeviceBundle<T,W,Es...>`
 * (`backend/cuda/DeviceBundle.h`) + its `bundleGet`/`bundleStore`/
 * `bundleAssign` protocol (`backend/cuda/bundle/{LoadStore,Assign}.h`)
 * and `View::operator[](BundleIndex<W>)` (the device analog of the CPU
 * packet layer; compiled in both modes, cpp-mode lowers to a scalar
 * loop); `bundleLaunchConfig` (`backend/cuda/Launch.h`); `make_work_view`
 * (`view/WorkView.h`, the shared-memory idiom).
 *
 * Residency: `Chunk::borrow()` (foreign-pointer, non-owning wrap —
 * `chunk/Chunk.h`) and `residency/` — `PartitionSpec` +
 * `padded_block_samples` + `partition_view` (`residency/Partition.h`,
 * slicing a batched view's sample mode into padded blocks, no
 * allocation); `ReplicaSet` (`residency/Replica.h`, N `Chunk`s across an
 * explicit `Device` list + `broadcast()`); the `transport<T>` concept +
 * `StreamTransport` (`residency/Transport.h`, the multi-node seam).
 *
 * Also: `Item`'s N-scalar constructor (`view/Item.h`); `aether::int_t`
 * (`dtype/DType.h`, the codegen Int value type, distinct from
 * `offset_t`); `TableHandle<T>` (`view/TableHandle.h`, the bounded
 * non-texture table-read handle — reuses the `__ldg` read-only policy).
 *
 * `math/explog.h` (`exp`/`exp2`/`exp10`/`log`/`log2`/`log10`) and
 * `math::sign` (in `math/math.h` alongside `fma`/`isfinite`);
 * `expr/nodes/Elementwise.h` (`cwiseMin`/`cwiseMax`/`cwiseAbs`/`clamp`,
 * free functions mirroring `CWiseScale`'s shape; `cwiseSign` alongside
 * `cwiseAbs`); `expr/nodes/Inverse.h` + `Expression::{trace,det,
 * inverse}` (small-matrix closed forms, 2x2/3x3 only, no LU/solve).
 *
 * `random/` — `aether::random::Generator` plus the `uniform`/`normal`
 * per-component distribution leaves, built on aether's own
 * Philox4x32-10 (`random/detail/Philox.h`). One counter-based, stateless
 * generator serves both arms, so a seed names one stream everywhere
 * (host draws == device draws, bit for bit) and no `<curand_kernel.h>` —
 * no CUDA toolkit dependency of any kind — enters the library. Every
 * draw is a pure function of `(seed, global sample id, sub-counter)`, so
 * a parallel fill and a serial fill agree exactly. See `random/random.h`.
 *
 * `aether/banded/` — the emulated 53-bit carrier `Band`, its 8-byte
 * codec word `BandCell8`, and the storage value type `BandedReal` with
 * its operator set and `numeric_limits` — is deliberately not included
 * here: it reaches the whole carrier and codec, which no TU that
 * computes `sin(double)` has any use for (`tests/headers/
 * check_header_diet.sh`'s BANDED arms pin the absence, with a control
 * arm proving the observation is real). What is reachable from here is
 * the declaration-only `aether/math/detail/BandedFwd.h`, which
 * `math/detail/MathDispatch.h` includes so the five certified
 * `aether::math` entry points (`abs`, `copysign`, `fmax`/`fmin`/`min`,
 * `pow`) route for a banded operand in every math TU while the
 * definitions stay out — include `aether/banded/banded.h` explicitly
 * when you need the type.
 *
 * `accum/` — `aether::accum::atomic` (`accum/atomic.h`: the two-sum
 * compensated-atomic sink, plus the `minAbsReal`/`orBool`/`maxInt`
 * siblings) and `aether::AccumPlane`/`AccumPlaneView`
 * (`accum/AccumPlane.h`: the per-sample accumulation plane,
 * `CompensatedAtomic`/`Serialized` policies only) — plus
 * `aether::accum::AppendCounter` (`accum/AppendCounter.h`: the
 * device-side atomic count over an `Array`'s reserved `spare()`
 * capacity) and its error-contract dependency `aether::DeviceFlag`
 * (`err/DeviceFlag.h`: the missing device-visible half of the
 * host-throw error contract `err/Error.h` states but cannot itself
 * express from device code).
 *
 * `view/TableView.h` (`TableView<T,Extents>`/`makeTableView` — a thin
 * alias over `View<const T,Extents,layout_right>`/`make_view`) and
 * `view/Tile.h` (`Tile<T,D,K>`/`slot<s>` over `Item`/
 * `Expression::segment<>`, plus the `PacketTile`/`slot<s>`
 * counterpart over `backend/cpu/PacketItem`, host-only) — both pure
 * facades over machinery this umbrella already carries (`view/View.h`,
 * `view/Item.h`), adding no members and no runtime.
 *
 * This umbrella splits into a device-safe half, `aether/device.h`
 * (NVRTC-parseable: register-resident + batched-view expression-template
 * arithmetic, no allocation, no throw), and everything below, the
 * host-only remainder (allocation, the `Make*` factories/checked-
 * promotion `view/{MakeView,MakeReshape,MakeRuntimeView}.h` split out of
 * `view/{View,Reshape,RuntimeView}.h`, the error contract, residency,
 * DLPack interop, `Tile.h`'s packet half) — see `aether/device.h`'s own
 * docstring for exactly what lives where.
 *
 * The const read path — `View::operator[](const SampleIndex&|const
 * BundleIndex<W>&) const` returning `ConstSampleRef`/`ConstBundleRef`
 * (`view/View.h`, `expr/Assign.h`, `backend/cuda/bundle/Assign.h`), `=
 * += -=`-free by construction — and `expr/nodes/Constant.h` (the
 * scalar-broadcast leaf `aether::constant<Extents>(v)` backing
 * `SampleRef`'s three scalar overloads) are deliberately not part of
 * `aether/device.h`'s NVRTC-audited surface (see that header's own
 * docstring and `Constant.h`'s/`ConstBundleRef`'s placement notes) —
 * every test in this tree reaches it through this umbrella, never
 * `device.h` alone.
 *
 * `chunk/TextureBinding.h` (`TextureBinding<T>` — a move-only RAII
 * `cudaTextureObject_t` lifetime object over a `Chunk`) and
 * `dtype/Fetch.h` (`aether::texture_handle_t` + `dtype::Fetch<T>`, the
 * texel-fetch trait: `double` -> `int2`, `float` -> `float`).
 * `view/TableHandle.h` gains a sibling `Texture` carrier
 * (`TableHandle<T, Texture>`, `tex1Dfetch`-routed) alongside its
 * original `Plain` carrier — see that header's own docstring.
 * `dtype/Fetch.h` also joins `aether/device.h` (see that header's own
 * docstring).
 *
 * `View::component<I>` (`view/View.h` — a writable rank-1 strided view
 * of static component `I` of a rank-2 view); `aether::simd::select`
 * (`backend/cpu/simd/Packet.h` — the lane-wise blend); and
 * `aether::copyAsync(View, View, Stream)` (`view/Copy.h` — view-level
 * transfer over the existing `Chunk`-pair machinery in `chunk/Copy.h`,
 * CUDA-only like every other `Stream`-taking call in this library).
 */

#include "aether/macros.h"

#include "aether/device.h"

#include "aether/accum/AccumPlane.h"
#include "aether/accum/AppendCounter.h"
#include "aether/accum/atomic.h"
#include "aether/array/Array.h"
#include "aether/backend/cpu/Tiled.h"
#include "aether/backend/cuda/DeviceBundle.h"
#include "aether/backend/cuda/Launch.h"
#include "aether/backend/cuda/bundle/Assign.h"
#include "aether/backend/cuda/bundle/LoadStore.h"
#include "aether/chunk/Chunk.h"
#include "aether/chunk/Copy.h"
#include "aether/chunk/TextureBinding.h"
#include "aether/dtype/Fetch.h"
#include "aether/err/DeviceFlag.h"
#include "aether/err/Error.h"
#include "aether/eval/Runtime.h"
#include "aether/expr/nodes/Arithmetic.h"
#include "aether/expr/nodes/Constant.h"
#include "aether/expr/nodes/Elementwise.h"
#include "aether/expr/nodes/Geometric.h"
#include "aether/expr/nodes/Inverse.h"
#include "aether/expr/nodes/Product.h"
#include "aether/expr/nodes/Quaternion.h"
#include "aether/expr/nodes/Structural.h"
#include "aether/interop/Buffer.h"
#include "aether/interop/DLPack.h"
#include "aether/math/math.h"
#include "aether/random/random.h"
#include "aether/residency/Partition.h"
#include "aether/residency/PartitionedArray.h"
#include "aether/residency/Replica.h"
#include "aether/residency/Transport.h"
#include "aether/typedefs.h"
#include "aether/version.h"
#include "aether/view/Copy.h"
#include "aether/view/MakeReshape.h"
#include "aether/view/MakeRuntimeView.h"
#include "aether/view/MakeTableView.h"
#include "aether/view/MakeView.h"
#include "aether/view/Tile.h"

namespace aether {

/** @brief Major version component of this aether checkout. */
inline constexpr int version_major = 0;

// ---------------------------------------------------------------------------
// Consumer alias surface.
// ---------------------------------------------------------------------------

/** @brief `Item<T,N>` — a register-resident N-component vector. */
template<class T, std::size_t N>
using Vec = Item<T, N>;
/** @brief `Item<T,R,C>` — a register-resident R x C matrix. */
template<class T, std::size_t R, std::size_t C>
using Mat = Item<T, R, C>;

/** @brief Same type as `eagle::Vec3R` (`eagle/typedefs.h`) under a different name. */
using Vec3d = Vec<double, 3>;
using Vec4d = Vec<double, 4>;
using Vec6d = Vec<double, 6>;

using Mat33d = Mat<double, 3, 3>;
using Mat36d = Mat<double, 3, 6>;

/** @brief Batched scalar view — every `*View` alias appends the dynamic SAMPLE mode. */
template<class T>
using ScalarView = View<T, extents<dyn>>;
/** @brief Batched N-component vector view. */
template<class T, std::size_t N>
using VecView = View<T, extents<N, dyn>>;
/** @brief Batched R x C matrix view. */
template<class T, std::size_t R, std::size_t C>
using MatView = View<T, extents<R, C, dyn>>;

using Vec3dView = VecView<double, 3>;
using Mat33dView = MatView<double, 3, 3>;

} // namespace aether
