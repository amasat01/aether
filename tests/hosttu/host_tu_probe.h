// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tests/hosttu/host_tu_probe.h — layout-identity harness, the SHARED
// surface both compilers see.
//
// A CUDA-mode aether build may contain translation units compiled by the
// HOST compiler (`g++`) as well as by nvcc, and both must see the SAME
// types with the SAME layout. That is a GATE, not a promise:
// `host_tu_probe.cpp` is compiled by g++ (tests/CMakeLists.txt forces
// `LANGUAGE CXX` on it inside the CUDA build) and linked into the same
// `aether_tests` binary as `test_HostTuLayout.cu`, which nvcc compiles.
//
// TWO INDEPENDENT CHECKS, both driven from THIS header so the two sides
// cannot drift apart by editing one and forgetting the other:
//
//   1. THE TABLE. `AETHER_R8_LAYOUT_TYPES(X)` below is the ONE list of
//      public types. Each side expands it in its OWN non-inline, uniquely
//      named function (`hostFillLayoutTable` / the `.cu`'s local filler) —
//      deliberately NOT an `inline` helper in this header, because a weak
//      COMDAT body would be emitted once and deduped by the linker, and the
//      "comparison" would then compare one compiler's table against itself.
//   2. THE OBJECTS. The host TU allocates real `Array`s, fills and uploads
//      them, and hands back both the objects and a POD summary of what IT
//      believes their pointers / strides / extents are. The `.cu` re-reads
//      the same objects with nvcc's view of the same headers and must agree
//      — and then launches a kernel over data a g++ TU uploaded.

#include <cstddef>
#include <cstdint>

#include <aether/aether.h>

namespace aether_r8 {

/**
 * @brief The public types whose `sizeof`/`alignof` must agree between the
 *        host compiler and nvcc. Add a row here and BOTH sides gain it.
 *
 * ★ Every consumer macro `X` must be VARIADIC (`X(...)`, using
 * `__VA_ARGS__`): several rows are templates whose argument lists contain
 * commas, which the preprocessor splits into separate macro arguments.
 *
 * Deliberately broad: the value types a consumer actually passes across a
 * TU boundary (views, indices, descriptors), the owning containers whose
 * members those views are carved from, and `Fetch<double>::type` — that
 * last one is `int2`, a CUDA vector type a host TU has no implicit
 * vocabulary for, and is exactly the member-set trap this harness exists
 * to catch rather than fence away (`aether/dtype/Fetch.h`).
 */
#define AETHER_R8_LAYOUT_TYPES(X)                                                                                      \
    X(aether::Device)                                                                                                  \
    X(aether::Chunk)                                                                                                   \
    X(aether::DType)                                                                                                   \
    X(aether::SampleIndex)                                                                                             \
    X(aether::BundleIndex<4>)                                                                                          \
    X(aether::RuntimeView)                                                                                             \
    X(aether::DeviceFlagView)                                                                                          \
    X(aether::DeviceFlag)                                                                                              \
    X(aether::extents<3, aether::dyn>)                                                                                  \
    X(aether::layout_right::mapping<aether::extents<3, aether::dyn>>)                                                   \
    X(aether::layout_stride::mapping<aether::extents<3, aether::dyn>>)                                                  \
    X(aether::View<double, aether::extents<3, aether::dyn>, aether::layout_right>)                                      \
    X(aether::View<const double, aether::extents<3, aether::dyn>, aether::layout_stride>)                               \
    X(aether::Array<float>)                                                                                            \
    X(aether::Array<float, 3>)                                                                                         \
    X(aether::Array<double, 3>)                                                                                        \
    X(aether::Array<float>::ViewT)                                                                                     \
    X(aether::Array<float, 3>::ViewT)                                                                                  \
    X(aether::Array<float, 3>::ConstViewT)                                                                             \
    X(aether::Stream)                                                                                                  \
    X(aether::texture_handle_t)                                                                                        \
    X(aether::dtype::Fetch<double>::type)                                                                              \
    X(aether::dtype::Fetch<float>::type)

/** @brief One `sizeof`/`alignof` measurement. Trivially copyable POD: it
 *         crosses the TU boundary itself. */
struct LayoutRow {
    const char* name;
    std::uint64_t size;
    std::uint64_t align;
};

/** @brief The row LABELS, stringised straight from the one list. Shared by
 *         both sides (only the measurements are compared, not the names). */
inline constexpr const char* kLayoutNames[] = {
#define AETHER_R8_NAME_ONE(...) #__VA_ARGS__,
    AETHER_R8_LAYOUT_TYPES(AETHER_R8_NAME_ONE)
#undef AETHER_R8_NAME_ONE
};

/** @brief Row count of `AETHER_R8_LAYOUT_TYPES`, derived from the list — so
 *         adding a row cannot leave a stale count behind. */
inline constexpr std::size_t kLayoutRowCount = sizeof(kLayoutNames) / sizeof(kLayoutNames[0]);

/** @brief Fills `out[0..kLayoutRowCount)` as the HOST COMPILER measures the
 *         list. Defined (non-inline, external linkage) in
 *         `host_tu_probe.cpp` — never here, so nvcc cannot supply its own
 *         body and the comparison stays a real cross-compiler one. */
void hostFillLayoutTable(LayoutRow* out);

/** @brief What the host TU believes about the `Array`s it built — the
 *         `.cu` recomputes every field from the SAME objects and must
 *         match. A pointer or a stride that disagrees IS a layout
 *         disagreement. */
struct HostArraySummary {
    const float* scalarHostData;
    const float* scalarDeviceData;
    const float* vectorHostData;
    const float* vectorDeviceData;
    std::uint64_t samples;
    std::uint64_t capacity;
    std::uint64_t vectorRank;
    std::int64_t vectorStride0;
    std::int64_t vectorStride1;
    std::uint64_t vectorExtent0;
    std::uint64_t vectorExtent1;
    std::uint64_t arrayFloatSize;   ///< `sizeof(Array<float>)` as g++ sees it.
    std::uint64_t arrayVec3Size;    ///< `sizeof(Array<float,3>)` as g++ sees it.
};

/** @brief Opaque handle to the `Array`s the host TU owns. The `.cu` never
 *         constructs or destroys them — that is the point: the objects are
 *         BUILT by g++ and USED by nvcc. */
struct HostOwned;

/**
 * @brief Allocate `Array<float>(n)` + `Array<float,3>(n)` in the HOST TU,
 *        fill the host side with `value(c,i) = 1000*c + i`, `upload()` on
 *        a stream the host TU also creates, synchronize, and report what it
 *        measured. Returns `nullptr` never (throws `aether::Error` on an
 *        allocation failure, exactly as any other host caller would).
 */
HostOwned* hostMakeArrays(std::size_t n, HostArraySummary* summary);

/** @brief Release everything `hostMakeArrays` allocated, in the same TU
 *         that allocated it. */
void hostDestroyArrays(HostOwned* owned);

/** @brief The `Array<float,3>`'s device view, returned BY VALUE across the
 *         TU boundary — a g++-built `View` object consumed by nvcc code
 *         (and, in `test_HostTuLayout.cu`, handed straight to a kernel). */
aether::Array<float, 3>::ViewT hostVectorDeviceView(HostOwned* owned);

/** @brief The `Array<float,3>`'s host view, same boundary crossing. */
aether::Array<float, 3>::ViewT hostVectorHostView(HostOwned* owned);

/** @brief `download()` the vector array through the host TU's own stream,
 *         so the `.cu` can read back what a kernel wrote. */
void hostDownloadVector(HostOwned* owned);

/** @brief A `aether::Device(kDLCUDA, 0)` built by the host TU. */
aether::Device hostMakeCudaDevice();

/** @brief The stream the host TU created (`aether::Stream`, i.e.
 *         `cudaStream_t`), handed to nvcc code. */
aether::Stream hostStream(HostOwned* owned);

} // namespace aether_r8
