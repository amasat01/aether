// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Fft.h
 * @brief `aether::fft`: batched fast Fourier transforms on the host, over interleaved (re, im) pairs.
 *
 * Complex data is a plain `float` or `double` array whose last axis has extent 2 (real part, imaginary
 * part), so there is no complex dtype: the layout is that of `std::complex<T>`, cuFFT's `float2` and
 * `torch.view_as_complex`. Arrays are dense and row-major. `shape` counts complex (or real, for the real
 * side of a real transform) elements and never includes the trailing 2.
 *
 * The engine is pocketfft (vendored in `pocketfft_hdronly.h`, BSD-3-Clause), driven over the independent lines of
 * each transformed axis (the batch, and the other axes of an N-D transform) in chunks. This header owns no threads:
 * each public entry takes the loop that runs the chunks as its last argument, a callable
 * `loop(std::size_t count, F&& chunkBody)` that must call `chunkBody(c)` exactly once for every `c` in `[0, count)`,
 * in any order, possibly concurrently, and return only when all calls are done. It need not catch exceptions: the
 * chunk body never throws, `forChunks` catches inside each chunk and rethrows the first failure after the loop
 * returns. `SerialLoop`, the default, runs the chunks in order on the calling thread. The split into chunks depends
 * only on the shape, so the result is bit-identical for any loop.
 *
 * Normalisation follows numpy's `norm` argument, stated once here:
 *  - `Norm::Backward` (the default): the forward transform is unscaled, the inverse is scaled by `1/n`;
 *  - `Norm::Forward`: the forward transform is scaled by `1/n`, the inverse is unscaled;
 *  - `Norm::Ortho`: both are scaled by `1/sqrt(n)`;
 * where `n` is the product of the lengths of the transformed axes (for a real transform, the lengths of the
 * REAL side). The adjoint of an unscaled transform is the unscaled transform of the other direction, i.e.
 * the other direction with the opposite `Norm::Backward` / `Norm::Forward`.
 *
 * Failures throw `aether::Error`: an empty or repeated axis list, an axis out of range, an extent
 * mismatch. In-place `c2c` (`in == out`) is allowed; the real transforms need distinct buffers.
 */

#include <cmath>
#include <complex>
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <span>
#include <type_traits>
#include <vector>

#include "aether/err/Error.h"

// plans are cached (thread-safe) and shared by the chunks below
#define POCKETFFT_NO_MULTITHREADING 1
#define POCKETFFT_CACHE_SIZE 16
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "aether/fft/pocketfft_hdronly.h"
#pragma GCC diagnostic pop

namespace aether::fft {

enum class Norm { Backward, Forward, Ortho };

/// The loop used when a caller gives none: runs the chunks in order on the calling thread.
struct SerialLoop {
    template <class F> void operator()(std::size_t count, F&& chunkBody) const {
        for (std::size_t c = 0; c < count; ++c) chunkBody(c);
    }
};

namespace detail {

inline void check(std::span<const std::size_t> shape, std::span<const std::size_t> axes) {
    if (axes.empty()) throw aether::Error("fft: no axes to transform");
    std::vector<bool> seen(shape.size(), false);
    for (std::size_t a : axes) {
        if (a >= shape.size()) throw aether::Error("fft: axis out of range");
        if (seen[a]) throw aether::Error("fft: repeated axis");
        seen[a] = true;
    }
}

/// the scale factor of one transform over `n` points
template <class T> T scale(Norm norm, bool forward, std::size_t n) {
    if (norm == Norm::Ortho) return T(1) / std::sqrt(T(n));
    return forward == (norm == Norm::Forward) ? T(1) / T(n) : T(1);
}

/// the product of the lengths of the transformed axes (`shape` is the REAL side for a real transform)
inline std::size_t points(std::span<const std::size_t> shape, std::span<const std::size_t> axes) {
    std::size_t n = 1;
    for (std::size_t a : axes) n *= shape[a];
    return n;
}

/// byte strides of a dense row-major array whose elements are `item` bytes
inline pocketfft::stride_t strides(std::span<const std::size_t> shape, std::size_t item) {
    pocketfft::stride_t s(shape.size());
    std::ptrdiff_t acc = std::ptrdiff_t(item);
    for (std::size_t i = shape.size(); i-- > 0;) {
        s[i] = acc;
        acc *= std::ptrdiff_t(shape[i]);
    }
    return s;
}

inline pocketfft::shape_t vec(std::span<const std::size_t> v) { return pocketfft::shape_t(v.begin(), v.end()); }

/// Lines per work chunk. The decomposition below depends only on the shape, never on the loop, so the result is
/// bit-identical for any loop.
inline constexpr std::size_t kChunkLines = 16;

/// Runs `body(sub_shape, byte_offset_of(dim, begin))` over independent chunks of the lines of a transform along `axis`
/// of an array of `shape`. A chunk is a rectangular block: the non-`axis` dimension with the largest extent is cut
/// into pieces of about `kChunkLines` lines (the other non-`axis` dimensions stay whole). `loop` runs the chunks (see
/// the file comment); a failure inside one chunk is caught there, the first is rethrown after the loop returns.
template <class Loop, class Body> void forChunks(const pocketfft::shape_t& shape, std::size_t axis, Loop& loop, Body&& body) {
    std::size_t dim = shape.size(), best = 0, lines = 1;
    for (std::size_t d = 0; d < shape.size(); ++d) {
        if (d == axis) continue;
        lines *= shape[d];
        if (shape[d] > best) { best = shape[d]; dim = d; }
    }
    if (lines == 0 || shape[axis] == 0) return;
    if (dim == shape.size()) { body(shape, std::size_t(0), std::size_t(0), std::size_t(shape.size())); return; }   // one line
    const std::size_t others = lines / shape[dim];
    const std::size_t piece = std::max<std::size_t>(1, kChunkLines / others);
    const std::size_t chunks = (shape[dim] + piece - 1) / piece;
    // the flag admits one writer of `failure`; the loop's return orders that write before the read below
    std::atomic_flag failed;
    std::exception_ptr failure;
    loop(chunks, [&](std::size_t c) {
        try {
            std::size_t begin = c * piece, count = std::min(piece, shape[dim] - begin);
            auto sub = shape;
            sub[dim] = count;
            body(sub, begin, count, dim);
        } catch (...) {
            if (!failed.test_and_set()) failure = std::current_exception();
        }
    });
    if (failure) std::rethrow_exception(failure);
}

template <class T> const std::complex<T>* at(const std::complex<T>* p, std::ptrdiff_t bytes) {
    return reinterpret_cast<const std::complex<T>*>(reinterpret_cast<const char*>(p) + bytes);
}
template <class T> std::complex<T>* at(std::complex<T>* p, std::ptrdiff_t bytes) {
    return reinterpret_cast<std::complex<T>*>(reinterpret_cast<char*>(p) + bytes);
}
template <class T> const T* atr(const T* p, std::ptrdiff_t bytes) {
    return reinterpret_cast<const T*>(reinterpret_cast<const char*>(p) + bytes);
}
template <class T> T* atr(T* p, std::ptrdiff_t bytes) { return reinterpret_cast<T*>(reinterpret_cast<char*>(p) + bytes); }

/// c2c along one `axis` of a dense array of `shape`, chunked, run by `loop`; `in == out` is allowed
template <class T, class Loop>
void c2cPass(const std::complex<T>* in, std::complex<T>* out, const pocketfft::shape_t& shape, std::size_t axis, bool forward, T fct, Loop& loop) {
    auto st = strides(shape, sizeof(std::complex<T>));
    forChunks(shape, axis, loop, [&](const pocketfft::shape_t& sub, std::size_t begin, std::size_t, std::size_t dim) {
        std::ptrdiff_t off = dim < shape.size() ? std::ptrdiff_t(begin) * st[dim] : 0;
        pocketfft::c2c(sub, st, st, pocketfft::shape_t{axis}, forward, at(in, off), at(out, off), fct);
    });
}

/// r2c along `axis`: real `shape` in, complex out whose `axis` extent is `n / 2 + 1`
template <class T, class Loop> void r2cPass(const T* in, std::complex<T>* out, const pocketfft::shape_t& shape, std::size_t axis, T fct, Loop& loop) {
    pocketfft::shape_t half = shape;
    half[axis] = shape[axis] / 2 + 1;
    auto sti = strides(shape, sizeof(T)), sto = strides(half, sizeof(std::complex<T>));
    forChunks(shape, axis, loop, [&](const pocketfft::shape_t& sub, std::size_t begin, std::size_t, std::size_t dim) {
        std::ptrdiff_t oi = dim < shape.size() ? std::ptrdiff_t(begin) * sti[dim] : 0;
        std::ptrdiff_t oo = dim < shape.size() ? std::ptrdiff_t(begin) * sto[dim] : 0;
        pocketfft::r2c(sub, sti, sto, axis, true, atr(in, oi), at(out, oo), fct);
    });
}

/// c2r along `axis`: complex in (extent `n / 2 + 1`), real `shape` out; the input is left unchanged
template <class T, class Loop> void c2rPass(const std::complex<T>* in, T* out, const pocketfft::shape_t& shape, std::size_t axis, T fct, Loop& loop) {
    pocketfft::shape_t half = shape;
    half[axis] = shape[axis] / 2 + 1;
    auto sti = strides(half, sizeof(std::complex<T>)), sto = strides(shape, sizeof(T));
    forChunks(shape, axis, loop, [&](const pocketfft::shape_t& sub, std::size_t begin, std::size_t, std::size_t dim) {
        std::ptrdiff_t oi = dim < shape.size() ? std::ptrdiff_t(begin) * sti[dim] : 0;
        std::ptrdiff_t oo = dim < shape.size() ? std::ptrdiff_t(begin) * sto[dim] : 0;
        pocketfft::c2r(sub, sti, sto, axis, false, at(in, oi), atr(out, oo), fct);
    });
}

}  // namespace detail

/// Complex to complex over `axes`: `in` and `out` hold `prod(shape)` pairs. `loop` runs the chunks (see the file
/// comment); the result is bit-identical for every loop.
template <class T, class Loop = SerialLoop>
void c2c(const T* in, T* out, std::span<const std::size_t> shape, std::span<const std::size_t> axes, bool forward,
         Norm norm = Norm::Backward, Loop&& loop = {}) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    detail::check(shape, axes);
    const T fct = detail::scale<T>(norm, forward, detail::points(shape, axes));
    auto sh = detail::vec(shape);
    auto src = reinterpret_cast<const std::complex<T>*>(in);
    auto dst = reinterpret_cast<std::complex<T>*>(out);
    for (std::size_t i = 0; i < axes.size(); ++i)
        detail::c2cPass<T>(i == 0 ? src : dst, dst, sh, axes[i], forward, i + 1 == axes.size() ? fct : T(1), loop);
}

/// Real to complex (forward) over `axes`. `shape` is the REAL input; the last of `axes` is the halved one: `out`
/// holds the shape with that extent replaced by `n / 2 + 1`, as pairs. `loop` as in `c2c`.
template <class T, class Loop = SerialLoop>
void r2c(const T* in, T* out, std::span<const std::size_t> shape, std::span<const std::size_t> axes,
         Norm norm = Norm::Backward, Loop&& loop = {}) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    detail::check(shape, axes);
    const T fct = detail::scale<T>(norm, true, detail::points(shape, axes));
    auto sh = detail::vec(shape);
    auto dst = reinterpret_cast<std::complex<T>*>(out);
    detail::r2cPass<T>(in, dst, sh, axes.back(), axes.size() == 1 ? fct : T(1), loop);
    sh[axes.back()] = shape[axes.back()] / 2 + 1;
    for (std::size_t i = 0; i + 1 < axes.size(); ++i)
        detail::c2cPass<T>(dst, dst, sh, axes[i], true, i + 2 == axes.size() ? fct : T(1), loop);
}

/// Complex to real (inverse) over `axes`. `shape` is the REAL output; the last of `axes` is the halved one: `in`
/// holds the shape with that extent replaced by `n / 2 + 1`, as pairs. The input is left unchanged. `loop` as in `c2c`.
template <class T, class Loop = SerialLoop>
void c2r(const T* in, T* out, std::span<const std::size_t> shape, std::span<const std::size_t> axes,
         Norm norm = Norm::Backward, Loop&& loop = {}) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    detail::check(shape, axes);
    const T fct = detail::scale<T>(norm, false, detail::points(shape, axes));
    auto src = reinterpret_cast<const std::complex<T>*>(in);
    auto half = detail::vec(shape);
    half[axes.back()] = shape[axes.back()] / 2 + 1;
    std::vector<std::complex<T>> work;
    if (axes.size() > 1) {
        std::size_t total = 1;
        for (auto v : half) total *= v;
        work.resize(total);
        for (std::size_t i = 0; i + 1 < axes.size(); ++i)
            detail::c2cPass<T>(i == 0 ? src : work.data(), work.data(), half, axes[i], false, T(1), loop);
        src = work.data();
    }
    detail::c2rPass<T>(src, out, detail::vec(shape), axes.back(), fct, loop);
}

}  // namespace aether::fft
