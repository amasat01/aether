// Kernels that take views: element access at the sample being run, by component, at a computed
// index (gather, scatter, accumulate), and a store by component under a data-dependent branch.
// Written once; the normal arm builds them against the library's own view headers.
#pragma once
#include <cstdint>

#include "aether/expr/Assign.h"
#include "aether/expr/nodes/Constant.h"
#include "aether/index/SampleIndex.h"
#include "aether/view/View.h"
#include "aether_probe/real.h"

namespace physics {

using aether::dyn;
using aether::extents;
using aether::Int;
using aether::Real;
using aether::SampleIndex;
using aether::View;

/// A 3-vector in, a 3-vector out: component reads and component writes.
inline void vscale(const View<const double, extents<3, dyn>> vec, View<double, extents<3, dyn>> scaled,
                   SampleIndex i, aether::Uniform<Real> gain) {
    scaled[i].at(0) = vec[i].eval<0>() * gain;
    scaled[i].at(1) = vec[i].eval<1>() * gain;
    scaled[i].at(2) = vec[i].eval<2>() * gain;
}

/// A per-sample index plane selects entries of a uniform lookup plane.
inline void gather(const View<const std::int32_t, extents<dyn>> row,
                   const View<const double, extents<dyn>> table, View<double, extents<dyn>> out,
                   SampleIndex i) {
    Int r = row[i].eval<>();
    Real first = table(r);
    Real second = table(r + 1);
    out[i] = first + second;
}

/// A store at a computed index.
inline void scatter(const View<const std::int32_t, extents<dyn>> slot,
                    const View<const double, extents<dyn>> value, View<double, extents<dyn>> out,
                    SampleIndex i) {
    out(slot[i].eval<>()) = 2.0 * value[i].eval<>();
}

/// An accumulate at a computed key.
inline void tally(const View<const std::int32_t, extents<dyn>> key,
                  const View<const double, extents<dyn>> value, View<double, extents<dyn>> bins,
                  SampleIndex i) {
    bins(key[i].eval<>()) += value[i].eval<>();
}

/// A component store under a data-dependent branch.
inline void vgate(const View<const double, extents<3, dyn>> vec,
                  const View<const double, extents<dyn>> gate, View<double, extents<3, dyn>> out,
                  SampleIndex i) {
    if (gate[i].eval<>() > 0.5) {
        out[i].at(1) = vec[i].eval<1>() * 2.0;
    }
}

/// A loop over the components of a vector.
inline void vsum(const View<const double, extents<3, dyn>> vec, View<double, extents<dyn>> out,
                 SampleIndex i) {
    Real total = 0.0;
    for (int k = 0; k < 3; ++k) {
        total = total + vec[i].at(k);
    }
    out[i] = total;
}

} // namespace physics
