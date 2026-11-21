// The scalar a kernel is written against, the role types of its planes and the math it may call.
// The arm is chosen at build: the normal arm computes with double, the trace arm builds the IR.
#pragma once
#include <cmath>

#ifdef AETHER_TRACE
#include "aether_probe/trace.h"
#endif

namespace aether {

#ifdef AETHER_TRACE
using Real = trace::Real;
using Int = trace::Int;
namespace math {
inline Real abs(const Real& x) { return Real(trace::makeOp("abs", {x.node()})); }
inline Real sqrt(const Real& x) { return Real(trace::makeOp("sqrt", {x.node()})); }
} // namespace math
#else
using Real = double;
using Int = int;
namespace math {
inline Real abs(Real x) { return std::fabs(x); }
inline Real sqrt(Real x) { return std::sqrt(x); }
} // namespace math
#endif

#ifdef AETHER_TRACE
/// The condition of a `while` with the most iterations it may run.
inline trace::Bool bounded(trace::Bool condition, long) { return condition; }
#endif
inline bool bounded(bool condition, long) { return condition; }

/// One value per sample, read. A role type is the scalar itself: the body computes with plain
/// scalars, and the role is read from the kernel's declaration.
template <class T>
using PerSample = T;

/// One value for the whole run, read.
template <class T>
using Uniform = T;

/// One value per sample, written.
template <class T>
using Mutable = T&;

} // namespace aether
