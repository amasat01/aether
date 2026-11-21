// What aether's math needs to know of the tracing scalar and of the role types of planes, by
// declaration only: the definitions live with the trace arm and with the role types.
#pragma once
#include <type_traits>

namespace aether {
namespace trace {
class Var;
/// A scalar of the trace arm: its operations build IR nodes.
template <class T>
concept IsTraced = std::is_base_of_v<Var, T>;
} // namespace trace

/// A role type of a plane: it names the scalar it reads as.
template <class T>
concept IsPlane = requires { typename T::Scalar; };

namespace math::detail {
/// The scalar an argument of a math call reads as: a plane's scalar, else the argument's own type.
template <class T>
struct ScalarOf {
    using type = T;
};
template <IsPlane T>
struct ScalarOf<T> {
    using type = typename T::Scalar;
};
template <class A, class B>
using CommonScalar = std::common_type_t<typename ScalarOf<A>::type, typename ScalarOf<B>::type>;
/// Whether a two-argument call needs its arguments brought to one scalar first.
template <class A, class B>
concept Mixed = (IsPlane<A> || IsPlane<B> || trace::IsTraced<A> || trace::IsTraced<B>) &&
                (IsPlane<A> || IsPlane<B> || !std::is_same_v<A, B>);
} // namespace math::detail
} // namespace aether
