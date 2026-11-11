// Variant host_vecmath: the vector-ABI arm of the old HostVectorMathRoute.h (pulls the HOST LEAF header).
#include "aether/math/detail/HostVectorMath.h"
#define AETHER_MATH_HOST_ROUTE(NAME, ...) ::aether::math::detail::hostvec::NAME(__VA_ARGS__)
#define AETHER_MATH_HOST_ROUTE_OR(NAME, FALLBACK, ...) ::aether::math::detail::hostvec::NAME(__VA_ARGS__)
#define AETHER_MATH_HOST_ABI [[gnu::abi_tag("aether_hvm")]]
