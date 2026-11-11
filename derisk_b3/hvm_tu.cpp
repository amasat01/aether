// Instantiates the host vector math facade on double and float (x86-64 vector function ABI).
#include <aether/math/detail/HostVectorMath.h>
namespace hv = aether::math::detail::hostvec;
double d1(double x) { return hv::exp<double>(x) + hv::log<double>(x) + hv::sin<double>(x) + hv::tanh<double>(x) + hv::cbrt<double>(x); }
double d2(double a, double b) { return hv::pow<double>(a, b) + hv::atan2<double>(a, b) + hv::hypot<double>(a, b) + hv::fdim<double>(a, b); }
float f1(float x) { return hv::exp(x) + hv::log1p(x) + hv::rsqrt(x); }
float f2(float a, float b) { return hv::pow(a, b) + hv::atan2(a, b); }
