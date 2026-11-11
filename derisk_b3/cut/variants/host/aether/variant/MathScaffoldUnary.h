// Variant host: unary scaffold (host arm), moved from MathDispatch.h. No #pragma once on purpose.
#define AETHER_MATH_UNARY(NAME, FLOATFN, DOUBLEFN, STDEXPR)                                                          \
    template<class T>                                                                                                \
    AETHER_MATH_HOST_ABI AETHER_DEVICEHOST() AETHER_FORCEINLINE() T NAME(T x)                                                             \
    {                                                                                                                \
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,                                        \
            "aether::math::" #NAME ": T must be float or double");                                                  \
        return STDEXPR;                                                                                              \
    }
