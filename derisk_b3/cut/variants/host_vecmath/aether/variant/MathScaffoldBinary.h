// Variant host: binary scaffold (host arm), moved from MathDispatch.h. No #pragma once on purpose.
#define AETHER_MATH_BINARY(NAME, FLOATFN, DOUBLEFN, STDEXPR)                                                         \
    template<class T>                                                                                                \
        requires(!aether::banded::IsBandedDoor<T> && !std::is_integral_v<T>)                                         \
    AETHER_MATH_HOST_ABI AETHER_DEVICEHOST() AETHER_FORCEINLINE() T NAME(T a, T b)                                                        \
    {                                                                                                                \
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,                                        \
            "aether::math::" #NAME ": T must be float or double");                                                  \
        return STDEXPR;                                                                                              \
    }
