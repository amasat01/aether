// Variant cuda: binary scaffold (device arm), moved from MathDispatch.h.
#define AETHER_MATH_BINARY(NAME, FLOATFN, DOUBLEFN, STDEXPR)                                                         \
    template<class T>                                                                                                \
        requires(!aether::banded::IsBandedDoor<T> && !std::is_integral_v<T>)                                         \
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() T NAME(T a, T b)                                                        \
    {                                                                                                                \
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,                                        \
            "aether::math::" #NAME ": T must be float or double");                                                  \
        if constexpr (std::is_same_v<T, float>) {                                                                    \
            return FLOATFN(a, b);                                                                                    \
        } else {                                                                                                      \
            return DOUBLEFN(a, b);                                                                                   \
        }                                                                                                            \
    }
