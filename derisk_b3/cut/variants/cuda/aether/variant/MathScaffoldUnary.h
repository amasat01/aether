// Variant cuda: unary scaffold (device arm), moved from MathDispatch.h.
#define AETHER_MATH_UNARY(NAME, FLOATFN, DOUBLEFN, STDEXPR)                                                          \
    template<class T>                                                                                                \
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() T NAME(T x)                                                             \
    {                                                                                                                \
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,                                        \
            "aether::math::" #NAME ": T must be float or double");                                                  \
        if constexpr (std::is_same_v<T, float>) {                                                                    \
            return FLOATFN(x);                                                                                       \
        } else {                                                                                                      \
            return DOUBLEFN(x);                                                                                      \
        }                                                                                                            \
    }
