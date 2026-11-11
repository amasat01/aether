// tex1Dfetch body moved verbatim from view/TableHandle.h (was #if defined(__CUDA_ARCH__) && !defined(AETHER_CPP_MODE)),
// now spelled as the variant's AETHER_TABLE_READ_BODY: the device pass fetches through the texture, the host pass and a
// CPP_MODE build read the pointer.
#pragma once
#if defined(__CUDA_ARCH__) && !defined(AETHER_CPP_MODE)
#define AETHER_TABLE_READ_BODY(T, tex, ptr, idx, off)                                          \
    using FetchT = typename ::aether::dtype::Fetch<T>::type;                                   \
    const FetchT r = tex1Dfetch<FetchT>((tex), static_cast<int>((idx) + (off)));               \
    if constexpr (::aether::dtype::Fetch<T>::needsConversion) {                                \
        /* only double (FetchT == int2) takes this branch: x = low word, y = high word */      \
        return __hiloint2double(r.y, r.x);                                                     \
    } else {                                                                                   \
        return r;                                                                              \
    }
#else
#define AETHER_TABLE_READ_BODY(T, tex, ptr, idx, off) return (ptr)[(idx) + (off)];
#endif
