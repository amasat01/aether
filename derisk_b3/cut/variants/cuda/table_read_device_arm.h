// tex1Dfetch body moved verbatim from view/TableHandle.h (was #if defined(__CUDA_ARCH__) && !defined(AETHER_CPP_MODE))
#if 0
#if defined(__CUDA_ARCH__) && !defined(AETHER_CPP_MODE)
        using FetchT = typename dtype::Fetch<T>::type;
        const FetchT r = tex1Dfetch<FetchT>(tex_, static_cast<int>(idx + offset_));
        if constexpr (dtype::Fetch<T>::needsConversion) {
            // Only `double` (FetchT == int2) takes this branch today — see
            // dtype/Fetch.h. `x` = low word, `y` = high word.
            return __hiloint2double(r.y, r.x);
        } else {
            return r;
        }
#else
        return ptr_[idx + offset_];
#endif
#endif
