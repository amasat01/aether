// specialisations moved verbatim from device/Device.h
#ifdef AETHER_HAS_CUDA
template<>
inline constexpr bool backend_enabled<kDLCUDA> = true;
template<>
inline constexpr bool backend_enabled<kDLCUDAHost> = true;
#endif
