// Variant host: std route (the default arm of the old HostVectorMathRoute.h).
#define AETHER_MATH_HOST_ROUTE(NAME, ...) std::NAME(__VA_ARGS__)
#define AETHER_MATH_HOST_ROUTE_OR(NAME, FALLBACK, ...) FALLBACK
#define AETHER_MATH_HOST_ABI
