// Variant cuda: device never routes to the host vector math.
#define AETHER_MATH_HOST_ROUTE(NAME, ...) std::NAME(__VA_ARGS__)
#define AETHER_MATH_HOST_ROUTE_OR(NAME, FALLBACK, ...) FALLBACK
#define AETHER_MATH_HOST_ABI
