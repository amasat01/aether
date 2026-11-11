// Variant host: no texture fetch types (the AETHER_CPP_MODE arm of dtype/Fetch.h); the handle type is kept.
#pragma once
namespace aether {
using texture_handle_t = unsigned long long;
template<class T>
struct Fetch;
} // namespace aether
