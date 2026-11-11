// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0
// ROOT seam (replaces the 287-line macros.h). The qualifier and hook macros are a NAMED INPUT: the variant's
// header `aether_variant.h` (its include root is the variant). With no variant on the path the neutral
// defaults below hold (every qualifier empty): that is what the root layer parses against.
#pragma once

#if __has_include(<aether_variant.h>)
#include <aether_variant.h>
#else
#define AETHER_DEVICE(...)
#define AETHER_HOST(...)
#define AETHER_DEVICEANDHOST(...)
#define AETHER_DEVICEHOST(...)
#define AETHER_SHARED(...)
#define AETHER_GRID_CONSTANT(...)
#define AETHER_FORCEINLINE(...) inline
#endif

#if defined(__cplusplus) && (__cplusplus >= 202302L)
#define AETHER_HAS_CXX23
#endif
