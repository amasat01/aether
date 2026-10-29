// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file geometry.h
 * @brief Umbrella header for `aether::geometry`: ray intersection primitives,
 *        identical on host and device.
 *
 *  - `rayTriangle` (Moller-Trumbore), `rayAabb` (slab) and `raySphere`
 *    (`Ray.h`): each returns one `Vec4` of `[hit, t, u, v]`-style values.
 */

#include "aether/geometry/Ray.h"
