// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file simd.h
 * @brief Umbrella header for the aether CPU SIMD packet subsystem. Include
 *        this single header to get `Packet`, `PacketMask`, and
 *        `PacketTraits`/`PreferredWidth`.
 *
 * No `Prefetch.h` counterpart yet: the trajectory audit shows only
 * capture/assign are live downstream, so no consumer needs software
 * prefetch yet.
 */

#include "aether/backend/cpu/simd/PacketTraits.h"
#include "aether/backend/cpu/simd/PacketMask.h"
#include "aether/backend/cpu/simd/Packet.h"
