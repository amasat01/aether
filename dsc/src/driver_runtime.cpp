// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0
// The CUDA runtime functions the plugin's headers call, over the driver API (see DriverRuntime.inl). A plain C++
// translation unit: nvcc's own front end would clash with the runtime names this file defines.
#include "aether/backend/cuda/DriverRuntime.inl"
