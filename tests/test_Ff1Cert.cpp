// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_Ff1Cert.cpp
 * @brief `Ff1` (w=24 rung) certification (CPP_MODE): the host DD-oracle certs
 *        (shared `test_Ff1Cert_common.h`), host-computable and registered in
 *        the CPU build.
 *
 * The CPP_MODE twin of `test_Ff1Cert.cu`: it registers the SAME
 * host-computable Ff1 cert suite via the shared common header. Under
 * `AETHER_CPP_MODE` the suite globs `test_*.cpp`, so this is the TU that
 * carries the suite in the CPU build (the `.cu`'s device-parity kernel is
 * CUDA-only and absent here).
 */

#include "test_Ff1Cert_common.h"
