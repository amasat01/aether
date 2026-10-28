// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file team.h
 * @brief Umbrella header for `aether::linalg::team`: dense linear algebra on ONE
 *        matrix by a team of threads, for matrices too large for one lane.
 *
 * One thread block (device) or one host thread (the size-1 twin) works on a
 * matrix held in caller-provided global scratch, `1 <= N <= 4096`. Where the
 * per-lane family (`aether/linalg/linalg.h`, `N <= 32`) keeps the matrix in
 * a lane's local memory, this one keeps it in memory the caller owns, so the
 * size is bounded by that memory. Every function is collective (see `Team.h`)
 * and every result is bitwise independent of the team size.
 *
 *  - `lu` / `luSolve` / `luSolveT` / `luDet` / `luInverse` (`Lu.h`).
 *  - `cholesky` / `choleskyFrom` / `choSolve` / `triLower` / `triUpperT` (`Cholesky.h`).
 *  - `eigh` (`Eigh.h`): the one-thread rotation order of `linalg::eigh`.
 *  - `qr` / `qrSolve` / `qrGramSolve` (`Qr.h`): Householder over row tiles, least squares and `(A^T A)^-1 B`.
 *  - `blockSum` (`BlockSum.h`): ordered `sum_r u_r v_r^T` (`J^T J`, `J^T r`).
 */

#include "aether/linalg/team/BlockSum.h"
#include "aether/linalg/team/Cholesky.h"
#include "aether/linalg/team/Eigh.h"
#include "aether/linalg/team/Lu.h"
#include "aether/linalg/team/Qr.h"
#include "aether/linalg/team/Team.h"
