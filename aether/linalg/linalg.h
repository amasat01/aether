// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file linalg.h
 * @brief Umbrella header for `aether::linalg`: per-lane dense linear algebra
 *        on small static matrices, identical on host and device.
 *
 * One lane owns its matrices: row-major `N x N` arrays, `float` or `double`,
 * `1 <= N <= 32`, no allocation and no data-dependent control flow beyond the
 * factorisation's own pivot search. Plain `a * b + c` arithmetic in a fixed
 * order, with no compensation inside a factorisation, so a target that does
 * not contract products gets the same bits on every path.
 *
 *  - `lu` / `luSolve` / `luSolveT` / `luDet` / `luInverse` / `solve` / `inv` / `det`
 *    (`Lu.h`): `P A = L U`, `info` = LAPACK `getrf` numbering.
 *  - `cholesky` / `choleskyFactor` / `choSolve` (`Cholesky.h`): `S = L L^T` of
 *    the symmetrised input, `info` = LAPACK `potrf` numbering.
 *  - `triLower` / `triUpperT` (`Triangular.h`): multi-right-hand-side solves
 *    against `L` and `L^T`, the pieces `choSolve` and the factor's derivative
 *    are made of.
 *  - `eigh` (`Eigh.h`): cyclic Jacobi for the symmetrised input, fixed sweeps.
 *
 * Matrices and right-hand sides are row-major pointers or per-sample operands
 * (`view[i]`, `Item`): a plane is read and written element by element, never
 * copied whole, and a right-hand side is solved one column at a time. Every
 * loop is kept rolled, so the working set sits in local memory at any size.
 *
 * Matrices too large for one lane (`N` up to 4096) are worked on by a team of
 * threads in caller-provided scratch: `aether/linalg/team/team.h`.
 *
 * Failure policy: a failed factorisation reports `info > 0` and every solve,
 * inverse and determinant built on it returns zeros.
 */

#include "aether/linalg/Cholesky.h"
#include "aether/linalg/Eigh.h"
#include "aether/linalg/Lu.h"
#include "aether/linalg/Triangular.h"
#include "aether/linalg/team/team.h"
