// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file golden_linalg.h
 * @brief Reference values for the `aether::linalg` tests, taken from LAPACK:
 *        `dgetrf` and `dpotrf` (scipy.linalg.lapack) for the factors, pivots
 *        and `info`, `numpy.linalg.solve/inv/det/eigh` for the rest. Row-major
 *        throughout; eigenvectors carry aether's sign convention (largest
 *        component positive).
 */

namespace aether_test_linalg {

// 4x4 with row exchanges: matrix, right-hand side, packed LU, 0-based pivots, solution, determinant, inverse
inline constexpr double kA4[16] = {2.0, 1.0, -1.0, 3.0, 4.0, -2.0, 5.0, 1.0, -6.0, 3.0, 1.0, 2.0, 1.0, 7.0, -3.0, -4.0};
inline constexpr double kB4[4] = {1.0, -2.0, 3.0, 0.5};
inline constexpr double kLu4[16] = {-6.0, 3.0, 1.0, 2.0, -0.16666666666666666, 7.5, -2.8333333333333335, -3.6666666666666665, -0.6666666666666666, 0.0, 5.666666666666667, 2.333333333333333, -0.3333333333333333, 0.26666666666666666, 0.015686274509803904, 4.607843137254902};
inline constexpr int kPiv4[4] = {2, 3, 3, 3};
inline constexpr double kX4[4] = {-0.2710638297872341, 0.25872340425531914, -0.1548936170212766, 0.37617021276595747};
inline constexpr double kDet4 = 1175.0;
inline constexpr double kInv4[16] = {0.09361702127659574, 0.06127659574468085, -0.0876595744680851, 0.04170212765957446, 0.0723404255319149, 0.06553191489361702, 0.08680851063829786, 0.11404255319148936, -0.08936170212765956, 0.1778723404255319, 0.09276595744680849, 0.023829787234042547, 0.2170212765957447, -0.0034042553191489327, 0.060425531914893624, -0.05787234042553192};

// Singular matrices with the LAPACK dgetrf info
inline constexpr double kSingA[4] = {1.0, 2.0, 2.0, 4.0};
inline constexpr int kSingAInfo = 2;
inline constexpr double kSingB[4] = {0.0, 0.0, 0.0, 1.0};
inline constexpr int kSingBInfo = 1;
inline constexpr double kSingC[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
inline constexpr int kSingCInfo = 3;
inline constexpr double kSingD[9] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
inline constexpr int kSingDInfo = 2;

// SPD 3x3: matrix, lower factor, solution of S x = b3
inline constexpr double kS3[9] = {4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0};
inline constexpr double kB3[3] = {1.0, -2.0, 3.0};
inline constexpr double kChol3[9] = {2.0, 0.0, 0.0, 6.0, 1.0, 0.0, -8.0, 5.0, 3.0};
inline constexpr double kChoX3[3] = {82.80555555555569, -22.777777777777814, 3.555555555555561};

// Not positive definite, with the LAPACK dpotrf info
inline constexpr double kNpdA[4] = {1.0, 2.0, 2.0, 1.0};
inline constexpr int kNpdAInfo = 2;
inline constexpr double kNpdB[4] = {-1.0, 0.0, 0.0, 1.0};
inline constexpr int kNpdBInfo = 1;
inline constexpr double kNpdC[9] = {4.0, 2.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0};
inline constexpr int kNpdCInfo = 2;
inline constexpr double kNpdD[4] = {0.0, 0.0, 0.0, 0.0};
inline constexpr int kNpdDInfo = 1;

// Symmetric matrices: eigenvalues ascending, eigenvectors as columns, largest component positive
inline constexpr double kE4[16] = {4.0, 1.0, -2.0, 2.0, 1.0, 2.0, 0.0, 1.0, -2.0, 0.0, 3.0, -2.0, 2.0, 1.0, -2.0, -1.0};
inline constexpr double kE4W[4] = {-2.1975169774394243, 1.0843644637732166, 2.2685314064312423, 6.844621107234965};
inline constexpr double kE4V[16] = {-0.17670517062447652, 0.6422600067660328, 0.20171109662094233, 0.7180459594506796, -0.17810046990630068, -0.5441878481946866, 0.7894499124874397, 0.2211529881551807, 0.2876680343197011, 0.5202218509386148, 0.5796341693167879, -0.5573513771375096, 0.9242849167461131, -0.14398227457619123, 0.010280998686330534, 0.3533564748939929};
inline constexpr double kE3[9] = {2.0, -1.0, 0.5, -1.0, 3.0, -1.0, 0.5, -1.0, 1.0};
inline constexpr double kE3W[3] = {0.5793889649749694, 1.3925808818720364, 4.028030153152994};
inline constexpr double kE3V[9] = {-0.08041679924024196, 0.8723860596385582, -0.4821573408631922, 0.3520240156207673, 0.477409601540651, 0.8050833278506182, 0.9325300160456951, -0.10498873894426285, -0.3454925959680018};

} // namespace aether_test_linalg
