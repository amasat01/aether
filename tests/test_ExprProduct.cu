// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Matrix product and structural expression tests (CUDA build;
// test_ExprProduct.cpp is the identical host-build twin).
// `MatVec`/`MatMat`/`outer`/`cwiseMul`/`matDot`/`transpose`/`row`/`col`/
// `block` (aether/expr/nodes/{Product,Structural}.h, member functions
// declared on Expression in aether/expr/Expression.h) and
// `Item::Identity()` (aether/view/Item.h) are DEVICEHOST-safe with no
// CUDA-specific behaviour a plain host build cannot already exercise.
// `cwiseMul`/`matDot` (Hadamard product, Frobenius inner product) are
// original tests with a hand-computed reference.

#include <cstddef>
#include <utility>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace {

using aether::Mat33d;
using aether::Mat36d;
using aether::Vec3d;
using aether::Vec6d;
using Mat63d = aether::Mat<double, 6, 3>;
using Mat22d = aether::Mat<double, 2, 2>;

// ---------------------------------------------------------------------
// Compile-time family/dimension trait gates (namespace scope).
// ---------------------------------------------------------------------

using MatVecT = decltype(std::declval<const Mat33d&>() * std::declval<const Vec3d&>());
static_assert(MatVecT::element_extents::Rank == 1, "A*x must be a rank-1 (vector) expression");
static_assert(MatVecT::element_extents::static_extent(0) == 3, "A*x must have length == A's row count");

using MatMatT = decltype(std::declval<const Mat33d&>() * std::declval<const Mat33d&>());
static_assert(MatMatT::element_extents::Rank == 2, "A*B must be a rank-2 (matrix) expression");
static_assert(MatMatT::element_extents::static_extent(0) == 3 && MatMatT::element_extents::static_extent(1) == 3, "A*B dims");

using OuterT = decltype(aether::outer(std::declval<const Vec3d&>(), std::declval<const Vec3d&>()));
static_assert(OuterT::element_extents::Rank == 2, "outer(x,y) must be a rank-2 (matrix) expression");
static_assert(OuterT::element_extents::static_extent(0) == 3 && OuterT::element_extents::static_extent(1) == 3, "outer(x,y) dims");

using TransposeT = decltype(std::declval<const Mat33d&>().transpose());
static_assert(
    TransposeT::element_extents::static_extent(0) == 3 && TransposeT::element_extents::static_extent(1) == 3, "3x3 transpose dims");

using Mat36TransposeT = decltype(std::declval<const Mat36d&>().transpose());
static_assert(Mat36TransposeT::element_extents::static_extent(0) == 6 && Mat36TransposeT::element_extents::static_extent(1) == 3,
    "3x6 transpose dims (Rows/Cols swap)");

using BlockT = decltype(std::declval<const Mat33d&>().block<0, 1, 2, 2>());
static_assert(BlockT::element_extents::static_extent(0) == 2 && BlockT::element_extents::static_extent(1) == 2, "block<0,1,2,2>() dims");

using RowT = decltype(std::declval<const Mat33d&>().row<0>());
static_assert(RowT::element_extents::Rank == 1 && RowT::element_extents::static_extent(0) == 3, "row<0>() dims");

using ColT = decltype(std::declval<const Mat33d&>().col<1>());
static_assert(ColT::element_extents::Rank == 1 && ColT::element_extents::static_extent(0) == 3, "col<1>() dims");

class ExprProductTest : public ::testing::Test { };

TEST_F(ExprProductTest, Item33IdentityZerosOnesMatchHandComputed)
{
    // Zeros(), Identity() and setZero() on a 3x3 matrix.
    const Mat33d z  = Mat33d::Zeros();
    const Mat33d o  = Mat33d::Ones();
    const Mat33d id = Mat33d::Identity();
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            EXPECT_DOUBLE_EQ(z(r, c), 0.0);
            EXPECT_DOUBLE_EQ(o(r, c), 1.0);
            EXPECT_DOUBLE_EQ(id(r, c), (r == c) ? 1.0 : 0.0);
        }
    }
}

TEST_F(ExprProductTest, MatVecBasicMatchesHandComputed)
{
    // Basic 3x3 matrix-vector product.
    Mat33d A;
    A(0, 0) = 1.0; A(0, 1) = 2.0; A(0, 2) = 3.0;
    A(1, 0) = 4.0; A(1, 1) = 5.0; A(1, 2) = 6.0;
    A(2, 0) = 7.0; A(2, 1) = 8.0; A(2, 2) = 9.0;
    Vec3d x;
    x(0) = 1.0; x(1) = 0.0; x(2) = -1.0;

    const Vec3d result = A * x;
    EXPECT_DOUBLE_EQ(result(0), A(0, 0) * x(0) + A(0, 1) * x(1) + A(0, 2) * x(2));
    EXPECT_DOUBLE_EQ(result(1), A(1, 0) * x(0) + A(1, 1) * x(1) + A(1, 2) * x(2));
    EXPECT_DOUBLE_EQ(result(2), A(2, 0) * x(0) + A(2, 1) * x(1) + A(2, 2) * x(2));

    // Fused-vs-per-component consistency: the lazy node's own eval<I>()
    // (per-component recompute, e.g. when the node is a Sum operand) must
    // agree EXACTLY with the fused-write path used to materialize `result`
    // above (aether/expr/nodes/Product.h's AssignDispatch specialization).
    const auto node                 = A * x;
    const aether::SampleIndex zero  = aether::SampleIndex::make(0);
    EXPECT_DOUBLE_EQ(node.template eval<0>(zero), result(0));
    EXPECT_DOUBLE_EQ(node.template eval<1>(zero), result(1));
    EXPECT_DOUBLE_EQ(node.template eval<2>(zero), result(2));
}

TEST_F(ExprProductTest, MatVecNonSquare36By6MatchesHandComputed)
{
    // Non-square matrix-vector product (3x6 * 6-vector -> 3-vector).
    Mat36d A;
    Vec6d x;
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 6; ++c)
            A(r, c) = static_cast<double>(r) * 6.0 + static_cast<double>(c) + 1.0;
    for (std::size_t c = 0; c < 6; ++c)
        x(c) = static_cast<double>(c) - 2.5;

    const Vec3d result = A * x;
    for (std::size_t r = 0; r < 3; ++r) {
        double expected = 0.0;
        for (std::size_t c = 0; c < 6; ++c)
            expected += A(r, c) * x(c);
        EXPECT_DOUBLE_EQ(result(r), expected);
    }
}

TEST_F(ExprProductTest, MatMatBasicMatchesHandComputed)
{
    // Basic fused matrix-matrix product.
    Mat33d A, B;
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            A(r, c) = static_cast<double>(r) - static_cast<double>(c) * 0.5;
            B(r, c) = static_cast<double>(r) * static_cast<double>(c) + 1.0;
        }
    }

    const Mat33d result = A * B;
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            double expected = 0.0;
            for (std::size_t k = 0; k < 3; ++k)
                expected += A(r, k) * B(k, c);
            EXPECT_DOUBLE_EQ(result(r, c), expected);
        }
    }

    // Fused-vs-per-component consistency (same rationale as the MatVec test above).
    const auto node                = A * B;
    const aether::SampleIndex zero = aether::SampleIndex::make(0);
    EXPECT_DOUBLE_EQ((node.template eval<0, 0>(zero)), result(0, 0));
    EXPECT_DOUBLE_EQ((node.template eval<1, 2>(zero)), result(1, 2));
    EXPECT_DOUBLE_EQ((node.template eval<2, 1>(zero)), result(2, 1));
}

TEST_F(ExprProductTest, MatMatNonSquare36By63MatchesHandComputed)
{
    // Non-square matrix-matrix product (3x6 * 6x3 -> 3x3).
    Mat36d A;
    Mat63d B;
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 6; ++c)
            A(r, c) = static_cast<double>(r) + static_cast<double>(c) * 0.25;
    for (std::size_t r = 0; r < 6; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            B(r, c) = static_cast<double>(r) * 0.5 - static_cast<double>(c);

    const Mat33d result = A * B;
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            double expected = 0.0;
            for (std::size_t k = 0; k < 6; ++k)
                expected += A(r, k) * B(k, c);
            EXPECT_DOUBLE_EQ(result(r, c), expected);
        }
    }
}

TEST_F(ExprProductTest, OuterBasicMatchesHandComputed)
{
    // Basic outer product.
    Vec3d u, v;
    u(0) = 2.0; u(1) = -1.0; u(2) = 0.5;
    v(0) = 1.0; v(1) = 3.0; v(2) = -2.0;

    const Mat33d result = aether::outer(u, v);
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            EXPECT_DOUBLE_EQ(result(r, c), u(r) * v(c));
}

TEST_F(ExprProductTest, TransposeComponentsMatchSwappedIndices)
{
    // Transpose components, including non-square dimensions.
    Mat36d A;
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 6; ++c)
            A(r, c) = static_cast<double>(r) * 10.0 + static_cast<double>(c);

    // `eval<Is...>()` needs compile-time indices, so this is a spot-check
    // over a representative subset rather than an exhaustive runtime loop.
    const auto AT                  = A.transpose();
    const aether::SampleIndex zero = aether::SampleIndex::make(0);
    EXPECT_DOUBLE_EQ((AT.template eval<0, 0>(zero)), (A.template eval<0, 0>(zero)));
    EXPECT_DOUBLE_EQ((AT.template eval<2, 1>(zero)), (A.template eval<1, 2>(zero)));
    EXPECT_DOUBLE_EQ((AT.template eval<5, 0>(zero)), (A.template eval<0, 5>(zero)));
    EXPECT_DOUBLE_EQ((AT.template eval<0, 2>(zero)), (A.template eval<2, 0>(zero)));
}

TEST_F(ExprProductTest, TransposeOfTransposeRestoresOriginalValues)
{
    // Transpose is an involution: (A^T)^T == A (not to be confused with the
    // identity matrix).
    Mat33d A;
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            A(r, c) = static_cast<double>(r) * 3.0 - static_cast<double>(c) * 2.0 + 1.0;

    const Mat33d roundTrip = A.transpose().transpose();
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            EXPECT_DOUBLE_EQ(roundTrip(r, c), A(r, c));
}

TEST_F(ExprProductTest, RowViewComponentsMatchSourceRow)
{
    // Row-view component extraction.
    Mat33d A;
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            A(r, c) = static_cast<double>(r) * 5.0 + static_cast<double>(c);

    const Vec3d row1 = A.row<1>();
    EXPECT_DOUBLE_EQ(row1(0), A(1, 0));
    EXPECT_DOUBLE_EQ(row1(1), A(1, 1));
    EXPECT_DOUBLE_EQ(row1(2), A(1, 2));
}

TEST_F(ExprProductTest, ColViewComponentsMatchSourceColumn)
{
    // Column-view component extraction.
    Mat33d A;
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            A(r, c) = static_cast<double>(r) - static_cast<double>(c) * 5.0;

    const Vec3d col2 = A.col<2>();
    EXPECT_DOUBLE_EQ(col2(0), A(0, 2));
    EXPECT_DOUBLE_EQ(col2(1), A(1, 2));
    EXPECT_DOUBLE_EQ(col2(2), A(2, 2));

    // ColView composes with the vector algebra: `A.col<C>.dot(x)`.
    Vec3d x;
    x(0) = 1.0; x(1) = 2.0; x(2) = 3.0;
    const double dotResult   = col2.dot(x);
    const double expectedDot = col2(0) * x(0) + col2(1) * x(1) + col2(2) * x(2);
    EXPECT_DOUBLE_EQ(dotResult, expectedDot);
}

TEST_F(ExprProductTest, BlockViewComponentsMatchSourceSubBlock)
{
    // Block-view component extraction.
    Mat33d A;
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            A(r, c) = static_cast<double>(r) * 3.0 + static_cast<double>(c) + 1.0;

    const Mat22d block = A.block<0, 1, 2, 2>();
    EXPECT_DOUBLE_EQ(block(0, 0), A(0, 1));
    EXPECT_DOUBLE_EQ(block(0, 1), A(0, 2));
    EXPECT_DOUBLE_EQ(block(1, 0), A(1, 1));
    EXPECT_DOUBLE_EQ(block(1, 1), A(1, 2));
}

TEST_F(ExprProductTest, CwiseMulMatrixMatchesHandComputed)
{
    // Hadamard (elementwise) matrix product against a hand-computed
    // reference.
    Mat33d A, B;
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            A(r, c) = static_cast<double>(r) + 1.0;
            B(r, c) = static_cast<double>(c) - 2.0;
        }
    }
    const Mat33d result = A.cwiseMul(B);
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            EXPECT_DOUBLE_EQ(result(r, c), A(r, c) * B(r, c));
}

TEST_F(ExprProductTest, CwiseMulIsRankGenericOnVectorsToo)
{
    // Proves cwiseMul is NOT matrix-specific despite living in Product.h —
    // same node, rank-1 operands.
    Vec3d a, b;
    a(0) = 2.0; a(1) = -3.0; a(2) = 0.5;
    b(0) = 1.5; b(1) = 4.0; b(2) = -2.0;
    const Vec3d result = a.cwiseMul(b);
    EXPECT_DOUBLE_EQ(result(0), a(0) * b(0));
    EXPECT_DOUBLE_EQ(result(1), a(1) * b(1));
    EXPECT_DOUBLE_EQ(result(2), a(2) * b(2));
}

TEST_F(ExprProductTest, MatDotFrobeniusMatchesHandComputed)
{
    // Frobenius inner product against a hand-computed sum_{r,c}
    // A(r,c)*B(r,c).
    Mat33d A, B;
    double expected = 0.0;
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            A(r, c) = static_cast<double>(r) * 2.0 - static_cast<double>(c);
            B(r, c) = static_cast<double>(c) + 1.0;
            expected += A(r, c) * B(r, c);
        }
    }
    EXPECT_DOUBLE_EQ(A.matDot(B), expected);
}

TEST_F(ExprProductTest, NestedMatMatThenMatVecCascadesTheFusedPath)
{
    // (A*B)*x must equal A*(B*x) — exercises the fused MatVec
    // specialization's cascade-on-nesting case, when the cached operand is
    // itself a composite MatMat, since `(A*B)*x`'s `m_` operand is
    // `MatMat<A,B>`.
    Mat33d A, B;
    Vec3d x;
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            A(r, c) = static_cast<double>(r) - static_cast<double>(c) + 2.0;
            B(r, c) = static_cast<double>(r) * static_cast<double>(c) - 1.0;
        }
        x(r) = static_cast<double>(r) * 0.5 - 1.0;
    }

    const Vec3d lhs = (A * B) * x;
    const Vec3d rhs = A * (B * x);
    for (std::size_t r = 0; r < 3; ++r)
        EXPECT_DOUBLE_EQ(lhs(r), rhs(r));
}

TEST_F(ExprProductTest, VectorAlgebraRegressionStillWorksAlongsideMatrixOps)
{
    // The vector operator set (dot/cross/norm) stays intact next to the
    // new matrix operators.
    Mat33d A = Mat33d::Identity();
    Vec3d x;
    x(0) = 3.0; x(1) = -4.0; x(2) = 0.0;

    const Vec3d Ax = A * x;
    EXPECT_DOUBLE_EQ(Ax(0), x(0));
    EXPECT_DOUBLE_EQ(Ax(1), x(1));
    EXPECT_DOUBLE_EQ(Ax(2), x(2));
    EXPECT_DOUBLE_EQ(Ax.norm(), 5.0);

    Vec3d y;
    y(0) = 1.0; y(1) = 0.0; y(2) = 0.0;
    const Vec3d crossResult = Ax.cross(y);
    EXPECT_DOUBLE_EQ(crossResult(0), Ax(1) * y(2) - Ax(2) * y(1));
    EXPECT_DOUBLE_EQ(crossResult(1), Ax(2) * y(0) - Ax(0) * y(2));
    EXPECT_DOUBLE_EQ(crossResult(2), Ax(0) * y(1) - Ax(1) * y(0));
}

TEST_F(ExprProductTest, ViewMaterializedMatVecMatchesRawAccessorArithmetic)
{
    // Batched path: the matrix-product surface reached via `view[i].get()`
    // materialization (mirrors test_ExprGeometric.cpp's analogous case).
    constexpr std::size_t N = 4;
    aether::Array<double, 3, 3> arrA(N);
    aether::Array<double, 3> arrX(N);
    auto av = arrA.hostView();
    auto xv = arrX.hostView();
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t r = 0; r < 3; ++r) {
            for (std::size_t c = 0; c < 3; ++c)
                av(r, c, i) = static_cast<double>(r) - static_cast<double>(c) + static_cast<double>(i) * 0.1;
            xv(r, i) = static_cast<double>(r) * 0.5 - static_cast<double>(i);
        }
    }

    const aether::SampleIndex i2 = aether::SampleIndex::make(2);
    const Mat33d A                = av[i2].get();
    const Vec3d x                 = xv[i2].get();

    const Vec3d result = A * x;
    for (std::size_t r = 0; r < 3; ++r) {
        double expected = 0.0;
        for (std::size_t c = 0; c < 3; ++c)
            expected += A(r, c) * x(c);
        EXPECT_DOUBLE_EQ(result(r), expected);
    }
}

TEST_F(ExprProductTest, MatVecTransposeChainMatchesHandComputed)
{
    // A.transpose() * x -- proves Transpose composes with MatVec (the
    // fused AssignDispatch<MatVec<M,V>,Op> specialization is generic over
    // M, so M = Transpose<Mat33d> hits the same fused path as a plain Item
    // M).
    Mat33d A;
    Vec3d x;
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c)
            A(r, c) = static_cast<double>(r) * 2.0 - static_cast<double>(c) + 1.0;
        x(r) = static_cast<double>(r) - 1.5;
    }

    const Vec3d result = A.transpose() * x;
    for (std::size_t r = 0; r < 3; ++r) {
        double expected = 0.0;
        for (std::size_t c = 0; c < 3; ++c)
            expected += A(c, r) * x(c); // transpose: (r,c) reads A(c,r)
        EXPECT_DOUBLE_EQ(result(r), expected);
    }
}

TEST_F(ExprProductTest, OuterThenMatMulMatchesHandComputed)
{
    // outer(u,v) * B -- proves Outer composes with MatMat.
    Vec3d u, v;
    u(0) = 1.0; u(1) = 2.0; u(2) = -1.0;
    v(0) = 0.5; v(1) = -1.5; v(2) = 2.0;
    Mat33d B;
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            B(r, c) = static_cast<double>(r) - static_cast<double>(c) * 0.5;

    const Mat33d result = aether::outer(u, v) * B;
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            double expected = 0.0;
            for (std::size_t k = 0; k < 3; ++k)
                expected += (u(r) * v(k)) * B(k, c);
            EXPECT_DOUBLE_EQ(result(r, c), expected);
        }
    }
}

} // namespace
} // namespace aether_tests
