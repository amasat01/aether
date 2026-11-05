// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// aether::accum::twoSumErr: one primitive for every compensated sum. Shared by
// test_TwoSum.cpp (C++ build) and test_TwoSum.cu (CUDA build, which adds the
// device arm). The rows pin: the residual is exact (against an independent
// Fast2Sum by fabs), a non-finite sum gives NaN, a zero residual is +0.

#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace twosum {

struct Triple { double a, b; };

// Mixed magnitudes, both zeros, cancellation, denormals, overflow, Inf and NaN.
inline std::vector<Triple> corpus()
{
    std::mt19937_64 gen(11);
    std::uniform_real_distribution<double> mant(0.5, 1.0), expo(-300.0, 300.0);
    std::vector<Triple> v;
    for (int i = 0; i < 20000; ++i) {
        const double a = (gen() & 1 ? -1 : 1) * mant(gen) * std::pow(10.0, expo(gen));
        const double b = (gen() & 1 ? -1 : 1) * mant(gen) * std::pow(10.0, i % 3 ? expo(gen) : std::floor(expo(gen) / 40) * 40);
        v.push_back({a, b});
        v.push_back({a, -a});
        v.push_back({a, a});
    }
    const double inf = std::numeric_limits<double>::infinity(), nan = std::numeric_limits<double>::quiet_NaN(),
                 big = 1.7e308, den = std::numeric_limits<double>::denorm_min();
    for (double a : {0.0, -0.0, 1.0, -1.0, big, -big, den, -den, inf, -inf, nan})
        for (double b : {0.0, -0.0, 1.0, -1.0, big, -big, den, -den, inf, -inf, nan})
            v.push_back({a, b});
    return v;
}

// The independent reference: Fast2Sum with the larger operand picked by fabs.
template<class E>
E reference(E a, E b, E s)
{
    if (!std::isfinite(s))
        return std::numeric_limits<E>::quiet_NaN();
    volatile E big = std::fabs(a) >= std::fabs(b) ? a : b, small = std::fabs(a) >= std::fabs(b) ? b : a;
    volatile E z = s - big;
    volatile E e = small - z;
    return e + E(0);
}

template<class E>
bool same(E x, E y)
{
    return (std::isnan(x) && std::isnan(y)) || (x == y && std::signbit(x) == std::signbit(y));
}

template<class E>
void checkHost()
{
    for (const auto& t : corpus()) {
        const E a = static_cast<E>(t.a), b = static_cast<E>(t.b);
        volatile E sv = a + b;
        const E s = sv, e = aether::accum::twoSumErr(a, b, s);
        ASSERT_TRUE(same(e, reference(a, b, s))) << a << " + " << b << " = " << s << ": " << e;
        if (!std::isfinite(s)) {
            ASSERT_TRUE(std::isnan(e));
        } else if (e == E(0)) {
            ASSERT_FALSE(std::signbit(e)) << "a zero residual must be +0";
        }
    }
}

template<class E>
void checkAtomicForm()
{
    for (const auto& t : corpus()) {
        const E a = static_cast<E>(t.a), b = static_cast<E>(t.b);
        volatile E sv = a + b;
        const E s = sv;
        ASSERT_TRUE(same(aether::accum::twoSumErrAtomic(a, b, s), aether::accum::twoSumErr(a, b, s)))
            << a << " + " << b << " = " << s;
    }
}

// Sequences summed into a +0 compensation, the way a run accumulates: the Accum form and the pinned form deliver
// the same bits, including sequences that reach Inf or NaN midway and all-zero or cancelling ones.
template<class E, class F>
std::vector<E> accumulate(const std::vector<std::vector<E>>& seqs, F resid)
{
    std::vector<E> out;
    for (const auto& q : seqs) {
        E s = E(0), c = E(0);
        for (E t : q) {
            volatile E sv = s + t;
            const E n = sv;
            c = c + resid(s, t, n);
            s = n;
        }
        out.push_back(c);
    }
    return out;
}

template<class E>
void checkAccumForm()
{
    std::mt19937_64 gen(5);
    const auto c = corpus();
    std::vector<std::vector<E>> seqs;
    for (std::size_t i = 0; i + 8 < c.size(); i += 5) {
        std::vector<E> q;
        for (int k = 0; k < 8; ++k)
            q.push_back(static_cast<E>(gen() & 1 ? c[i + k].a : c[i + k].b));
        seqs.push_back(q);
    }
    seqs.push_back({E(0), E(-0.0), E(0), E(-0.0)});
    seqs.push_back({E(-0.0), E(-0.0)});
    seqs.push_back({E(1), E(-1), E(1), E(-1)});
    seqs.push_back({E(1e30), E(-1e30), E(3), E(-3)});
    seqs.push_back({std::numeric_limits<E>::max(), std::numeric_limits<E>::max(), E(1)});
    seqs.push_back({E(1), std::numeric_limits<E>::infinity(), E(2), -std::numeric_limits<E>::infinity()});
    seqs.push_back({E(1), std::numeric_limits<E>::quiet_NaN(), E(2)});
    const auto a = accumulate<E>(seqs, [](E x, E y, E s) { return aether::accum::twoSumErrAccum(x, y, s); });
    const auto b = accumulate<E>(seqs, [](E x, E y, E s) { return aether::accum::twoSumErr(x, y, s); });
    for (std::size_t i = 0; i < seqs.size(); ++i)
        ASSERT_TRUE(same(a[i], b[i])) << "sequence " << i << ": " << a[i] << " vs " << b[i];
}

TEST(TwoSum, AccumulatedAccumFormEqualsThePinnedFormDouble) { checkAccumForm<double>(); }
TEST(TwoSum, AccumulatedAccumFormEqualsThePinnedFormFloat) { checkAccumForm<float>(); }
TEST(TwoSum, AtomicFormEqualsTheFast2SumFormDouble) { checkAtomicForm<double>(); }
TEST(TwoSum, AtomicFormEqualsTheFast2SumFormFloat) { checkAtomicForm<float>(); }
TEST(TwoSum, HostResidualIsExactAndPinnedDouble) { checkHost<double>(); }
TEST(TwoSum, HostResidualIsExactAndPinnedFloat) { checkHost<float>(); }

TEST(TwoSum, TheFiniteTestReadsTheExponentField)
{
    const double inf = std::numeric_limits<double>::infinity();
    for (double x : {0.0, -0.0, 1.0, 1e308, std::numeric_limits<double>::denorm_min()})
        EXPECT_TRUE(aether::accum::isFiniteBits(x));
    for (double x : {inf, -inf, std::numeric_limits<double>::quiet_NaN()})
        EXPECT_FALSE(aether::accum::isFiniteBits(x));
    EXPECT_FALSE(aether::accum::isFiniteBits(std::numeric_limits<float>::infinity()));
    EXPECT_TRUE(aether::accum::isFiniteBits(3.0f));
}

} // namespace twosum
} // namespace aether_tests
