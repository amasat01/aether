// The normal arm of the view kernels: the example's kernels, unchanged, built on the library's own
// view headers and run on deterministic inputs. It prints each input and output array, in memory
// order, for the hawk check to run the raised kernels on the same data.
#include <cstdint>
#include <cstdio>
#include <vector>

#include "physics/views.h"

namespace {

constexpr std::size_t kSamples = 64;
constexpr std::size_t kTable = 40;
constexpr std::size_t kBins = 8;

/// A view over `data` with `samples` samples.
template <class V, class T>
V viewOver(T* data, std::size_t samples) {
    return V(data, typename V::mapping_type(typename V::extents_type(samples)), aether::Device{});
}

std::uint64_t state = 0x9e3779b97f4a7c15ull;
double uniform(double low, double high) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return low + (high - low) * static_cast<double>(state >> 11) / 9007199254740992.0;
}

template <class T>
void print(const char* name, const std::vector<T>& values) {
    std::printf("%s %zu", name, values.size());
    for (T v : values) std::printf(" %.17g", static_cast<double>(v));
    std::printf("\n");
}

} // namespace

int main() {
    using aether::dyn;
    using aether::extents;
    using aether::SampleIndex;
    using aether::View;
    using Scalar = extents<dyn>;
    using Vec3 = extents<3, dyn>;
    const std::size_t n = kSamples;

    std::vector<double> vec(3 * n), scaled(3 * n, 0.0), gate(n), table(kTable), value(n);
    std::vector<double> out(n, 0.0), bins(kBins, 0.0), gated(3 * n, 0.0), sums(n, 0.0), scattered(n, 0.0);
    std::vector<std::int32_t> row(n), slot(n), key(n);
    for (auto& v : vec) v = uniform(-2.0, 2.0);
    for (auto& v : gate) v = uniform(0.0, 1.0);
    for (auto& v : table) v = uniform(-1.0, 1.0);
    for (auto& v : value) v = uniform(-1.0, 1.0);
    for (std::size_t s = 0; s < n; ++s) {
        row[s] = static_cast<std::int32_t>(uniform(0.0, kTable - 1.0));
        slot[s] = static_cast<std::int32_t>((s * 7) % n);   // a permutation: no two samples store to one place
        key[s] = static_cast<std::int32_t>(uniform(0.0, kBins));
    }
    const double gain = 0.7;

    for (std::size_t s = 0; s < n; ++s) {
        const auto i = SampleIndex::make(s);
        physics::vscale(viewOver<View<const double, Vec3>>(vec.data(), n), viewOver<View<double, Vec3>>(scaled.data(), n), i, gain);
        physics::gather(viewOver<View<const std::int32_t, Scalar>>(row.data(), n), viewOver<View<const double, Scalar>>(table.data(), kTable),
                        viewOver<View<double, Scalar>>(out.data(), n), i);
        physics::scatter(viewOver<View<const std::int32_t, Scalar>>(slot.data(), n), viewOver<View<const double, Scalar>>(value.data(), n),
                         viewOver<View<double, Scalar>>(scattered.data(), n), i);
        physics::tally(viewOver<View<const std::int32_t, Scalar>>(key.data(), n), viewOver<View<const double, Scalar>>(value.data(), n),
                       viewOver<View<double, Scalar>>(bins.data(), kBins), i);
        physics::vgate(viewOver<View<const double, Vec3>>(vec.data(), n), viewOver<View<const double, Scalar>>(gate.data(), n),
                       viewOver<View<double, Vec3>>(gated.data(), n), i);
        physics::vsum(viewOver<View<const double, Vec3>>(vec.data(), n), viewOver<View<double, Scalar>>(sums.data(), n), i);
    }
    print("vec", vec); print("gain", std::vector<double>{gain}); print("scaled", scaled);
    print("row", row); print("table", table); print("gathered", out);
    print("slot", slot); print("value", value); print("scattered", scattered);
    print("key", key); print("bins", bins);
    print("gate", gate); print("gated", gated); print("sums", sums);
}
