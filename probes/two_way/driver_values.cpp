// The normal arm of the example project: the values its kernels give.
#include <cstdio>
#include <initializer_list>
#include <physics/drag.h>
#include <physics/gear.h>
#include <physics/relax.h>
int main() {
    for (double v : {-3.0, -0.7, 0.2, 0.9, 2.5, 7.0, 40.0}) {
        double d = 0, x = 0, y = 0, g = 0;
        physics::drag(v, 0.47, d);
        physics::relax(v < 0 ? -v : v, x);
        physics::band(v, 0.5, y);
        const int k = static_cast<int>(v < 0 ? -v : v) % 5;   // gears 0 to 4 from the same samples
        physics::gear(k, v, g);
        std::printf("%.17g %.17g %.17g %.17g %d %.17g\n", v, d, x, y, k, g);
    }
}
