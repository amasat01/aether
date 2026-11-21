// The normal arm: the author's files, untouched, computing with double.
#include <cstdio>

#include "model/kernels.h"

int main() {
    const double vs[] = {0.25, 1.0, 2.0, 9.0, 123.456};
    for (double v : vs) {
        double out = 0, y = 0, x = 0;
        model::drag({v}, {0.47}, {&out});
        model::clip({v}, {1.5}, {&y});
        model::newton({v}, {&x});
        std::printf("%.17g %.17g %.17g %.17g\n", v, out, y, x);
    }
}
