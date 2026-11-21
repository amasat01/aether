// The normal arm of the second set: the author's file, untouched.
#include <cstdio>

#include "model/more.h"

int main() {
    const double vs[] = {-3.0, -0.2, 0.25, 1.0, 2.0, 9.0, 123.456};
    for (double v : vs) {
        double band = 0, sides = 0, horner = 0, x = 0, w = 0;
        int iters = 0;
        model::band({v}, {0.5}, {&band});
        model::horner({v}, {&horner});
        if (v > 0) {
            model::sides({v}, {&sides});
            model::counted({v}, {&x}, {&iters});
            model::whileRoot({v}, {&w});
        }
        std::printf("%.17g %.17g %.17g %.17g %.17g %d %.17g\n", v, band, sides, horner, x, iters, w);
    }
}
