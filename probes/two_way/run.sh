#!/bin/sh
# Two-way probe: kernel bodies written in native C++ are raised to the IR and compared.
#
# needs   CXX        a C++20 compiler
#         RAPTOR_IR  a raptor checkout holding rs/ir/include and a built target/release/libraptor_ir.a
#         cargo      for the statement reader
# option  PYTHON     an interpreter that imports hawk with a cpu backend: adds the build-and-run legs
#         LDLIBS     extra link libraries for the trace arm
# Every leg prints one line that starts with LEG; the last line is ALL OK or the count of red legs.
cd "$(dirname "$0")" || exit 2
: "${CXX:?set CXX to a C++20 compiler}" "${RAPTOR_IR:?set RAPTOR_IR to a raptor checkout with a built libraptor_ir.a}"
aether=$(cd ../.. && pwd)
ir="-I$RAPTOR_IR/rs/ir/include"
lib="$RAPTOR_IR/target/release/libraptor_ir.a ${LDLIBS:-}"
flags="-std=c++20 -O1 -Wall -Wextra"
red=0
leg() { if [ "$2" = 0 ]; then echo "LEG $1: PASS"; else echo "LEG $1: RED"; red=$((red + 1)); fi; }
rm -rf out && mkdir -p out/shadow/model

(cd reader && cargo test --release > ../out/reader_tests.log 2>&1 && cargo build --release >> ../out/reader_tests.log 2>&1)
leg "reader builds and its tests pass" $?

status=0
for name in helper kernels more; do
    reader/target/release/reader rewrite src/model/$name.h out/shadow/model/$name.h >> out/rewrite.log || status=1
done
leg "reader rewrites the three kernel files" $status

status=0
for set in "" 2; do
    $CXX $flags -Iinclude -Isrc main_values$set.cpp -o out/values$set && ./out/values$set > out/values$set.txt || status=1
done
leg "normal arm builds and runs" $status

status=0
for set in "" 2; do
    $CXX $flags -DAETHER_TRACE -Iout/shadow -Iinclude -Isrc $ir main_trace$set.cpp $lib -o out/trace$set \
        && ./out/trace$set > out/trace$set.log || status=1
done
leg "trace arm builds and raises eight kernels" $status

status=0
$CXX $flags "-I$aether" -Iinclude wave.cpp -o out/wave_values && ./out/wave_values > out/values3.txt || status=1
$CXX $flags -DAETHER_TRACE "-I$aether" -Iinclude $ir wave.cpp $lib -o out/wave_trace && ./out/wave_trace > out/wave.log || status=1
leg "the library math headers build in both arms" $status

status=0
for reference in expected/*.ir; do
    cmp -s "$reference" "out/$(basename "$reference")" || { echo "  differs: $(basename "$reference")"; status=1; }
done
leg "raised IR text equals the reference of the other architecture" $status

$CXX $flags -DAETHER_TRACE -Iinclude $ir leaky.cpp $lib -o out/leaky && ./out/leaky > out/leaky.log 2>&1
[ $? = 3 ]
leg "a plain variable assigned under a traced condition is refused" $?

if [ -n "${PYTHON:-}" ]; then
    "$PYTHON" e2e2.py > out/e2e.log 2>&1 && [ "$(grep -c 'difference to the normal C++ arm 0$' out/e2e.log)" = 8 ]
    leg "eight raised kernels build and run through hawk, exact against the normal arm" $?
    "$PYTHON" check3.py > out/wave_values.log 2>&1
    leg "the math kernel evaluates (see out/wave_values.log for the ulp figure)" $?
fi
if [ $red = 0 ]; then echo "ALL OK"; else echo "RED LEGS: $red"; exit 1; fi
