#!/bin/sh
# Two-way probe: kernel bodies written in native C++ are raised to the IR and compared.
#
# needs   CXX        a C++20 compiler
#         RAPTOR_IR  a raptor checkout holding rs/ir/include and a built target/release/libraptor_ir.a
#         cargo      for the statement reader and the raise driver
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

$CXX $flags -DAETHER_TRACE -Iout/shadow -Iinclude -Isrc $ir main_trace_all.cpp $lib -o out/trace \
    && ./out/trace > out/trace.log && [ "$(grep -c '^raised ' out/trace.log)" = 8 ]
leg "trace arm builds and raises eight kernels, each recorded twice with equal result" $?

$CXX $flags -DAETHER_TRACE -Iout/shadow -Iinclude -Isrc $ir rules.cpp $lib -o out/rules && ./out/rules > out/rules.log 2>&1
[ $? = 5 ] && grep -q "^raised flagged" out/rules.log && grep -q "kernel counting records differently" out/rules.log
leg "a condition that only converts to bool is taken; a kernel on a global counter is refused" $?

status=0
$CXX $flags "-I$aether" -Iinclude wave.cpp -o out/wave_values && ./out/wave_values > out/values3.txt || status=1
$CXX $flags -DAETHER_TRACE "-I$aether" -Iinclude $ir wave.cpp $lib -o out/wave_trace && ./out/wave_trace > out/wave.log || status=1
leg "a kernel on the library math headers, unchanged, builds in both arms" $status

status=0
$CXX $flags "-I$aether" -Iinclude bucket.cpp -o out/bucket_values && ./out/bucket_values > out/values_bucket.txt || status=1
$CXX $flags -DAETHER_TRACE "-I$aether" -Iinclude $ir bucket.cpp $lib -o out/bucket_trace && ./out/bucket_trace > out/bucket.log || status=1
leg "an integer kernel on the library integer header builds in both arms and raises" $status

(cd driver && cargo test --release > ../out/driver_tests.log 2>&1 && cargo build --release >> ../out/driver_tests.log 2>&1)
leg "raise driver builds and its tests pass" $?

example=driver/example
mkdir -p out/driver
link=""; for word in ${LDLIBS:-}; do link="$link --link $word"; done
driver/target/release/raise --entry $example/physics/kernels.cpp --kernel physics::drag --kernel physics::relax \
    --kernel physics::band --root $example/physics --root $example/numerics -I $example/physics/include \
    -I $example/numerics/include --cxx "$CXX" --trace-include include --ir "$RAPTOR_IR" --out out/driver $link \
    > out/driver.log 2>&1 && [ "$(ls out/driver/*.ir | wc -l | tr -d ' ')" = 3 ] && [ ! -s out/driver/left.txt ]
leg "raise driver raises the three kernels of a project with two roots, nothing left" $?

driver/target/release/raise --entry $example/negative/bad.h --kernel bad::unbounded --root $example/negative \
    --cxx "$CXX" --trace-include include --ir "$RAPTOR_IR" --out out/driver $link > out/driver_negative.log 2>&1
[ $? != 0 ] && grep -q "bad.h:9" out/driver_negative.log
leg "raise driver stops an unbounded while on a traced value, naming the source line" $?

$CXX $flags -DAETHER_TRACE -Iinclude $ir leaky.cpp $lib -o out/leaky && ./out/leaky > out/leaky.log 2>&1
[ $? = 3 ]
leg "a plain variable assigned under a traced condition is refused" $?

if [ -n "${PYTHON:-}" ]; then
    "$PYTHON" e2e2.py > out/e2e.log 2>&1 && [ "$(grep -c 'difference to the normal C++ arm 0$' out/e2e.log)" = 8 ]
    leg "eight raised kernels build and run through hawk, exact against the normal arm" $?
    grep "^digest " out/e2e.log > out/digests.txt; cmp -s expected/digests.txt out/digests.txt
    leg "canonical digests of the eleven raised kernels equal the reference" $?
    "$PYTHON" check_bucket.py > out/bucket_values.log 2>&1 && grep -q "mismatches against the normal C++ arm: 0 of" out/bucket_values.log
    leg "the raised integer kernel builds and runs through hawk, equal to the normal arm" $?
    $CXX $flags -I$example/physics/include -I$example/numerics/include -Iinclude driver_values.cpp -o out/driver_values \
        && ./out/driver_values > out/driver/values.txt && "$PYTHON" driver_check.py > out/driver_values.log 2>&1 \
        && [ "$(grep -c 'worst absolute difference 0.0$' out/driver_values.log)" = 3 ]
    leg "the three kernels the driver raised run through hawk, exact against the normal arm" $?
    "$PYTHON" wave_unit.py > out/wave_values.log 2>&1 && grep -q "worst difference 0.0 ulp" out/wave_values.log
    leg "the math kernel as a built unit equals the normal arm to the last bit" $?
fi
if [ $red = 0 ]; then echo "ALL OK"; else echo "RED LEGS: $red"; exit 1; fi
