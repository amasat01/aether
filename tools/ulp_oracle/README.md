# ulp_oracle — MPFR-backed ULP-oracle harness

Standalone, host-only. Mints golden headers for the transcendentals Banded
math certifies against. Not wired into aether's root CMakeLists.txt
— build out-of-tree.
```
cmake -S . -B <scratch-build>   # needs $CONDA_PREFIX/{include/mpfr.h,lib/libmpfr.so,libgmp.so}
cmake --build <scratch-build>
ulp_oracle <function> <corpus-spec-file> <out-header>
```
Functions: `exp exp2 log log2 log1p expm1 sqrt rsqrt cbrt sin cos tan sincos
atan asin acos sinh cosh tanh` (unary), `pow hypot atan2 fmod` (binary —
`atan2`'s inputs are `(y,x)`, libm order). Reference = MPFR @ 256 bits,
correctly rounded to double (RNDN). An ALWAYS-ON specials block (±0, ±inf,
qNaN, ±min/max normal, ±min subnormal, ±1, per-function domain edges) is
added to every corpus and cannot be disabled.
Corpus-spec grammar (one directive/line, `#` comments):
```
seed <uint64>                                # default: the project's shared DEFAULT_SEED
uniform <lo> <hi> <n>                        # unary: n uniform-random points
loguniform <lo> <hi> <n> [neg]               # unary: n log-uniform points (lo>0)
points <v...>                                # unary: literals (decimal/hex-float/inf/nan)
reduction <pi_over_2|pi_over_4|ln2> <K>      # k*boundary, k=-K..K, +-1ULP neighbors
reduction pow2 <elo> <ehi>                   # 2^e, both signs, +-1ULP neighbors
reduction subnormal_edge                     # curated subnormal/normal boundary set
uniform2/loguniform2/points2/reduction2      # binary-function counterparts
```
One seeded splitmix64 stream, draws in file order; points dedup + sort by bit
pattern, so directive order never affects the minted content.
`ulp.h`'s `ulpDistance(ref, got)`: signed ordered-integer distance, correct
across signed zeros/subnormals/binades. NaN/inf = class-equality (both NaN
→ 0, same-signed inf → 0, any other pairing → sentinel `kMismatch`).
Fence check:
```
BEGIN='^// ---8<--- GOLDEN ROWS BEGIN'; END='^// ---8<--- GOLDEN ROWS END'
sed -n "/$BEGIN/,/$END/p" <header> | sed '1d;$d' | md5sum   # vs BEGIN line's md5=
```
Consuming from a Band math test: `#include "golden_<fn>.h"` + `"ulp.h"`; for each
`kRows[i]`, `memcpy` the `uint64_t in[]`/`ref` fields to `double`, check
`ulp::ulpDistanceAbs(ref, candidate) <= tolerance`.
