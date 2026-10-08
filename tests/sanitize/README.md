# aether sanitize step

Wraps `aether_tests` under valgrind (+ NVIDIA compute-sanitizer when CUDA is
available) and prints a consolidated, human-readable report.

**WIRING ONLY:** this target is registered so `make sanitize` exists; no
sanitizer sweep has been RUN as part of any gate here.

## Run it

```bash
# From the aether repo root, configured with -DAETHER_BUILD_SANITIZE=ON
cmake --build build --target sanitize                      # full mode, valgrind + compute-sanitizer
cmake --build build --target sanitize -- AETHER_SANITIZE_MINIMAL=ON
```

or reconfigure with `-DAETHER_SANITIZE_TOOL=valgrind` / `computesan` to
restrict which tool runs.

Artifacts land in `<build>/sanitize/`:
- `valgrind.xml`, `valgrind.stdout.log`, `valgrind.stderr.log`
- `computesan.{memcheck,initcheck,racecheck,synccheck}.log`

## Reading the report

Five categories are summarised per valgrind run:
- **errors** — invalid reads/writes, uninitialised values. Always fail.
- **leak definitely / indirectly / possibly** — real leaks. Always fail.
- **leak still-reachable** — benign (OpenMP TLS, CUDA driver globals). Shown
  as a warning by default; pass `--fail-on-reachable` to escalate.

Compute-sanitizer is run in four modes; each produces its own error count
and (when failing) a short stack excerpt lifted from the log.

## Adding suppressions

Drop new `.supp` files into `suppressions/`. Anything matching `*.supp` is
picked up automatically. Valgrind's own `--gen-suppressions=all` output can
be pasted directly; keep one scenario per file so future debugging is easy.
The two generic suppression files here (CUDA driver globals, OpenMP/glibc
TLS machinery) suppress third-party libraries aether links against, not
aether code.

## Minimal mode — wired, one consumer so far

The driver sets `AETHER_TEST_MINIMAL=1` when invoked with `--minimal`; the
free function `aether_tests::isMinimalMode()` in
`tests/banded/minimal_mode.h` reads it, and the Banded-core batteries are
its first consumer (`tests/test_BandCell8_common.h`; only the BULK arms
consult it, never the enumerated corpora, see the header's own doc
comment). `--minimal` is accepted and forwarded (parity with the driver's
CLI) and now shrinks whatever batteries have opted in. Wire the helper
into other expensive aether tests as they land; do not wire it
speculatively ahead of a genuinely expensive test needing it.
