# Contributing

The manifesto behind [How we build](https://amasat01.github.io/raptor/content/principles.html):

<!-- manifesto:begin -->
Bloat is future technical debt.<br>
Excessive abstraction hides reality.<br>
Focus and clear boundaries promote flexibility and seamless integration.<br>
Efficiency safeguards portability and scale.<br>
Performance unlocks capabilities.<br>
Rigor earns trust.<br>
Simplicity enables discovery.
<!-- manifesto:end -->

Read [How we build](https://amasat01.github.io/raptor/content/principles.html) first: the twelve rules every pull request in the
family is read against, plus the style, release and license terms.

**This repo:** the header-only C++23 expression library (host and CUDA, one API). The layer every other repo builds on.

## Build and run the gate

From the repo root (CPU-only shown; leave `cpp=ON` off for a CUDA build). `Makefile` has the same commands.

```bash
make test cpp=ON                 # configure, build, then tests/check_gate.sh cpp build/tests/aether_tests
make remint cpp=ON               # after adding or removing a test: rewrite tests/expected_tests_cpp.txt
```

Never run the test binary bare. An empty `--gtest_filter` exits 0 from the binary alone; `tests/check_gate.sh` compares the run
and the listing with the committed name manifest `tests/expected_tests_{cuda,cpp}.txt`. Commit the manifest diff with the tests.
`remint` is refused when `$CI` is set.

CI configures with `-DAETHER_WARN_AS_ERR=ON`; use it too (`.github/workflows/ci.yml`).

Other gates, each a Makefile target: `make sanitize-gate` (CUDA sanitizers, local), `make faithful-gate` (CPU packet math rounding).

Docs (`pip install -r docs/requirements.txt`; the gate needs `doxygen`):

```bash
tests/docs/check_docs.sh         # doxygen allowlist, cited symbols exist, sphinx -W, examples match their tests
cd docs && make nbexec && make nbcheck && make linkcheck
```

A new Doxygen warning is a regression; do not add it to `docs/doxygen_allowlist.txt`.

## What a PR here must show

- The code stays header-only C++23 and builds in both modes, CUDA and `AETHER_CPP_MODE`, with the same API.
- Device-reachable headers stay NVRTC-clean (`python -m aether_dsc.seal` checks them).
- The host-compiler matrix stays green: the `host-gcc` job builds with g++ 12, 13 and 14. A compiler bug is worked around here and
  pinned by a check that fails on the bad compiler.
- Math changes carry their rounding bound and a test that fails when the code is wrong.
- Every public symbol has a Doxygen `@brief`; docs cite only symbols that exist.
- The family checklist from [How we build](https://amasat01.github.io/raptor/content/principles.html), nothing more.

## Release

Record changes under `## Unreleased` in `CHANGELOG.md`; release with `python tools/version.py bump X.Y.Z` (never edit version strings by hand); the whole flow is in [RELEASING.md](https://github.com/amasat01/raptor/blob/main/RELEASING.md).
