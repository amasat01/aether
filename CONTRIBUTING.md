# Contributing

Contributions — issues and pull requests — are welcome on GitHub, under the
terms below.

## Terms

Contributions are accepted under the Apache License 2.0, the license this
project ships under (inbound = outbound). Every contribution must carry a
Developer Certificate of Origin sign-off (`git commit -s`); see
https://developercertificate.org/ for what that certifies. There is no
Contributor License Agreement and no relicensing right.

## Using the code

The code is available under the Apache License 2.0 (see `LICENSE`).
Contributions are accepted under the same terms (inbound = outbound).

## Releasing

`aether-dsc` (the sealed AETHER header payload — this repo's C++ library
itself is not published to PyPI) publishes through
`.github/workflows/publish.yml`, triggered by pushing a tag `v<version>`
that matches `dsc/pyproject.toml`'s `project.version` (which tracks
`aether/version.h`). `workflow_dispatch` runs the same build, check and
test-wheel steps without publishing.

1. Bump `version` in `dsc/pyproject.toml` (keep it in step with
   `aether/version.h`), update `CHANGELOG.md`.
2. `git tag vX.Y.Z && git push origin vX.Y.Z`.
3. The workflow checks out the sibling `eagle` repository (`seal()` needs
   its `plugin/gref_layout.h` / `plugin/gref_abi.h`), runs
   `python -m aether_dsc.seal` — which also runs the NVRTC-clean gate over
   every device-reachable header — builds the sdist + wheel, runs
   `twine check --strict`, verifies the wheel ships the sealed payload (and
   no readable `aether/` source), tests the installed wheel on every
   supported CPython, then publishes via a PyPI Trusted Publisher
   (environment `pypi`; no token in this repository).

**Publish order across the family:** `aether-dsc` and `raptor-core` (in the
`raptor` repo) have no family dependencies and publish first, in either
order. `raptor-hawk` depends on `aether-dsc` and publishes after it.
