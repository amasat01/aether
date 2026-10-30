#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# An aether-dsc wheel in a fresh venv, as a user gets it: installs with its [test] extra only, imports, makes a host
# array, refuses a device one by name, keeps the payload without the arrays, and passes the Python tests from outside
# the checkout (dsc/aether, which has no _core, would shadow the installed wheel). CI test-wheel and `make gate-wheel`.
# Usage: tools/wheel_check.sh VENV_DIR WHEEL_DIR TESTS_DIR [PYTEST_ARG...]    env: PYTHON, RAPTOR_SRC (else PyPI)
set -euo pipefail
venv=${1:?venv dir}; wheels=$(cd "${2:?wheel dir}" && pwd); tests=$(cd "${3:?tests dir}" && pwd); shift 3
"${PYTHON:-python3}" -m venv "$venv"
py=$(cd "$venv" && pwd)/bin/python
cd "$(mktemp -d)"
[ -z "${RAPTOR_SRC:-}" ] || "$py" -m pip install --quiet "$RAPTOR_SRC"
"$py" -m pip install --quiet --no-index --find-links "$wheels" aether-dsc
"$py" -m pip install --quiet --prefer-binary "aether-dsc[test]"   # the installed wheel satisfies the name: only the extra is fetched
"$py" -m pip check
"$py" - <<'PY'
import sys
import aether_dsc, aether
p = aether_dsc.payload()
assert p.headers and p.digest
a = aether.zeros((2, 3), "float32")
assert a.shape == (2, 3) and memoryview(a).nbytes == 24
# no GPU (and here no driver) on this runner: the plugin loads, asks the driver, and the error names it
try:
    aether.empty(8, device="cuda")
except aether.CudaUnavailable as exc:
    assert sys.platform == "darwin" or "CUDA driver" in str(exc), exc   # macOS wheels carry no plugin
print("OK")
PY
# the payload does not depend on the arrays
site=$("$py" -c "import aether_dsc, os; print(os.path.dirname(os.path.dirname(aether_dsc.__file__)))")
rm "$site"/aether/_core*.so
"$py" -c "import aether_dsc; aether_dsc.payload()"
out=$("$py" -c "import aether" 2>&1 || true)
grep -q "_core is not built" <<< "$out"
"$py" -m pip install --quiet --no-index --find-links "$wheels" --force-reinstall --no-deps aether-dsc
cd "$tests"
"$py" -m pytest -q -c /dev/null --rootdir . -p no:cacheprovider "$@"
