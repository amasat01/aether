#!/bin/sh
# derisk-b3, one leg per call: sh run.sh V | X | F   (prints one RC line per step; the first error line is the report)
# V: every device variant header parses with nvcc.  X: the x86-64 host vector math header and a TU that instantiates it.
# F: the Rust core as a free-threaded wheel; the GIL must stay off after the import.
cd "$(dirname "$0")" || exit 1
cut_dir=cut
case "$1" in
V)
  for f in $(find $cut_dir/variants/cuda -name '*.h' | sort); do
    ${NVCC:-nvcc} -std=c++14 -x cu -arch=sm_61 -fsyntax-only -I $cut_dir/components -I $cut_dir/root -I $cut_dir/variants/cuda "$f"
    echo "V RC=$? $f"
  done ;;
X)
  inc="-I $cut_dir/hostleaf -I $cut_dir/components -I $cut_dir/root -I $cut_dir/variants/host_vecmath"
  for m in native x86-64; do
    ${CXX:-g++} -std=c++20 -O2 -march=$m -fsyntax-only -x c++ $inc $cut_dir/hostleaf/aether/math/detail/HostVectorMath.h
    echo "X RC=$? header -march=$m"
    ${CXX:-g++} -std=c++20 -O2 -march=$m -c $inc hvm_tu.cpp -o hvm_$m.o
    echo "X RC=$? tu -march=$m vector-ABI symbols=$(nm hvm_$m.o 2>/dev/null | grep -c _ZGV)"
  done ;;
F)
  py=${PYTHON_FT:-python3.14t}
  command -v "$py" >/dev/null || { echo "F NOT RUN: no $py"; exit 0; }
  "$py" -m venv ft && ft/bin/pip install -q maturin pytest; echo "F RC=$? venv"
  (cd raptor_src && ../ft/bin/maturin build --release -j 6 -o ../wheels); echo "F RC=$? build"
  ft/bin/pip install -q wheels/raptor-*.whl; echo "F RC=$? install"
  ft/bin/python -X gil=0 -c "import sys; a = sys._is_gil_enabled(); import raptor, raptor._core; print('F gil before', a, 'after import', sys._is_gil_enabled())"
  echo "F RC=$? import"
  (cd raptor_src && ../ft/bin/python -X gil=0 -m pytest -q -p no:cacheprovider tests/test_keytext.py tests/test_table.py); echo "F RC=$? tests" ;;
*) echo "usage: sh run.sh V|X|F"; exit 2 ;;
esac
