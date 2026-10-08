#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
# Compile-fail gate for the `BandedReal` TYPING CONTRACT: the storage-facing
# scalar wrapper around `Band` must not admit an implicit path to or from
# `double`, must not open cross-representation operators `Band` itself
# refuses, and must route egress/ingress and math dispatch through named,
# explicit functions only.
#
# ## What this instrument can see, and what it cannot
# Every claim below is that some expression MUST NOT COMPILE. None of them can
# live inside the test binary: a test that compiles is a test whose subject
# compiled. The COMPLEMENTS do live there — `tests/test_BandedReal_common.h`
# carries the trait-level rows (`!is_convertible_v<BandedReal, double>` and its
# controls). A trait row and a live spelling are different observations: a trait
# can be right while some other conversion sequence still reaches the value.
#
# g++ only, CPP_MODE (`AETHER_CPP_MODE`), no CUDA toolchain, no GPU involved.
#
# ## Arms — every one mandatory, and RC alone is NEVER the verdict
#
#   ARM D10-EGRESS-IMPLICIT   `double d = br;` MUST fail; stderr MUST name
#                             `cannot convert`.
#                             ★ This matters for portability, not just style. A carrier with an
#                             implicit `operator double()` makes EVERY
#                             `aether::math` entry point resolve silently on the
#                             HOST through `std::`, in native FP64 — while the
#                             DEVICE arm hard-errors on the identical call.
#                             Host-green, device-red, and the host answer is
#                             computed on hardware the banded kernels are
#                             targeted at NOT having.
#   ARM D10-EGRESS-CAST       `static_cast<double>(br)` MUST fail; stderr MUST
#                             name `invalid 'static_cast'`. Egress is by NAMED
#                             function only, so even an explicit conversion
#                             operator is refused — the sanctioned spelling is
#                             `br.toDouble()`, and the POSITIVE arm uses it.
#   ARM D10-INGRESS-IMPLICIT  `BandedReal r = 1.5;` MUST fail. Ingest is
#                             EXPLICIT and host-only, so no `double` reaches
#                             device code by accident. Without this arm the two
#                             egress arms above are satisfied by a type that is
#                             merely awkward in one direction.
#   ARM WALL-DIRECT           `br * 2.0` MUST fail. The storage face must not
#                             open a cross-representation operator route that
#                             `Band` itself refuses.
#   ARM WALL-VIA-CARRIER      `Band(br) * 2.0` MUST fail; stderr MUST name
#                             `deleted function`. This is the operator wall
#                             proper (`BandWallFamily`), and the arm exists to
#                             show that entering the carrier through the storage
#                             type does not launder a value around it.
#   ARM BANDRAW-NO-IMPLICIT-BAND
#                             `Band b = BandRaw{...};` MUST fail. `normalize` is
#                             the ONLY bridge, because it is the only operation
#                             that makes the carrier's ordered-limb promise true.
#                             The whole point of `BandRaw` being a TYPE is that
#                             this is a compile error rather than a convention.
#   ARM POW-SANCTIONED        `aether::math::pow(br, br)` MUST compile. `pow` routes the
#                             fifth certified entry point; the arm pins that an
#                             unimplemented entry point fails LOUDLY and by name,
#                             rather than falling through to the scaffold's "T
#                             must be float or double", which would say nothing
#                             true about why.
#   ARM MATH-UNSUPPORTED      `aether::math::tan(band)` MUST fail. Everything outside
#                             the five certified entry points is a hard error on a
#                             banded operand — the honest answer while the
#                             transcendental family remains unported.
#   ARM DTYPE-BAND            `aether::dtype_of<Band>()` MUST fail; stderr MUST
#                             name `BandedReal`. `Band` is a register carrier
#                             with no memory form, so the diagnostic points at
#                             the type that DOES export.
#   ARM LADDER-GAP-BAND       `BandedDemandT<30, BandedReal>` MUST fail; stderr
#                             MUST name `certified set`. `Ladder.h`
#                             restricts a Banded-domain demand to the ladder's
#                             certified set {24, 45, 53} — 30 lands in the
#                             UNCERTIFIED GAP between the Ff1 and Ff2 rungs, and
#                             this is the negative leg of that rule: a
#                             must-not-compile claim that cannot live inside a
#                             `static_assert` in `Ladder.h` itself (the whole
#                             TU would fail to compile rather than pinning ONE
#                             instantiation). The Ladder.h `ladder_asserts`
#                             battery carries the POSITIVE companion claims
#                             (each rung self-selects at its own certified
#                             width).
#   ARM CONTROL               an undeclared symbol. MUST fail (any diagnostic).
#                             Proves the invocation genuinely compiles the file:
#                             a mis-ordered flag returns RC=0 having compiled
#                             nothing, which would make every rejection above
#                             vacuous.
#   ARM POSITIVE              every SANCTIONED spelling: `toDouble()`,
#                             `fromDouble()`, the implicit demote to `Band` and
#                             the implicit pack back, the four live facade entry
#                             points, `dtype_of<BandedReal>()`, and an element
#                             write through a `View<BandedReal, …>`. MUST
#                             compile. Without it every rejection above is also
#                             satisfied by a type that does nothing at all.
#
# Usage: `tests/compile_fail/check_bandedreal_typing_rejected.sh [<aether repo root>]`
# Exit code IS the verdict.

set -uo pipefail

REPO_ROOT="${1:-${AETHER_REPO_ROOT:-}}"
if [[ -z "${REPO_ROOT}" ]]; then
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

CXX="${CXX:-g++}"
WORK=$(mktemp -d)
trap 'rm -rf "${WORK}"' EXIT

FAIL=0

if ! command -v "${CXX}" > /dev/null 2>&1; then
    echo "WARNING [fatal]: compiler '${CXX}' not found; this gate cannot run"
    exit 1
fi

# ---------------------------------------------------------------------------
# ★ PRECONDITION — gate the SOURCE TREE, never an installed copy. A stale
# $CONDA_PREFIX/include/aether answers every question below about headers this
# checkout has never seen. Two locks: the entry headers must exist under
# REPO_ROOT, and every snippet includes them BY ABSOLUTE PATH.
# ---------------------------------------------------------------------------
AETHER_H="${REPO_ROOT}/aether/aether.h"
BAND_H="${REPO_ROOT}/aether/banded/banded.h"
for f in "${AETHER_H}" "${BAND_H}"; do
    if [[ ! -f "${f}" ]]; then
        echo "WARNING [fatal]: '${f}' does not exist — REPO_ROOT is wrong, and" \
             "this gate would silently test an INSTALLED aether instead" >&2
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# arm <TAG> <fail|compile> <required-stderr-substring-or-'-'> <source file>
# ---------------------------------------------------------------------------
arm() {
    local tag="$1" expect="$2" pattern="$3" src="$4"
    local log="${WORK}/${tag}.log"
    local gen="${WORK}/${tag}.gen.cpp"
    local rc=0

    sed -e "s|@AETHER_H@|${AETHER_H}|" -e "s|@BAND_H@|${BAND_H}|" "${src}" > "${gen}"

    "${CXX}" -std=c++23 -DAETHER_CPP_MODE=1 -I"${REPO_ROOT}" -c "${gen}" \
        -o "${WORK}/${tag}.o" > "${log}" 2>&1 || rc=$?

    if [[ "${expect}" == "fail" ]]; then
        if [[ ${rc} -eq 0 ]]; then
            echo "ARM ${tag}: VERDICT RED — compiled, but must be rejected (rc=0)"
            FAIL=1
            return
        fi
        if [[ "${pattern}" != "-" ]] && ! grep -qF -- "${pattern}" "${log}"; then
            echo "ARM ${tag}: VERDICT RED — rejected (rc=${rc}) but for the" \
                 "WRONG reason: diagnostic does not contain '${pattern}'"
            sed -n '1,20p' "${log}" >&2
            FAIL=1
            return
        fi
        if [[ "${pattern}" == "-" ]]; then
            echo "ARM ${tag}: PASS (rc=${rc}, any diagnostic accepted)"
        else
            echo "ARM ${tag}: PASS (rc=${rc}, matched '${pattern}')"
        fi
    else
        if [[ ${rc} -ne 0 ]]; then
            echo "ARM ${tag}: VERDICT RED — a SANCTIONED path stopped compiling (rc=${rc})"
            sed -n '1,40p' "${log}" >&2
            FAIL=1
            return
        fi
        echo "ARM ${tag}: PASS (rc=${rc})"
    fi
}

echo "=========================================="
echo "  BandedReal typing gate (${CXX}, AETHER_CPP_MODE)"
echo "  repo root: ${REPO_ROOT}"
echo "=========================================="

# ---- ARM D10-EGRESS-IMPLICIT ----------------------------------------------
cat > "${WORK}/D10-EGRESS-IMPLICIT.cpp" <<'EOF'
#include "@BAND_H@"
int main()
{
    const aether::banded::BandedReal r = aether::banded::BandedReal::fromDouble(1.5);
    const double d = r;   /* MUST NOT COMPILE — explicit egress only */
    return d > 0.0 ? 0 : 1;
}
EOF
arm D10-EGRESS-IMPLICIT fail "cannot convert" "${WORK}/D10-EGRESS-IMPLICIT.cpp"

# ---- ARM D10-EGRESS-CAST --------------------------------------------------
cat > "${WORK}/D10-EGRESS-CAST.cpp" <<'EOF'
#include "@BAND_H@"
int main()
{
    const aether::banded::BandedReal r = aether::banded::BandedReal::fromDouble(1.5);
    const double d = static_cast<double>(r);  /* MUST NOT COMPILE — NAMED only */
    return d > 0.0 ? 0 : 1;
}
EOF
arm D10-EGRESS-CAST fail "invalid 'static_cast'" "${WORK}/D10-EGRESS-CAST.cpp"

# ---- ARM D10-INGRESS-IMPLICIT ---------------------------------------------
cat > "${WORK}/D10-INGRESS-IMPLICIT.cpp" <<'EOF'
#include "@BAND_H@"
int main()
{
    const aether::banded::BandedReal r = 1.5;  /* MUST NOT COMPILE — explicit ingest */
    return static_cast<int>(r.toBits() & 1u);
}
EOF
arm D10-INGRESS-IMPLICIT fail "conversion from 'double'" "${WORK}/D10-INGRESS-IMPLICIT.cpp"

# ---- ARM WALL-DIRECT ------------------------------------------------------
cat > "${WORK}/WALL-DIRECT.cpp" <<'EOF'
#include "@BAND_H@"
int main()
{
    const aether::banded::BandedReal r = aether::banded::BandedReal::fromDouble(1.5);
    const auto bad = r * 2.0;   /* MUST NOT COMPILE — the operator wall */
    return static_cast<int>(bad.hi);
}
EOF
arm WALL-DIRECT fail "-" "${WORK}/WALL-DIRECT.cpp"

# ---- ARM WALL-VIA-CARRIER -------------------------------------------------
cat > "${WORK}/WALL-VIA-CARRIER.cpp" <<'EOF'
#include "@BAND_H@"
int main()
{
    const aether::banded::BandedReal r = aether::banded::BandedReal::fromDouble(1.5);
    const aether::banded::Band b = r;          /* sanctioned demote */
    const auto bad = b * 2.0;                  /* MUST NOT COMPILE — deleted */
    return static_cast<int>(bad.hi);
}
EOF
arm WALL-VIA-CARRIER fail "deleted function" "${WORK}/WALL-VIA-CARRIER.cpp"

# ---- ARM BANDRAW-NO-IMPLICIT-BAND -----------------------------------------
cat > "${WORK}/BANDRAW-NO-IMPLICIT-BAND.cpp" <<'EOF'
#include "@BAND_H@"
int main()
{
    const aether::banded::detail::BandRaw raw{ 1.0f, 0.0f, 0.0f };
    const aether::banded::Band b = raw;  /* MUST NOT COMPILE — normalize() only */
    return static_cast<int>(b.hi);
}
EOF
arm BANDRAW-NO-IMPLICIT-BAND fail "-" "${WORK}/BANDRAW-NO-IMPLICIT-BAND.cpp"

# ---- ARM POW-SANCTIONED ----------------------------------------------------
cat > "${WORK}/POW-DEFERRED.cpp" <<'EOF'
#include "@BAND_H@"
#include "@AETHER_H@"
int main()
{
    const aether::banded::BandedReal a = aether::banded::BandedReal::fromDouble(2.0);
    const aether::banded::BandedReal b = aether::banded::BandedReal::fromDouble(3.0);
    const auto ok = aether::math::pow(a, b);  /* MUST COMPILE — pow is a certified Band entry point */
    return static_cast<int>(ok.hi);
}
EOF
arm POW-SANCTIONED pass "-" "${WORK}/POW-DEFERRED.cpp"

# ---- ARM MATH-UNSUPPORTED -------------------------------------------------
cat > "${WORK}/MATH-UNSUPPORTED.cpp" <<'EOF'
#include "@BAND_H@"
#include "@AETHER_H@"
int main()
{
    const aether::banded::Band b = aether::banded::BandedReal::fromDouble(2.0);
    const auto bad = aether::math::tan(b);  /* MUST NOT COMPILE — `tan` is NOT certified on the Band family; re-subject this arm when it lands. */
    return static_cast<int>(bad.hi);
}
EOF
arm MATH-UNSUPPORTED fail "-" "${WORK}/MATH-UNSUPPORTED.cpp"

# ---- ARM DTYPE-BAND -------------------------------------------------------
cat > "${WORK}/DTYPE-BAND.cpp" <<'EOF'
#include "@BAND_H@"
#include "@AETHER_H@"
int main()
{
    /* MUST NOT COMPILE — Band has no memory form; the diagnostic must point at
     * the STORAGE type that does export. */
    const aether::DType dt = aether::dtype_of<aether::banded::Band>();
    return static_cast<int>(dt.bits());
}
EOF
arm DTYPE-BAND fail "BandedReal" "${WORK}/DTYPE-BAND.cpp"

# ---- ARM LADDER-GAP-BAND ---------------------------------------------------
cat > "${WORK}/LADDER-GAP-BAND.cpp" <<'EOF'
#include "@BAND_H@"
int main()
{
    /* 30 lands in the UNCERTIFIED GAP between the Ff1 (24) and Ff2 (45)
     * rungs -- MUST NOT COMPILE (no ladder rung certifies a 30-bit demand). */
    using Bad = aether::banded::BandedDemandT<30, aether::banded::BandedReal>;
    Bad x{};
    return sizeof(x) > 0 ? 0 : 1;
}
EOF
arm LADDER-GAP-BAND fail "certified set" "${WORK}/LADDER-GAP-BAND.cpp"

# ---- ARM CONTROL ----------------------------------------------------------
cat > "${WORK}/CONTROL.cpp" <<'EOF'
#include "@BAND_H@"
int main()
{
    return thisSymbolDoesNotExist();  /* MUST NOT COMPILE — proves the gate compiles */
}
EOF
arm CONTROL fail "-" "${WORK}/CONTROL.cpp"

# ---- ARM POSITIVE ---------------------------------------------------------
cat > "${WORK}/POSITIVE.cpp" <<'EOF'
#include "@BAND_H@"
#include "@AETHER_H@"
#include <vector>
int main()
{
    using aether::banded::Band;
    using aether::banded::BandedReal;

    /* the sanctioned host ingest and egress, both NAMED */
    const BandedReal a = BandedReal::fromDouble(1.5);
    const BandedReal b = BandedReal::fromDouble(-0.25);
    const double back  = a.toDouble();
    /* the sanctioned explicit ingest constructor */
    const BandedReal c(2.5);

    /* the implicit demote to the working carrier, and the implicit pack back */
    const Band x       = a;
    const BandedReal p = x;

    /* the namespace-scope operator set: chain in Band, ONE pack at the `=` */
    const BandedReal q = (a - b) * c / a;

    /* the four LIVE certified facade entry points */
    const Band m  = aether::math::fmax(a, b);
    const Band n  = aether::math::fmin(x, b);
    const Band v  = aether::math::abs(b);
    const Band cs = aether::math::copysign(a, b);

    /* the storage face exports bit-transparently */
    const aether::DType dt = aether::dtype_of<BandedReal>();

    /* an element write through a real View<BandedReal, ...> */
    std::vector<BandedReal> buf(3 * 4);
    auto view = aether::make_view<BandedReal, 3, aether::dyn>(
        buf.data(), aether::Device(kDLCPU), 4);
    const aether::SampleIndex i = aether::SampleIndex::make(0);
    view[i] = view[i].get() + view[i].get();

    return (back > 0.0 && p.toBits() != 0ull && q.toBits() != 0ull
               && dt.bits() == 64 && m.hi + n.hi + v.hi + cs.hi != 0.0f)
        ? 0
        : 1;
}
EOF
arm POSITIVE compile "-" "${WORK}/POSITIVE.cpp"

echo "=========================================="
if [[ "${FAIL}" -ne 0 ]]; then
    echo "BANDEDREAL TYPING GATE: RED"
    exit 1
fi
echo "BANDEDREAL TYPING GATE: GREEN"
