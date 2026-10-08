#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# tests/check_gate.sh — aether C++ suite integrity gate.
#
# WHY THIS EXISTS. gtest exits 0 having run NOTHING when a filter matches nothing or the
# source glob came up empty: `--gtest_filter=NoSuchTest` prints "[  PASSED  ] 0 tests." and
# returns 0. The exit code cannot tell "everything passed" from "there was nothing to run".
# Worse, an ambient GTEST_FILTER env var silences the run and the listing IDENTICALLY, so a
# self-referential "ran == listed" invariant agrees with itself while running a subset. The
# only non-self-referential reference is a committed name manifest — this script compares
# both the run and an ENV-SCRUBBED listing against git, never against each other.
#
# Usage:
#   tests/check_gate.sh <cuda|cpp> <path/to/aether_tests> [--config debug|release]
#   tests/check_gate.sh <cuda|cpp> <path/to/aether_tests> [--config ...] --remint [--allow-removals]
#
# There is deliberately NO count argument: the expectation is the manifest data file
# tests/expected_tests_<mode>.txt, never a literal on a command line or in CI YAML.
#
# MANIFEST GRAMMAR (config-tagged rows). The compiled test SET depends on AETHER_DEBUG_MODE:
# tests wrapped in `#ifdef AETHER_DEBUG_MODE` exist only in a debug build. One manifest per
# mode remains the single source of truth; it now DECLARES that config dimension instead of
# pretending it does not exist. A row is either
#     Suite.Test            present in BOTH configs
#     Suite.Test @debug     present ONLY when AETHER_DEBUG_MODE=ON
# with exactly one space before the literal tag. A DEBUG build must list bare AND tagged; a
# RELEASE build must list the bare rows and NONE of the tagged ones — a @debug row that
# turns up in a release binary is a WRONG TAG and is RED.
#
# The config of the gated tree is read from its own build record (AETHER_DEBUG_MODE:BOOL in
# the CMakeCache.txt above the binary) and declared by the lane via --config; when both are
# available they MUST agree, which is what catches a mislabelled CI lane. The config is
# NEVER inferred from the observed listing: that is circular and would vacuously bless a
# debug build that silently lost its debug-only tests.
#
# Checks (C1-C7):
#   C1  a gtest summary line exists and reports ran >= 1
#   C2  ran == this lane's expected count
#   C3  env-scrubbed live listing set == this lane's expected set, BOTH directions
#   C4  no summary line => RED (abort path)
#   C5  the binary's own rc != 0 => RED even when every count matches
#   C6  manifest hygiene: non-empty, newline-terminated, no blanks/comments, sorted, unique,
#       at most one ' @debug' suffix per row, and no NAME declared both bare and tagged
#   C7  the lane's config is DECLARED (CMakeCache.txt and/or --config) and the two agree
#
set -u -o pipefail

PROG="tests/check_gate.sh"
BIN_LABEL="aether_tests"

usage() {
    cat >&2 <<EOF
usage: $PROG <cuda|cpp> <path/to/$BIN_LABEL> [--config debug|release] [--remint [--allow-removals]]

  <cuda|cpp>          selects tests/expected_tests_<mode>.txt as the pinned manifest
  --config C          declares the build config of the gated tree (the per-lane toggle);
                      cross-checked against AETHER_DEBUG_MODE:BOOL in that tree's CMakeCache.txt
  --remint            regenerate that manifest from this binary (local only; refused in CI)
  --allow-removals    required when a re-mint would DELETE manifest lines

No count is ever passed on the command line — the manifest file is the expectation.
Rows tagged ' @debug' are expected ONLY in a AETHER_DEBUG_MODE=ON build.
EOF
    exit 2
}

MODE=""
BIN=""
REMINT=0
ALLOW_REMOVALS=0
CONFIG_FLAG=""
while [ $# -gt 0 ]; do
    case "$1" in
        --remint)         REMINT=1 ;;
        --allow-removals) ALLOW_REMOVALS=1 ;;
        --config)         [ $# -ge 2 ] || { echo "$PROG: --config needs a value" >&2; usage; }
                          CONFIG_FLAG="$2"; shift ;;
        --config=*)       CONFIG_FLAG="${1#--config=}" ;;
        -h|--help)        usage ;;
        -*)               echo "$PROG: unknown option '$1'" >&2; usage ;;
        *)
            if   [ -z "$MODE" ]; then MODE="$1"
            elif [ -z "$BIN"  ]; then BIN="$1"
            else echo "$PROG: unexpected argument '$1'" >&2; usage
            fi ;;
    esac
    shift
done
[ -n "$MODE" ] && [ -n "$BIN" ] || usage
case "$MODE" in cuda|cpp) ;; *) echo "$PROG: mode must be 'cuda' or 'cpp', got '$MODE'" >&2; usage ;; esac
case "$CONFIG_FLAG" in
    ""|debug|release) ;;
    *) echo "$PROG: --config must be 'debug' or 'release', got '$CONFIG_FLAG'" >&2; usage ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$SCRIPT_DIR/expected_tests_${MODE}.txt"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ ! -x "$BIN" ]; then
    echo "GATE RED: '$BIN' is not an executable file" >&2
    exit 1
fi
BIN_ABS="$(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")"
BIN_DIR="$(dirname "$BIN_ABS")"
BIN_EXE="./$(basename "$BIN_ABS")"
# aether_tests lives at <build-tree>/tests/aether_tests.
BUILD_TREE="$(dirname "$BIN_DIR")"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

if [ "$MODE" = "cuda" ] && [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "$PROG: CUDA_VISIBLE_DEVICES is unset; GPU selection and serialisation are the" \
         "caller's responsibility and this run may collide with other GPU processes" >&2
fi

# ---------------------------------------------------------------------------
# C7 — the lane's config, from two sources, NEITHER of them the test listing:
#   1. the tree's OWN build record — AETHER_DEBUG_MODE:BOOL in the CMakeCache.txt at or above
#      the binary (build/tests/aether_tests -> build/CMakeCache.txt), searched up 3 levels;
#   2. --config, the lane's declaration (CI passes it explicitly, so a lane that is
#      mislabelled in YAML reds HERE rather than as a puzzling set mismatch).
# Both available => they MUST agree. Exactly one => it rules. Neither => RED: a gate that
# guesses which set to expect is a gate that expects whatever it was given.
# ---------------------------------------------------------------------------
CACHE_FILE=""
CONFIG_CACHE=""
CACHE_PROBE="$BIN_DIR"
for _ in 0 1 2 3; do
    if [ -f "$CACHE_PROBE/CMakeCache.txt" ]; then CACHE_FILE="$CACHE_PROBE/CMakeCache.txt"; break; fi
    [ "$CACHE_PROBE" = "/" ] && break
    CACHE_PROBE="$(dirname "$CACHE_PROBE")"
done
if [ -n "$CACHE_FILE" ]; then
    if CACHE_LINE="$(grep -m1 '^AETHER_DEBUG_MODE:BOOL=' "$CACHE_FILE")"; then
        case "$(printf '%s' "${CACHE_LINE#*=}" | tr '[:lower:]' '[:upper:]')" in
            ON|1|TRUE|YES|Y)   CONFIG_CACHE="debug" ;;
            OFF|0|FALSE|NO|N)  CONFIG_CACHE="release" ;;
            *) echo "GATE RED: [C7] $CACHE_FILE carries an unreadable '$CACHE_LINE'" >&2; exit 1 ;;
        esac
    else
        echo "$PROG: note: $CACHE_FILE has no AETHER_DEBUG_MODE:BOOL entry" >&2
    fi
fi
if [ -n "$CONFIG_FLAG" ] && [ -n "$CONFIG_CACHE" ]; then
    if [ "$CONFIG_FLAG" != "$CONFIG_CACHE" ]; then
        echo "GATE RED: [C7] CONFIG MISMATCH — the lane declares --config $CONFIG_FLAG but this tree" >&2
        echo "         was CONFIGURED $CONFIG_CACHE (AETHER_DEBUG_MODE:BOOL in $CACHE_FILE)." >&2
        echo "         One of the two is wrong, and a mislabelled lane gates the WRONG test set." >&2
        echo "" >&2
        echo "VERDICT: RED" >&2
        exit 1
    fi
    CONFIG="$CONFIG_FLAG"
    CONFIG_SRC="--config, agreed by $CACHE_FILE"
elif [ -n "$CONFIG_FLAG" ]; then
    CONFIG="$CONFIG_FLAG"
    CONFIG_SRC="--config (no AETHER_DEBUG_MODE:BOOL found near the binary to cross-check)"
elif [ -n "$CONFIG_CACHE" ]; then
    CONFIG="$CONFIG_CACHE"
    CONFIG_SRC="$CACHE_FILE"
else
    echo "GATE RED: [C7] cannot determine this tree's config: no AETHER_DEBUG_MODE:BOOL in a" >&2
    echo "         CMakeCache.txt at or above '$BIN_DIR', and no --config given." >&2
    echo "         Pass --config debug or --config release — the lane knows what it built." >&2
    exit 1
fi

REMINT_CMD="$PROG $MODE $BIN --config $CONFIG --remint"

# ---------------------------------------------------------------------------
# Listing normalisation. gtest prints
#     Suite.                       [# TypeParam = ...]
#       TestName                   [# GetParam() = ...]
# Comment tails are stripped (they carry TypeParam/GetParam text and would make the
# manifest churn on unrelated edits); the result is one full Suite.Test name per line.
# ---------------------------------------------------------------------------
normalize() {
    awk '
        /^[^ \t]/ {
            line = $0
            sub(/[ \t]*#.*$/, "", line); sub(/[ \t]+$/, "", line)
            if (line ~ /\.$/) { suite = line } else { suite = "" }
            next
        }
        /^[ \t]+[^ \t]/ {
            if (suite == "") next
            line = $0
            sub(/^[ \t]+/, "", line); sub(/[ \t]*#.*$/, "", line); sub(/[ \t]+$/, "", line)
            if (line != "") print suite line
        }
    '
}

# LOCK: the LISTING is env-scrubbed. An ambient GTEST_FILTER filters the listing too,
# which is exactly how a self-referential invariant is defeated.
list_tests_scrubbed() {
    ( cd "$BIN_DIR" && env -u GTEST_FILTER -u TESTBRIDGE_TEST_ONLY \
        "$BIN_EXE" --gtest_list_tests ) 2>"$TMP/list.err"
}

# ---------------------------------------------------------------------------
# Manifest row classes. Both helpers emit plain Suite.Test names: the ' @debug' tag is a
# DECLARATION about the config dimension, never part of a test's name.
# ---------------------------------------------------------------------------
manifest_bare()   { grep -v ' @debug$' "$MANIFEST" | LC_ALL=C sort -u; }
manifest_tagged() { grep    ' @debug$' "$MANIFEST" | sed 's/ @debug$//' | LC_ALL=C sort -u; }
MANIFEST_SHAPE_RE='^[^[:space:]#]+\.[^[:space:]#]+( @debug)?$'

# ---------------------------------------------------------------------------
# --remint
# ---------------------------------------------------------------------------
if [ "$REMINT" -eq 1 ]; then
    # LOCK: CI compares, never regenerates. Regeneration inside the gate is the
    # self-fulfilling trap the whole mechanism exists to prevent.
    if [ -n "${CI:-}" ]; then
        echo "REFUSED: --remint is a LOCAL command; \$CI is set." >&2
        echo "         CI compares against the committed manifest and never regenerates it." >&2
        exit 2
    fi
    list_tests_scrubbed | normalize | LC_ALL=C sort -u > "$TMP/observed.txt"
    LIST_RC=${PIPESTATUS[0]}
    if [ "$LIST_RC" -ne 0 ]; then
        echo "REFUSED: listing the binary failed (rc $LIST_RC); refusing to mint from it." >&2
        sed 's/^/  | /' "$TMP/list.err" >&2
        exit 1
    fi
    if [ ! -s "$TMP/observed.txt" ]; then
        echo "REFUSED: the binary listed ZERO tests — refusing to mint an empty manifest." >&2
        exit 1
    fi

    # A tree speaks ONLY for the row class it can see, so the existing manifest is split
    # first — and a malformed manifest cannot be split at all.
    : > "$TMP/bare_old.txt"
    : > "$TMP/tagged_old.txt"
    if [ -f "$MANIFEST" ]; then
        if grep -nvE "$MANIFEST_SHAPE_RE" "$MANIFEST" > "$TMP/shape.txt"; then
            echo "REFUSED: $MANIFEST has malformed rows; fix them before minting from a tree:" >&2
            sed 's/^/  | /' "$TMP/shape.txt" >&2
            exit 1
        fi
        manifest_bare   > "$TMP/bare_old.txt"
        manifest_tagged > "$TMP/tagged_old.txt"
    fi

    if [ "$CONFIG" = "release" ]; then
        # A RELEASE tree can only speak for the BARE rows: a @debug row is by construction
        # absent from this build, so it is preserved verbatim and reported as unverified.
        ROW_CLASS="bare"
        LC_ALL=C comm -23 "$TMP/bare_old.txt" "$TMP/observed.txt" > "$TMP/removed.txt"
        LC_ALL=C comm -13 "$TMP/bare_old.txt" "$TMP/observed.txt" > "$TMP/added.txt"
        # A name tagged @debug yet PRESENT here is a wrong tag, not a new bare row; minting
        # it would produce a name declared in both classes.
        LC_ALL=C comm -12 "$TMP/tagged_old.txt" "$TMP/observed.txt" > "$TMP/wrongtag.txt"
        if [ -s "$TMP/wrongtag.txt" ]; then
            echo "REFUSED: these rows are tagged ' @debug' yet PRESENT in this RELEASE build:" >&2
            sed 's/^/  ! /' "$TMP/wrongtag.txt" >&2
            echo "         They are unconditional — drop the tag by hand, then re-mint." >&2
            exit 1
        fi
    else
        # A DEBUG tree speaks for the TAGGED rows: observed MINUS the bare rows. It must
        # contain every bare row, or the two classes disagree on what 'unconditional' means.
        ROW_CLASS="@debug"
        LC_ALL=C comm -23 "$TMP/bare_old.txt" "$TMP/observed.txt" > "$TMP/missing_bare.txt"
        if [ -s "$TMP/missing_bare.txt" ]; then
            echo "REFUSED: a DEBUG build must be a SUPERSET of the bare rows; these are absent:" >&2
            sed 's/^/  - /' "$TMP/missing_bare.txt" >&2
            echo "         --allow-removals does NOT apply here: bare rows are minted on a RELEASE" >&2
            echo "         tree. Re-mint the release tree first, then re-run this mint." >&2
            exit 1
        fi
        LC_ALL=C comm -13 "$TMP/bare_old.txt" "$TMP/observed.txt" > "$TMP/tagged_new.txt"
        LC_ALL=C comm -23 "$TMP/tagged_old.txt" "$TMP/tagged_new.txt" > "$TMP/removed.txt"
        LC_ALL=C comm -13 "$TMP/tagged_old.txt" "$TMP/tagged_new.txt" > "$TMP/added.txt"
    fi

    if [ -s "$TMP/removed.txt" ] && [ "$ALLOW_REMOVALS" -ne 1 ]; then
        echo "" >&2
        echo "  ####################################################################" >&2
        echo "  #  RE-MINT REFUSED: this would REMOVE $(wc -l < "$TMP/removed.txt") $ROW_CLASS test(s) from the manifest." >&2
        echo "  #  Tests do not normally disappear. Read this list before you agree:" >&2
        echo "  ####################################################################" >&2
        sed 's/^/  - /' "$TMP/removed.txt" >&2
        echo "" >&2
        echo "  If every removal above is intended, re-run with:" >&2
        echo "      $REMINT_CMD --allow-removals" >&2
        echo "  and commit the manifest diff IN THE SAME COMMIT as the test change." >&2
        exit 1
    fi
    sed 's/^/  - /' "$TMP/removed.txt"
    sed 's/^/  + /' "$TMP/added.txt"

    # Rewrite: this tree's class from the LISTING, the other class carried over verbatim.
    if [ "$CONFIG" = "release" ]; then
        { cat "$TMP/observed.txt"; sed 's/$/ @debug/' "$TMP/tagged_old.txt"; } \
            | LC_ALL=C sort -u > "$TMP/new.txt"
    else
        { cat "$TMP/bare_old.txt"; sed 's/$/ @debug/' "$TMP/tagged_new.txt"; } \
            | LC_ALL=C sort -u > "$TMP/new.txt"
    fi
    cp "$TMP/new.txt" "$MANIFEST"
    N_BARE=$(manifest_bare   | awk 'END{print NR}')
    N_TAG=$(manifest_tagged  | awk 'END{print NR}')
    echo "re-minted $MANIFEST from a $CONFIG tree: $N_BARE bare + $N_TAG @debug = $(wc -l < "$MANIFEST") rows"
    if [ "$CONFIG" = "release" ] && [ "$N_TAG" -gt 0 ]; then
        echo "NOT VERIFIED BY THIS MINT — a release tree cannot see these; preserved verbatim:"
        manifest_tagged | sed 's/^/  ~ /'
        echo "Re-mint on a AETHER_DEBUG_MODE=ON tree to re-derive them."
    fi
    echo "Commit this manifest diff IN THE SAME COMMIT as the test change."
    exit 0
fi

# ---------------------------------------------------------------------------
# C6 — manifest hygiene. A malformed manifest is a broken instrument, not a pass.
# ---------------------------------------------------------------------------
RED=0
red() { echo "GATE RED: $*" >&2; RED=1; }

if [ ! -f "$MANIFEST" ]; then
    echo "GATE RED: no manifest at $MANIFEST — mint it with: $REMINT_CMD" >&2
    exit 1
fi
ROWS=$(awk 'END{print NR}' "$MANIFEST")
if [ "$ROWS" -lt 1 ]; then
    red "[C6] manifest $MANIFEST is EMPTY — an empty expectation certifies nothing"
fi
if [ -s "$MANIFEST" ] && [ "$(tail -c1 "$MANIFEST" | wc -l)" -ne 1 ]; then
    red "[C6] manifest $MANIFEST is not newline-terminated"
fi
# One 'Suite.Test' name per line, optionally followed by the single literal ' @debug' tag,
# and nothing else: this one shape test rejects blank lines, comments, stray whitespace,
# a mistyped or repeated tag, and any hand-editing that smuggled a count in.
if grep -nvE "$MANIFEST_SHAPE_RE" "$MANIFEST" > "$TMP/shape.txt"; then
    red "[C6] manifest $MANIFEST has lines that are not 'Suite.Test' or 'Suite.Test @debug':"
    sed 's/^/         /' "$TMP/shape.txt" >&2
fi
if ! LC_ALL=C sort -c "$MANIFEST" 2>"$TMP/sortc.txt"; then
    red "[C6] manifest $MANIFEST is NOT sorted (LC_ALL=C): $(cat "$TMP/sortc.txt")"
fi
UNIQ=$(LC_ALL=C sort -u "$MANIFEST" | awk 'END{print NR}')
if [ "$UNIQ" -ne "$ROWS" ]; then
    red "[C6] manifest $MANIFEST has duplicate lines ($ROWS lines, $UNIQ unique):"
    LC_ALL=C sort "$MANIFEST" | uniq -d | sed 's/^/         /' >&2
fi
# A NAME may not be declared in both classes: "present always AND only in debug" is not a
# statement about the world, it is an expectation that cannot be wrong.
{ manifest_bare; manifest_tagged; } | LC_ALL=C sort | uniq -d > "$TMP/bothclass.txt"
if [ -s "$TMP/bothclass.txt" ]; then
    red "[C6] manifest $MANIFEST declares these names BOTH bare and ' @debug':"
    sed 's/^/         /' "$TMP/bothclass.txt" >&2
fi

# ---------------------------------------------------------------------------
# This lane's expectation, exact (rule 3 of the tagged-manifest design):
#   release: the bare rows and NOTHING else — every ' @debug' row is REQUIRED ABSENT, so a
#            wrong tag surfaces instead of being quietly tolerated;
#   debug:   bare + tagged exactly — a debug build that lost a debug-only test still reds,
#            and so does one that lost a bare one.
# ---------------------------------------------------------------------------
manifest_bare   > "$TMP/bare.txt"
manifest_tagged > "$TMP/tagged.txt"
N_BARE=$(awk 'END{print NR}' "$TMP/bare.txt")
N_TAG=$(awk 'END{print NR}' "$TMP/tagged.txt")
if [ "$CONFIG" = "debug" ]; then
    TAG_ROLE="required-present"
    LC_ALL=C sort -u "$TMP/bare.txt" "$TMP/tagged.txt" > "$TMP/expected.txt"
else
    TAG_ROLE="required-absent"
    cp "$TMP/bare.txt" "$TMP/expected.txt"
fi
EXPECTED=$(awk 'END{print NR}' "$TMP/expected.txt")

# ---------------------------------------------------------------------------
# C3 — env-scrubbed live listing vs the manifest, both directions.
# ---------------------------------------------------------------------------
list_tests_scrubbed | normalize | LC_ALL=C sort -u > "$TMP/listed.txt"
LIST_RC=${PIPESTATUS[0]}
LISTED=$(awk 'END{print NR}' "$TMP/listed.txt")
if [ "$LIST_RC" -ne 0 ]; then
    red "[C3] '--gtest_list_tests' failed with rc $LIST_RC"
    sed 's/^/         | /' "$TMP/list.err" >&2
fi
LC_ALL=C comm -23 "$TMP/expected.txt" "$TMP/listed.txt" > "$TMP/missing.txt"
LC_ALL=C comm -13 "$TMP/expected.txt" "$TMP/listed.txt" > "$TMP/extra.txt"
if [ -s "$TMP/missing.txt" ] || [ -s "$TMP/extra.txt" ]; then
    red "[C3] the built binary's test SET differs from $MANIFEST for a $CONFIG build (listed $LISTED, pinned $EXPECTED)"
    if [ -s "$TMP/missing.txt" ]; then
        echo "         MISSING (pinned but NOT in the binary — deleted, renamed, or NOT BUILT):" >&2
        sed 's/^/         - /' "$TMP/missing.txt" >&2
    fi
    if [ -s "$TMP/extra.txt" ]; then
        echo "         UNEXPECTED (in the binary but NOT pinned — new tests, un-minted):" >&2
        sed 's/^/         + /' "$TMP/extra.txt" >&2
    fi
fi
# The release lane's POSITIVE statement about the tagged rows. Saying it out loud keeps the
# wrong-tag guard visible: a ' @debug' row that shows up in a release build is not a
# harmless extra, it is a false declaration about when that test exists.
if [ "$CONFIG" = "release" ] && [ "$N_TAG" -gt 0 ]; then
    LC_ALL=C comm -12 "$TMP/tagged.txt" "$TMP/listed.txt" > "$TMP/tagpresent.txt"
    if [ -s "$TMP/tagpresent.txt" ]; then
        red "[C3] $(wc -l < "$TMP/tagpresent.txt") row(s) declared ' @debug' are PRESENT in this RELEASE build — WRONG TAG:"
        sed 's/^/         ! /' "$TMP/tagpresent.txt" >&2
        echo "         Either the test is unconditional (drop the tag) or its #ifdef is wrong." >&2
    else
        echo "[C3] $N_TAG ' @debug' row(s) REQUIRED ABSENT, confirmed absent from this release build:"
        sed 's/^/       ~ /' "$TMP/tagged.txt"
    fi
fi

# ---------------------------------------------------------------------------
# Run the suite. The RUN is deliberately NOT env-scrubbed: an ambient
# GTEST_FILTER must make this gate RED, not be laundered into a clean run.
# ---------------------------------------------------------------------------
LOG="$BIN_DIR/gtest.log"
XML="$(basename "$BIN_ABS").xml"
echo "== $PROG: running $BIN_ABS (mode=$MODE, config=$CONFIG, pinned $EXPECTED tests) =="
( cd "$BIN_DIR" && time "$BIN_EXE" --gtest_repeat=1 --gtest_break_on_failure \
    --gtest_shuffle --gtest_output="xml:$XML" ) 2>&1 | tee "$LOG"
BIN_RC=${PIPESTATUS[0]}

# C5 — the binary's own rc, judged on its own. `| tee` above would otherwise
# hand the pipeline's status to tee and mask a failing suite entirely.
if [ "$BIN_RC" -ne 0 ]; then
    red "[C5] the test binary exited with rc $BIN_RC — the suite FAILED (counts are irrelevant)"
fi

# C1/C4 — the summary line must exist and report ran >= 1.
RAN=$(grep -oP '^\[==========\] \K[0-9]+(?= tests? from .* ran)' "$LOG" | tail -1)
if [ -z "$RAN" ]; then
    red "[C4] no gtest summary line in $LOG — the suite never reported a run (aborted, crashed, or not a gtest binary)"
elif [ "$RAN" -lt 1 ]; then
    red "[C1] the suite ran $RAN tests — gtest exits 0 having run NOTHING when a filter matches nothing"
elif [ "$RAN" -ne "$EXPECTED" ]; then
    # C2
    red "[C2] ran $RAN tests, manifest pins $EXPECTED"
    if [ "$RAN" -lt "$EXPECTED" ]; then
        echo "         A run smaller than the listing is the FILTER class: check GTEST_FILTER" >&2
        echo "         (GTEST_FILTER='${GTEST_FILTER:-<unset>}', TESTBRIDGE_TEST_ONLY='${TESTBRIDGE_TEST_ONLY:-<unset>}')" >&2
    fi
fi

echo "-----------------------------------------------------------------------"
echo "mode=$MODE  config=$CONFIG  [from: $CONFIG_SRC]"
echo "manifest=$MANIFEST  rows=$ROWS (bare=$N_BARE, @debug=$N_TAG $TAG_ROLE)"
echo "pinned=$EXPECTED  listed=$LISTED  ran=${RAN:-<none>}  binary_rc=$BIN_RC"
if [ "$RED" -ne 0 ]; then
    echo "" >&2
    echo "VERDICT: RED" >&2
    echo "If — and only if — the suite legitimately changed, re-mint DELIBERATELY with:" >&2
    echo "    $REMINT_CMD" >&2
    echo "and commit the manifest diff in the SAME commit as the test change. Never silence." >&2
    exit 1
fi
echo "VERDICT: GREEN"
exit 0
