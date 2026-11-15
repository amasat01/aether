#!/bin/sh
# The Rust legs of the gate. Usage: sh ci/gate_rs.sh [leg ...]   (no argument: every leg, in this order)
#   fmt deps globals test clippy modules
# RAPTOR_IR_LIB names the built raptor_ir static library (default: the sibling raptor checkout's release build).
# Each leg prints what it examined and is RED when it examined nothing. The exit status is non-zero when any leg is red.
cd "$(dirname "$0")/.." || exit 2
JOBS=${JOBS:-2}
CRATE=rs/writers
RAPTOR_IR_LIB=${RAPTOR_IR_LIB:-$(cd .. && pwd)/raptor/target/release/libraptor_ir.a}
export RAPTOR_IR_LIB
out=$(mktemp "${TMPDIR:-/tmp}/gate_rs.XXXXXX")
trap 'rm -f "$out" "$out".*' EXIT

need() { [ "$2" -gt 0 ] || { echo "RED: $1 examined nothing"; return 1; }; }

leg_fmt() {
  cargo fmt --manifest-path $CRATE/Cargo.toml --check >"$out" 2>&1 || { head -5 "$out"; echo "RED: fmt differs"; return 1; }
  n=$(git ls-files $CRATE | grep -c '\.rs$'); echo "examined: $n rust files"; need fmt "$n"
}

# the crate names no raptor crate: its [dependencies] table is empty and no dependency table mentions raptor
leg_deps() {
  n=$(awk '/^\[/ { sec = $0 } sec ~ /dependencies\]$/ && $0 !~ /^(\[|#|[ \t]*$)/ { c++ } END { print c + 0 }' $CRATE/Cargo.toml)
  r=$(awk '/^\[/ { sec = $0 } sec ~ /dependencies\]$/ && /raptor/ { c++ } END { print c + 0 }' $CRATE/Cargo.toml)
  has=$(grep -c '^\[dependencies\]' $CRATE/Cargo.toml)
  echo "examined: Cargo.toml with $has [dependencies] table, $n dependency entries, $r naming raptor"
  need deps "$has" || return 1
  [ "$n" -eq 0 ] && [ "$r" -eq 0 ] || { echo "RED: the dependency tables are not empty"; return 1; }
}

leg_globals() {
  files=$(git ls-files $CRATE/src | grep '\.rs$'); n=$(echo "$files" | grep -c .)
  echo "examined: $n rust files in $CRATE/src"; need globals "$n" || return 1
  # shellcheck disable=SC2086
  hits=$(grep -nE 'static +mut|thread_local!|OnceLock|LazyLock|OnceCell|lazy_static|static +[A-Za-z_0-9]+ *: *[A-Za-z_:]*(Mutex|RwLock|Atomic[A-Za-z0-9]*)' $files)
  [ -z "$hits" ] || { echo "$hits" | head -10; echo "RED: global mutable state: $(echo "$hits" | wc -l | tr -d ' ') site(s)"; return 1; }
}

# cargo test against the built raptor library; the door test must run at least one test
leg_test() {
  [ -f "$RAPTOR_IR_LIB" ] || { echo "RED: RAPTOR_IR_LIB not found: $RAPTOR_IR_LIB (build raptor_ir first)"; return 1; }
  (cd $CRATE && cargo test --release -j "$JOBS") >"$out" 2>&1
  rc=$?
  total=$(awk '/^test result:/ { p += $4 } END { print p + 0 }' "$out")
  door=$(awk '/Running .*door/ { d = 1; next } d && /^test result:/ { print $4; exit }' "$out")
  grep -E "FAILED|panicked|^error" "$out" | head -5
  echo "examined: $total tests passed, door test passed ${door:-0}"
  [ $rc -eq 0 ] || { echo "RED: cargo test"; return 1; }
  need "door test" "${door:-0}" || return 1; need test "$total"
}

# clippy with every warning denied, in the three feature configurations the crate has (none, the replay tool's, development)
leg_clippy() {
  n=0
  for feats in "" "old-contract" "old-contract,shadow"; do
    (cd $CRATE && cargo clippy --release -j "$JOBS" --all-targets ${feats:+--features "$feats"} -- -D warnings) >"$out" 2>&1
    rc=$?
    n=$((n + 1))
    echo "examined: clippy on $CRATE, features '${feats:-none}', warnings denied"
    [ $rc -eq 0 ] || { grep -E '^(warning|error)' -A3 "$out" | head -12; echo "RED: clippy with features '${feats:-none}'"; return 1; }
  done
  need clippy "$n"
}

# The edges between the crate's modules (see WRITERS_DESIGN.md section 1): one line per `crate::` path in code, as
# "file line from to". `from` and `to` are top-level modules; `unit/host` and `unit/device` are their own modules, and the
# contract's reader and old-shape converter are told apart from its typed `types` (target `contract::<name>`).
module_edges() {
  for f in $(git ls-files $CRATE/src | grep '\.rs$'); do
    rel=${f#$CRATE/src/}
    [ "$rel" = lib.rs ] && continue
    awk -v file="$rel" '
      BEGIN { n = split(file, p, "/"); from = p[1]; sub(/\.rs$/, "", from)
              if (from == "unit" && (p[2] == "host" || p[2] == "device")) from = "unit/" p[2] }
      /^[ \t]*\/\// { next }
      { code = $0; sub(/\/\/.*/, "", code)
        while (match(code, /crate::[A-Za-z_0-9]+(::[A-Za-z_0-9]+)?/)) {
          path = substr(code, RSTART + 7, RLENGTH - 7); code = substr(code, RSTART + RLENGTH)
          split(path, q, "::"); to = q[1]
          if (to == "contract") to = "contract::" q[2]
          else if (to == "unit" && (q[2] == "host" || q[2] == "device")) to = "unit/" q[2]
          print file, NR, from, to } }' "$f"
  done
}

# whether `from` may use `to`: refuse uses nothing; contract, ir and spell use only refuse; body uses the leaves (the
# contract only through its typed `types`); lanes uses body and the leaves; unit uses all; unit/host and unit/device
# never use each other. A module may always use itself, and unit/host and unit/device their parent.
leg_modules() {
  module_edges >"$out.edges"
  n=$(grep -c . "$out.edges"); echo "examined: $n crate:: paths in $CRATE/src"
  need modules "$n" || return 1
  awk '{ c[$3 " -> " $4]++ } END { for (k in c) printf "  %-32s %d\n", k, c[k] }' "$out.edges" | sort
  awk '
    function leaf(t) { return t == "refuse" || t == "ir" || t == "spell" || t == "contract::types" }
    function json(t) { return t ~ /^contract::(Contract|Json|json|old|wire)/ }
    function ok(f, t,   top) { top = f; sub(/\/.*/, "", top)
      if (f == t || (top == "unit" && t == "unit") ) return 1
      if (f == "refuse") return 0
      if (f == "contract" || f == "ir" || f == "spell") return t == "refuse"
      if (json(t)) return top == "unit" && f == "unit"
      if (f == "body") return leaf(t)
      if (f == "lanes") return leaf(t) || t == "body"
      if (f == "unit/host") return t != "unit/device"
      if (f == "unit/device") return t != "unit/host"
      return 1 }
    { if (!ok($3, $4)) c[$1 ":" $3 ">" $4]++ }
    END { for (k in c) print k ":" c[k] }' "$out.edges" | sort >"$out.now"
  # a bare mention of the JSON reader below the unit entry is as bad as a path to it
  for f in $(git ls-files $CRATE/src | grep '\.rs$' | grep -vE "/(contract/|unit/entry.rs|unit/team_stage.rs|lib.rs)"); do
    grep -nE '\b(Json|Contract)\b' "$f" | grep -vE '^[0-9]+: *//' | sed -E "s|^([0-9]+):.*|${f#$CRATE/src/}:\1|" ; done >"$out.json"
  [ ! -s "$out.json" ] || { head -5 "$out.json"; echo "RED: the JSON reader is named below the unit entry"; return 1; }
  # the lane model and its entry points belong to lanes/: a file under body/ that names one of them knows the lanes
  lanes_words='CutLoop|LaneSegments|LaneSplit|lane_model|CutEmitter|SplitEmitter|lane_segments|lane_split|emit_cut|emit_split|segments_of'
  nb=$(git ls-files $CRATE/src/body | grep -c '\.rs$')
  echo "examined: $nb files under $CRATE/src/body for the lane vocabulary ($lanes_words)"
  for f in $(git ls-files $CRATE/src/body | grep '\.rs$'); do
    grep -nE "\\b($lanes_words)\\b" "$f" | grep -vE '^[0-9]+: *//' | sed -E "s|^([0-9]+):.*|${f#$CRATE/src/}:\1|" ; done >"$out.lanes"
  [ ! -s "$out.lanes" ] || { head -5 "$out.lanes"; echo "RED: the body names the lane vocabulary"; return 1; }
  sed 's/ *#.*//' ci/rs_modules_allowed.txt | grep . | sort >"$out.allow"
  echo "allow-list: $(grep -c . "$out.allow") entries (file:from>to:count)"
  bad=$(diff "$out.allow" "$out.now" | grep -E '^[<>]' | sed -e 's/^</gone or smaller, shrink the list:/' -e 's/^>/new or larger:/')
  [ -z "$bad" ] || { echo "$bad" | head -15; echo "RED: module edges differ from ci/rs_modules_allowed.txt"; return 1; }
}

legs=${*:-fmt deps globals test clippy modules}
green=; red=
for l in $legs; do
  echo "== $l"
  if "leg_$l"; then green="$green $l"; else red="$red $l"; fi
done
echo "GREEN:$green"; [ -z "$red" ] || echo "RED:$red"
[ -z "$red" ]
