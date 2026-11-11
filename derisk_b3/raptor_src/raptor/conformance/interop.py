# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The interop certification declaration and reusable matrix-hygiene scanners.

raptor is the framework-free contract owner: this module names every
framework the family will ever bridge to (:data:`UNIVERSE`), which subset is
certified today (:data:`CERTIFIED`) vs. declared-but-not-yet-built
(:data:`ROADMAP`), and the row-level ``ROWS`` catalogue, declared here in
full, owned here and registered against by the packages that write the
matrix tests — so that :func:`assert_rows_complete` has a real gap to catch
instead of declared and collected always landing together. Framework objects NEVER enter raptor —
pointers and callables are passed in duck-typed; this module itself imports
no framework, ever (``find_spec`` never imports).

Pure-Python, numpy-free by design — raptor has zero hard dependencies.
"""

from __future__ import annotations

import ast
import importlib.util
import re
from dataclasses import dataclass
from pathlib import Path

#: every framework the family will ever bridge to, day-0 and roadmap together.
UNIVERSE = ("numpy", "cupy", "torch", "warp", "jax", "tensorflow")

#: frameworks with an executable, certified round-trip today.
CERTIFIED = ("numpy", "cupy", "torch", "warp", "jax")

#: frameworks declared absent-by-design — no row certifies them yet.
ROADMAP = ("tensorflow",)

#: the vocabulary a :class:`RowDecl` may declare for how a row moves data.
#: ``ALIAS`` — the crossing shares the same underlying buffer, no copy (e.g.
#: DLPack zero-copy into another framework's tensor).
#: ``COPY`` — the crossing makes an independent copy of the data.
#: ``STREAM`` — the crossing's ordering relative to other work on the same
#: stream/queue (not whether data is shared or copied).
#: ``ENV`` — a precondition on the execution environment/state that gates
#: whether a crossing is legal (e.g. CUDA graph capture mode, device
#: placement, a layout self-check) — certifying a loud refusal under the
#: wrong state as much as acceptance under the right one.
#: ``INVARIANCE`` — the same plugin body produces equivalent results under
#: different execution structures (bit-exact for some structures, within a
#: tolerance band for others) — distinct from ALIAS/COPY (which certify a
#: framework crossing's pointer semantics) and from STREAM (ordering).
ROW_KINDS = frozenset({"ALIAS", "COPY", "STREAM", "ENV", "INVARIANCE"})

#: the platforms a row can be certified on. Both are legs of ONE GPU-capable
#: job (a GPU-less job deselects the whole matrix), so a row never skips a leg.
LEGS = ("cpu", "cuda")


@dataclass(frozen=True)
class RowDecl:
    """One declared interop-matrix row.

    ``framework`` must be a member of :data:`CERTIFIED`, OR ``None`` for a row
    that bridges to no array framework at all — an execution-structure
    invariant internal to eagle rather than a crossing to one of those.
    Every prior row, even ``CP-CAPTURE-REJECT``/``T-DEVICE-FAILLOUD``, still
    names the framework whose crossing it guards; widening ``framework`` to
    ``Optional[str]`` — rather than inventing a sixth pseudo-framework entry
    in :data:`CERTIFIED`/:data:`UNIVERSE` — keeps that vocabulary honestly
    scoped to actual array frameworks. ``owner_repo`` is the repo whose test
    suite carries the executable assertion, so the matrix rows live with the
    seam they certify; ``kind`` is one of :data:`ROW_KINDS`. ``legs`` names
    the platforms the row must pass on (a subset of :data:`LEGS`, default
    both): :func:`assert_rows_complete` needs a passing test on every one.
    """

    framework: str | None
    owner_repo: str
    kind: str
    legs: tuple[str, ...] = LEGS

    def __post_init__(self) -> None:
        if self.kind not in ROW_KINDS:
            raise ValueError(
                f"RowDecl.kind={self.kind!r} not in {sorted(ROW_KINDS)}"
            )
        if not self.legs or not set(self.legs) <= set(LEGS):
            raise ValueError(
                f"RowDecl.legs={self.legs!r} not a non-empty subset of {LEGS}"
            )


#: row_id -> RowDecl — the full row catalogue, declared here, before the
#: packages that write the tests. This is deliberate: if declaring and
#: collecting happened in the same later package, declared and collected
#: would always agree by construction and :func:`assert_rows_complete` could
#: never bite. Every row named elsewhere in the family exists below under its
#: catalogue ID.
ROWS: dict[str, RowDecl] = {
    # numpy (host) rows.
    # cupy — kernel-level rows and adapter/pipeline rows, split by owner.
    # CP-STREAM-ORDER is a named cross-reference to STREAM-CUPY-PRODUCER-ORDER
    # below, not a second row.
    "CP-CAPTURE-REJECT": RowDecl("cupy", "eagle", "ENV"),
    # torch — adapter+stream rows, kernel-level rows, and bridge+backend
    # rows, split by owner.
    "T-IN-CUDA-ALIAS": RowDecl("torch", "eagle", "ALIAS"),
    "T-OUT-CUDA-ALIAS": RowDecl("torch", "eagle", "ALIAS"),
    "T-IN-CPU-COPY": RowDecl("torch", "eagle", "COPY"),
    # A row separate from T-IN-CPU-COPY immediately above: that row stays
    # owner=eagle and keeps certifying eagle's own adapter crossing. This one
    # certifies a different seam: a native host provider round-trips a
    # conforming (float64, C-contiguous) torch CPU tensor by pointer-alias —
    # no upload, no eagle, no cupy. The declaration here is authoritative, an
    # explicit re-declaration rather than a quiet one.
    # Note eagle's adapter ALSO copies float32 (``eagle/python/eagle/interop.py``);
    # this row certifies the kernel-level crossing, per the grouping above.
    "T-BRIDGE-COPY": RowDecl("torch", "learning_runtime", "COPY"),
    "T-BACKEND-CPU-IN-ALIAS": RowDecl("torch", "learning_runtime", "ALIAS"),
    # Renamed from "T-BACKEND-CUDA": kind=ALIAS is true for the INPUT crossing
    # only — the learning runtime's bridge copies unconditionally on export
    # (any live replay buffer is copied off), so no backend row can be an
    # alias in the OUT direction — that copy is itself certified by
    # T-BRIDGE-COPY. The unqualified name would have published as "the CUDA
    # backend aliases", which is false for outputs; the ID now names its
    # direction, matching its CPU sibling.
    "T-BACKEND-CUDA-IN-ALIAS": RowDecl("torch", "learning_runtime", "ALIAS"),
    "T-DEVICE-FAILLOUD": RowDecl("torch", "learning_runtime", "ENV"),
    # stream rows — owner eagle, GPU-serial.
    "STREAM-TORCH-PRODUCER-ORDER": RowDecl("torch", "eagle", "STREAM"),
    "STREAM-CUPY-PRODUCER-ORDER": RowDecl("cupy", "eagle", "STREAM"),
    "STREAM-TORCH-CONSUMER-ORDER": RowDecl("torch", "eagle", "STREAM"),
    "STREAM-IDENTITY": RowDecl("torch", "eagle", "STREAM"),
    # warp — NVIDIA Warp crosses through eagle's generic zero-copy buffer
    # layer (eagle.interop.import_buffer), not a dedicated Adapter: any
    # __dlpack__ + __dlpack_device__ producer already routes through
    # eagle.interop._DlpackAdapter, so warp needed no new adapter code, only
    # these rows. Warp speaks legacy (pre-1.0) DLPack, so its access flag is
    # always "unknown" — that is part of WP-IN-CUDA-ALIAS, not a separate
    # row (same bundling convention T-IN-CUDA-ALIAS already uses for its own
    # three legs). It joins the other STREAM-* rows.
    "WP-IN-CUDA-ALIAS": RowDecl("warp", "eagle", "ALIAS"),
    "WP-OUT-CUDA-ALIAS": RowDecl("warp", "eagle", "ALIAS"),
    "STREAM-WARP-PRODUCER-ORDER": RowDecl("warp", "eagle", "STREAM"),
    # jax — also crosses through eagle's generic DLPack layer, no dedicated
    # array code. Contract: a jax array in a read-only role is aliased on
    # CUDA (uploaded on CPU); the jax result IS eagle's fresh result buffer
    # and eagle never writes it again; a jax array in a role the kernel
    # writes (out=, terminated, a Mutable plane) is refused by name, because
    # jax arrays are immutable and a silent copy would drop the update. jax
    # speaks legacy DLPack, so its access flag is "unknown" (or "read-only"
    # once it ships versioned capsules). jax rows run where jax is installed
    # (Python 3.12 or newer); every other environment deselects
    # ``-m "not interop_matrix"``.
    "JX-IN-CUDA-ALIAS": RowDecl("jax", "eagle", "ALIAS"),
    "JX-IN-WRITE-REFUSED": RowDecl("jax", "eagle", "ENV"),
    "JX-OUT-CUDA-ALIAS": RowDecl("jax", "eagle", "ALIAS"),
    "JX-IN-CPU-COPY": RowDecl("jax", "eagle", "COPY"),
    "STREAM-JAX-PRODUCER-ORDER": RowDecl("jax", "eagle", "STREAM"),
    # jax — the FFI bridge (``eagle.frameworks.jax``): a hawk kernel run as an
    # XLA custom call on XLA's own buffers and stream. Legs are platforms.
    # CPU: XLA's host buffers are read in place and the result is XLA's own.
    "JX-FFI-CPU-ALIAS": RowDecl("jax", "eagle", "ALIAS", ("cpu",)),
    # Outputs are XLA-allocated and zero where the kernel writes nothing, on a
    # reused (dirty) allocator, except carried-state planes and the mask, which
    # alias their operand; a donated mask comes back in place.
    "JX-OUT-XLA-OWNED": RowDecl("jax", "eagle", "ALIAS", ("cpu", "cuda")),
    # Eager, jit, a vmap-folded batch and a scan of steps give one answer.
    "JX-JIT-INVARIANCE": RowDecl("jax", "eagle", "INVARIANCE", ("cpu", "cuda")),
    # In one jit, a kernel launched after another XLA op rides XLA's stream:
    # no host wait, and it reads the finished values.
    "STREAM-JAX-FFI-ORDER": RowDecl("jax", "eagle", "STREAM", ("cuda",)),
    # the reader's own framework code inside an eagle graph: one seam, three
    # tiers (captured, stream-ordered, host-synchronized), a framework-captured
    # graph as a child node, per-step hooks, gradients through the reader's
    # model and weights bound live.
    "INJ-TIER-NAMED": RowDecl(None, "eagle", "ENV", ("cpu", "cuda")),
    "STREAM-INJECT-STREAM-ORDER": RowDecl("torch", "eagle", "STREAM", ("cuda",)),
    "STREAM-INJECT-HOST-ORDER": RowDecl("torch", "eagle", "STREAM", ("cuda",)),
    "T-CHILD-OWNED": RowDecl("torch", "eagle", "ALIAS", ("cuda",)),
    "INJ-LOOP-HOOK": RowDecl("torch", "eagle", "INVARIANCE", ("cpu", "cuda")),
    "INJ-VJP-CHAIN": RowDecl("torch", "eagle", "INVARIANCE", ("cpu", "cuda")),
    "INJ-RUN-VJP": RowDecl("torch", "eagle", "INVARIANCE", ("cpu", "cuda")),
    "INJ-TABLE-LIVE": RowDecl("torch", "eagle", "ALIAS", ("cuda",)),
    "STREAM-JAX-INJECT-ORDER": RowDecl("jax", "eagle", "STREAM", ("cuda",)),
    "JX-INJECT-HOST": RowDecl("jax", "eagle", "STREAM", ("cuda",)),
    "JX-LOOP-INVARIANCE": RowDecl("jax", "eagle", "INVARIANCE", ("cpu", "cuda")),
    # gradients through an eagle kernel, read by each framework's own machinery
    # (both legs), each against the same law written in that framework alone.
    "JX-GRAD-TRANSFORMS": RowDecl("jax", "eagle", "INVARIANCE"),
    "T-GRAD-AUTOGRAD": RowDecl("torch", "eagle", "INVARIANCE"),
    "T-GRAD-FUNC": RowDecl("torch", "eagle", "INVARIANCE"),
    "WP-GRAD-TAPE": RowDecl("warp", "eagle", "INVARIANCE"),
    # host/device hand-over rows for numpy and cupy.
    "NP-HOST-ALIAS": RowDecl("numpy", "eagle", "ALIAS", ("cpu",)),
    "NP-IN-CPU-COPY": RowDecl("numpy", "eagle", "COPY", ("cuda",)),
    "CP-IN-ALIAS": RowDecl("cupy", "eagle", "ALIAS", ("cuda",)),
    "CP-OUT-ALIAS": RowDecl("cupy", "eagle", "ALIAS", ("cuda",)),
    # execution rows: the closed RowDecl framework/kind sets extended (above)
    # so these four are declared here with the eagle-owned consumer
    # implementation — the same declare-before-build convention every other
    # eagle-owned row above already uses (e.g. CP-CAPTURE-REJECT): raptor's
    # own suite never calls ``assert_rows_complete("eagle")``, so a
    # declared-but-uncollected eagle row is not a gap this repo's gate can
    # see — that completeness gate is eagle's own, in its own suite. Row ids
    # are given in eagle's own snake_case naming, not this file's
    # UPPER-HYPHEN convention — kept as given rather than renamed to fit.
    "execution_partition_identity": RowDecl(None, "eagle", "INVARIANCE"),
    "execution_host_device_twin": RowDecl(None, "eagle", "INVARIANCE"),
    "execution_illegal_placement_refused": RowDecl(None, "eagle", "ENV"),
    "execution_layout_selfcheck_refused": RowDecl(None, "eagle", "ENV"),
    # hawk rows: rows that exist only because hawk exists, declared
    # owner_repo="hawk" — same convention as the execution_* rows above,
    # checked by hawk's own consumer implementation.
    # The array-framework crossing rows are owned by eagle, above, not here.
    # framework=None for all
    # five: none bridges to an array framework (see RowDecl's docstring);
    # every check is deferred to its own consumer implementation. Row ids
    # cite the row they carry in hawk's own documentation, which gives no
    # snake_case name for them (contrast the execution_* rows' eagle-given
    # names above) — this file's own UPPER-HYPHEN convention applies,
    # prefixed HAWK- for the family.
    "HAWK-DIFF-POSITIVE-ROUNDTRIP": RowDecl(None, "hawk", "INVARIANCE"),
    "HAWK-EXEC-DEPLOY-LOADABLE": RowDecl(None, "hawk", "ENV"),
    "HAWK-EXEC-REFERENCE-HOST": RowDecl(None, "hawk", "INVARIANCE"),
    "HAWK-EXEC-REFERENCE-RANKPARTITION": RowDecl(None, "hawk", "INVARIANCE"),
}


def certified_framework_problem(fw: str) -> str | None:
    """The failure text when ``fw`` (a name in :data:`CERTIFIED`) is not importable here, else ``None``.

    An absent certified framework is a hard failure of the gate environment, never a skip (a skip is RC=0 and
    certifies nothing); the test-side gate fails on the text."""
    if importlib.util.find_spec(fw) is None:
        return "certified framework absent from the gate environment"
    return None


def roadmap_framework(fw: str) -> None:
    """Assert ``fw`` (a name in :data:`ROADMAP`) is ABSENT here.

    Certified-absent and roadmap-present are BOTH failures — the declaration
    and reality must agree in both directions, or the matrix certifies
    nothing.
    """
    assert importlib.util.find_spec(fw) is None, (
        "declared ROADMAP (absent-by-declaration) but INSTALLED — certify it "
        "or remove it from the env"
    )


#: row IDs actually collected this pytest session (populated by
#: :func:`register_row` at decoration time, i.e. on import).
_COLLECTED_ROWS: set[str] = set()


def register_row(row_id: str, leg: str | None = None):
    """Decorator marking a matrix test as covering declared row ``row_id``.

    ``leg`` (one of the row's :attr:`RowDecl.legs`) says which platform this
    test certifies; ``None`` means the test covers every declared leg.

    Raises :class:`KeyError` immediately if ``row_id`` is not in :data:`ROWS`
    (a typo'd row id should fail loud at collection, not silently pass
    :func:`assert_rows_complete`), and :class:`ValueError` for a ``leg`` the
    row does not declare.
    """
    if row_id not in ROWS:
        raise KeyError(f"register_row: {row_id!r} is not a declared row in ROWS")
    if leg is not None and leg not in ROWS[row_id].legs:
        raise ValueError(
            f"register_row: {row_id!r} declares legs {ROWS[row_id].legs}, not {leg!r}"
        )

    def _decorate(fn):
        _COLLECTED_ROWS.add(row_id)
        fn.__dict__.setdefault("raptor_rows", []).append((row_id, leg))
        return fn

    return _decorate


def assert_rows_complete(owner_repo: str, ran: bool = True) -> None:
    """Assert every :data:`ROWS` entry owned by ``owner_repo`` is certified.

    ``ran=True`` (the default): every declared row has a test that RAN and
    passed on every declared leg in this pytest session, as recorded by the
    :mod:`raptor.conformance.pytest_rows` plugin (which must be loaded, and
    this check must run after the row tests: mark it ``rows_gate``). A row
    whose test was deselected, skipped, failed or never collected is RED.

    ``ran=False``: the weaker import-time check only, that each declared row
    has a registered test. A module that merely imports satisfies it, so a
    run that executed nothing still passes; keep it for tools that read
    declarations, not for a gate.
    """
    declared = {rid for rid, decl in ROWS.items() if decl.owner_repo == owner_repo}
    missing = declared - _COLLECTED_ROWS
    assert not missing, (
        f"rows declared for {owner_repo!r} but never collected: {sorted(missing)}"
    )
    if ran:
        from . import pytest_rows

        problems = pytest_rows.unfinished(
            (rid, leg) for rid in sorted(declared) for leg in ROWS[rid].legs
        )
        assert not problems, (
            f"rows declared for {owner_repo!r} without a passing run:\n"
            + "\n".join(problems)
        )


#: token names that must never appear in a matrix module — skip machinery is
#: banned from matrix modules, mechanically.
_SKIP_TOKEN_ATTRS = frozenset({"skip", "skipif", "importorskip"})


def assert_no_skip_tokens(*module_paths) -> None:
    """Assert zero ``pytest.skip``/``skipif``/``importorskip`` tokens.

    AST-based over each path in ``module_paths`` (a matrix module file):
    catches direct calls (``pytest.skip(...)``, ``pytest.importorskip(...)``)
    and decorator uses (``@pytest.mark.skip``/``@pytest.mark.skipif``) alike,
    since decorators are ordinary ``Attribute``/``Call`` nodes in the AST.
    Absence of a certified framework must be a FAILURE by construction, never
    a silent skip.
    """
    hits: list[str] = []
    for raw_path in module_paths:
        path = Path(raw_path)
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in _SKIP_TOKEN_ATTRS:
                hits.append(f"{path}:{node.lineno}: pytest.{node.attr}")
    assert not hits, "banned skip tokens found in matrix modules:\n" + "\n".join(hits)


#: markers ANY conftest in the family wires into ``pytest_collection_modifyitems``'s
#: skip-on-absence machinery — the union across every family member's own
#: availability markers (verified against each repo's own
#: ``conftest.py``/``pyproject.toml`` directly, not assumed). Checking for a
#: marker a given repo's conftest doesn't define is always a zero-hit, never
#: a false negative — the union is safe to use everywhere.
_AVAILABILITY_MARKERS = frozenset({"gpu", "torch", "torch_gpu", "jax_gpu", "tf_gpu"})


def _calls_register_row(tree: ast.AST) -> bool:
    return any(
        isinstance(node, ast.Name) and node.id == "register_row"
        for node in ast.walk(tree)
    )


def discover_matrix_modules(test_dir) -> list[Path]:
    """Every ``.py`` file under ``test_dir`` that calls :func:`register_row`
    (AST-detected, not textually grepped) — the matrix modules a repo actually
    ships, DISCOVERED rather than hand-passed.

    A hand-passed module list is exactly how a matrix row can go unscanned by
    accident — a row registered from a file its scanner's caller forgot to
    name still counts as "collected" (``register_row`` fires at DECORATION
    time, independent of whether pytest later skips the test), so
    completeness and skip-freedom silently diverge. Discovery closes that gap
    by construction: there is no second list to forget to update.
    """
    return sorted(p for p in Path(test_dir).rglob("*.py") if _calls_register_row(
        ast.parse(p.read_text(), filename=str(p))
    ))


def _registered_function_defs(tree: ast.AST):
    """Yield ``(row_id, FunctionDef)`` for every function decorated
    ``@register_row("...")`` in ``tree`` — the exact, narrow locus a matrix
    row's own hygiene must hold at (not the whole file: a file mixing matrix
    rows with ordinary, legitimately-marker-gated tests must not have its
    unrelated tests' markers flagged)."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            if (
                isinstance(dec, ast.Call)
                and isinstance(dec.func, ast.Name)
                and dec.func.id == "register_row"
            ):
                row_id = (
                    dec.args[0].value
                    if dec.args and isinstance(dec.args[0], ast.Constant)
                    else "<unresolved>"
                )
                yield row_id, node
                break


def _is_marker_decorator(dec: ast.expr, banned_markers: frozenset[str]) -> str | None:
    """Return the banned marker name ``dec`` applies (``@pytest.mark.<name>``,
    bare or called), or ``None``."""
    target = dec.func if isinstance(dec, ast.Call) else dec
    if (
        isinstance(target, ast.Attribute)
        and target.attr in banned_markers
        and isinstance(target.value, ast.Attribute)
        and target.value.attr == "mark"
    ):
        return target.attr
    return None


def assert_no_skip_machinery(
    test_dir, *, banned_markers: frozenset[str] = _AVAILABILITY_MARKERS
) -> None:
    """Assert every DISCOVERED matrix row (any ``@register_row``-decorated
    function in any ``.py`` under ``test_dir``) is free of skip machinery:
    zero ``pytest.skip``/``skipif``/``importorskip`` tokens in its body, AND
    zero conftest availability-marker decorators (``banned_markers``, default
    the cross-repo union) on the registered function itself.

    Scoped to the REGISTERED function, not the whole file (see
    :func:`_registered_function_defs`) — a file may legitimately mix matrix
    rows with ordinary marker-gated tests; only the declared rows must be
    unable to skip. Discovery (:func:`discover_matrix_modules`) means there is
    no hand-passed module list to fall out of sync with reality: that is
    exactly how a row once skipped silently while
    :func:`assert_rows_complete` still passed — ``register_row`` collects at
    decoration time, not at execution.
    """
    modules = discover_matrix_modules(test_dir)
    assert modules, f"no register_row-bearing modules found under {test_dir}"
    skip_hits: list[str] = []
    marker_hits: list[str] = []
    for path in modules:
        tree = ast.parse(path.read_text(), filename=str(path))
        for row_id, fn in _registered_function_defs(tree):
            for dec in fn.decorator_list:
                marker = _is_marker_decorator(dec, banned_markers)
                if marker is not None:
                    marker_hits.append(
                        f"{path}:{fn.lineno}: row {row_id!r} carries "
                        f"@pytest.mark.{marker} (skip-capable)"
                    )
            for sub in ast.walk(fn):
                if isinstance(sub, ast.Attribute) and sub.attr in _SKIP_TOKEN_ATTRS:
                    skip_hits.append(
                        f"{path}:{sub.lineno}: row {row_id!r} uses pytest.{sub.attr}"
                    )
    assert not skip_hits, "banned skip tokens found on registered rows:\n" + "\n".join(
        skip_hits
    )
    assert not marker_hits, (
        "banned availability-marker decorators found on registered rows:\n"
        + "\n".join(marker_hits)
    )


def assert_doc_rows_match(
    doc_path,
    rows: dict[str, RowDecl],
    *,
    row_id_pattern: str = r"`([A-Za-z][A-Za-z0-9_.:/-]*)`",
) -> None:
    """Assert the doc's row IDs and ``rows``'s keys are the same set.

    Mirrors ``test_roles_vocab.py``'s technique (``eagle/python/tests/
    test_roles_vocab.py``): the document is read as raw TEXT and scanned with
    a regex — no markdown parser, no import of anything the doc describes —
    then compared by set equality against the declaration, in both
    directions. ``row_id_pattern`` defaults to backtick-quoted identifiers
    (the row-id convention a table of ``` `row_id` ``` cells would use);
    override it if the shipped document's convention differs.
    """
    text = Path(doc_path).read_text()
    doc_ids = set(re.findall(row_id_pattern, text))
    declared_ids = set(rows)
    missing_from_doc = declared_ids - doc_ids
    extra_in_doc = doc_ids - declared_ids
    assert not missing_from_doc, (
        f"rows declared but missing from {doc_path}: {sorted(missing_from_doc)}"
    )
    assert not extra_in_doc, (
        f"{doc_path} references undeclared rows: {sorted(extra_in_doc)}"
    )
