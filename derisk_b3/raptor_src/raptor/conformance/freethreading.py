# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Free-threaded CPython conformance harness (the ``FT-*`` row family).

Pure standard library, importable on every interpreter. It names no test framework: the skip and fail gates that
turn :func:`masking_problem`, :func:`free_threaded_problem` and :func:`not_vacuous_problem` into outcomes live with
the tests.

The harness is written for reuse by every package in the family that ships a
module to be run under 3.14t / 3.15t. Its pieces:

* :func:`masking_problem` -- a test-side gate skips on a GIL build and **fails** when the GIL
  state is forced from outside (``-X gil=...`` or ``PYTHON_GIL``): a forced
  state hides a missing ``FREE_THREADED`` declaration, so a run under it
  certifies nothing.
* :func:`hammer` -- N threads released together by a barrier.
* :func:`assert_gil_free` (row class FT-1) -- import a module in a clean
  subprocess under ``-W error::RuntimeWarning`` and require that the GIL is
  still off afterwards.
* :func:`measure_loss` / :func:`canary_python` / :func:`not_vacuous_problem`
  (row class FT-2) -- a *planted* race that the schedule must expose. A
  schedule that cannot lose updates on a known-racy counter cannot certify
  that a real counter is race-free, so the run is RED as vacuous, never green.
  A package supplies its own compiled canary (conventionally
  ``_unsynchronised_bump``) through :func:`measure_loss`.

FT-3..FT-6 (global state, distinct objects, shared-object misuse, GPU) are
package-specific; packages build them from :func:`hammer` and register their
row ids with :func:`declare_ft_row` / :func:`register_ft_row`.
"""

from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

__all__ = [
    "FT_ROWS",
    "CanaryResult",
    "FtRowDecl",
    "assert_ft_rows_complete",
    "assert_gil_free",
    "canary_python",
    "declare_ft_row",
    "free_threaded_problem",
    "hammer",
    "measure_loss",
    "register_ft_row",
    "masking_problem",
    "not_vacuous_problem",
]

#: Default hammer shape of the vacuity canary (threads x iterations per thread).
CANARY_THREADS = 8
CANARY_ITERATIONS = 200_000
#: A canary that loses less than this fraction of its updates is vacuous.
MIN_LOST_FRACTION = 0.10
#: Wall-clock cap (seconds) on the adaptive escalation in :func:`measure_loss`.
CANARY_TIME_CAP = 5.0


def usable_cores() -> int:
    """Cores this process may run on (affinity-aware where the OS has it)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def _is_free_threaded_build() -> bool:
    return bool(sysconfig.get_config_var("Py_GIL_DISABLED")) and hasattr(
        sys, "_is_gil_enabled"
    )


def free_threaded_problem() -> str | None:
    """Return why this process cannot certify free-threading, or ``None``.

    Does not need pytest. ``None`` means: a free-threaded build with the GIL
    state left to the interpreter and the modules.
    """
    if not _is_free_threaded_build():
        return "not a free-threaded CPython build (GIL build)"
    return None


def _masking_reason() -> str | None:
    flag = getattr(sys.flags, "gil", None)
    if flag is not None:
        return f"-X gil={flag} is in force (sys.flags.gil={flag})"
    if "PYTHON_GIL" in os.environ:
        return f"PYTHON_GIL={os.environ['PYTHON_GIL']!r} is set"
    return None


def masking_problem() -> str | None:
    """The reason the GIL state is forced from outside (``-X gil=0`` / ``PYTHON_GIL``), or ``None``.

    Both ``=0`` and ``=1`` mask what the free-threading rows exist to catch, so a gate must FAIL on this, never skip.
    A test-side gate turns :func:`free_threaded_problem` into a skip and this into a failure."""
    return _masking_reason()


def hammer(
    fn: Callable[[], object],
    threads: int = CANARY_THREADS,
    iterations: int = CANARY_ITERATIONS,
) -> None:
    """Run ``fn()`` ``iterations`` times on each of ``threads`` threads.

    All threads wait on a barrier and start together. The first exception
    raised on any thread is re-raised here after every thread has finished.
    """
    barrier = threading.Barrier(threads)
    errors: list[BaseException] = []

    def work() -> None:
        try:
            barrier.wait()
            for _ in range(iterations):
                fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            errors.append(exc)

    pool = [threading.Thread(target=work) for _ in range(threads)]
    for t in pool:
        t.start()
    for t in pool:
        t.join()
    if errors:
        raise errors[0]


_GIL_PROBE = (
    "import importlib, sys\n"
    "importlib.import_module(sys.argv[1])\n"
    "print('GIL_ENABLED=' + str(sys._is_gil_enabled()))\n"
)


def assert_gil_free(
    module_name: str, python_args: tuple[str, ...] = (), timeout: float = 120.0
) -> None:
    """FT-1: importing ``module_name`` leaves the GIL disabled.

    The import runs in a fresh subprocess under ``-W error::RuntimeWarning``
    (the interpreter's GIL re-enable notice is a ``RuntimeWarning``, so an
    undeclared extension fails the import itself) and the child then reports
    ``sys._is_gil_enabled()``. A subprocess keeps the check clean: the parent
    may already have re-enabled the GIL through some other import.
    ``python_args`` is for tests of the check itself.
    """
    if _masking_reason() is not None:
        raise AssertionError(f"assert_gil_free refused: {_masking_reason()}")
    cmd = [
        sys.executable,
        *python_args,
        "-W",
        "error::RuntimeWarning",
        "-c",
        _GIL_PROBE,
        module_name,
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=False
    )
    out = proc.stdout.strip()
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-1:] or ["(no stderr)"]
        raise AssertionError(
            f"FT-1: importing {module_name!r} failed under "
            f"-W error::RuntimeWarning: {tail[0]}"
        )
    if out != "GIL_ENABLED=False":
        raise AssertionError(
            f"FT-1: the GIL is enabled after importing {module_name!r} ({out})"
        )


@dataclass(frozen=True)
class CanaryResult:
    """Outcome of a planted-race measurement (the final attempt).

    ``threads``, ``iterations``, ``attempts`` and ``cores`` are evidence for
    telling a weak runner from a broken instrument; they default to "unknown"
    so a hand-built ``CanaryResult(expected, observed)`` stays valid.
    """

    expected: int
    observed: int
    threads: int = 0
    iterations: int = 0
    attempts: int = 1
    cores: int = 0

    @property
    def lost(self) -> int:
        return self.expected - self.observed

    @property
    def lost_fraction(self) -> float:
        return self.lost / self.expected if self.expected else 0.0


def measure_loss(
    bump: Callable[[], object],
    read: Callable[[], int],
    threads: int = CANARY_THREADS,
    iterations: int = CANARY_ITERATIONS,
    min_lost: float = MIN_LOST_FRACTION,
    time_cap: float = CANARY_TIME_CAP,
) -> CanaryResult:
    """Hammer a deliberately unsynchronised ``bump`` and count lost updates.

    ``read`` returns the counter ``bump`` increments. A package's compiled
    canary plugs in here: ``measure_loss(core._unsynchronised_bump,
    core.canary_count)``.

    Contention adapts to the machine. Attempt 1 uses the given shape. While
    the loss is below ``min_lost``, each further attempt doubles the iterations
    and raises the threads to :func:`usable_cores` (never fewer than before),
    until the floor is cleared or ``time_cap`` seconds are spent (an attempt
    that would not fit in the remaining time is not started). The last attempt
    is returned, with its shape and the attempt count as evidence. The floor
    itself is never lowered.
    """
    cores = usable_cores()
    start = time.monotonic()
    attempts = 0
    while True:
        attempts += 1
        t0 = time.monotonic()
        before = read()
        hammer(bump, threads, iterations)
        res = CanaryResult(
            threads * iterations, read() - before, threads, iterations,
            attempts, cores,
        )
        took = time.monotonic() - t0
        if res.lost_fraction >= min_lost:
            return res
        # Next attempt costs about twice this one (double the iterations).
        if time.monotonic() - start + 2 * took > time_cap:
            return res
        threads = max(threads, cores)
        iterations *= 2


#: Busy-loop steps between the canary's read and its write. A bare
#: ``d[k] += 1`` races in a window a few bytecodes wide, so how much it loses
#: depends on interpreter speed and machine load (measured 3-84 % on one
#: machine); widening the window makes the planted race visible everywhere.
CANARY_GAP = 16


def canary_python(
    threads: int = CANARY_THREADS, iterations: int = CANARY_ITERATIONS,
    gap: int = CANARY_GAP,
) -> CanaryResult:
    """FT-2 canary: an unsynchronised read-modify-write of one shared value,
    with ``gap`` busy steps between the read and the write."""
    d = {"k": 0}

    def _unsynchronised_bump() -> None:
        v = d["k"]
        for _ in range(gap):
            pass
        d["k"] = v + 1

    return measure_loss(_unsynchronised_bump, lambda: d["k"], threads, iterations)


_PREFLIGHT: dict[str, CanaryResult] = {}


def not_vacuous_problem(
    result: CanaryResult | None = None, min_lost: float = MIN_LOST_FRACTION
) -> tuple[CanaryResult, str | None]:
    """The canary result and, unless it lost at least ``min_lost``, the reason the run is vacuous.

    With no argument the Python canary is measured once per process and cached (it escalates contention first, see
    :func:`measure_loss`). Pass a package's own :func:`measure_loss` result to judge its compiled canary. A test-side
    gate fails (RED, never skips) on the reason."""
    if result is None:
        if "python" not in _PREFLIGHT:
            _PREFLIGHT["python"] = canary_python()
        result = _PREFLIGHT["python"]
    if result.lost_fraction < min_lost:
        return result, (
            f"free-threading run is VACUOUS: the planted race lost only "
            f"{result.lost_fraction:.1%} of {result.expected} updates "
            f"(< {min_lost:.0%}) after {result.attempts} attempt(s) "
            f"[usable cores={result.cores}, threads={result.threads}, "
            f"iterations/thread={result.iterations}]; a schedule that cannot "
            "expose a known race cannot certify the real code. Few usable "
            "cores points at a weak runner, many at a broken instrument."
        )
    return result, None


@dataclass(frozen=True)
class FtRowDecl:
    """One declared free-threading row (``owner_repo`` carries its test)."""

    owner_repo: str
    summary: str


#: row_id -> FtRowDecl. raptor-core owns the harness-level rows below;
#: packages add theirs with :func:`declare_ft_row`. Kept apart from
#: :data:`raptor.conformance.interop.ROWS` because these rows certify an
#: execution property, not an array-framework crossing.
FT_ROWS: dict[str, FtRowDecl] = {
    "FT-0-REFUSE-MASKED-GIL": FtRowDecl(
        "raptor", "the gate fails under -X gil=... and PYTHON_GIL"
    ),
    "FT-1-GIL-FREE-PURE": FtRowDecl(
        "raptor", "a pure-Python module imports with the GIL still off"
    ),
    "FT-1-GIL-REQUIRED-FAILS": FtRowDecl(
        "raptor", "FT-1 is red when the GIL is on after import"
    ),
    "FT-2-CANARY-PYTHON": FtRowDecl(
        "raptor", "the plain-Python planted race loses >= 10% of updates"
    ),
}

_COLLECTED_FT_ROWS: set[str] = set()


def declare_ft_row(row_id: str, owner_repo: str, summary: str) -> None:
    """Declare a package's FT row ahead of its test (idempotent for equal args)."""
    decl = FtRowDecl(owner_repo, summary)
    if FT_ROWS.setdefault(row_id, decl) != decl:
        raise ValueError(f"declare_ft_row: {row_id!r} already declared differently")


def register_ft_row(row_id: str):
    """Decorator marking a test as covering declared FT row ``row_id``."""
    if row_id not in FT_ROWS:
        raise KeyError(f"register_ft_row: {row_id!r} is not a declared FT row")

    def _decorate(fn):
        _COLLECTED_FT_ROWS.add(row_id)
        return fn

    return _decorate


def assert_ft_rows_complete(owner_repo: str) -> None:
    """Assert every FT row owned by ``owner_repo`` was collected."""
    declared = {r for r, d in FT_ROWS.items() if d.owner_repo == owner_repo}
    missing = declared - _COLLECTED_FT_ROWS
    assert not missing, (
        f"FT rows declared for {owner_repo!r} but never collected: "
        f"{sorted(missing)}"
    )
