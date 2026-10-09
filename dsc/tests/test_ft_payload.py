# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Free-threading rows for aether-dsc (FT-1 and FT-3: one ``Payload`` identity).

On a GIL build the rows SKIP with a reason; the name manifest keeps "skipped"
from degrading into "silently not collected".
"""

from __future__ import annotations

import sys
import threading
import time

import pytest

import aether_dsc
from raptor.conformance import freethreading as ft
from raptor.conformance.freethreading import declare_ft_row, register_ft_row

pytestmark = pytest.mark.ft

declare_ft_row(
    "FT-1-GIL-FREE-AETHER-DSC", "aether-dsc",
    "importing aether_dsc leaves the GIL off")
declare_ft_row(
    "FT-3-PAYLOAD-IDENTITY", "aether-dsc",
    "16 threads on a cold payload() cache get one Payload object")

EXPECTED_TESTS = frozenset(
    {
        "test_manifest_all_tests_collected",
        "test_rows_declared_and_collected",
        "test_gil_free_import",
        "test_payload_cold_cache_single_identity",
        "test_payload_cold_cache_single_identity_slow_unpack",
    }
)

THREADS = 16


def _cold_race(rounds: int) -> None:
    for _ in range(rounds):
        aether_dsc._reset_payload_cache()
        barrier = threading.Barrier(THREADS)
        got: list[object] = []
        errors: list[BaseException] = []

        def work() -> None:
            try:
                barrier.wait()
                got.append(aether_dsc.payload())
            except BaseException as exc:  # noqa: BLE001 - checked below
                errors.append(exc)

        pool = [threading.Thread(target=work) for _ in range(THREADS)]
        for t in pool:
            t.start()
        for t in pool:
            t.join()
        assert not errors, errors
        assert len(got) == THREADS
        assert len({id(p) for p in got}) == 1, "payload() returned distinct objects"
        assert all(p is got[0] for p in got)
        assert aether_dsc.payload() is got[0]


def test_manifest_all_tests_collected(request):
    here = {
        i.name for i in request.session.items if i.module is sys.modules[__name__]
    }
    assert EXPECTED_TESTS <= here, sorted(EXPECTED_TESTS - here)
    assert {n for n in here if "[" not in n} <= EXPECTED_TESTS


def test_rows_declared_and_collected():
    ft.assert_ft_rows_complete("aether-dsc")


@register_ft_row("FT-1-GIL-FREE-AETHER-DSC")
def test_gil_free_import():
    ft.require_free_threaded()
    ft.assert_gil_free("aether_dsc")


@register_ft_row("FT-3-PAYLOAD-IDENTITY")
def test_payload_cold_cache_single_identity():
    try:
        _cold_race(50)
    finally:
        aether_dsc._reset_payload_cache()


def test_payload_cold_cache_single_identity_slow_unpack(monkeypatch):
    # Widen the check-then-set window: a slow unpack makes an unlocked
    # cold-cache race certain on every interpreter, GIL or not.
    real = aether_dsc._unpack

    def slow(data):
        time.sleep(0.02)
        return real(data)

    monkeypatch.setattr(aether_dsc, "_unpack", slow)
    try:
        _cold_race(3)
    finally:
        aether_dsc._reset_payload_cache()
