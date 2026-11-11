# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The kernel-level launch/marshal protocols.

Names mirror eagle's EXISTING public surface 1:1 (``eagle.launch`` /
``eagle.marshal``) so binding an implementation is mechanical.
``DEFAULT_BLOCK`` is deliberately NOT lifted here — it belongs to a later
LaunchPolicy layer. ``LaunchMixin`` has NO protocol twin: hawk's
``CompiledKernel(LaunchMixin)`` inheritance becomes composition over
:class:`KernelLauncher`.

``dt=None`` below stands in for eagle's ``np.float64`` default (the resolved
Real dtype) — raptor has zero hard deps, so no module here imports numpy;
``None`` means "the implementation's default Real dtype".

A toy caller, just enough to show the shape (a real :class:`KernelLauncher`
is eagle's; this is not a runnable implementation)::

    launcher: KernelLauncher = ...  # e.g. eagle's launch module
    launcher.launch("pure", my_fn, arg_spec, vector_inputs={"r": r},
                     params={}, per_sample={}, kw={}, grid=blocks, block=256)
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class KernelLauncher(Protocol):
    """Mirrors ``eagle.launch``'s capturable launch verb + the pure
    allocate-and-launch body (``eagle/python/eagle/launch.py``: ``launch``:253,
    ``assemble_args``:67, ``pure_origin``:284, ``pure_prepare``:299)."""

    def launch(
        self,
        kind,
        fn,
        arg_spec,
        vector_inputs,
        params,
        per_sample,
        *,
        kw,
        grid,
        block,
        dt=None,
        **extra,
    ) -> Any:
        """Issue ONE capturable launch of any kernel kind on the current stream."""
        ...

    def assemble_args(
        self,
        arg_spec,
        *,
        out,
        vec,
        per_sample,
        terminated,
        uniforms,
        n,
        tables=None,
        mutables=None,
        mutable_vec=None,
        mats=None,
    ) -> Any:
        """Build the kernel argument tuple in the order the signature expects."""
        ...

    def pure_origin(self, kw, *, vector_inputs, mutable_names, per_sample) -> Any:
        """Return the caller's framework, detected from every per-sample source."""
        ...

    def pure_prepare(
        self,
        fn,
        arg_spec,
        *,
        vector_inputs,
        per_sample,
        params,
        mutables_decl,
        mutable_defaults,
        lookup_counts,
        kw,
        mat_shapes=None,
        vec_widths=None,
        dt=None,
    ) -> Any:
        """Coerce inputs, bind writable Mutables, assemble args, launch (no sync)."""
        ...


@runtime_checkable
class Marshal(Protocol):
    """Mirrors ``eagle.marshal``'s per-role input coercion functions
    (``eagle/python/eagle/marshal.py``)."""

    def coerce_vec_inputs(self, kw, names, n=None, widths=None, dt=None) -> Any:
        """State vectors -> contiguous ``(W, N)`` device arrays.

        Returns ``(dict, n)``."""
        ...

    def coerce_mat_inputs(self, kw, shapes, n=None, dt=None) -> Any:
        """Per-sample matrix inputs -> contiguous flat ``(R*C, N)`` device arrays."""
        ...

    def coerce_per_sample(self, kw, names, n=None, dt=None) -> Any:
        """Per-sample scalars -> contiguous ``(N,)`` device arrays."""
        ...

    def coerce_uniforms(self, kw, params, dt=None) -> Any:
        """Broadcast ``Param`` constants -> by value."""
        ...

    def coerce_tables(self, kw, lookup_counts, dt=None) -> Any:
        """Lookup tables / shared constants -> flat ``(count,)`` handles."""
        ...

    def coerce_terminated(self, kw, n) -> Any:
        """The ``terminated`` mask -> a ``(N,)`` bool device array."""
        ...

    def require_n(self, n, *, hint="state-vector input") -> Any:
        """The canonical 'cannot size the batch' guard, with a kind-appropriate hint."""
        ...
