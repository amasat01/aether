# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The program-level execution protocol.

``program`` (the argument to :meth:`Backend.compile`) is typed as a duck
(structural) reference to a downstream package's spec layer — documented
here, never imported (raptor depends on nothing). Naming is LOCKED:
``compile``/``Executable`` is the torch.compile / JAX-AOT verb;
the internal template-layer ``.build()`` word stays distinct and is not used
here. ``vjp``/``jvp`` + ``cotangents``/``tangents`` are the JAX/torch.func
canonical functional-AD pair; ``backward`` is a documented ALIAS of ``vjp``
(torch parity) that implementations should also expose, bound to the same
method — the capability string itself stays ``"vjp"``, never ``"backward"``.

A toy implementation, just enough to satisfy the protocol::

    class EchoExecutable:
        def forward(self, inputs): return inputs
        def vjp(self, cotangents): raise CapabilityError("vjp not supported")
        def jvp(self, tangents): raise CapabilityError("jvp not supported")
        def parameters(self): return []

    class EchoBackend:
        name = "echo"
        def capabilities(self): return frozenset({"forward"})
        def compile(self, program): return EchoExecutable()
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

#: the capability names a Backend/Executable may advertise.
CAPABILITIES = frozenset({"forward", "vjp", "jvp", "train_step"})


class CapabilityError(NotImplementedError):
    """Raised when a requested capability is not implemented.

    The fail-loud law: a missing capability raises this, never returns
    a silent ``None``."""


@runtime_checkable
class Executable(Protocol):
    """A compiled/wired program instance, ready to run.

    Implementations conventionally also expose ``__call__`` delegating to
    :meth:`forward`, and ``backward`` as a documented alias of :meth:`vjp`
    (torch parity)."""

    def forward(self, inputs: Any) -> Any:
        """Run the forward pass over ``inputs``."""
        ...

    def vjp(self, cotangents: Any) -> Any:
        """Evaluate the reverse-mode (VJP) derivative against ``cotangents``.

        Raise :class:`CapabilityError` if ``"vjp"`` is not one of this
        Executable's capabilities. ``backward`` is a documented alias of this
        method (torch parity) — ONE implementation, the capability string
        stays ``"vjp"``."""
        ...

    def jvp(self, tangents: Any) -> Any:
        """Evaluate the forward-mode (JVP) derivative against ``tangents``.

        Raise :class:`CapabilityError` if ``"jvp"`` is not declared."""
        ...

    def parameters(self) -> Any:
        """Return this Executable's trainable parameters."""
        ...


@runtime_checkable
class Backend(Protocol):
    """A program-level execution substrate.

    ``EagleBackend`` / ``TorchBackend`` / ``ReferenceBackend`` implement this;
    ``Layer.to(name)`` selects one by :attr:`name`. ``TorchBackend`` COMPOSES
    the existing engines by tensor device — CPU tensors route to the
    host-JIT engine, CUDA tensors route to eagle, and any other device raises
    :class:`CapabilityError`; it is not an interpretive substrate of its
    own."""

    name: str

    def capabilities(self) -> frozenset:
        """Return the subset of :data:`CAPABILITIES` this backend supports."""
        ...

    def compile(self, program: Any) -> Executable:
        """Compile/wire ``program`` into an :class:`Executable`.

        ``program`` is a duck-typed reference to a downstream package's spec
        layer — documented
        here, never imported. Mirrors torch.compile / JAX AOT
        ``.lower().compile()``; distinct from the internal
        template-layer ``.build()`` word."""
        ...
