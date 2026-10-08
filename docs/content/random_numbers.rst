Random numbers
===============

How do I fill an array with reproducible randoms?

Construct an ``aether::random::Generator`` with a seed and call
``.normal<Real, D>(mean, sd)`` or ``.uniform<Real, D>(lo, hi)`` per sample
— the same seed always produces the exact same stream, whether you draw it
on the host or inside a GPU kernel, and regardless of what order the draws
happen in.

One generator, both arms
----------------------------

``aether::random::Generator`` (``aether/random/Generator.h``) is a tiny,
trivially-copyable value carrying one mixed 64-bit key — pass it by value
into a kernel exactly like a ``View``. There is no global mutable RNG
state anywhere in this module: every draw is a pure function of ``(seed,
sample id, sub-counter)``, built on aether's own counter-based
Philox4x32-10 generator, so a parallel fill and a serial fill agree
exactly, and a host build and a device build draw the identical stream for
the same seed. ``rng.substream(k)`` derives an independent, decorrelated
sub-stream from the same generator when you need more than one distinct
stream from one seed.

The distribution surface covers ``uniform``, ``normal``, ``lognormal``,
``exponential``, ``bernoulli``, ``uniformInt`` and a correlated
``multivariateNormal`` (given a mean ``Item`` and a Cholesky factor,
``aether::random::LowerTriangular``). Parameters are always a scalar or a
small ``Item`` — never a full expression — since that already covers every
real call site this module was built to serve.

The free-function shortcut
-------------------------------

For the common case where you either do not care about the seed or want
one chosen once, ``aether/random/free.h`` offers the same distributions as
free functions: ``aether::random::normal<double,3>(mean, sd)`` uses a
fixed, reproducible default seed, and ``aether::random::seed(s)`` is a
named ``Generator`` factory for when the seed does matter — there is
deliberately no settable AMBIENT seed (a host-set device global runs into
real trouble in multi-library GPU builds), so ``seed(s)`` or a
``Generator`` you build yourself are the only ways to control it.

A runnable example
-------------------

The free function and the equivalent explicit ``Generator`` produce the
SAME stream, element for element:

.. aether-example:: tests/test_Random.cpp:298-302

.. code-block:: cpp

       aether::Array<double, 3> a(pn);
       fillSerial(a, pn, [](auto& v, SampleIndex i) { v[i] = fr::normal<double, 3>(0.5, 2.0); });
       aether::Array<double, 3> b(pn);
       fillSerial(b, pn, [](auto& v, SampleIndex i) { v[i] = fr::Generator(fr::DEFAULT_SEED).normal<double, 3>(0.5, 2.0); });
       expectIdentical(a.hostView(), b.hostView(), pn);

.. aether-example-end::

``fr`` here is ``aether::random``; ``fr::normal<double,3>(0.5, 2.0)`` (the
default-seed free function) and ``fr::Generator(fr::DEFAULT_SEED).normal<
double,3>(0.5, 2.0)`` (the same seed spelled out explicitly) are exactly
the same call underneath.
