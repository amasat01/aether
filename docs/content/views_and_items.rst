Views and Items
================

What is a ``View``, and why is ``Item`` register-resident?

A ``View`` is a lightweight, non-owning handle over memory that lets you
index it like a small tensor — ``v(component, sample)`` — while the bytes
themselves live somewhere else (a ``Chunk``, or any raw pointer you hand
it); an ``Item`` is the opposite number, a small, all-static-shape value
(one vector or matrix) that the compiler is free to keep entirely in a
thread's own registers, with no memory address at all.

``View``: a typed, zero-copy tensor
-------------------------------------

``aether::View`` (``aether/view/View.h``) wraps a pointer plus a shape
(``aether::extents``, ``aether/layout/Extents.h``) plus a layout mapping
(``aether/layout/Layout.h``) that turns a multi-index into a flat offset.
It owns nothing and copies nothing — building one is just arithmetic. You
build one with ``aether::make_view<T, Es..., aether::dyn>(...)`` over a
``Chunk`` or a raw pointer (see :doc:`memory_ownership`); the trailing
``aether::dyn`` mode is the dynamic SAMPLE dimension every batched view
carries.

The default layout, ``aether::layout_right``, is what makes a ``View``
structure-of-arrays (SoA): for a 3-component view over ``N`` samples,
``v(c, i)`` maps to flat offset ``c*N + i`` — every component's ``N``
samples are stored contiguously, one component block after another,
rather than one sample's 3 components interleaved (array-of-structures).
That layout is what lets a GPU kernel read the SAME component from many
threads' samples in one coalesced memory transaction.

``Item``: a register-resident value
--------------------------------------

``aether::Item<T, Es...>`` (``aether/view/Item.h``) is what one sample's
worth of data materializes into: ``Item<double,3>`` (aliased ``Vec3d``) is
a 3-vector, ``Item<double,3,3>`` (``Mat33d``) a 3x3 matrix. Every extent in
``Es...`` must be static — no ``aether::dyn`` mode is allowed, because an
``Item`` never carries a batch dimension; that is exactly what makes it
small and shape-known enough for a compiler to keep it in registers rather
than spilling it to memory, on host or device alike. Reading ``v[i].get()``
off a ``View`` at one ``SampleIndex`` is how you turn a batched sample into
a register-resident ``Item`` to work with directly.

A runnable example
-------------------

The SoA layout claim above is not just a description — it is a checkable
identity, ``&v(c, i) - v.data() == c*N + i``, over a full sweep of every
component and sample:

.. aether-example:: tests/test_View.cpp:27-38

.. code-block:: cpp

   TEST_F(ViewTest, SoaProofPointerArithmeticMatchesClosedFormOverAFullSweep)
   {
       // THE SoA PROOF: &v(c,i) - v.data() == c*N + i.
       constexpr std::size_t C = 3, N = 11;
       auto chunk = aether::Chunk::allocate(aether::Device(kDLCPU), C * N * sizeof(double));
       auto v     = aether::make_view<double, C, aether::dyn>(chunk, N);
       for (std::size_t c = 0; c < C; ++c) {
           for (std::size_t i = 0; i < N; ++i) {
               EXPECT_EQ(&v(c, i) - v.data(), static_cast<std::ptrdiff_t>(c * N + i));
           }
       }
   }

.. aether-example-end::

From here, :doc:`expression_templates` covers what happens when you combine
several ``View``\ s/``Item``\ s with ``+``/``-``/``*`` instead of reading
them one at a time.
