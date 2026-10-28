Linear algebra for large matrices
=================================

``aether::linalg`` solves small systems inside a lane: one thread, the matrix in its registers, ``N <= 32``.
``aether::linalg::team`` is for the rest. A **team** of threads works on ONE matrix, which lives in scratch
you provide, so the size is bounded by memory (``N`` up to 4096) and not by what a thread can hold.

.. code-block:: cpp

   #include <aether/aether.h>
   using namespace aether::linalg::team;

   constexpr std::size_t N = 512;
   // a_in, b, x: N*N, N and N elements. f_a, piv, info: scratch you own.
   Team tm;                                   // host: rank 0, size 1
   TeamLu<double, N> f{f_a, piv, info};
   lu<N>(tm, f, a_in);                        // f_a may be a_in: factor in place
   luSolve<N>(tm, f, b, x);                   // *info != 0 gives zeros in x

On the device the same code runs with ``Team tm{threadIdx.x, blockDim.x};``, one block per matrix. Every
routine is **collective**: all threads of the team call it with the same arguments, and the results are
visible to the whole team on return.

What you can call
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Routine
     - What it does
   * - ``lu``, ``luSolve``, ``luSolveT``, ``luDet``, ``luInverse``
     - LU with partial pivoting; solve ``A X = B`` or ``A^T X = B``; determinant; inverse
   * - ``cholesky``, ``choleskyFrom``, ``choSolve``, ``triLower``, ``triUpperT``
     - Cholesky of the symmetric part; a factor you built; the solves
   * - ``eigh``
     - eigenvalues ascending and eigenvectors, in the one-thread rotation order
   * - ``qr``, ``qrSolve``, ``qrGramSolve``
     - Householder QR over row tiles; least squares ``min |A X - B|``; ``(A^T A)^-1 B`` from the factor
   * - ``blockSum``
     - the ordered ``sum_r u_r v_r^T`` behind ``J^T J`` and ``J^T r``

``info`` means what it means in the per-lane family (LAPACK numbering for LU and Cholesky; a bit mask for
``eigh``; for ``qr`` the first column with a zero ``R`` diagonal). A failed factorisation makes every solve
built on it write zeros.

Same bits for any team size
---------------------------

The team size is a performance knob, not a numerical one. Pivot searches break ties by lowest index,
substitution is column by column, and ``blockSum`` adds each entry's rows serially in order, so no sum ever
crosses threads. A team of 1 (the host) and a team of 256 give the same bits. LU here is right-looking where
the one-thread form is Crout, so the two forms agree to rounding (a relative residual within ``50 N`` machine
epsilon), not bit for bit.

Scratch
-------

All arrays are row-major, ``T`` is ``float`` or ``double``, and ``k`` is the number of right-hand sides.

.. list-table::
   :header-rows: 1
   :widths: 20 52 28

   * - Routine
     - Elements you provide
     - Bytes for ``N = 512``, ``double``
   * - ``lu``
     - ``N*N`` factor, ``N`` pivots (``int32``), 1 ``info`` (``int32``)
     - 2.1 MB
   * - ``cholesky``
     - ``N*N`` factor, 1 ``info``
     - 2.1 MB
   * - ``eigh``
     - ``w`` ``N``, ``v`` ``N*N``, ``work`` ``eighWorkSize(N) = N*N + 4``, 1 ``info``
     - 4.2 MB
   * - ``qr`` of ``M x N``
     - ``a`` ``M*N``, ``tau`` ``N``, ``work`` ``qrWorkSize(M, max(N, k))``, 1 ``info``
     - ``M = 2048``: 8.4 MB
   * - ``blockSum``
     - ``g`` ``P*Q``; nothing else
     - 2.1 MB for ``P = Q = 512``

``qrWorkSize(m, cols) = cols * ceil(m / 64) + 1`` (the tile is 64 rows, fixed, so a sum never depends on the
team size). The byte column is plain arithmetic, ``elements * sizeof(T)``.

Two worked sizes, again arithmetic: 130 matrices of ``N = 1800`` need 3.4 GB for each plane of ``N*N``
doubles, and one matrix of ``N = 800`` needs 5 MB. Allocate for all the lanes you launch at once.

.. seealso::

   :doc:`numerics_contract` for the order of arithmetic in the per-lane family, and
   :doc:`views_and_items` for the per-lane operands.
