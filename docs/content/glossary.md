# Glossary

The words aether's pages use, in one or two sentences each. Each entry links
to the page that explains it in full. Words shared by the whole RAPTOR family
(kernel, sample, manifest, DLPack, ...) are in the
[family glossary](https://amasat01.github.io/raptor/content/glossary.html).

```{glossary}
structure of arrays (SoA)
  A layout that stores each component of a batch contiguously (all the x
  values, then all the y values, ...) instead of one record per sample, so
  neighbouring GPU threads or CPU vector lanes read neighbouring memory.

Chunk
  One byte allocation on exactly one device. It frees itself when it goes
  out of scope, or wraps memory you already own without freeing it. See
  {doc}`memory_ownership`.

View
  A lightweight, non-owning handle over memory, indexed like a small tensor
  as `v(component, sample)`. Building one copies nothing. See
  {doc}`views_and_items`.

Item
  One small, fixed-shape value (a vector or a matrix) that the compiler can
  keep entirely in a thread's registers. See {doc}`views_and_items`.

expression template
  An expression such as `a + b*s` that builds a tiny description of the
  computation instead of computing a temporary vector; the work happens
  once, when it is assigned. See {doc}`expression_templates`.

host, device
  The host is the CPU and its memory; the device is a GPU and its memory.
  `AETHER_DEVICEHOST()` marks a function that runs correctly on either.
  See {doc}`host_vs_device`.

kernel entry point
  A function the CPU launches to start many GPU threads at once. In aether
  it is the exception: most code is written once for both sides. See
  {doc}`host_vs_device`.

replica, partition
  A replica is a copy of the same data on every device in a list
  (`ReplicaSet`); a partition slices one view's samples into
  non-overlapping blocks, one per device or worker, without copying. See
  {doc}`multi_device_residency`.

packet, lane
  A packet is one CPU vector register holding several values
  (`aether::simd::Packet`); each value's slot in it is a lane. See
  {doc}`cpu_simd`.

BandedReal
  An emulated double-precision value with 53 significand bits built from
  three ordinary `float` limbs, for GPUs with little native FP64
  throughput. See {doc}`banded_emulated_real`.

accumulation plane
  A per-sample running total that many producers add into without a data
  race, from a CUDA kernel or an OpenMP loop with the same call. See
  {doc}`accumulation`.

append counter
  A shared counter that lets an unknown number of writers each claim the
  next free slot of one destination array. See {doc}`accumulation`.

numerics contract
  The fixed order and spelling of aether's arithmetic (fold order, division
  as one reciprocal, compensated sums, non-finite rules) that every target
  computes against. See {doc}`numerics_contract`.

language core
  The numerics contract written out in a language that cannot compile
  aether's C++23 templates, generated and checked against the same reference
  set. See {doc}`numerics_contract` and {doc}`c_core`.
```
