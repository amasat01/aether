```{raw} html
<div class="raptor-hero">
  <img class="raptor-reveal dark-light" src="_static/brand/wide_family_aether.svg" alt="aether">
</div>
```

# aether

**One array, two machines.** aether is a header-only C++23 tensor and
expression-template core: typed views over your own memory, arithmetic
that composes into one assignment instead of allocating a temporary per
operator, and a dual-mode evaluator so the SAME kernel source compiles for
OpenMP threads or CUDA threads. It is the numerics base layer the
[eagle](https://amasat01.github.io/eagle/) execution runtime and the
[hawk](https://amasat01.github.io/hawk/) code generator build on.

```{image} _static/ecosystem/ecosystem_aether_light.svg
:alt: The RAPTOR family: hawk (write it), eagle (run it), aether (the C++/CUDA numerics underneath) and raptor (the shared contract); you are looking at aether.
:class: only-light
:align: center
```

```{image} _static/ecosystem/ecosystem_aether_dark.svg
:alt: The RAPTOR family: hawk (write it), eagle (run it), aether (the C++/CUDA numerics underneath) and raptor (the shared contract); you are looking at aether.
:class: only-dark
:align: center
```

[aether](https://amasat01.github.io/aether/) · [hawk](https://amasat01.github.io/hawk/) · [eagle](https://amasat01.github.io/eagle/) · [raptor](https://amasat01.github.io/raptor/) · [the family](https://amasat01.github.io/)

## 30 seconds

```cpp
#include <aether/aether.h>

aether::Array<double, 3> arr(10);   // 10 samples, 3 components each
auto v = arr.hostView();            // a View: a typed window over arr's own memory, introduced next
v(0, 0) = 1.0;                      // (component, sample)
// arr.samples() == 10, arr.size() == 30
```

No separate install step: it is header-only, so a consumer just
`#include`s it and links nothing extra beyond the compiler's own runtime.
{doc}`content/quickstart` walks through reading and writing a whole array.

## Where aether sits

```text
aether → eagle ← raptor → hawk
```

aether is the numerics core: eagle's GPU execution layer builds directly
on it, and hawk's code generator emits kernels against the same device-safe
slice. raptor holds the family's shared contracts, with no dependencies — the
manifest schema and interop contracts eagle and hawk both certify against.

::::{grid} 2
:gutter: 3

:::{grid-item-card} Start here
:link: content/tutorials/01_one_array_two_machines
:link-type: doc
Build an array, read it through a view, and move it to the device and
back — about six minutes.
:::

:::{grid-item-card} Tutorials
:link: content/tutorials
:link-type: doc
Arrays and views, expression templates, device math, your own kernel,
and the macro scaffold behind it all.
:::

:::{grid-item-card} How-to guides
:link: content/examples
:link-type: doc
Quaternions, random numbers, the vector/matrix cookbook, banded emulated
double precision.
:::

:::{grid-item-card} Reference
:link: content/api/api_reference
:link-type: doc
Every documented C++ class, function and macro.
:::
::::

New to the whole RAPTOR family? The landing page's
[Start here](https://amasat01.github.io/start_here.html) walks through one
kernel, a million samples and a PyTorch fit in ten minutes, across every
repo at once.

```{toctree}
:maxdepth: 1
:caption: Start here
:hidden:

content/installation
content/quickstart
```

```{toctree}
:maxdepth: 1
:caption: Tutorials
:hidden:

content/tutorials
```

```{toctree}
:maxdepth: 1
:caption: How-to guides
:hidden:

content/examples
```

```{toctree}
:maxdepth: 1
:caption: Explanation
:hidden:

content/views_and_items
content/expression_templates
content/host_vs_device
content/memory_ownership
content/interop
```

```{toctree}
:maxdepth: 1
:caption: Explanation: advanced
:hidden:

content/multi_device_residency
content/cpu_simd
content/banded_emulated_real
content/accumulation
```

```{toctree}
:maxdepth: 1
:caption: Reference
:hidden:

content/api/api_reference
content/python/aether_dsc
```

```{toctree}
:maxdepth: 1
:caption: Contributing
:hidden:

content/contributing
```
