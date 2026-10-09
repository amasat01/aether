<p align="center">
  <img src="https://raw.githubusercontent.com/amasat01/aether/main/docs/_static/brand/glyph_aether.svg" alt="" height="56">
</p>
<h1 align="center">aether</h1>
<p align="center">Header-only C++ numerics core. Part of the RAPTOR family.</p>

<h3 align="center">aether: one array, two machines</h3>
<p align="center">A header-only C++23 numerics core. Write the arithmetic once; it compiles for CUDA threads or for
CPU threads (OpenMP), and <code>a + b + c</code> becomes one loop with no temporary arrays.</p>

<p align="center">
  <a href="https://github.com/amasat01/aether/actions/workflows/ci.yml"><img src="https://github.com/amasat01/aether/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://amasat01.github.io/aether/"><img src="https://github.com/amasat01/aether/actions/workflows/docs.yml/badge.svg" alt="Docs"></a>
  <a href="https://github.com/amasat01/aether/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License"></a>
  <a href="https://pypi.org/project/aether-dsc/"><img src="https://img.shields.io/pypi/v/aether-dsc.svg" alt="PyPI"></a>
  <a href="https://doi.org/10.5281/zenodo.23250238"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.23250238.svg" alt="DOI"></a>
</p>

<p align="center">
  <a href="https://amasat01.github.io/"><b>The RAPTOR family</b></a><br>
  <a href="https://amasat01.github.io/hawk/"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/amasat01/aether/main/docs/_static/ecosystem/ecosystem_card_hawk_aether_dark.svg"><img src="https://raw.githubusercontent.com/amasat01/aether/main/docs/_static/ecosystem/ecosystem_card_hawk_aether_light.svg" alt="hawk" width="430"></picture></a>
  <a href="https://amasat01.github.io/eagle/"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/amasat01/aether/main/docs/_static/ecosystem/ecosystem_card_eagle_aether_dark.svg"><img src="https://raw.githubusercontent.com/amasat01/aether/main/docs/_static/ecosystem/ecosystem_card_eagle_aether_light.svg" alt="eagle" width="430"></picture></a><br>
  <a href="https://amasat01.github.io/aether/"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/amasat01/aether/main/docs/_static/ecosystem/ecosystem_card_aether_aether_dark.svg"><img src="https://raw.githubusercontent.com/amasat01/aether/main/docs/_static/ecosystem/ecosystem_card_aether_aether_light.svg" alt="aether" width="430"></picture></a>
  <a href="https://amasat01.github.io/raptor/"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/amasat01/aether/main/docs/_static/ecosystem/ecosystem_card_raptor_aether_dark.svg"><img src="https://raw.githubusercontent.com/amasat01/aether/main/docs/_static/ecosystem/ecosystem_card_raptor_aether_light.svg" alt="raptor" width="430"></picture></a>
</p>

aether is the numerics core of the [RAPTOR family](https://amasat01.github.io/): the array layout and dual-mode
evaluator that keep the family's kernels efficient on the GPU and portable to CPU threads. See
[where RAPTOR fits](https://amasat01.github.io/where_raptor_fits.html) for the full four-part story.

| **0** | **1** | **156 ms** |
|:---:|:---:|:---:|
| libraries to link (header-only) | source, two targets: CUDA or OpenMP | a million per-sample steps on a Quadro P2000 GPU, on the stack aether sits under — [eagle's card](https://amasat01.github.io/eagle/content/performance.html) |

```cpp
#include <aether/aether.h>

int main() {
    aether::Array<double, 3> arr(10);     // 10 samples, 3 components each
    auto v = arr.hostView();
    for (std::size_t i = 0; i < arr.samples(); ++i)
        v(0, i) = static_cast<double>(i); // (component, sample)

    aether::Vec3d a, b;                   // register-resident, no batch dim
    a(0) = 1.0; b(0) = 2.0;
    aether::Vec3d sum = a + b;            // lazy: one assignment, no temporaries
    return 0;
}
```

No separate install step beyond the below: it is header-only, so a consumer just `#include`s it and links nothing
extra. See the
[Quickstart](https://amasat01.github.io/aether/content/quickstart.html).

<details>
<summary>Install</summary>

`PREFIX` is the install directory (inside a conda environment, use `${CONDA_PREFIX}`):

```bash
cmake -DCMAKE_PREFIX_PATH=${PREFIX} -DAETHER_BUILD_TESTS=ON -B build .   # CUDA mode
cmake --build build
cmake --install build --prefix ${PREFIX}
```

A CPU-only build (no CUDA toolchain needed) adds `-DAETHER_CPP_MODE=ON`; see
[Installation](https://amasat01.github.io/aether/content/installation.html) for requirements, the CPU-only
invocation, and the verification one-liner.

`-DAETHER_CUDA_ARCHS=` picks the CUDA architectures when `CMAKE_CUDA_ARCHITECTURES` is not given: `native` (the
default, the GPUs in this machine), `all` (every architecture from Pascal to Blackwell that your nvcc compiles), or one
architecture or a list such as `"80;90"`.

Host (pure C++) execution is built and measured today on x86-64 Linux (Xeon W-2125, AVX-512); the default
`-march` is `x86-64-v3` (AVX2 + FMA). Support for ARM (aarch64) and for x86 CPUs with other vector widths is
planned. Device builds are tested with CUDA 12.6 and CUDA 13.0; later CUDA 12 releases are expected to work.

`aether-dsc` on PyPI is the sealed header payload that hawk compiles against. It is installed with hawk
automatically (`pip install raptor-hawk`); it is not something C++ users install. aether itself is used through
CMake, as above. From a clone, `pip install ./aether/dsc` builds it (Python 3.9 or newer).
</details>

---
<p align="center">
  <img src="https://raw.githubusercontent.com/amasat01/raptor/main/docs/_static/brand/wide_family_raptor.svg" height="28" alt="">
  <b>hawk</b> write it ·
  <b>eagle</b> run it ·
  <b>aether</b> the numerics core (this repo) ·
  <b>raptor</b> the shared contracts —
  <a href="https://amasat01.github.io/">the RAPTOR family</a>
</p>

Apache-2.0 (see [`LICENSE`](https://github.com/amasat01/aether/blob/main/LICENSE) and
[`NOTICE`](https://github.com/amasat01/aether/blob/main/NOTICE)) · cite via "Cite this repository"
(`CITATION.cff`; every tagged release is archived on Zenodo: [doi:10.5281/zenodo.23250238](https://doi.org/10.5281/zenodo.23250238)) · built to
make GPU computing accessible on modest hardware, for research and education. Collaboration is the point, and a
citation is the currency — get in touch.
