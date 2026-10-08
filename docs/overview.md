# aether — folder map

One row per top-level `aether/` folder: what it holds, and the actual C++
namespace(s) declared under it, checked against the source. Namespaces
mostly stay flat under `aether::`; a nested namespace appears only where a
folder's contents are genuinely a distinct sub-vocabulary.

| Folder | Namespace(s) | Contents |
|---|---|---|
| `aether.h`, `macros.h` | `aether` | Umbrella header, `AETHER_*` device/host macros |
| `dtype/` | `aether` | `DType`, `dtype_of<T>()`, `int_t`, vendored DLPack (`dlpack.h`), host-only `Format.h` |
| `device/` | `aether` | `Device`, the `kDLCPU`/`kDLCUDA`/`kDLCUDAHost` backend-enabled traits |
| `err/` | `aether`, `aether::err` | `Error`, `checkCuda()` |
| `chunk/` | `aether`, `aether::detail` | `Chunk` (owning bytes), `borrow()`, `copy`/`copyAsync` (`Copy.h`) |
| `layout/` | `aether`, `aether::detail` | `extents`/`dyn`, `layout_right` mapping (`Layout.h`); `layout/detail/Carray.h` is the internal fixed-array aggregate |
| `index/` | `aether` | `offset_t`, `SampleIndex`, `BundleIndex<W>` |
| `view/` | `aether`, `aether::detail` | `View` (the tensor type), `Item` (register-resident, all-static), `TableHandle`, `RuntimeView`, `Reshape`, `WorkView` (shared-memory) |
| `array/` | `aether` | `Array<T,Es...>` — the owning host/device-pair convenience |
| `expr/` | `aether`, `aether::detail` | `Expression` CRTP base, `Operations`/`Assign`/`Reduce`; `expr/nodes/` holds the node types (`Arithmetic`, `Geometric`, `Product`, `Quaternion`, `Structural`) |
| `eval/` | `aether`, `aether::detail`, `aether::eval` | The dual-mode runtime evaluator (`Runtime.h`) |
| `math/` | `aether`, `aether::math`, `aether::detail` | Device-legal scalar dispatch (`abs`/`fma`/`sqrt`/trig/…); `math/detail/BoundedTrig.h` is the bounded-argument FP64 trig path |
| `accum/` | `aether`, `aether::accum`, `aether::accum::atomic` | `AccumPlane`/`AccumPlaneView` (`AccumPlane.h`, `CompensatedAtomic`/`Serialized` policies), `AppendCounter`/`AppendCounterView` (`AppendCounter.h`), `atomic::compensatedSum`/`minAbsReal`/`orBool`/`maxInt` (`atomic.h`) — narrative: [Accumulation](content/accumulation.rst) |
| `backend/cpu/` | `aether`, `aether::detail`, `aether::simd` | `PacketItem`, `Tiled`; `backend/cpu/simd/` is the SIMD `Packet`/`PacketMask`/`PacketTraits` layer; `backend/cpu/packet/` holds `Assign`/`LoadStore` |
| `backend/cuda/` | `aether`, `aether::cuda`, `aether::detail` | `Launch.h` (`aether::cuda::launchConfig`/`checkLastLaunch`), `DeviceBundle<T,W,Es...>`; `backend/cuda/bundle/` holds `Assign`/`LoadStore` |
| `interop/` | `aether`, `aether::detail`, `aether::interop` | `interop::fromDLPack`/`toDLPack` (`DLPack.h`); `Mdspan.h` (host-only `std::mdspan` adapter, not included by the umbrella) |
| `residency/` | `aether`, `aether::detail` | `PartitionSpec`/`partition_view` (`Partition.h`), `ReplicaSet` (`Replica.h`), the `transport<T>` concept + `StreamTransport` (`Transport.h`) |

## Where to look for X

- **Consumer alias surface** (`Vec3d`, `Mat33d`, `Vec3dView`, …): the bottom
  of `aether/aether.h` itself, not any one folder above.
- **What the umbrella header pulls in**: `aether/aether.h`'s own
  top-of-file doc comment.
- **Build/test conventions and the test-integrity gate**:
  `tests/check_gate.sh`'s own header comment.
