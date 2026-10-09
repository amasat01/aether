# aether-dsc

The sealed AETHER header payload.

## What the payload is

aether ships to Python-only users as ONE opaque, digest-named blob instead of
readable headers inside a wheel's include tree. `aether_dsc.payload()`
returns a `Payload(digest, headers)`: `digest` is a sha256 over the sorted
`(name, bytes)` pairs of `headers`; `headers` maps include-relative names
(`aether/...`, `rtc/<std name>`, `plugin/gref_layout.h`, `plugin/gref_abi.h`)
to bytes — the exact set an NVRTC-compiled HAWK device kernel needs, plus
what a host compiler needs through `Payload.serve()`.

Two consumers: **device** — NVRTC compiles a kernel straight from
`payload().headers`, in memory, nothing written to disk; **host, or any
file-only compiler** — `payload().serve()` is a context manager materialising
the headers into a private, owner-only (`0700`) RAM directory (`/dev/shm`
when it is a tmpfs, else `$TMPDIR`), removed on both the normal exit path
AND a `with`-block exception.

## How to seal one

```bash
python -m aether_dsc.seal
```
Reads this checkout's aether headers — scoped to the committed manifest
`aether_dsc/payload_headers.txt` (package data, not every file under
`aether/` — see that file's comment for why) — the seven `rtc/` shims, and
the sibling `eagle` checkout's `plugin/gref_layout.h` + `plugin/gref_abi.h`
(host path only — NVRTC gets an alias to the layout header's bytes instead);
runs the NVRTC-clean gate (every device-reachable header compiles under
NVRTC with zero diagnostics); only then writes
`aether_dsc/_payload/<digest>.bin` — a payload failing the gate is
refused, never shipped silently broken.

## Opacity, not secrecy

The blob is a zlib-compressed tar — not obfuscation against a determined
reader, just enough that `strings`/`cat` on the wheel does not hand back
readable C++ source.

## Threading

Free-threaded CPython (3.13t/3.14t) is supported: the compiled modules declare GIL-free operation and the GIL stays disabled after import. Any number of threads may call module-level functions, build, compile, plan and cache concurrently. Distinct objects may be used from distinct threads without synchronisation. One stateful object (a stream, capture, launcher, graph, composer, plan, pipeline, active set, host kernel or arg block) shared by several threads is memory-safe — its calls serialise and a consumed object raises — but the ORDER of those calls is the caller's responsibility, exactly as for a NumPy array or a CuPy stream. CUDA adds two rules of its own: a stream capture is begun, filled and ended by one thread, and while any capture is open no thread may synchronise the whole device (stream-level synchronisation is fine). GPU routes are supported on free-threaded 3.14; on 3.13t the CPU route runs.
