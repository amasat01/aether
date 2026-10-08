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
