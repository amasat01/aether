// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// tools/nvrtc/device_umbrella_probe.cpp — gate subjects: does
// `aether/device.h` (the new device-safe umbrella) parse+compile AND
// materialize an ET assignment under NVRTC, and does the full
// `aether/aether.h` umbrella (the seeded-RED control, proving the audit can
// still fail) stay RED as documented?
//
// Host-only, no GPU: `nvrtcCompileProgram` is host-side compilation to
// PTX/CUBIN — this probe never creates a CUDA context, never launches
// anything (house convention, tools/nvrtc_parse_probe.cpp's own precedent,
// which this driver mirrors structurally).
//
// CLI: device_umbrella_probe <device|control> [nvrtc-option ...]
//   mode = "device"  — `#include <aether/device.h>` ONLY, then a
//          MATERIALIZING `Item c = a + b` (not `auto c = a + b` — the
//          whole point is proving Item's out-of-line
//          expression ctor, which used to drag in host-only `err/Error.h`
//          via `view/View.h`, now resolves cleanly through `device.h`
//          alone). This is the GATING GREEN target.
//   mode = "control"  — the literal `#include <aether/aether.h>` full
//          umbrella. Expected RED (host-only headers like
//          `accum/AccumPlane.h`'s `<algorithm>` are still unconditionally
//          in `aether.h`) — this is the non-vacuity proof: an audit that
//          cannot fail is certifying nothing. ALSO GATES (unlike
//          tools/nvrtc_parse_audit.sh's own non-gating "umbrella" arm,
//          which predates aether/device.h existing at all).
//
// Every remaining argv entry is passed to nvrtcCompileProgram VERBATIM as
// one NVRTC compile option — toolchain-resolution logic lives in
// tools/nvrtc/audit_device_umbrella.sh, not here (same split as
// tools/nvrtc_parse_audit.sh / tools/nvrtc_parse_probe.cpp).
//
// ALL toolchain paths passed to NVRTC MUST be derived at runtime by the
// caller from the active toolchain — no machine-specific path may be
// hard-coded here or in the audit script.

#include <cstdio>
#include <string>
#include <vector>

#include <nvrtc.h>

namespace {

const char* kDeviceSrc = R"AETHER_NVRTC(
#include <aether/device.h>

extern "C" __global__ void probeKernel(double* out)
{
    aether::Item<double, 3> a{ 1.0, 2.0, 3.0 };
    aether::Item<double, 3> b{ 4.0, 5.0, 6.0 };
    aether::Item<double, 3> c = a + b; // materializing ET assignment (Item's
                                        // out-of-line expr ctor, expr/Assign.h)
                                        // -- exercised here.
    out[0] = c.dot(c);
}
)AETHER_NVRTC";

const char* kControlSrc = R"AETHER_NVRTC(
#include <aether/aether.h>

extern "C" __global__ void probeKernel(double* out)
{
    aether::Vec3d a{ 1.0, 2.0, 3.0 };
    aether::Vec3d b{ 4.0, 5.0, 6.0 };
    aether::Vec3d c = a + b;
    out[0] = c.dot(c);
}
)AETHER_NVRTC";

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <device|control> [nvrtc-option ...]\n", argv[0]);
        return 2;
    }
    const std::string mode = argv[1];
    const char* src;
    const char* label;
    if (mode == "device") {
        src   = kDeviceSrc;
        label = "aether/device.h (device-safe umbrella, materializing Item c = a + b)";
    } else if (mode == "control") {
        src   = kControlSrc;
        label = "aether/aether.h (full umbrella, seeded-RED control)";
    } else {
        std::fprintf(stderr, "%s: unknown mode '%s' (want device|control)\n", argv[0], mode.c_str());
        return 2;
    }

    std::vector<std::string> optStrs;
    for (int i = 2; i < argc; ++i)
        optStrs.emplace_back(argv[i]);
    std::vector<const char*> opts;
    opts.reserve(optStrs.size());
    for (const auto& s : optStrs)
        opts.push_back(s.c_str());

    nvrtcProgram prog;
    const nvrtcResult createSt = nvrtcCreateProgram(&prog, src, "device_umbrella_probe.cu", 0, nullptr, nullptr);
    if (createSt != NVRTC_SUCCESS) {
        std::fprintf(stderr, "RED: nvrtcCreateProgram failed: %s\n", nvrtcGetErrorString(createSt));
        return 1;
    }

    const nvrtcResult compileSt = nvrtcCompileProgram(prog, static_cast<int>(opts.size()), opts.data());

    std::size_t logSize = 0;
    nvrtcGetProgramLogSize(prog, &logSize);
    std::string log(logSize > 0 ? logSize : 1, '\0');
    if (logSize > 1)
        nvrtcGetProgramLog(prog, log.data());

    if (compileSt != NVRTC_SUCCESS) {
        std::fprintf(stderr, "RED: nvrtcCompileProgram failed: %s\n", nvrtcGetErrorString(compileSt));
        std::fprintf(stderr, "--- NVRTC log (names the first offending header/line) ---\n%s\n"
                              "-----------------------------------------------------------\n",
            log.c_str());
        nvrtcDestroyProgram(&prog);
        return 1;
    }

    std::fprintf(stdout, "PASS: %s parses+compiles under NVRTC\n", label);
    if (logSize > 1) {
        std::fprintf(stdout, "--- NVRTC log (warnings) ---\n%s\n----------------------------\n", log.c_str());
    }
    nvrtcDestroyProgram(&prog);
    return 0;
}
