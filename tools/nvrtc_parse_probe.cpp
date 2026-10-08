// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// tools/nvrtc_parse_probe.cpp — de-risk probe: does aether device code
// parse and compile under NVRTC (the CUDA runtime-compilation API), not
// just nvcc's own frontend?
//
// Headers exercised only through nvcc's own frontend are not guaranteed
// to parse under NVRTC's narrower built-in-header set; a PASS here
// green-lights runtime JIT for a future codegen path, a RED names
// the first offending header via NVRTC's own compile log (which reports
// file/line).
//
// Host-only, no GPU: `nvrtcCompileProgram` is host-side compilation to
// PTX/CUBIN — this probe never creates a CUDA context, never launches
// anything (NVRTC compile is explicitly allowed here, launching is not).
//
// Standalone TU, compiled+run directly by tools/nvrtc_parse_audit.sh
// (mirrors tools/volatile_probe.cu's placement/rationale) — never wired
// into tests/CMakeLists.txt's glob (this is a de-risk probe, not a
// permanent regression test; see tools/nvrtc/README.md for its findings).
//
// CLI: nvrtc_parse_probe <mode> [nvrtc-option ...]
//   mode = "device-safe" — a minimal kernel over aether's own register-
//          resident expression-template arithmetic (Item construction, `+`,
//          `.dot()`) reached WITHOUT going through Item's out-of-line
//          expression constructor (kept as `auto`, never materialized into
//          an `Item`) — this is the achievable, in-scope GREEN target; see
//          tools/nvrtc/README.md for why materializing into an `Item`, or
//          `#include <aether/aether.h>` wholesale, is currently blocked by
//          aether-source-level issues outside this tool's edit scope.
//   mode = "umbrella"    — the literal `#include <aether/aether.h>` ask,
//          reported for transparency; expected RED today (informational
//          only, see tools/nvrtc/README.md).
// Every remaining argv entry is passed to nvrtcCompileProgram VERBATIM as
// one NVRTC compile option (the audit script owns deriving the actual
// values from the active toolchain — this probe stays a dumb executor so
// the toolchain-resolution logic lives in exactly one place).
//
// ALL toolchain paths passed to NVRTC MUST be derived at runtime by the
// caller from the active toolchain (nvcc location / conda prefix layout) —
// no machine-specific path may be hard-coded here or in the audit script.

#include <cstdio>
#include <string>
#include <vector>

#include <nvrtc.h>

namespace {

const char* kDeviceSafeSrc = R"AETHER_NVRTC(
#include <aether/view/Item.h>
#include <aether/expr/Operations.h>

extern "C" __global__ void probeKernel(double* out)
{
    aether::Item<double, 3> a{ 1.0, 2.0, 3.0 };
    aether::Item<double, 3> b{ 4.0, 5.0, 6.0 };
    auto c = a + b; // kept as an expression leaf — NOT materialized into an
                     // Item, whose out-of-line expr-ctor pulls in View.h ->
                     // err/Error.h (documented HOST-ONLY, see README).
    out[0] = c.dot(c);
}
)AETHER_NVRTC";

const char* kUmbrellaSrc = R"AETHER_NVRTC(
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
        std::fprintf(stderr, "usage: %s <device-safe|umbrella> [nvrtc-option ...]\n", argv[0]);
        return 2;
    }
    const std::string mode = argv[1];
    const char* src;
    const char* label;
    if (mode == "device-safe") {
        src   = kDeviceSafeSrc;
        label = "aether device-safe arithmetic kernel (Item/expr .dot() surface)";
    } else if (mode == "umbrella") {
        src   = kUmbrellaSrc;
        label = "aether/aether.h (full umbrella)";
    } else {
        std::fprintf(stderr, "%s: unknown mode '%s' (want device-safe|umbrella)\n", argv[0], mode.c_str());
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
    const nvrtcResult createSt = nvrtcCreateProgram(&prog, src, "aether_nvrtc_probe.cu", 0, nullptr, nullptr);
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
