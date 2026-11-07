// Compile one source with the real NVRTC library: exit 0 = compiled, 1 = compile error
// (log on stderr), 2 = usage/API failure. The include budget is the shims directory.
#include <nvrtc.h>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

int main(int argc, char** argv)
{
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s <source> <aether root> <shims dir> <cccl include>\n", argv[0]);
        return 2;
    }
    std::ifstream in(argv[1]);
    if (!in) {
        std::fprintf(stderr, "cannot read %s\n", argv[1]);
        return 2;
    }
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string src = ss.str();
    const std::string incAether = std::string("-I") + argv[2];
    const std::string incShims  = std::string("-I") + argv[3];
    const std::string incCccl   = std::string("-I") + argv[4];
    const char* opts[]          = { "-std=c++20", "-default-device", "--gpu-architecture=compute_61",
        incAether.c_str(), incShims.c_str(), incCccl.c_str() };
    nvrtcProgram prog;
    if (nvrtcCreateProgram(&prog, src.c_str(), "row.cu", 0, nullptr, nullptr) != NVRTC_SUCCESS)
        return 2;
    const nvrtcResult rc = nvrtcCompileProgram(prog, 6, opts);
    std::size_t n = 0;
    nvrtcGetProgramLogSize(prog, &n);
    std::string log(n, '\0');
    if (n > 1)
        nvrtcGetProgramLog(prog, &log[0]);
    nvrtcDestroyProgram(&prog);
    if (rc == NVRTC_SUCCESS)
        return 0;
    std::fprintf(stderr, "%s\n", log.c_str());
    return 1;
}
