// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file DriverRuntime.inl
 * @brief The CUDA runtime functions aether's CUDA-mode headers call, implemented over the CUDA DRIVER API, so a
 *        library built from them (the aether-dsc plugin) needs `libcuda.so.1` and no CUDA runtime library.
 *
 * Include this file ONCE, in ONE translation unit of a shared library linked with `nvcc -cudart none` (and
 * nowhere else: it defines the runtime's own function names). The library's version script must keep every symbol
 * local, so these definitions never interpose another library's CUDA runtime (cupy, torch) in the same process.
 *
 * DRIVER. `libcuda.so.1` is opened on first use; every driver function is resolved one by one through
 * `cuGetProcAddress` at the CUDA version whose ABI the `PFN_*` typedef below names, so a newer driver hands back
 * that same ABI. A missing entry returns `cudaErrorCallRequiresNewerDriver`; no driver at all returns
 * `cudaErrorInsufficientDriver`, as the runtime does. `AETHER_CUDA_DRIVER_HIDE` (comma separated entry names, or
 * `libcuda.so.1`) treats them as absent: the test hook for both paths.
 *
 * CONTEXTS follow the runtime's rules, so the device's PRIMARY context is shared with every other runtime in the
 * process: the current device is the device of the thread's current driver context; a call that needs a context and
 * finds none makes the primary context of the thread's selected device (initially 0) current.
 *
 * STREAMS. The plain spellings: stream 0 is the legacy stream, as in a build without `--default-stream per-thread`.
 * The `_ptsz` spellings (what a `--default-stream per-thread` build references) follow: there stream 0 is the
 * per-thread default stream (`CU_STREAM_PER_THREAD`). Each is a one-line wrapper over the plain function.
 *
 * KERNELS. nvcc registers a library's own kernels through the `__cudaRegister*` hooks defined here: the fat binary
 * is loaded with `cuModuleLoadData` once per context, and a launch is a `cuLaunchKernel`.
 */

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>

#include <dlfcn.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace aether_driver_runtime {

// name, ABI version: the slot's type is PFN_<name>_v<version>
#define AETHER_DRIVER_ENTRIES(X)                                                                            \
    X(cuInit, 2000) X(cuDeviceGetCount, 2000) X(cuDeviceGet, 2000) X(cuDevicePrimaryCtxRetain, 7000)          \
    X(cuCtxGetCurrent, 4000) X(cuCtxSetCurrent, 4000) X(cuCtxGetDevice, 2000) X(cuGetErrorString, 6000)        \
    X(cuDriverGetVersion, 2020) X(cuMemAlloc, 3020) X(cuMemFree, 3020) X(cuMemAllocAsync, 11020)               \
    X(cuMemFreeAsync, 11020) X(cuMemHostAlloc, 2020) X(cuMemFreeHost, 2000) X(cuMemcpy, 4000)                  \
    X(cuMemcpyAsync, 4000) X(cuMemsetD8Async, 3020) X(cuMemsetD2D8Async, 3020) X(cuStreamSynchronize, 2000)    \
    X(cuStreamWaitEvent, 3020) X(cuStreamIsCapturing, 10000) X(cuThreadExchangeStreamCaptureMode, 10010)       \
    X(cuEventCreate, 2000) X(cuEventDestroy, 4000) X(cuEventRecord, 2000) X(cuEventSynchronize, 2000)          \
    X(cuPointerGetAttribute, 4000) X(cuDeviceCanAccessPeer, 4000) X(cuCtxEnablePeerAccess, 4000)               \
    X(cuModuleLoadData, 2000) X(cuModuleGetFunction, 2000) X(cuLaunchKernel, 4000) X(cuFuncGetAttribute, 2020)                           \
    /* streams, capture, graphs, host functions (the eagle plugin) */                                          \
    X(cuGetErrorName, 6000) X(cuStreamGetCtx, 9020) X(cuDeviceGetAttribute, 2000) X(cuDeviceGetName, 2000)    \
    X(cuDeviceTotalMem, 3020) X(cuKernelGetFunction, 12000) X(cuStreamCreate, 2000) X(cuStreamDestroy, 4000)  \
    X(cuStreamBeginCapture, 10010) X(cuStreamEndCapture, 10000) X(cuStreamBeginCaptureToGraph, 12030)          \
    X(cuStreamGetCaptureInfo, 12030) X(cuStreamUpdateCaptureDependencies, 11030) X(cuEventElapsedTime, 2000)   \
    X(cuGraphCreate, 10000) X(cuGraphDestroy, 10000) X(cuGraphExecDestroy, 10000)                              \
    X(cuGraphInstantiateWithFlags, 11040) X(cuGraphLaunch, 10000) X(cuGraphGetNodes, 10000)                    \
    X(cuGraphNodeGetType, 10000) X(cuGraphAddKernelNode, 12000) X(cuGraphKernelNodeGetParams, 12000)          \
    X(cuGraphExecKernelNodeSetParams, 12000) X(cuGraphAddEmptyNode, 10000) X(cuGraphAddChildGraphNode, 10000)  \
    X(cuGraphChildGraphNodeGetGraph, 10000) X(cuGraphNodeSetEnabled, 11060) X(cuGraphDebugDotPrint, 11030)     \
    X(cuGraphConditionalHandleCreate, 12030) X(cuGraphAddNode, 12020) X(cuLaunchHostFunc, 10000)

struct Slots {
#define AETHER_SLOT(name, ver) PFN_##name##_v##ver name = nullptr;
    AETHER_DRIVER_ENTRIES(AETHER_SLOT)
#undef AETHER_SLOT
    bool loaded = false;
    CUresult (*getProc)(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*) = nullptr;   // cuGetProcAddress_v2
};

#ifndef AETHER_DRIVER_HIDE_ENV
#define AETHER_DRIVER_HIDE_ENV "AETHER_CUDA_DRIVER_HIDE"   // a plugin may name its own (eagle: EAGLE_CUDA_DRIVER_HIDE)
#endif

inline bool hidden(const char* name) {
    const char* list = std::getenv(AETHER_DRIVER_HIDE_ENV);
    if (list == nullptr) return false;
    const std::size_t n = std::strlen(name);
    for (const char* p = list; *p != '\0';) {
        const char* end = std::strchr(p, ',');
        const std::size_t len = end ? static_cast<std::size_t>(end - p) : std::strlen(p);
        if (len == n && std::strncmp(p, name, n) == 0) return true;
        if (end == nullptr) break;
        p = end + 1;
    }
    return false;
}

inline const Slots& driver() {
    static const Slots slots = [] {
        Slots out;
        void* lib = hidden("libcuda.so.1") ? nullptr : dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        if (lib == nullptr) return out;
        using GetProc = CUresult (*)(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*);
        auto get = reinterpret_cast<GetProc>(dlsym(lib, "cuGetProcAddress_v2"));
        if (get == nullptr) return out;
        out.getProc = get;
        auto resolve = [&](const char* name, int version) -> void* {
            void* fn = nullptr;
            CUdriverProcAddressQueryResult found{};
            if (hidden(name) || get(name, &fn, version, CU_GET_PROC_ADDRESS_LEGACY_STREAM, &found) != CUDA_SUCCESS ||
                found != CU_GET_PROC_ADDRESS_SUCCESS)
                return nullptr;
            return fn;
        };
#define AETHER_RESOLVE(name, ver) out.name = reinterpret_cast<PFN_##name##_v##ver>(resolve(#name, ver));
        AETHER_DRIVER_ENTRIES(AETHER_RESOLVE)
#undef AETHER_RESOLVE
        out.loaded = out.cuInit != nullptr;
        return out;
    }();
    return slots;
}

// ---- runtime error state -----------------------------------------------------------------------------------
inline thread_local cudaError_t t_lastError = cudaSuccess;
inline cudaError_t record(cudaError_t e) {
    if (e != cudaSuccess) t_lastError = e;
    return e;
}
// the runtime's codes take the driver's values for every condition the two share
inline cudaError_t fromDriver(CUresult r) { return record(static_cast<cudaError_t>(r)); }

inline cudaError_t initDriver() {
    static const cudaError_t status = [] {
        const Slots& d = driver();
        return d.loaded ? static_cast<cudaError_t>(d.cuInit(0)) : cudaErrorInsufficientDriver;
    }();
    return record(status);
}

/// `slot` of the driver, or nullptr after recording why it cannot be called
template <class Fn> Fn need(Fn Slots::*slot, cudaError_t& err) {
    err = initDriver();
    if (err != cudaSuccess) return nullptr;
    Fn f = driver().*slot;
    if (f == nullptr) err = record(cudaErrorCallRequiresNewerDriver);
    return f;
}
#define AETHER_DRV(var, slot)                                                                               \
    cudaError_t var##_err = cudaSuccess;                                                                    \
    const auto var = aether_driver_runtime::need(&aether_driver_runtime::Slots::slot, var##_err);           \
    if (var == nullptr) return var##_err

// ---- contexts ----------------------------------------------------------------------------------------------
inline thread_local int t_device = 0;   // the thread's selected device while no context is current

struct Primary {
    std::mutex mutex;
    std::map<int, CUcontext> contexts;   // retained once and kept, as the runtime does
};
inline Primary& primary() {
    static Primary* p = new Primary();
    return *p;
}

inline cudaError_t primaryOf(int device, CUcontext* ctx) {
    AETHER_DRV(retain, cuDevicePrimaryCtxRetain);
    AETHER_DRV(get, cuDeviceGet);
    Primary& pr = primary();
    std::lock_guard<std::mutex> lock(pr.mutex);
    auto it = pr.contexts.find(device);
    if (it == pr.contexts.end()) {
        CUdevice dev = 0;
        if (CUresult r = get(&dev, device); r != CUDA_SUCCESS)
            return record(r == CUDA_ERROR_INVALID_VALUE ? cudaErrorInvalidDevice : static_cast<cudaError_t>(r));
        if (CUresult r = retain(ctx, dev); r != CUDA_SUCCESS) return fromDriver(r);
        pr.contexts.emplace(device, *ctx);
    } else {
        *ctx = it->second;
    }
    return cudaSuccess;
}

inline cudaError_t bindPrimary(int device) {
    AETHER_DRV(set, cuCtxSetCurrent);
    CUcontext ctx = nullptr;
    if (cudaError_t e = primaryOf(device, &ctx); e != cudaSuccess) return e;
    if (CUresult r = set(ctx); r != CUDA_SUCCESS) return fromDriver(r);
    t_device = device;
    return cudaSuccess;
}

/// the runtime's lazy initialisation: the calling thread has a current context afterwards
inline cudaError_t ensureContext() {
    AETHER_DRV(getCurrent, cuCtxGetCurrent);
    CUcontext ctx = nullptr;
    if (CUresult r = getCurrent(&ctx); r != CUDA_SUCCESS) return fromDriver(r);
    return ctx != nullptr ? cudaSuccess : bindPrimary(t_device);
}
#define AETHER_CTX()                                                                                        \
    do {                                                                                                    \
        if (cudaError_t ctx_err_ = aether_driver_runtime::ensureContext(); ctx_err_ != cudaSuccess)         \
            return ctx_err_;                                                                                \
    } while (0)

inline CUstream stream(cudaStream_t s) { return reinterpret_cast<CUstream>(s); }

// ---- kernels (nvcc's registration hooks) -------------------------------------------------------------------
struct FatbinWrapper { int magic; int version; const void* data; void* filenameOrFatbins; };
struct Kernel { std::size_t image; std::string name; };
struct Registry {
    std::mutex mutex;
    std::vector<const void*> images;
    std::map<const void*, Kernel> kernels;                           // host stub -> device kernel
    std::map<std::pair<CUcontext, std::size_t>, CUmodule> modules;   // per context and image
    std::map<std::pair<CUcontext, const void*>, CUfunction> functions;
    std::map<CUfunction, const void*> stubs;   // reverse: a kernel node's CUfunction -> its host stub
};
inline Registry& registry() {
    static Registry* r = new Registry();   // registration runs from static initialisers of other units
    return *r;
}
struct LaunchConfig { dim3 grid, block; std::size_t shared; CUstream_st* stream; };
inline thread_local std::vector<LaunchConfig> t_launchConfigs;

inline cudaError_t functionOf(const void* stub, CUfunction* out) {
    AETHER_CTX();
    AETHER_DRV(getCurrent, cuCtxGetCurrent);
    AETHER_DRV(load, cuModuleLoadData);
    AETHER_DRV(getFunction, cuModuleGetFunction);
    CUcontext ctx = nullptr;
    getCurrent(&ctx);
    Registry& reg = registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    auto k = reg.kernels.find(stub);
    if (k == reg.kernels.end()) return record(cudaErrorInvalidDeviceFunction);
    if (auto f = reg.functions.find({ctx, stub}); f != reg.functions.end()) { *out = f->second; return cudaSuccess; }
    CUmodule& mod = reg.modules[{ctx, k->second.image}];
    if (mod == nullptr) {
        if (CUresult r = load(&mod, reg.images[k->second.image]); r != CUDA_SUCCESS) { mod = nullptr; return fromDriver(r); }
    }
    CUfunction f = nullptr;
    if (CUresult r = getFunction(&f, mod, k->second.name.c_str()); r != CUDA_SUCCESS) return fromDriver(r);
    reg.functions[{ctx, stub}] = f;
    reg.stubs[f] = stub;
    *out = f;
    return cudaSuccess;
}

}  // namespace aether_driver_runtime

using namespace aether_driver_runtime;

extern "C" {

void** __cudaRegisterFatBinary(void* fatCubin) {
    const auto* w = static_cast<const FatbinWrapper*>(fatCubin);
    Registry& reg = registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    reg.images.push_back(w->data);
    return reinterpret_cast<void**>(reg.images.size());   // only ever handed back to the hooks below
}
void __cudaRegisterFatBinaryEnd(void**) {}
void __cudaUnregisterFatBinary(void**) {}   // modules die with their contexts
void __cudaRegisterFunction(void** handle, const char* hostFun, char*, const char* deviceName, int, uint3*, uint3*,
                            dim3*, dim3*, int*) {
    Registry& reg = registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    reg.kernels[hostFun] = Kernel{reinterpret_cast<std::size_t>(handle) - 1, deviceName};
}
void __cudaRegisterVar(void**, char*, char*, const char*, int, std::size_t, int, int) {}
unsigned __cudaPushCallConfiguration(dim3 grid, dim3 block, std::size_t shared, CUstream_st* s) {
    t_launchConfigs.push_back(LaunchConfig{grid, block, shared, s});
    return 0;
}
cudaError_t __cudaPopCallConfiguration(dim3* grid, dim3* block, std::size_t* shared, void* s) {
    if (t_launchConfigs.empty()) return record(cudaErrorMissingConfiguration);
    const LaunchConfig c = t_launchConfigs.back();
    t_launchConfigs.pop_back();
    *grid = c.grid; *block = c.block; *shared = c.shared;
    *static_cast<cudaStream_t*>(s) = c.stream;
    return cudaSuccess;
}

cudaError_t cudaLaunchKernel(const void* func, dim3 grid, dim3 block, void** args, std::size_t shared, cudaStream_t s) {
    CUfunction f = nullptr;
    if (cudaError_t e = functionOf(func, &f); e != cudaSuccess) return e;
    AETHER_DRV(launch, cuLaunchKernel);
    return fromDriver(launch(f, grid.x, grid.y, grid.z, block.x, block.y, block.z, static_cast<unsigned>(shared), stream(s), args, nullptr));
}

// ---- errors and versions -----------------------------------------------------------------------------------
cudaError_t cudaGetLastError(void) {
    const cudaError_t e = t_lastError;
    t_lastError = cudaSuccess;
    return e;
}
const char* cudaGetErrorString(cudaError_t error) {
    switch (error) {
        case cudaSuccess: return "no error";
        case cudaErrorInsufficientDriver: return "no usable CUDA driver: libcuda.so.1 could not be loaded";
        case cudaErrorCallRequiresNewerDriver: return "the installed CUDA driver does not provide this entry point";
        case cudaErrorNoDevice: return "no CUDA-capable device is detected";
        case cudaErrorInvalidDeviceFunction: return "invalid device function";
        case cudaErrorMissingConfiguration: return "__global__ function call is not configured";
        default: break;
    }
    const cudaError_t saved = t_lastError;   // a lookup is not the caller's error
    const char* text = nullptr;
    const Slots& d = driver();
    if (initDriver() == cudaSuccess && d.cuGetErrorString != nullptr &&
        d.cuGetErrorString(static_cast<CUresult>(error), &text) == CUDA_SUCCESS && text != nullptr) {
        t_lastError = saved;
        return text;
    }
    t_lastError = saved;
    return "unrecognized error code";
}
// the version of CUDA the DRIVER supports: there is no runtime library to ask
cudaError_t cudaRuntimeGetVersion(int* v) {
    AETHER_DRV(get, cuDriverGetVersion);
    return fromDriver(get(v));
}

// ---- devices -----------------------------------------------------------------------------------------------
cudaError_t cudaGetDeviceCount(int* count) {
    AETHER_DRV(getCount, cuDeviceGetCount);
    int n = 0;
    if (CUresult r = getCount(&n); r != CUDA_SUCCESS) return fromDriver(r);
    *count = n;
    return n == 0 ? record(cudaErrorNoDevice) : cudaSuccess;
}
cudaError_t cudaGetDevice(int* device) {
    AETHER_DRV(getCurrent, cuCtxGetCurrent);
    AETHER_DRV(getDevice, cuCtxGetDevice);
    CUcontext ctx = nullptr;
    if (CUresult r = getCurrent(&ctx); r != CUDA_SUCCESS) return fromDriver(r);
    if (ctx == nullptr) { *device = t_device; return cudaSuccess; }
    CUdevice dev = 0;
    if (CUresult r = getDevice(&dev); r != CUDA_SUCCESS) return fromDriver(r);
    *device = static_cast<int>(dev);
    return cudaSuccess;
}
cudaError_t cudaSetDevice(int device) {
    int n = 0;
    if (cudaError_t e = cudaGetDeviceCount(&n); e != cudaSuccess) return e;
    if (device < 0 || device >= n) return record(cudaErrorInvalidDevice);
    return bindPrimary(device);
}
cudaError_t cudaDeviceCanAccessPeer(int* can, int device, int peer) {
    AETHER_DRV(access, cuDeviceCanAccessPeer);
    AETHER_DRV(get, cuDeviceGet);
    CUdevice a = 0, b = 0;
    if (CUresult r = get(&a, device); r != CUDA_SUCCESS) return fromDriver(r);
    if (CUresult r = get(&b, peer); r != CUDA_SUCCESS) return fromDriver(r);
    return fromDriver(access(can, a, b));
}
cudaError_t cudaDeviceEnablePeerAccess(int peer, unsigned int flags) {
    AETHER_CTX();
    AETHER_DRV(enable, cuCtxEnablePeerAccess);
    CUcontext ctx = nullptr;
    if (cudaError_t e = primaryOf(peer, &ctx); e != cudaSuccess) return e;
    return fromDriver(enable(ctx, flags));
}

// ---- memory ------------------------------------------------------------------------------------------------
cudaError_t cudaMalloc(void** p, size_t bytes) {
    AETHER_CTX();
    AETHER_DRV(alloc, cuMemAlloc);
    return fromDriver(alloc(reinterpret_cast<CUdeviceptr*>(p), bytes));
}
cudaError_t cudaFree(void* p) {
    if (p == nullptr) return cudaSuccess;
    AETHER_CTX();
    AETHER_DRV(release, cuMemFree);
    return fromDriver(release(reinterpret_cast<CUdeviceptr>(p)));
}
cudaError_t cudaMallocAsync(void** p, size_t bytes, cudaStream_t s) {
    AETHER_CTX();
    AETHER_DRV(alloc, cuMemAllocAsync);
    return fromDriver(alloc(reinterpret_cast<CUdeviceptr*>(p), bytes, stream(s)));
}
cudaError_t cudaFreeAsync(void* p, cudaStream_t s) {
    if (p == nullptr) return cudaSuccess;
    AETHER_CTX();
    AETHER_DRV(release, cuMemFreeAsync);
    return fromDriver(release(reinterpret_cast<CUdeviceptr>(p), stream(s)));
}
cudaError_t cudaHostAlloc(void** p, size_t bytes, unsigned int flags) {
    AETHER_CTX();
    AETHER_DRV(alloc, cuMemHostAlloc);
    return fromDriver(alloc(p, bytes, flags));
}
cudaError_t cudaMallocHost(void** p, size_t bytes) { return cudaHostAlloc(p, bytes, 0); }
cudaError_t cudaFreeHost(void* p) {
    AETHER_CTX();
    AETHER_DRV(release, cuMemFreeHost);
    return fromDriver(release(p));
}
// unified addressing: one call copies every direction
cudaError_t cudaMemcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind) {
    AETHER_CTX();
    AETHER_DRV(copy, cuMemcpy);
    return fromDriver(copy(reinterpret_cast<CUdeviceptr>(dst), reinterpret_cast<CUdeviceptr>(src), bytes));
}
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t bytes, cudaMemcpyKind, cudaStream_t s) {
    AETHER_CTX();
    AETHER_DRV(copy, cuMemcpyAsync);
    return fromDriver(copy(reinterpret_cast<CUdeviceptr>(dst), reinterpret_cast<CUdeviceptr>(src), bytes, stream(s)));
}
cudaError_t cudaMemsetAsync(void* dst, int value, size_t bytes, cudaStream_t s) {
    AETHER_CTX();
    AETHER_DRV(set, cuMemsetD8Async);
    return fromDriver(set(reinterpret_cast<CUdeviceptr>(dst), static_cast<unsigned char>(value), bytes, stream(s)));
}
cudaError_t cudaMemset2DAsync(void* dst, size_t pitch, int value, size_t width, size_t height, cudaStream_t s) {
    AETHER_CTX();
    AETHER_DRV(set, cuMemsetD2D8Async);
    return fromDriver(set(reinterpret_cast<CUdeviceptr>(dst), pitch, static_cast<unsigned char>(value), width, height, stream(s)));
}
cudaError_t cudaPointerGetAttributes(cudaPointerAttributes* a, const void* p) {
    AETHER_CTX();
    AETHER_DRV(query, cuPointerGetAttribute);
    std::memset(a, 0, sizeof(*a));
    unsigned int type = 0;
    CUresult r = query(&type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, reinterpret_cast<CUdeviceptr>(p));
    if (r == CUDA_ERROR_INVALID_VALUE) {   // plain pageable host memory
        a->type = cudaMemoryTypeUnregistered;
        a->device = -1;
        a->hostPointer = const_cast<void*>(p);
        return cudaSuccess;
    }
    if (r != CUDA_SUCCESS) return fromDriver(r);
    int managed = 0, ordinal = -1;
    query(&managed, CU_POINTER_ATTRIBUTE_IS_MANAGED, reinterpret_cast<CUdeviceptr>(p));
    query(&ordinal, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, reinterpret_cast<CUdeviceptr>(p));
    a->type = managed ? cudaMemoryTypeManaged : type == CU_MEMORYTYPE_DEVICE ? cudaMemoryTypeDevice : cudaMemoryTypeHost;
    a->device = ordinal;
    if (type == CU_MEMORYTYPE_DEVICE || managed) a->devicePointer = const_cast<void*>(p);
    if (type == CU_MEMORYTYPE_HOST || managed) a->hostPointer = const_cast<void*>(p);
    return cudaSuccess;
}

// ---- streams, events, capture ------------------------------------------------------------------------------
cudaError_t cudaStreamSynchronize(cudaStream_t s) {
    AETHER_CTX();
    AETHER_DRV(sync, cuStreamSynchronize);
    return fromDriver(sync(stream(s)));
}
cudaError_t cudaStreamWaitEvent(cudaStream_t s, cudaEvent_t e, unsigned int flags) {
    AETHER_CTX();
    AETHER_DRV(wait, cuStreamWaitEvent);
    return fromDriver(wait(stream(s), reinterpret_cast<CUevent>(e), flags));
}
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e, unsigned int flags) {
    AETHER_CTX();
    AETHER_DRV(create, cuEventCreate);
    return fromDriver(create(reinterpret_cast<CUevent*>(e), flags));
}
cudaError_t cudaEventCreate(cudaEvent_t* e) { return cudaEventCreateWithFlags(e, cudaEventDefault); }
cudaError_t cudaEventDestroy(cudaEvent_t e) {
    AETHER_DRV(destroy, cuEventDestroy);
    return fromDriver(destroy(reinterpret_cast<CUevent>(e)));
}
cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t s) {
    AETHER_CTX();
    AETHER_DRV(rec, cuEventRecord);
    return fromDriver(rec(reinterpret_cast<CUevent>(e), stream(s)));
}
cudaError_t cudaEventSynchronize(cudaEvent_t e) {
    AETHER_DRV(sync, cuEventSynchronize);
    return fromDriver(sync(reinterpret_cast<CUevent>(e)));
}
cudaError_t cudaStreamIsCapturing(cudaStream_t s, cudaStreamCaptureStatus* status) {
    AETHER_CTX();
    AETHER_DRV(query, cuStreamIsCapturing);
    CUstreamCaptureStatus st = CU_STREAM_CAPTURE_STATUS_NONE;
    if (CUresult r = query(stream(s), &st); r != CUDA_SUCCESS) return fromDriver(r);
    *status = static_cast<cudaStreamCaptureStatus>(st);
    return cudaSuccess;
}
cudaError_t cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode* mode) {
    AETHER_CTX();
    AETHER_DRV(exchange, cuThreadExchangeStreamCaptureMode);
    auto m = static_cast<CUstreamCaptureMode>(*mode);
    CUresult r = exchange(&m);
    *mode = static_cast<cudaStreamCaptureMode>(m);
    return fromDriver(r);
}

// ---- the per-thread default stream spellings (`--default-stream per-thread` builds) -----------------------------
inline cudaStream_t perThreadStream(cudaStream_t s) { return s == nullptr ? reinterpret_cast<cudaStream_t>(CU_STREAM_PER_THREAD) : s; }
cudaError_t cudaLaunchKernel_ptsz(const void* f, dim3 g, dim3 b, void** a, std::size_t sh, cudaStream_t s) { return cudaLaunchKernel(f, g, b, a, sh, perThreadStream(s)); }
cudaError_t cudaMallocAsync_ptsz(void** p, size_t n, cudaStream_t s) { return cudaMallocAsync(p, n, perThreadStream(s)); }
cudaError_t cudaFreeAsync_ptsz(void* p, cudaStream_t s) { return cudaFreeAsync(p, perThreadStream(s)); }
cudaError_t cudaMemcpyAsync_ptsz(void* d, const void* src, size_t n, cudaMemcpyKind k, cudaStream_t s) { return cudaMemcpyAsync(d, src, n, k, perThreadStream(s)); }
cudaError_t cudaMemsetAsync_ptsz(void* d, int v, size_t n, cudaStream_t s) { return cudaMemsetAsync(d, v, n, perThreadStream(s)); }
cudaError_t cudaMemset2DAsync_ptsz(void* d, size_t pitch, int v, size_t w, size_t h, cudaStream_t s) { return cudaMemset2DAsync(d, pitch, v, w, h, perThreadStream(s)); }
cudaError_t cudaStreamSynchronize_ptsz(cudaStream_t s) { return cudaStreamSynchronize(perThreadStream(s)); }
cudaError_t cudaStreamWaitEvent_ptsz(cudaStream_t s, cudaEvent_t e, unsigned int f) { return cudaStreamWaitEvent(perThreadStream(s), e, f); }
cudaError_t cudaEventRecord_ptsz(cudaEvent_t e, cudaStream_t s) { return cudaEventRecord(e, perThreadStream(s)); }
cudaError_t cudaStreamIsCapturing_ptsz(cudaStream_t s, cudaStreamCaptureStatus* st) { return cudaStreamIsCapturing(perThreadStream(s), st); }

}  // extern "C"
