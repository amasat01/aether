// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0
// aether._core: CPU and GPU arrays for Python. Plain C++ (no CUDA, no numpy); device memory goes through a
// CUDA plugin behind the C seam in seam.h, loaded on the first device call.
#include <dlfcn.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "aether/chunk/Chunk.h"
#include "aether/chunk/Fill.h"
#include "aether/interop/DLPack.h"
#include "aether/interop/DLPackLegacy.h"
#include "seam.h"

namespace nb = nanobind;
using namespace aether;

namespace {

nb::object g_error, g_unavailable;   // aether.Error, aether.CudaUnavailable
nb::object g_device_cls, g_dtype_cls;

[[noreturn]] void fail(const std::string& m) { PyErr_SetString(g_error.ptr(), m.c_str()); throw nb::python_error(); }
[[noreturn]] void unavailable(const std::string& m) {
    PyErr_SetString(g_unavailable.ptr(), m.c_str());
    throw nb::python_error();
}

// ---------------------------------------------------------------- dtypes
struct DT { const char* name; std::uint8_t code, bits; const char* cai; const char* fmt; };
// fmt / cai are null where no buffer format / __cuda_array_interface__ typestr exists (bfloat16)
const DT kDtypes[] = {
    {"float64", kDLFloat, 64, "<f8", "d"},   {"float32", kDLFloat, 32, "<f4", "f"},
    {"int64", kDLInt, 64, "<i8", "q"},       {"int32", kDLInt, 32, "<i4", "i"},
    {"uint64", kDLUInt, 64, "<u8", "Q"},     {"uint32", kDLUInt, 32, "<u4", "I"},
    {"uint8", kDLUInt, 8, "|u1", "B"},       {"bool", kDLBool, 8, "|b1", "?"},
    {"float16", kDLFloat, 16, "<f2", "e"},   {"bfloat16", kDLBfloat, 16, nullptr, nullptr},
    {"int8", kDLInt, 8, "|i1", "b"},         {"int16", kDLInt, 16, "<i2", "h"},
    {"uint16", kDLUInt, 16, "<u2", "H"},
};
constexpr int kNumDtypes = sizeof(kDtypes) / sizeof(kDtypes[0]);
enum { F16 = 8, BF16 = 9 };
constexpr const char* kSupported = "float64 float32 float16 bfloat16 int64 int32 int16 int8 uint64 uint32 uint16 uint8 bool";
int dtype_index(const std::string& name) {
    for (int i = 0; i < kNumDtypes; ++i) if (name == kDtypes[i].name) return i;
    fail("dtype '" + name + "' is not supported (" + kSupported + ")");
}
int dtype_index(DLDataType dt) {
    for (int i = 0; i < kNumDtypes; ++i) if (kDtypes[i].code == dt.code && kDtypes[i].bits == dt.bits && dt.lanes == 1) return i;
    fail("DLPack dtype code " + std::to_string(dt.code) + " bits " + std::to_string(dt.bits) + " is not supported");
}
int dtype_from_format(const char* f) {
    while (*f == '<' || *f == '=' || *f == '@' || *f == '|') ++f;
    std::string c = (*f == 'l') ? "q" : (*f == 'L') ? "Q" : f;
    for (int i = 0; i < kNumDtypes; ++i) if (kDtypes[i].fmt && c == kDtypes[i].fmt) return i;
    fail(std::string("buffer format '") + f + "' is not supported");
}
DType aether_dtype(int i) { return DType(kDtypes[i].code, kDtypes[i].bits, 1); }
std::size_t itemsize(int i) { return kDtypes[i].bits / 8; }

// ---------------------------------------------------------------- the plugin
struct Plugin {
    const AetherMemoryV1* t = nullptr;
    std::string name;
    std::string failure;
};
Plugin g_plugin;
std::once_flag g_load_once;

std::string module_dir() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&module_dir), &info) || !info.dli_fname) return ".";
    std::string p = info.dli_fname;
    auto cut = p.find_last_of('/');
    return cut == std::string::npos ? "." : p.substr(0, cut);
}

void load_plugin() {
    std::vector<int> order = {13, 12};
    if (const char* forced = std::getenv("AETHER_CUDA_MAJOR")) {
        std::string f = forced;
        if (f != "12" && f != "13") fail("AETHER_CUDA_MAJOR must be 12 or 13, not '" + f + "'");
        order = {std::stoi(f)};
    }
    std::string why;
    for (int major : order) {
        std::string path = module_dir() + "/libaether_cuda" + std::to_string(major) + ".so";
        void* h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!h) { const char* e = dlerror(); why += "cuda" + std::to_string(major) + ": " + (e ? e : "not loadable") + "; "; continue; }
        auto fn = reinterpret_cast<aether_memory_v1_fn>(dlsym(h, "aether_memory_v1"));
        if (!fn || fn()->abi != AETHER_MEMORY_ABI) { why += "cuda" + std::to_string(major) + ": not an aether-memory/1 plugin; "; continue; }
        int n = 0;
        int code = fn()->probe(&n);
        if (code != 0) { why += "cuda" + std::to_string(major) + " runtime: " + fn()->last_error() + "; "; continue; }
        g_plugin.t = fn();
        g_plugin.name = "cuda" + std::to_string(major);
        return;
    }
    g_plugin.failure = why;
}

const AetherMemoryV1* cuda() {
    std::call_once(g_load_once, load_plugin);
    if (!g_plugin.t) unavailable("no CUDA runtime available: " + g_plugin.failure);
    return g_plugin.t;
}
void check(int code, const char* what) {
    if (code != 0) fail(std::string(what) + ": " + cuda()->last_error());
}

// ---------------------------------------------------------------- storage
enum Kind { CPU = 0, CUDA = 1, PINNED = 2 };

struct Storage {
    void* ptr = nullptr;
    std::size_t bytes = 0;
    int kind = CPU, dev = 0;
    void* handle = nullptr;        // allocation stream
    Chunk host;                    // the owning aether Chunk of a CPU array
    void* chunk = nullptr;         // the plugin's owning Chunk of a device or pinned array
    nb::object stream;             // the stream OBJECT: it outlives the array
    nb::object keep;               // a foreign object whose memory this borrows
    Py_buffer* pybuf = nullptr;
    interop::DLPackOwner owner;    // an imported DLPack capsule

    // the Chunk frees on its allocation stream, queued after the work already there, and defers the free
    // while a capture makes that unsafe (all in aether's Chunk)
    ~Storage() {
        if (pybuf) { PyBuffer_Release(pybuf); delete pybuf; pybuf = nullptr; }
        owner.release();
        if (chunk) g_plugin.t->chunk_release(chunk);
    }
};

void* stream_handle(const nb::object& s) {
    if (s.is_none()) return nullptr;
    if (PyLong_Check(s.ptr())) return reinterpret_cast<void*>(static_cast<std::intptr_t>(nb::cast<std::int64_t>(s)));
    nb::object proto = nb::getattr(s, "__cuda_stream__", nb::none());
    if (proto.is_none()) fail("stream= wants an int handle or an object with __cuda_stream__");
    nb::object v = PyCallable_Check(proto.ptr()) ? proto() : proto;
    return reinterpret_cast<void*>(static_cast<std::intptr_t>(nb::cast<std::int64_t>(v[1])));
}

// release the frees a capture deferred (and, at exit, any still queued)
void drain() { if (g_plugin.t) g_plugin.t->drain(); }

// ---------------------------------------------------------------- Array
struct Array {
    std::shared_ptr<Storage> st;
    RuntimeView view;
    int dt = 0;

    std::int64_t size() const { std::int64_t n = 1; for (std::size_t i = 0; i < view.rank; ++i) n *= view.extents[i]; return n; }
    std::size_t nbytes() const { return std::size_t(size()) * itemsize(dt); }
    bool contiguous() const {
        std::int64_t acc = 1;
        for (std::size_t i = view.rank; i-- > 0;) {
            if (view.extents[i] != 1 && view.strides[i] != acc) return false;
            acc *= view.extents[i];
        }
        return true;
    }
    bool host() const { return st->kind != CUDA; }
    void need_contiguous(const char* what) const { if (!contiguous()) fail(std::string(what) + " needs a contiguous array; this one is strided"); }
};

Device device_of(int kind, int dev) {
    return kind == CPU ? Device(kDLCPU) : kind == PINNED ? Device(kDLCUDAHost) : Device(kDLCUDA, dev);
}

void set_shape(Array& a, const std::vector<std::int64_t>& shape) {
    if (shape.size() > RuntimeRankCap) fail("rank " + std::to_string(shape.size()) + " exceeds " + std::to_string(RuntimeRankCap));
    a.view.rank = shape.size();
    std::int64_t acc = 1;
    for (std::size_t i = shape.size(); i-- > 0;) {
        if (shape[i] < 0) fail("negative extent");
        a.view.extents[i] = shape[i];
        a.view.strides[i] = acc;
        acc *= std::max<std::int64_t>(shape[i], 1);
    }
}

// fill `n` bytes at `ptr` with the `item`-byte element `raw`: device memory through the plugin (on `h`), host
// memory (pageable or pinned) in place
void fill_bytes(int kind, int dev, void* ptr, std::size_t n, const unsigned char* raw, std::size_t item, void* h) {
    if (kind == CUDA) { check(cuda()->fill(kind, dev, ptr, n, raw, item, h), "fill"); return; }
    Chunk host = Chunk::borrow(ptr, n, Device(kDLCPU));
    aether::fill(host, std::span<const std::byte>(reinterpret_cast<const std::byte*>(raw), item), nullptr);
}

Array make(int kind, int dev, const std::vector<std::int64_t>& shape, int dt, void* handle, const nb::object& stream, bool zero) {
    auto st = std::make_shared<Storage>();
    Array a;
    a.dt = dt;
    set_shape(a, shape);
    std::size_t bytes = a.nbytes();
    st->kind = kind; st->dev = dev; st->bytes = bytes; st->handle = handle; st->stream = stream;
    if (bytes) {
        if (kind == CPU) {
            st->host = Chunk::allocate(Device(kDLCPU), bytes, 64);
            st->ptr = st->host.data();
        } else {
            check(cuda()->chunk_alloc(kind, dev, bytes, handle, &st->chunk, &st->ptr), "alloc");
        }
        if (zero) {
            const unsigned char zero_byte = 0;
            fill_bytes(kind, dev, st->ptr, bytes, &zero_byte, 1, handle);
        }
    }
    a.view.data = st->ptr; a.view.dtype = aether_dtype(dt); a.view.device = device_of(kind, dev);
    a.st = std::move(st);
    return a;
}

// IEEE half bits of a double (round to nearest even, overflow to inf, subnormals kept)
std::uint16_t half_bits(double d) {
    float f = float(d);
    std::uint32_t x; std::memcpy(&x, &f, 4);
    std::uint16_t sign = std::uint16_t((x >> 16) & 0x8000u);
    int e = int((x >> 23) & 0xff) - 127 + 15;
    std::uint32_t m = x & 0x7fffffu;
    if (((x >> 23) & 0xff) == 0xff) return std::uint16_t(sign | 0x7c00u | (m ? 0x200u : 0u));   // inf, nan
    if (e >= 31) return std::uint16_t(sign | 0x7c00u);
    if (e <= 0) {
        if (e < -10) return sign;
        m |= 0x800000u;
        int shift = 14 - e;
        std::uint32_t h = m >> shift, rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1))) ++h;
        return std::uint16_t(sign | h);
    }
    std::uint32_t h = (std::uint32_t(e) << 10) | (m >> 13), rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) ++h;   // a carry into the exponent is the right rounding
    return std::uint16_t(sign | h);
}
std::uint16_t bfloat_bits(float f) {
    std::uint32_t x; std::memcpy(&x, &f, 4);
    if ((x & 0x7fffffffu) > 0x7f800000u) return std::uint16_t((x >> 16) | 0x40u);   // nan
    return std::uint16_t((x + 0x7fffu + ((x >> 16) & 1u)) >> 16);
}

// pack a Python scalar as one element of this dtype
void pack(int dt, const nb::object& v, unsigned char* out) {
    switch (dt) {
        case 0: { double x = nb::cast<double>(v); std::memcpy(out, &x, 8); break; }
        case 1: { float x = float(nb::cast<double>(v)); std::memcpy(out, &x, 4); break; }
        case 2: { std::int64_t x = nb::cast<std::int64_t>(v); std::memcpy(out, &x, 8); break; }
        case 3: { std::int32_t x = std::int32_t(nb::cast<std::int64_t>(v)); std::memcpy(out, &x, 4); break; }
        case 4: { std::uint64_t x = nb::cast<std::uint64_t>(v); std::memcpy(out, &x, 8); break; }
        case 5: { std::uint32_t x = std::uint32_t(nb::cast<std::uint64_t>(v)); std::memcpy(out, &x, 4); break; }
        case 6: { out[0] = (unsigned char) nb::cast<std::uint64_t>(v); break; }
        case 7: out[0] = PyObject_IsTrue(v.ptr()) ? 1 : 0; break;
        case F16: { std::uint16_t x = half_bits(nb::cast<double>(v)); std::memcpy(out, &x, 2); break; }
        case BF16: { std::uint16_t x = bfloat_bits(float(nb::cast<double>(v))); std::memcpy(out, &x, 2); break; }
        case 10: { std::int8_t x = std::int8_t(nb::cast<std::int64_t>(v)); std::memcpy(out, &x, 1); break; }
        case 11: { std::int16_t x = std::int16_t(nb::cast<std::int64_t>(v)); std::memcpy(out, &x, 2); break; }
        default: { std::uint16_t x = std::uint16_t(nb::cast<std::uint64_t>(v)); std::memcpy(out, &x, 2); }   // uint16
    }
}

void fill(Array& a, const nb::object& value, const nb::object& stream, bool zero) {
    a.need_contiguous(zero ? "zero" : "fill");
    std::size_t n = a.nbytes(), item = itemsize(a.dt);
    if (!n) return;
    unsigned char raw[8] = {0};
    if (!zero) pack(a.dt, value, raw);
    if (a.host() && !stream.is_none() && a.st->kind == PINNED) check(cuda()->stream_sync(a.st->dev, stream_handle(stream)), "fill");
    void* h = a.host() ? nullptr : stream.is_none() ? a.st->handle : stream_handle(stream);
    fill_bytes(a.st->kind, a.st->dev, a.st->ptr, n, raw, item, h);
}

// ---------------------------------------------------------------- DLPack export
std::atomic<int> g_exports{0};
template <class T> struct Keep { void* ctx; void (*del)(T*); std::shared_ptr<Storage> keep; };
template <class T> void keep_deleter(T* self) {
    auto* k = static_cast<Keep<T>*>(self->manager_ctx);
    self->manager_ctx = k->ctx;
    auto del = k->del;
    {
        nb::gil_scoped_acquire gil;
        --g_exports;
        delete k;                       // releases the array (a stream object may be dropped with it)
    }
    if (del) del(self);
}
template <class T> T* wrap(T* t, std::shared_ptr<Storage> keep) {
    ++g_exports;
    t->manager_ctx = new Keep<T>{t->manager_ctx, t->deleter, std::move(keep)};
    t->deleter = &keep_deleter<T>;
    return t;
}
template <class T> void capsule_destructor(PyObject* cap) {
    const char* name = PyCapsule_GetName(cap);
    if (name && name[0] == 'd') {       // never consumed (a consumer renames it used_*)
        auto* t = static_cast<T*>(PyCapsule_GetPointer(cap, name));
        if (t && t->deleter) t->deleter(t);
    }
}

nb::object dlpack(const Array& a, nb::object stream, nb::object max_version, nb::object dl_device, nb::object copy) {
    if (!copy.is_none() && PyObject_IsTrue(copy.ptr())) throw nb::buffer_error("aether.Array does not copy on export (copy=True refused)");
    if (!dl_device.is_none()) {
        int dt_ = nb::cast<int>(dl_device[0]), di = nb::cast<int>(dl_device[1]);
        if (dt_ != int(a.view.device.type()) || di != a.view.device.id()) throw nb::buffer_error("cannot export on another device");
    }
    if (!stream.is_none() && a.st->kind == CUDA) {
        auto s = nb::cast<std::intptr_t>(stream);
        if (s == 0) throw nb::value_error("__dlpack__(stream=0) is not a valid CUDA stream code");
        if (s != -1) {
            void* consumer = reinterpret_cast<void*>(s);
            void* producer = a.st->handle;
            bool same = producer == consumer || (producer == nullptr && s == 1);
            if (!same) check(cuda()->stream_wait(a.st->dev, producer, consumer), "dlpack stream ordering");
        }
    }
    bool versioned = !max_version.is_none() && nb::cast<int>(max_version[0]) >= 1;
    if (versioned)
        return nb::steal(PyCapsule_New(wrap(interop::toDLPack(a.view), a.st), "dltensor_versioned",
                                       &capsule_destructor<DLManagedTensorVersioned>));
    return nb::steal(PyCapsule_New(wrap(interop::toDLPackLegacy(a.view), a.st), "dltensor",
                                   &capsule_destructor<DLManagedTensor>));
}

Array from_import(interop::DLPackImport&& imp) {
    Array a;
    a.view = imp.view;
    a.dt = dtype_index(imp.view.dtype.raw);
    auto st = std::make_shared<Storage>();
    DLDeviceType type = imp.view.device.type();
    st->kind = type == kDLCPU ? CPU : type == kDLCUDAHost ? PINNED : type == kDLCUDA ? CUDA : -1;
    if (st->kind < 0) fail("DLPack device type " + std::to_string(int(type)) + " is not supported (cpu, cuda, pinned)");
    st->dev = imp.view.device.id();
    st->ptr = imp.view.data;
    st->owner = std::move(imp.owner);
    a.st = std::move(st);
    return a;
}

Array from_dlpack(nb::object x, nb::object stream) {
    nb::tuple dev = nb::cast<nb::tuple>(x.attr("__dlpack_device__")());
    int type = nb::cast<int>(dev[0]);
    nb::object cap;
    nb::object ask = nb::none();
    if (!stream.is_none()) {
        std::intptr_t h = reinterpret_cast<std::intptr_t>(stream_handle(stream));
        ask = nb::int_(h ? h : 1);       // the DLPack code for the legacy stream
    }
    auto call = [&](bool versioned) {
        nb::dict kw;
        if (versioned) kw["max_version"] = nb::make_tuple(1, 0);
        if (!ask.is_none() && type == kDLCUDA) kw["stream"] = ask;
        return x.attr("__dlpack__")(**kw);
    };
    try { cap = call(true); }
    catch (nb::python_error& e) {
        if (!e.matches(PyExc_TypeError)) throw;
        cap = call(false);
    }
    const bool versioned = PyCapsule_IsValid(cap.ptr(), "dltensor_versioned");
    if (!versioned && !PyCapsule_IsValid(cap.ptr(), "dltensor")) fail("not a DLPack capsule (or one already consumed)");
    void* p = PyCapsule_GetPointer(cap.ptr(), versioned ? "dltensor_versioned" : "dltensor");
    PyCapsule_SetName(cap.ptr(), versioned ? "used_dltensor_versioned" : "used_dltensor");
    try {
        if (versioned) return from_import(interop::fromDLPack(static_cast<DLManagedTensorVersioned*>(p)));
        return from_import(interop::fromDLPack(static_cast<DLManagedTensor*>(p)));
    } catch (...) {
        // the tensor was ours to release the moment we took the capsule
        if (versioned) { auto* t = static_cast<DLManagedTensorVersioned*>(p); if (t->deleter) t->deleter(t); }
        else { auto* t = static_cast<DLManagedTensor*>(p); if (t->deleter) t->deleter(t); }
        throw;
    }
}

Array from_buffer(nb::object obj) {
    auto* pb = new Py_buffer();
    if (PyObject_GetBuffer(obj.ptr(), pb, PyBUF_FORMAT | PyBUF_STRIDES) != 0) { delete pb; throw nb::python_error(); }
    Array a;
    auto st = std::make_shared<Storage>();
    st->pybuf = pb;
    try {
        a.dt = dtype_from_format(pb->format ? pb->format : "B");
        std::vector<std::int64_t> shape(pb->shape, pb->shape + pb->ndim);
        set_shape(a, shape);
        for (int i = 0; i < pb->ndim; ++i) {
            if (pb->strides[i] % pb->itemsize) fail("buffer strides are not a whole number of elements");
            a.view.strides[i] = pb->strides[i] / pb->itemsize;
        }
    } catch (...) { throw; }       // ~Storage releases the buffer
    st->kind = CPU; st->ptr = pb->buf; st->bytes = pb->len; st->keep = obj;
    a.view.data = pb->buf; a.view.dtype = aether_dtype(a.dt); a.view.device = Device(kDLCPU);
    a.st = std::move(st);
    return a;
}

Array from_cai(nb::object obj) {
    nb::dict cai = nb::cast<nb::dict>(obj.attr("__cuda_array_interface__"));
    nb::tuple data = nb::cast<nb::tuple>(cai["data"]);
    auto ptr = reinterpret_cast<void*>(nb::cast<std::uintptr_t>(data[0]));
    std::string typestr = nb::cast<std::string>(cai["typestr"]);
    int dt = -1;
    for (int i = 0; i < kNumDtypes; ++i) if (kDtypes[i].cai && typestr == kDtypes[i].cai) dt = i;
    if (dt < 0) fail("__cuda_array_interface__ typestr '" + typestr + "' is not supported");
    Array a;
    a.dt = dt;
    set_shape(a, nb::cast<std::vector<std::int64_t>>(cai["shape"]));
    nb::object strides = cai.contains("strides") ? cai["strides"] : nb::none();
    if (!strides.is_none()) {
        auto s = nb::cast<std::vector<std::int64_t>>(strides);
        for (std::size_t i = 0; i < s.size(); ++i) a.view.strides[i] = s[i] / std::int64_t(itemsize(dt));
    }
    int dev = 0;
    if (ptr) check(cuda()->pointer_device(ptr, &dev), "__cuda_array_interface__ pointer");
    auto st = std::make_shared<Storage>();
    st->kind = CUDA; st->dev = dev; st->ptr = ptr; st->keep = obj;
    a.view.data = ptr; a.view.dtype = aether_dtype(dt); a.view.device = Device(kDLCUDA, dev);
    a.st = std::move(st);
    return a;
}

void copy(const Array& src, Array& dst, nb::object stream) {
    if (src.dt != dst.dt || src.size() != dst.size()) fail("copy: shapes/dtypes differ");
    if (!src.contiguous() || !dst.contiguous()) fail("copy needs contiguous arrays");
    std::size_t n = src.nbytes();
    if (!n) return;
    if (src.host() && dst.host()) {
        if (!stream.is_none() && (src.st->kind == PINNED || dst.st->kind == PINNED))
            check(cuda()->stream_sync(0, stream_handle(stream)), "copy");
        std::memmove(dst.st->ptr, src.st->ptr, n);
        return;
    }
    const AetherMemoryV1* t = cuda();
    int dev = dst.host() ? src.st->dev : dst.st->dev;
    void* h = stream_handle(stream);
    check(t->copy(dst.st->kind, dst.st->dev, dst.st->ptr, src.st->kind, src.st->dev, src.st->ptr, n, h), "copy");
    if (stream.is_none()) check(t->stream_sync(dev, nullptr), "copy");
}

// ---------------------------------------------------------------- buffer protocol (host-accessible memory)
int arr_getbuffer(PyObject* self, Py_buffer* buf, int) {
    Array* a = nb::inst_ptr<Array>(self);
    if (!a->host()) { PyErr_SetString(PyExc_BufferError, "device memory has no host buffer"); buf->obj = nullptr; return -1; }
    std::size_t r = a->view.rank;
    auto* mem = new Py_ssize_t[2 * (r ? r : 1)];
    std::size_t item = itemsize(a->dt);
    if (!kDtypes[a->dt].fmt) { PyErr_SetString(PyExc_BufferError, "this dtype has no buffer format"); buf->obj = nullptr; delete[] mem; return -1; }
    for (std::size_t i = 0; i < r; ++i) { mem[i] = a->view.extents[i]; mem[r + i] = a->view.strides[i] * Py_ssize_t(item); }
    buf->buf = a->st->ptr; buf->obj = self; Py_INCREF(self);
    buf->len = Py_ssize_t(a->nbytes()); buf->readonly = 0; buf->itemsize = Py_ssize_t(item);
    buf->format = const_cast<char*>(kDtypes[a->dt].fmt);
    buf->ndim = int(r); buf->shape = mem; buf->strides = mem + r; buf->suboffsets = nullptr; buf->internal = mem;
    return 0;
}
void arr_releasebuffer(PyObject*, Py_buffer* buf) { delete[] static_cast<Py_ssize_t*>(buf->internal); }
PyType_Slot arr_slots[] = {{Py_bf_getbuffer, (void*) arr_getbuffer}, {Py_bf_releasebuffer, (void*) arr_releasebuffer}, {0, nullptr}};

nb::object py_device(const Array& a) {
    return g_device_cls(a.st->kind == CUDA ? "cuda" : a.st->kind == PINNED ? "pinned" : "cpu", a.st->kind == CUDA ? a.st->dev : 0);
}

}  // namespace

NB_MODULE(_core, m) {
    g_error = nb::exception<aether::Error>(m, "Error", PyExc_RuntimeError);
    g_unavailable = nb::steal(PyErr_NewException("aether.CudaUnavailable", g_error.ptr(), nullptr));
    m.attr("CudaUnavailable") = g_unavailable;
    g_device_cls = nb::none();
    m.def("_register", [](nb::object device_cls, nb::object dtype_cls) { g_device_cls = device_cls; g_dtype_cls = dtype_cls; });
    m.def("_current_device", [] { int d = 0; check(cuda()->current_device(&d), "current device"); return d; });
    m.def("cuda_runtime_version", [] { int v = 0; check(cuda()->runtime_version(&v), "runtime version"); return v; });
    m.def("cuda_plugin", [] { cuda(); return g_plugin.name; });
    m.def("_new", [](int kind, int dev, std::vector<std::int64_t> shape, const std::string& dtype, nb::object stream, bool zero) {
        void* h = stream_handle(stream);
        return make(kind, dev, shape, dtype_index(dtype), h, stream, zero);
    }, nb::arg("kind"), nb::arg("dev"), nb::arg("shape"), nb::arg("dtype"), nb::arg("stream").none(), nb::arg("zero"));
    m.def("_from_dlpack", &from_dlpack, nb::arg("x"), nb::arg("stream").none() = nb::none());
    m.def("_from_buffer", &from_buffer);
    m.def("_from_cai", &from_cai);
    m.def("_copy", [](const Array& s, Array& d, nb::object stream) { copy(s, d, stream); }, nb::arg("src"), nb::arg("dst"), nb::arg("stream").none());
    m.def("_drain", &drain);
    m.def("_exports_live", [] { return g_exports.load(); });
    nb::module_::import_("atexit").attr("register")(nb::cpp_function(&drain));

    nb::class_<Array>(m, "Array", nb::type_slots(arr_slots))
        .def_prop_ro("shape", [](const Array& a) { nb::list l; for (std::size_t i = 0; i < a.view.rank; ++i) l.append(std::int64_t(a.view.extents[i])); return nb::tuple(l); })
        .def_prop_ro("strides", [](const Array& a) { nb::list l; for (std::size_t i = 0; i < a.view.rank; ++i) l.append(std::int64_t(a.view.strides[i]) * std::int64_t(itemsize(a.dt))); return nb::tuple(l); })
        .def_prop_ro("dtype", [](const Array& a) { return g_dtype_cls(kDtypes[a.dt].name); })
        .def_prop_ro("ndim", [](const Array& a) { return a.view.rank; })
        .def_prop_ro("size", [](const Array& a) { return a.size(); })
        .def_prop_ro("nbytes", [](const Array& a) { return a.nbytes(); })
        .def_prop_ro("ptr", [](const Array& a) { return reinterpret_cast<std::uintptr_t>(a.st->ptr); })
        .def_prop_ro("device", &py_device)
        .def_prop_ro("is_contiguous", [](const Array& a) { return a.contiguous(); })
        .def("zero", [](Array& a, nb::object stream) -> Array& { fill(a, nb::none(), stream, true); return a; }, nb::arg("stream").none() = nb::none(), nb::rv_policy::reference)
        .def("fill", [](Array& a, nb::object v, nb::object stream) -> Array& { fill(a, v, stream, false); return a; }, nb::arg("value"), nb::arg("stream").none() = nb::none(), nb::rv_policy::reference)
        .def("_set_flat", [](Array& a, std::size_t i, nb::object v) {
            if (!a.host() || i >= std::size_t(a.size())) fail("_set_flat: host arrays only, index in range");
            pack(a.dt, v, static_cast<unsigned char*>(a.st->ptr) + i * itemsize(a.dt));
        })
        .def("synchronize", [](const Array& a) { if (a.st->kind != CPU) check(cuda()->stream_sync(a.st->dev, a.st->handle), "synchronize"); })
        .def("__dlpack_device__", [](const Array& a) { return nb::make_tuple(int(a.view.device.type()), a.view.device.id()); })
        .def("__dlpack__", [](const Array& a, nb::object stream, nb::object mv, nb::object dd, nb::object cp) { return dlpack(a, stream, mv, dd, cp); },
             nb::kw_only(), nb::arg("stream").none() = nb::none(), nb::arg("max_version").none() = nb::none(),
             nb::arg("dl_device").none() = nb::none(), nb::arg("copy").none() = nb::none())
        .def_prop_ro("__cuda_array_interface__", [](const Array& a) {
            if (a.st->kind != CUDA) throw nb::attribute_error("only device arrays have __cuda_array_interface__");
            if (!kDtypes[a.dt].cai) throw nb::attribute_error("this dtype has no __cuda_array_interface__ typestr");
            nb::dict d;
            nb::list shape, strides;
            for (std::size_t i = 0; i < a.view.rank; ++i) { shape.append(std::int64_t(a.view.extents[i])); strides.append(std::int64_t(a.view.strides[i]) * std::int64_t(itemsize(a.dt))); }
            d["shape"] = nb::tuple(shape); d["typestr"] = kDtypes[a.dt].cai; d["data"] = nb::make_tuple(reinterpret_cast<std::uintptr_t>(a.st->ptr), false);
            d["version"] = 3; d["strides"] = a.contiguous() ? nb::none() : nb::object(nb::tuple(strides));
            auto h = reinterpret_cast<std::intptr_t>(a.st->handle);
            d["stream"] = h ? nb::object(nb::int_(h)) : nb::none();
            return d;
        })
        .def("__repr__", [](const Array& a) {
            std::string shape = "(";
            for (std::size_t i = 0; i < a.view.rank; ++i) shape += std::to_string(a.view.extents[i]) + (a.view.rank == 1 ? "," : i + 1 < a.view.rank ? ", " : "");
            shape += ")";
            return "aether.Array(shape=" + shape + ", dtype=" + kDtypes[a.dt].name + ", device='" + nb::cast<std::string>(nb::str(py_device(a))) + "')";
        });
}
