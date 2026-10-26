// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0
// The C seam `aether-memory/1` between aether._core (plain C++, no CUDA) and a CUDA plugin
// (libaether_cuda12.so / libaether_cuda13.so, each linked to its own static cudart). The plugin is a thin
// wrapper over aether's CUDA-mode C++ (`Chunk`, `fill`, `StreamTransport`): every lifetime rule lives there.
// Every entry returns 0 on success; on failure `last_error` holds the message (per thread). `dev` is the CUDA
// device the call runs on; `stream` is a raw stream handle (0 = legacy).
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AETHER_MEMORY_ABI 1
enum { AETHER_MEM_CPU = 0, AETHER_MEM_DEVICE = 1, AETHER_MEM_PINNED = 2 };

typedef struct AetherMemoryV1 {
    uint32_t abi;
    int (*probe)(int* device_count);                 // 0 only if a driver answers and >= 1 device exists
    int (*runtime_version)(int* version);
    int (*current_device)(int* dev);
    const char* (*last_error)(void);
    // an owning aether::Chunk, stream-ordered on `stream`; `chunk` is its handle, `ptr` its data
    int (*chunk_alloc)(int kind, int dev, size_t bytes, void* stream, void** chunk, void** ptr);
    void (*chunk_release)(void* chunk);              // frees it on its stream; deferred while a capture blocks it
    void (*drain)(void);                             // release the deferred frees whose capture has ended
    // fill `bytes` at `ptr` with the `width`-byte element `value`, on `stream` (capture-legal)
    int (*fill)(int kind, int dev, void* ptr, size_t bytes, const void* value, size_t width, void* stream);
    // ordered copy of `bytes` between any two memory kinds, on `stream` (pageable host staged)
    int (*copy)(int dst_kind, int dst_dev, void* dst, int src_kind, int src_dev, const void* src, size_t bytes,
                void* stream);
    int (*stream_wait)(int dev, void* producer, void* consumer);   // consumer waits for producer's queued work
    int (*stream_sync)(int dev, void* stream);
    int (*pointer_device)(const void* ptr, int* dev);   // device that owns a device pointer
} AetherMemoryV1;

typedef const AetherMemoryV1* (*aether_memory_v1_fn)(void);

#ifdef __cplusplus
}
#endif
