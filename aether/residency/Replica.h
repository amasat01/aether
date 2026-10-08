// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Replica.h
 * @brief `aether::ReplicaSet`: N `Chunk`s of the same byte size,
 *        one per `Device` — the REPLICATED counterpart to `Partition.h`'s
 *        PARTITIONED case. Minimal, not a framework. Plus
 *        `aether::Replica<T, Transport>`: the SAME shape, but TEMPLATED on
 *        a `transport`-satisfying type — the seam this file's own
 *        docstring named as not-yet-done (see below).
 *
 * v1 construction is an explicit `Device` LIST (no auto-discovery, no
 * topology). the `broadcast()` member copies ONE source chunk's bytes into every
 * replica via `aether::copy`/`copyAsync` (`aether/chunk/Copy.h`) — same
 * legal device-pair matrix as everywhere else in this library (CPU<->CUDA
 * direct is illegal; stage through CUDAHost). Multi-node/NCCL/MPI broadcast
 * implementations are CONSUMERS of the `transport` concept
 * (`aether/residency/Transport.h`); `ReplicaSet` (below) calls
 * `aether::copy`/`copyAsync` directly (equivalent to using the default
 * `StreamTransport`) rather than being templated on a transport —
 * genuinely minimal, and unaffected by the transport abstraction below.
 *
 * `Replica<T, Transport = StreamTransport>` closes that "not templated" gap
 * with a SEPARATE type, rather than rewriting `ReplicaSet` in place —
 * `ReplicaSet` stays exactly as it was (a pre-change bit-identity reference
 * for `Replica`'s default-`StreamTransport` behaviour,
 * `tests/test_Replica*.{cpp,cu}`), while `Replica` routes every move
 * through `Transport::copy`/`copyAsync` instead of calling the free
 * functions itself — for the default `StreamTransport`,
 * `Route::SAME/P2P/HOST` still delegate verbatim to
 * `aether::copy`/`copyAsync`, so the `broadcast` of
 * `Replica<T, StreamTransport>` is BIT-IDENTICAL to that of `ReplicaSet`
 * over the same devices/bytes — the property
 * `tests/test_Replica*.{cpp,cu}`'s bit-identity rows hold both types to.
 */

#include <cstddef>
#include <vector>

#include "aether/chunk/Chunk.h"
#include "aether/chunk/Copy.h"
#include "aether/device/Device.h"
#include "aether/residency/Transport.h"

namespace aether {

/**
 * @brief N `Chunk`s of `bytes` bytes, one per `Device` in `devices`.
 *        Move-only (implicit — `std::vector<Chunk>` cannot be copied since
 *        `Chunk` itself is move-only, so `ReplicaSet`'s copy members are
 *        implicitly deleted and its move members implicitly usable).
 */
class ReplicaSet {
public:
    ReplicaSet() = default;

    /** @brief Allocate one `Chunk` of `bytes` bytes (aligned to `alignment`)
     *         on each device in `devices`, in order (v1: an explicit
     *         `Device` list). */
    ReplicaSet(const std::vector<Device>& devices, std::size_t bytes, std::size_t alignment = 256)
    {
        chunks_.reserve(devices.size());
        for (const Device& d : devices)
            chunks_.push_back(Chunk::allocate(d, bytes, alignment));
    }

    /** @brief Number of replicas. */
    std::size_t size() const { return chunks_.size(); }

    /** @brief The `i`-th replica's chunk. */
    Chunk& chunk(std::size_t i) { return chunks_[i]; }
    /** @brief The `i`-th replica's chunk (read-only). */
    const Chunk& chunk(std::size_t i) const { return chunks_[i]; }

    /**
     * @brief Blocking broadcast: `src`'s bytes into EVERY replica
     *        in this set, via `aether::copy` — legal device-pair paths only
     *        (`aether/chunk/Copy.h`'s matrix; CPU<->CUDA direct throws).
     */
    void broadcast(const Chunk& src)
    {
        for (Chunk& dst : chunks_)
            copy(dst, src);
    }

#ifdef AETHER_HAS_CUDA
    /** @brief Asynchronous broadcast on `stream` (CUDA-backend-only —
     *         mirrors `aether::copyAsync`'s own narrower legal-pair set:
     *         no plain (pageable) CPU endpoint). */
    void broadcast(const Chunk& src, Stream stream)
    {
        for (Chunk& dst : chunks_)
            copyAsync(dst, src, stream);
    }
#endif

private:
    std::vector<Chunk> chunks_;
};

/**
 * @brief N `Chunk`s of `count * sizeof(T)` bytes, one per `Device` in
 *        `devices` — `ReplicaSet`'s SAME shape, but TEMPLATED on a
 *        `transport`-satisfying `Transport` (default `StreamTransport`, so
 *        `Replica<T>` behaves exactly like `ReplicaSet` unless a caller
 *        supplies a different transport). `broadcast()` routes every move
 *        through
 *        `Transport::copy`/`copyAsync` rather than calling
 *        `aether::copy`/`copyAsync` itself — the seam `ReplicaSet`'s own
 *        docstring named as not-yet-done.
 *
 * Move-only, same reasoning as `ReplicaSet` (`std::vector<Chunk>` cannot
 * be copied) — and, once `Transport` is `StreamTransport`, doubly so: that
 * type is itself move-only (`aether/residency/Transport.h`).
 */
template<class T, transport Transport = StreamTransport>
class Replica {
public:
    Replica() = default;

    /** @brief Allocate one `Chunk` of `count * sizeof(T)` bytes (aligned to
     *         `alignment`) on each device in `devices`, in order — same
     *         construction contract as `ReplicaSet` (an explicit
     *         `Device` list). */
    Replica(std::vector<Device> devices, std::size_t count, std::size_t alignment = 256)
        : devices_(std::move(devices))
    {
        chunks_.reserve(devices_.size());
        for (const Device& d : devices_)
            chunks_.push_back(Chunk::allocate(d, count * sizeof(T), alignment));
    }

    /** @brief Number of replicas. */
    std::size_t size() const { return chunks_.size(); }

    /** @brief The `i`-th replica's chunk. */
    Chunk& chunk(std::size_t i) { return chunks_[i]; }
    /** @brief The `i`-th replica's chunk (read-only). */
    const Chunk& chunk(std::size_t i) const { return chunks_[i]; }

    /** @brief This replica set's transport instance (its `route()`/
     *         `forceStaging()`/event state, for `StreamTransport`). */
    Transport& transport() { return transport_; }
    /** @brief Read-only accessor for `transport()`. */
    const Transport& transport() const { return transport_; }

    /**
     * @brief Blocking broadcast: `src`'s bytes into EVERY replica in this
     *        set, via `transport_.copy` — legal device-pair paths only, same as
     *        `ReplicaSet::broadcast` (`aether/chunk/Copy.h`'s matrix;
     *        CPU<->CUDA direct throws), routed through whatever `Route`
     *        `transport_` decides for each pair.
     */
    void broadcast(const Chunk& src)
    {
        for (Chunk& dst : chunks_)
            transport_.copy(dst, src);
    }

#ifdef AETHER_HAS_CUDA
    /** @brief Asynchronous broadcast on `stream` (CUDA-backend-only) —
     *         mirrors `ReplicaSet::broadcast(src, stream)`, routed through
     *         `transport_.copyAsync` (which records a `cudaEvent` after
     *         each move when `Transport` is `StreamTransport`). */
    void broadcast(const Chunk& src, Stream stream)
    {
        for (Chunk& dst : chunks_)
            transport_.copyAsync(dst, src, stream);
    }
#endif

private:
    std::vector<Device> devices_;
    std::vector<Chunk> chunks_;
    Transport transport_{};
};

} // namespace aether
