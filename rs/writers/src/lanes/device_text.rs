//! Fixed C++ text of the device lane split: the fast persistent entries.
/// The prelude of the fast persistent entries.
pub const FAST_PRELUDE: &str = r####"static __device__ __forceinline__ void hawk_steps_max(unsigned int* cell,
                                                      unsigned int v)
{
    const unsigned hawk_m = __activemask();
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    v = __reduce_max_sync(hawk_m, v);
#else
    if (hawk_m != 0xffffffffu) {
        if (v != 0u) atomicMax(cell, v);
        return;
    }
    for (int o = 16; o > 0; o >>= 1) v = max(v, __shfl_xor_sync(hawk_m, v, o));
#endif
    unsigned hawk_lid;
    asm("mov.u32 %0, %%laneid;" : "=r"(hawk_lid));
    if (v != 0u && hawk_lid == static_cast<unsigned>(__ffs(hawk_m) - 1))
        atomicMax(cell, v);
}

static __device__ __forceinline__ void hawk_steps_sum(unsigned long long* cell,
                                                      unsigned long long v)
{
    const unsigned hawk_m = __activemask();
    if (hawk_m != 0xffffffffu) {
        if (v != 0ull) atomicAdd(cell, v);
        return;
    }
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(hawk_m, v, o);
    unsigned hawk_lid;
    asm("mov.u32 %0, %%laneid;" : "=r"(hawk_lid));
    if (v != 0ull && hawk_lid == 0u) atomicAdd(cell, v);
}

template <class ViewT>
static __device__ __forceinline__ void hawk_finish_sum(const ViewT& counter, unsigned v)
{
    unsigned int* cell = const_cast<unsigned int*>(
        reinterpret_cast<const unsigned int*>(counter.data()));
    const unsigned hawk_m = __activemask();
    if (hawk_m != 0xffffffffu) {
        if (v != 0u) atomicAdd(cell, v);
        return;
    }
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(hawk_m, v, o);
    unsigned hawk_lid;
    asm("mov.u32 %0, %%laneid;" : "=r"(hawk_lid));
    if (v != 0u && hawk_lid == 0u) atomicAdd(cell, v);
}

static __device__ __forceinline__ void hawk_persist_util(unsigned long long* util,
                                                         unsigned ran)
{
    if (util != nullptr) atomicAdd(&util[0], static_cast<unsigned long long>(ran));
}

// Out of line: the flush runs on a lane's wrap only, and inlined its null
// tests would have the claim loop compiled twice, once per pointer state;
// a call keeps the rare path's instructions out of the chunk head.
static __device__ __noinline__ void hawk_persist_flush(unsigned long long* util,
                                                       unsigned long long* stepsum, unsigned ran)
{
    if (util != nullptr) atomicAdd(&util[0], static_cast<unsigned long long>(ran));
    if (stepsum != nullptr) atomicAdd(stepsum, static_cast<unsigned long long>(ran));
}

// Out of line for the same reason: the chunk head keeps one predicated call.
static __device__ __noinline__ void hawk_persist_slots(unsigned long long* util, unsigned chunk)
{
    unsigned hawk_lid;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(hawk_lid));
    if (hawk_lid == 0u) atomicAdd(&util[1], static_cast<unsigned long long>(chunk) * 32ull);
}"####;
/// The parameters of a persistent entry.
pub const PERSIST_PARAMS: &str = r####"    unsigned int* hawk_next,
    unsigned long long* hawk_util,
    unsigned int* hawk_steps,
    unsigned long long* hawk_stepsum,
    unsigned int hawk_chunk"####;
/// The parameters of a range entry.
pub const RANGE_PARAMS: &str = r####"    unsigned int* hawk_steps,
    unsigned long long* hawk_stepsum"####;
/// The unroll line of the persistent loop.
pub const PERSIST_UNROLL_LINE: &str = r####"        #pragma unroll 4"####;
/// The word that marks a lane as ended.
pub const ENDED: &str = r####"0x80000000u"####;
