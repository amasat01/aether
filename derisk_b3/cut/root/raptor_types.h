/* Neutral types header: index width, sample index, view and plane layout.
   Pure data layout. No hardware fact, no vendor type, no allocation. C99 and C++14. */
#ifndef RAPTOR_TYPES_H
#define RAPTOR_TYPES_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Index width: the build-wide switch. A variant states its width as data and the
   generated launch header asserts it equals sizeof(raptor_index_t). */
#ifndef RAPTOR_INDEX_T
#define RAPTOR_INDEX_T uint32_t
#endif
typedef RAPTOR_INDEX_T raptor_index_t;

/* Sample index: signed 64-bit, never narrower than the index. */
typedef int64_t raptor_sample_t;

/* Compile-time check usable from C99 and C++14. The tag must be an identifier. */
#ifdef __cplusplus
#define RAPTOR_STATIC_ASSERT(cond, tag) static_assert(cond, #tag)
#else
#define RAPTOR_STATIC_ASSERT(cond, tag) typedef char raptor_sa_##tag[(cond) ? 1 : -1]
#endif

/* Rank-2 view: a (components, samples) plane set, structure of arrays.
   Element (c, s) is data[c * comp_stride + s * sample_stride]. */
typedef struct raptor_view2 {
    void*    data;
    uint64_t samples;
    uint64_t comp_stride;
    uint64_t sample_stride;
    int32_t  device_type; /* the variant gives the codes */
    int32_t  device_id;
} raptor_view2_t;

/* Rank-1 view: one value per sample. Element s is data[s * stride]. */
typedef struct raptor_view1 {
    void*    data;
    uint64_t samples;
    uint64_t stride;
    int32_t  device_type;
    int32_t  device_id;
} raptor_view1_t;

/* Int-valued rank-1 view: same layout, a name of its own at the call site. */
typedef raptor_view1_t raptor_intview_t;

/* The launch triple: samples [base, base + count) of n_samples. */
typedef struct raptor_partition {
    raptor_sample_t base;
    raptor_sample_t count;
    raptor_sample_t n_samples;
} raptor_partition_t;

/* Contiguous (width, n) buffer: component pitch n, sample pitch 1. */
static inline raptor_view2_t raptor_make_view2(void* data, uint64_t n, int32_t dev) {
    raptor_view2_t v;
    v.data = data; v.samples = n; v.comp_stride = n; v.sample_stride = 1;
    v.device_type = dev; v.device_id = 0;
    return v;
}

static inline raptor_view1_t raptor_make_view1(void* data, uint64_t n, int32_t dev) {
    raptor_view1_t v;
    v.data = data; v.samples = n; v.stride = 1;
    v.device_type = dev; v.device_id = 0;
    return v;
}

#ifdef __cplusplus
}
#endif
#endif /* RAPTOR_TYPES_H */
