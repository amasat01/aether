// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// `TextureBinding<T>` lifetime rows for `aether::TextureBinding<T>`.
//
// ## The defect these rows were minted against
//
// A naive `TextureBinding::operator=(TextureBinding&&)` that overwrites
// `tex_` with a bare `std::exchange` and never destroys the destination's
// own live texture object leaks it. A leaked texture object is not the
// benign kind of leak: the handle is a descriptor that keeps pointing at
// the device address range it was bound to, so once the owning `Chunk` is
// freed and the allocator hands the block to the next request, the
// orphaned handle silently serves the new occupant's bytes with no error
// of any kind.
//
// ## The instrument, and why it is not the obvious one
//
// `cudaDestroyTextureObject` called a second time on an already-destroyed
// (or never-created) handle reports `cudaSuccess` on this driver — a test
// built on it would report "leaked" for every handle, fixed or not.
// `cudaGetTextureObjectResourceDesc` is the observable with a known
// answer: `cudaSuccess` on a live object, `cudaErrorInvalidValue`
// otherwise; non-destructive, and it returns the bound device pointer, so
// the adoption half of a move can be checked against the `Chunk`'s own
// storage rather than by inference. `LivenessProbeHasAKnownAnswer` pins
// all three answers inside the suite: if a future driver makes the query
// permissive the way `cudaDestroyTextureObject` already is, that row goes
// red before the rows that depend on it can go quietly vacuous.
//
// `TextureBinding<T>` does not own the `Chunk` it is bound over (mirrors
// `View` never owning the `Chunk` it is built over) — `Filled` below pairs
// a `Chunk` and a `TextureBinding<double>` bound over it, and moving a
// `Filled` moves both members (memberwise), exercising
// `TextureBinding::operator=(&&)`/the move ctor. aether keeps the memory
// and the texture object as two separate types rather than one owning
// container, so this test pairs them back together to exercise the move.

#include <cstddef>
#include <cstring>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace {

using aether::Chunk;
using aether::Device;
using aether::TextureBinding;

/** @brief Element count of every filled binding in this suite. */
constexpr std::size_t kN = 256;
/** @brief Repetitions in the leak-counter row — under the defect this is
 *  also the number of leaked handles, so that row reports a COUNT (0 of
 *  kReps), not a boolean: a fix that released only the first one cannot
 *  hide. */
constexpr int kReps = 32;

/** @brief Read `n` doubles through a RAW texture handle, bypassing
 *  `TableHandle` entirely — the point is to observe the HANDLE, not any
 *  higher-level view of it. */
__global__ void readRawTexKernel(aether::texture_handle_t tex, double* out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const int2 r = tex1Dfetch<int2>(tex, i);
        out[i] = __hiloint2double(r.y, r.x);
    }
}

/** @brief Fixture: owns the kernel's output buffer and keeps the CUDA error
 *  state clean between rows (several rows deliberately provoke a failed
 *  query). */
class TextureMoveAssignTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        ASSERT_EQ(cudaMalloc(&out_, kN * sizeof(double)), cudaSuccess);
        cudaGetLastError();
    }

    void TearDown() override
    {
        cudaFree(out_);
        out_ = nullptr;
        cudaGetLastError();
    }

    /** @brief A device `Chunk` of `kN` doubles PLUS a `TextureBinding`
     *  bound over it, moved together (memberwise) — the pairing this
     *  suite's move rows exercise. Implicitly move-only/move-assignable
     *  (both members are), copy implicitly deleted. */
    struct Filled {
        Chunk chunk;
        TextureBinding<double> binding;
    };

    /** @brief A `Filled` holding `base, base+1, base+2, ...`. */
    static Filled makeFilled(double base)
    {
        std::vector<double> host(kN);
        for (std::size_t i = 0; i < kN; ++i)
            host[i] = base + static_cast<double>(i);

        Chunk hostStage = Chunk::allocate(Device(kDLCUDAHost), kN * sizeof(double));
        std::memcpy(hostStage.data(), host.data(), kN * sizeof(double));

        Chunk device = Chunk::allocate(Device(kDLCUDA), kN * sizeof(double));
        aether::copy(device, hostStage);
        [[maybe_unused]] cudaError_t syncSt = cudaDeviceSynchronize();

        TextureBinding<double> binding = TextureBinding<double>::bind(device, kN);
        return Filled{ std::move(device), std::move(binding) };
    }

    /**
     * @brief `true` when `tex` still names a live texture object.
     *
     * Non-destructive. MUST be asked before any further texture is
     * created: handle values are recycled, so a deferred question can be
     * answered by a DIFFERENT object wearing the same number.
     */
    static bool handleIsLive(aether::texture_handle_t tex)
    {
        cudaResourceDesc desc{};
        const cudaError_t e
            = cudaGetTextureObjectResourceDesc(&desc, static_cast<cudaTextureObject_t>(tex));
        cudaGetLastError(); /* a failed query leaves a sticky error */
        return e == cudaSuccess;
    }

    /** @brief Device address the texture object is bound to, or `nullptr`
     *  if the handle is not live. */
    static const void* boundPointer(aether::texture_handle_t tex)
    {
        cudaResourceDesc desc{};
        const cudaError_t e
            = cudaGetTextureObjectResourceDesc(&desc, static_cast<cudaTextureObject_t>(tex));
        cudaGetLastError();
        return (e == cudaSuccess) ? desc.res.linear.devPtr : nullptr;
    }

    /** @brief Read every element through `tex`; `true` when the read
     *  succeeded and delivered `base, base+1, ...`. Only ever called on a
     *  handle already asserted LIVE — a fetch through a destroyed handle
     *  raises an illegal memory access and poisons the context for the
     *  rest of the binary. */
    bool deliversRun(aether::texture_handle_t tex, double base)
    {
        std::vector<double> host(kN, 0.0);
        if (cudaMemset(out_, 0, kN * sizeof(double)) != cudaSuccess)
            return false;
        readRawTexKernel<<<static_cast<unsigned>((kN + 63) / 64), 64>>>(tex, out_, static_cast<int>(kN));
        cudaError_t e = cudaGetLastError();
        if (e == cudaSuccess)
            e = cudaDeviceSynchronize();
        if (e != cudaSuccess)
            return false;
        if (cudaMemcpy(host.data(), out_, kN * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess)
            return false;
        for (std::size_t i = 0; i < kN; ++i)
            if (host[i] != base + static_cast<double>(i))
                return false;
        return true;
    }

    double* out_ = nullptr;
};

/**
 * @brief CONTROL — the liveness instrument answers all three known cases
 *        correctly. Every other row here reads a verdict out of
 *        `handleIsLive`; without this row a driver that made
 *        `cudaGetTextureObjectResourceDesc` permissive would make every
 *        leak row pass identically before and after a regression.
 */
TEST_F(TextureMoveAssignTest, LivenessProbeHasAKnownAnswer)
{
    /* keeper stays alive for the whole row so nothing can recycle the
     * value released below before the query reaches it */
    Filled keeper = makeFilled(1.0);
    const aether::texture_handle_t hAlive = keeper.binding.handle();
    ASSERT_NE(hAlive, aether::texture_handle_t{ 0 });

    aether::texture_handle_t hReleased = 0;
    {
        Filled doomed = makeFilled(2.0);
        hReleased = doomed.binding.handle();
        ASSERT_NE(hReleased, hAlive);
    } /* doomed's destructor destroys its texture */
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    EXPECT_TRUE(handleIsLive(hAlive)) << "the probe cannot see a LIVE object";
    EXPECT_FALSE(handleIsLive(hReleased))
        << "the probe reports a cleanly destroyed object as live - it is blind "
           "and every leak row in this suite is vacuous";
    EXPECT_FALSE(handleIsLive(static_cast<aether::texture_handle_t>(987654321ULL)))
        << "the probe reports a never-created handle as live - it is blind";
}

/**
 * @brief CONTROL — the textured path works at all. Without this row the
 *        leak rows could pass by never having bound a texture.
 */
TEST_F(TextureMoveAssignTest, TexturedReadIsLive)
{
    Filled f = makeFilled(1.0);
    const aether::texture_handle_t h = f.binding.handle();
    ASSERT_NE(h, aether::texture_handle_t{ 0 }) << "no texture object was bound - the instrument is blind";
    EXPECT_TRUE(handleIsLive(h));
    EXPECT_TRUE(deliversRun(h, 1.0)) << "a freshly bound texture did not read back";
}

/**
 * @brief THE row: move-assignment releases the destination's texture, and
 *        the consumer ends up on the adopted one. Both halves matter: the
 *        release half is the defect; the adoption half is what must NOT
 *        regress while fixing it — a repair that destroyed the wrong
 *        handle would trade a leak for a broken reader.
 */
TEST_F(TextureMoveAssignTest, MoveAssignReleasesDestinationTexture)
{
    Filled dst = makeFilled(1.0);
    const aether::texture_handle_t hDst = dst.binding.handle();
    aether::texture_handle_t hSrc = 0;
    const void* srcDevPtr = nullptr;
    {
        Filled src = makeFilled(1000.0);
        hSrc = src.binding.handle();
        srcDevPtr = src.chunk.data();
        ASSERT_NE(hSrc, aether::texture_handle_t{ 0 });
        ASSERT_NE(hDst, hSrc) << "both bindings got the same handle - they are not distinct "
                                 "texture objects";

        dst = std::move(src); /* TextureBinding::operator=(TextureBinding&&) (+ Chunk's own) */
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    /* --- release half: the displaced handle must be gone. Asked FIRST and
     * immediately, while nothing has had the chance to recycle its value. --- */
    const bool displacedSurvived = handleIsLive(hDst);

    /* --- adoption half --- */
    EXPECT_EQ(dst.binding.handle(), hSrc)
        << "after move-assignment the binding does not expose the adopted handle";
    EXPECT_TRUE(handleIsLive(hSrc)) << "the adopted handle was destroyed";
    EXPECT_EQ(boundPointer(hSrc), srcDevPtr)
        << "the adopted texture is not bound to the moved-in chunk's own storage";
    EXPECT_TRUE(deliversRun(dst.binding.handle(), 1000.0))
        << "the adopted handle does not deliver the source's values";

    EXPECT_FALSE(displacedSurvived)
        << "the destination's texture object " << hDst
        << " is STILL LIVE after move-assignment: it was leaked, and a leaked texture object "
           "silently serves whatever memory later reuses the block it is bound to";
}

/** @brief The same defect as a COUNT rather than a boolean: one leak per
 *  assignment. */
TEST_F(TextureMoveAssignTest, MoveAssignLeaksNoTextureOverRepeatedAssignment)
{
    Filled dst = makeFilled(1.0);
    int leaked = 0;
    int inspected = 0;

    for (int k = 0; k < kReps; ++k) {
        const aether::texture_handle_t displaced = dst.binding.handle();
        {
            Filled src = makeFilled(2000.0 + 1000.0 * k);
            dst = std::move(src);
        }
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ASSERT_NE(dst.binding.handle(), displaced) << "iteration " << k << " did not adopt a new handle";
        ++inspected;
        if (handleIsLive(displaced))
            ++leaked;
    }

    ASSERT_EQ(inspected, kReps) << "the loop did not run - vacuous row";
    EXPECT_EQ(leaked, 0) << leaked << " of " << kReps << " displaced texture objects survived their "
                                                          "move-assignment";
}

/**
 * @brief `TextureBinding` is decoupled from any owning container, so the
 *        public spelling a caller reaches for to release a binding's
 *        device resource is assigning it a fresh, default-constructed
 *        (unbound) `TextureBinding`, which routes through the same
 *        move-assignment operator `MoveAssignReleasesDestinationTexture`
 *        exercises, in the "release without replacing with a live one"
 *        shape.
 */
TEST_F(TextureMoveAssignTest, ClearWorkableReleasesTexture)
{
    Filled a = makeFilled(1.0);
    const aether::texture_handle_t hA = a.binding.handle();
    ASSERT_NE(hA, aether::texture_handle_t{ 0 });

    a.binding = TextureBinding<double>();
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    EXPECT_FALSE(handleIsLive(hA)) << "clearing the binding (assigning a fresh, unbound one) left "
                                       "texture object "
                                    << hA << " alive after freeing the memory it is bound to";
}

/**
 * @brief Move CONSTRUCTION transfers sole ownership. It cannot leak — there
 *        is no destination resource yet — but it must not lose the handle
 *        either. This row exists because the move constructor sits beside
 *        the defective operator and shares its `std::exchange` shape: it
 *        pins that the two are NOT the same class of hole, so a later
 *        reader neither "fixes" the constructor nor breaks it while fixing
 *        the operator.
 */
TEST_F(TextureMoveAssignTest, MoveConstructTransfersSoleOwnership)
{
    aether::texture_handle_t hSrc = 0;
    {
        Filled src = makeFilled(1.0);
        hSrc = src.binding.handle();
        ASSERT_NE(hSrc, aether::texture_handle_t{ 0 });

        Filled moved = std::move(src);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_EQ(moved.binding.handle(), hSrc) << "move construction did not carry the handle across";
        EXPECT_TRUE(handleIsLive(hSrc));
        EXPECT_TRUE(deliversRun(moved.binding.handle(), 1.0))
            << "the moved-to binding does not read through the transferred handle";
    } /* the moved-to object dies here and must release the handle */
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    EXPECT_FALSE(handleIsLive(hSrc)) << "texture object " << hSrc << " outlived the move-constructed owner";
}

// ---------------------------------------------------------------------------
// TextureWriteInvariant.Anchor — mode-agnostic content, IDENTICAL to
// test_TextureBinding.cpp's own copy (house convention: paired files carry
// identical shared content). See that file's own docstring for the full
// rationale.
// ---------------------------------------------------------------------------

using aether::Plain;
using aether::TableHandle;
using aether::Texture;

template<class H, class T, class = void>
struct IsElementWritable : std::false_type { };
template<class H, class T>
struct IsElementWritable<H, T,
    std::void_t<decltype(std::declval<H&>()[aether::offset_t{ 0 }] = std::declval<T>())>>
    : std::true_type { };

TEST(TextureWriteInvariant, Anchor)
{
    // `TableHandle<T, Texture>::operator[]` declares an explicit `const T`
    // return type (never `decltype(auto)` — this declaration is stable
    // across every pass and every build mode). `decltype` of the call
    // expression nonetheless reports plain `double`, not `const double`:
    // "a prvalue of a non-class type has its cv-qualification stripped by
    // decltype ([expr.type])". Measured here, not merely cited: an earlier
    // version of this row asserted `const double` and was red on this
    // exact line (nvcc, this tree) until fixed to match [expr.type] — the
    // const on the declaration still matters
    // (it is what makes `operator[]`'s body itself non-assignable, C++'s
    // own rule for why a named `const`-returning function can't be used as
    // an assignment target), but `decltype` cannot observe it for a
    // scalar `T`, and asserting otherwise would silently pass on a
    // build where the const had been dropped from the declaration too.
    static_assert(
        std::is_same_v<decltype(std::declval<const TableHandle<double, Texture>&>()[aether::offset_t{ 0 }]),
            double>,
        "#119 I-ACCESS - decltype(h[i]) on the texture carrier must be plain `double` (the "
        "declared `const double` return type's top-level const is stripped from a scalar "
        "prvalue's expression type by [expr.type] — this row pins that fact is still true, not "
        "that the declaration lost its const)");

    static_assert(!IsElementWritable<TableHandle<double, Texture>, double>::value,
        "TextureWriteInvariant ANCHOR - `h[i] = v` compiles on the texture carrier; texture "
        "memory is read-only and no write spelling may exist");

    static_assert(!IsElementWritable<TableHandle<double, Plain>, double>::value,
        "CONTROL - the plain carrier's operator[] is unexpectedly writable; TableHandle was "
        "always read-only in BOTH carriers, so the rows above would be trivially satisfied by "
        "any handle at all");

    static_assert(IsElementWritable<std::vector<double>, double>::value,
        "CONTROL - IsElementWritable does not detect a genuinely writable operator[]; the rows "
        "above are therefore vacuous");

    SUCCEED();
}

} // namespace
} // namespace aether_tests
