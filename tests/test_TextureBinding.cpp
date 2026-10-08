// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Host / CPP_MODE pairing for test_TextureBinding.cu.
// `TextureMoveAssignTest`'s 6 lifetime rows are CUDA-only content (they
// need real `cudaTextureObject_t` liveness queries) and live only in the
// `.cu` file — this file carries the one row that is mode-agnostic:
// `TextureWriteInvariant.Anchor`, a type-surface re-observation of the
// same contract that
// `tests/compile_fail/check_texture_write_rejected.sh` independently pins
// as a must-not-compile expression. See that script's own header comment
// for why the two are complements, not duplicates: a decltype/SFINAE check
// like this one never instantiates a function body, so it observes the
// type surface (`operator[]` returns `const T`, no non-const overload
// exists at all) rather than a body guard — there is no body guard here to
// check (`TableHandle<T, Texture>` has no non-const `operator[]` at all),
// so this row is the whole of the in-binary half of the contract.

#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <aether/aether.h>

namespace aether_tests {
namespace {

using aether::Plain;
using aether::TableHandle;
using aether::Texture;

/** @brief `true` iff `h[offset_t{0}] = std::declval<T>()` is a well-formed
 *  expression for handle type `H`. SFINAE on the EXPRESSION itself (not a
 *  `decltype`-then-`is_assignable` indirection), so it directly answers
 *  "does a write spelling compile", the same question the compile-fail
 *  script asks with a real compiler invocation. */
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

    // ANCHOR: no write spelling compiles on the texture carrier. This is
    // the SAME expression tests/compile_fail/check_texture_write_rejected.sh
    // proves red with a real compiler invocation (its WRITE-REJECTED arm);
    // this row re-observes it inside the ORDINARY test binary, so a
    // regression here shows up on every normal build, not only when the
    // specialized gate is run separately.
    static_assert(!IsElementWritable<TableHandle<double, Texture>, double>::value,
        "TextureWriteInvariant ANCHOR - `h[i] = v` compiles on the texture carrier; texture "
        "memory is read-only and no write spelling may exist");

    // CONTROL 1: `TableHandle` was ALWAYS a read-only handle (its own file
    // docstring: "a bounded, read-only ... table-read handle") — this pins
    // that the texture carrier's read-only contract is not a NEW
    // restriction relative to the plain one; both carriers agree.
    static_assert(!IsElementWritable<TableHandle<double, Plain>, double>::value,
        "CONTROL - the plain carrier's operator[] is unexpectedly writable; TableHandle was "
        "always read-only in BOTH carriers, so the rows above would be trivially satisfied by "
        "any handle at all");

    // CONTROL 2: the predicate machinery itself is not blind — a genuinely
    // writable container's operator[] IS detected as assignable. Without
    // this, `IsElementWritable` could have stopped binding the assignment
    // expression it names and every row above would pass vacuously.
    static_assert(IsElementWritable<std::vector<double>, double>::value,
        "CONTROL - IsElementWritable does not detect a genuinely writable operator[]; the "
        "rows above are therefore vacuous");

    SUCCEED();
}

} // namespace
} // namespace aether_tests
