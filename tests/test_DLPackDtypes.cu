// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// DLPack round trips of the added dtypes (CUDA build; test_DLPackDtypes.cpp is the host twin over the same
// body in test_DLPackDtypes_common.h), plus the same round trip over device memory.

#include "test_DLPackDtypes_common.h"

#include <aether/chunk/Chunk.h>

namespace aether_tests {
namespace {

TEST_F(DLPackDtypesTest, DeviceRoundTripKeepsTheDtypeAndPointer)
{
    for (const DtypeCase& c : kAddedDtypes) {
        auto chunk = aether::Chunk::allocate(aether::Device(kDLCUDA), 12 * c.bits / 8);
        std::vector<std::byte> unused;
        aether::RuntimeView v = hostView(unused, c);
        v.data   = chunk.data();
        v.device = aether::Device(kDLCUDA);
        auto imp = aether::interop::fromDLPack(aether::interop::toDLPack(v));
        EXPECT_EQ(imp.view.dtype, v.dtype);
        EXPECT_EQ(imp.view.data, chunk.data());
        EXPECT_EQ(imp.view.device, v.device);
    }
}

} // namespace
} // namespace aether_tests
