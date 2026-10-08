// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#include "test_ItemRank0_common.h"
TEST(ItemRank0Test, Rank0ItemReadsAndWrites) { aether_tests::itemRank0ReadsAndWrites(); }
TEST(ItemRank0Test, ScalarViewGetMaterializesARank0Item) { aether_tests::scalarViewGetMaterializesARank0Item(); }
