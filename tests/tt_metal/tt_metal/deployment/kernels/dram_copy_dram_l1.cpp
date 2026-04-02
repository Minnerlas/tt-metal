// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "args.hpp"
#include "timestamp.hpp"

#define ARGS(X)                  \
    X(uint32_t, dram_start_addr) \
    X(uint32_t, dram_end_addr)   \
    X(uint32_t, transfer_size)   \
    X(uint32_t, delta_addr)      \
    X(uint32_t, buffer0)         \
    X(uint32_t, src_bank)

void kernel_main() {
    ARG_INIT(ARGS);

    uint64_t start = timestamp();

    for (uint32_t curr_daddr = dram_start_addr, curr_baddr = buffer0; curr_daddr < dram_end_addr;
         curr_daddr += transfer_size, curr_baddr += transfer_size) {
        uint64_t noc_src_addr = get_noc_addr_from_bank_id<true>(src_bank, curr_daddr, 1);
        noc_async_read(noc_src_addr, curr_baddr, transfer_size);

        noc_async_read_barrier(1);
    }

    uint64_t delta = timestamp() - start;

    *(uint64_t*)delta_addr = delta;
}
