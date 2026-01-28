// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/debug/dprint.h"

static void read_from_dram(uint64_t* total, uint32_t dram_addr, uint32_t l1_addr, uint32_t sz) {
    noc_async_read(dram_addr, l1_addr, sz);
    *total += sz;
}

const uint32_t tile_size_bytes = sizeof(uint32_t) * 32 * 32;

static void mmain(
    uint32_t l1_buffer_addr,
    uint32_t dram_buffer_src_addr,
    uint32_t dram_buffer_dst_addr,
    uint32_t num_tiles,
    uint32_t num_l1_buffers) {
    // Arbitrary DRAM addresses
    // noc_addr_2 should be on a separate channel
    uint64_t noc_addrs[num_l1_buffers] = {0};
    uint64_t l1_addrs[num_l1_buffers] = {0};

    for (unsigned i = 0; i < num_l1_buffers; i++) {
        // noc_addrs[i] = get_noc_addr_from_bank_id<true>(0, i * tile_size_bytes);
        const uint64_t channel2_offset = 2 << 30;  // 2GB
        uint64_t dram_addr = i * tile_size_bytes + (i & 1 ? channel2_offset : 0);
        noc_addrs[i] = get_noc_addr_from_bank_id<true>(0, dram_addr);
    }

    constexpr auto in0_args = TensorAccessorArgs<0>();
    const auto in0 = TensorAccessor(in0_args, dram_buffer_src_addr, tile_size_bytes);

    const auto in1 = TensorAccessor(in0_args, noc_addrs[0], tile_size_bytes);
    const auto in2 = TensorAccessor(in0_args, noc_addrs[1], tile_size_bytes);

    constexpr auto out0_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    const auto out0 = TensorAccessor(out0_args, dram_buffer_dst_addr, tile_size_bytes);

    uint64_t readbytes = 0;

    noc_async_read_tile(0, in0, l1_buffer_addr);
    noc_async_read_barrier();
    for (uint32_t i = 0; i < num_l1_buffers; i++) {
        l1_addrs[i] = ((uint32_t*)l1_buffer_addr)[i];
        // DPRINT << "Read " << l1_addrs[i] << " from dram\n" << ENDL();
    }

    for (uint32_t j = 0; j < 1000; j++) {
        for (uint32_t i = 0; i < num_tiles; i++) {
            for (uint32_t i = 0; i < num_l1_buffers; i++) {
                read_from_dram(&readbytes, noc_addrs[i], l1_addrs[i], tile_size_bytes);
            }

            noc_async_read_barrier();
        }
    }

    *(uint64_t*)l1_buffer_addr = readbytes;

    noc_async_write_tile(0, out0, l1_buffer_addr);
    noc_async_write_barrier();

    DPRINT << "Read " << readbytes << " from dram\n";  // << ENDL();
}

void kernel_main() {
    mmain(
        // Read parameters from the kernel arguments
        get_arg_val<uint32_t>(0),

        // Address and the DRAM bank ID of the source buffer
        get_arg_val<uint32_t>(1),

        // Address and the DRAM bank ID of the destination buffer
        get_arg_val<uint32_t>(2),

        // Size of the buffer in bytes
        get_arg_val<uint32_t>(3),

        // Read parameters from the kernel arguments
        get_arg_val<uint32_t>(4));
}
