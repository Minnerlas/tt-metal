// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/debug/dprint.h"

static void read_from_dram(uint64_t* total, uint32_t dram_addr, uint32_t l1_addr, uint32_t sz) {
    noc_async_read(dram_addr, l1_addr, sz);
    *total += sz;
}

static void mmain(
    uint32_t l1_buffer_addr,
    uint32_t dram_buffer_src_addr,
    uint32_t dram_buffer_dst_addr,
    uint32_t num_tiles,
    uint32_t l1_buffer1_addr,
    uint32_t l1_buffer2_addr) {
    // Arbitrary DRAM addresses
    // noc_addr_2 should be on a separate channel
    uint64_t noc_addr_1 = get_noc_addr_from_bank_id<true>(0, 0x1000);
    uint64_t noc_addr_2 = get_noc_addr_from_bank_id<true>(0, 0x1000 + (2 << 30));

    // Each tile is 32x32 elements of bfloat16, which is 2 bytes per element.
    // So the tile size in bytes is 32 * 32 * 2 = 2048 bytes.
    // Note that this is the same as the tile size used in the host code
    // when creating the buffers.
    const uint32_t tile_size_bytes = 32 * 32 * 2;
    constexpr auto in0_args = TensorAccessorArgs<0>();
    const auto in0 = TensorAccessor(in0_args, dram_buffer_src_addr, tile_size_bytes);

    const auto in1 = TensorAccessor(in0_args, noc_addr_1, tile_size_bytes);
    const auto in2 = TensorAccessor(in0_args, noc_addr_2, tile_size_bytes);

    constexpr auto out0_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    const auto out0 = TensorAccessor(out0_args, dram_buffer_dst_addr, tile_size_bytes);

    uint64_t readbytes = 0;

    for (uint32_t j = 0; j < 1000; j++) {
        for (uint32_t i = 0; i < num_tiles; i++) {
            // Issue a read to the NoC and write to the L1 buffer. This operation
            // is asynchronous.  thus a barrier is needed to ensure that the read
            // is complete before the write.
            // noc_async_read_tile(i, in0, l1_buffer_addr);
            // readbytes += tile_size_bytes;

            read_from_dram(&readbytes, noc_addr_1, l1_buffer1_addr, tile_size_bytes);
            read_from_dram(&readbytes, noc_addr_2, l1_buffer2_addr, tile_size_bytes);

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
        get_arg_val<uint32_t>(4),

        // Read parameters from the kernel arguments
        get_arg_val<uint32_t>(5));
}
