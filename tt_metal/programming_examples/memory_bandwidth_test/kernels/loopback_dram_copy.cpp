// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/debug/dprint.h"

static void read_from_dram(uint64_t* total, uint32_t dram_addr, uint32_t l1_addr, uint32_t sz) {
    noc_async_read(dram_addr, l1_addr, sz);
    *total += sz;
}

#define WALL_CLOCK_L (*(volatile uint32_t*)0xFFB121F0)
#define WALL_CLOCK_H (*(volatile uint32_t*)0xFFB121F8)

static uint64_t timestamp() {
    uint64_t low = WALL_CLOCK_L;
    asm volatile("" ::: "memory");
    uint64_t high = WALL_CLOCK_H;

    return (high << 32) | low;
}

static void mmain(
    uint32_t l1_buffer_addr,
    uint32_t dram_buffer_src_addr,
    uint32_t dram_buffer_dst_addr,
    uint32_t num_tiles,
    uint32_t num_l1_buffers,
    uint32_t num_iter,
    uint32_t tile_size_bytes,
    uint32_t kernel_id) {
    // Arbitrary DRAM addresses
    // noc_addr_2 should be on a separate channel
    uint64_t noc_addrs[num_l1_buffers] = {0};
    uint64_t l1_addrs[num_l1_buffers] = {0};
    uint32_t dram_bank = kernel_id;

    for (unsigned i = 0; i < num_l1_buffers; i++) {
        uint64_t base = i & 1 ? 0 : 2ULL << 30;

        uint32_t controller = kernel_id;
        //   kernel_id == 0 ? 2
        // : kernel_id == 1 ? 3
        // : 4;
        uint64_t addr = i * 0x1000 + base;
        int noc = 0;  // kernel_id == 2 ? 1 : 0;

        noc_addrs[i] = get_noc_addr_from_bank_id<true>(controller, addr, noc);
        //
        // const uint64_t channel2_offset = 2 << 30;  // 2GB
        // uint64_t dram_addr = i * tile_size_bytes + (i & 1 ? channel2_offset : 0);
        // noc_addrs[i] = get_noc_addr_from_bank_id<true>(dram_bank, dram_addr);
    }

    constexpr auto in0_args = TensorAccessorArgs<0>();
    const auto in0 = TensorAccessor(in0_args, dram_buffer_src_addr, tile_size_bytes);

    // const auto in1 = TensorAccessor(in0_args, noc_addrs[0], tile_size_bytes);
    // const auto in2 = TensorAccessor(in0_args, noc_addrs[1], tile_size_bytes);

    constexpr auto out0_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    const auto out0 = TensorAccessor(out0_args, dram_buffer_dst_addr, tile_size_bytes);

    uint64_t readbytes = 0;

    noc_async_read_tile(0, in0, l1_buffer_addr);
    noc_async_read_barrier();
    for (uint32_t i = 0; i < num_l1_buffers; i++) {
        l1_addrs[i] = ((uint32_t*)l1_buffer_addr)[i];
        // DPRINT << "Read " << l1_addrs[i] << " from dram\n" << ENDL();
    }

    // num_iter = kernel_id == 6 ? num_iter / 2 : num_iter;

    uint64_t start = timestamp();
    DPRINT << start << " starting kernel: " << kernel_id << ENDL();

#define NUM 100
    uint64_t deltas[NUM] = {0};

    for (uint32_t j = 0; j < num_iter; j++) {
        for (uint32_t k = 0; k < num_tiles; k++) {
            for (uint32_t i = 0; i < num_l1_buffers; i++) {
                read_from_dram(&readbytes, noc_addrs[i], l1_addrs[i], tile_size_bytes);
            }

            uint64_t start = timestamp();
            noc_async_read_barrier();
            uint64_t delta = timestamp() - start;
            if (j < NUM) {
                deltas[j] = delta;
            }

            // if (j < 10)
            // 	DPRINT << "read_from_dram " << delta << ENDL();
        }
    }

    // for (int i = 0; i < NUM; i++)
    // 	DPRINT << "read_from_dram " << deltas[i] << ENDL();

    ((uint64_t*)l1_buffer_addr)[kernel_id] = readbytes;

    noc_async_write_tile(0, out0, l1_buffer_addr);
    noc_async_write_barrier();

    uint64_t end = timestamp();
    uint64_t delta = (end - start) / 1350000;

    DPRINT << end << " kernel: " << kernel_id;
    DPRINT << " Read " << readbytes << " bytes from dram in " << num_iter;
    DPRINT << " iterations, took " << delta << " ms" << ENDL();
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

        // Number of L1 buffers
        get_arg_val<uint32_t>(4),

        // Number of loop iterations
        get_arg_val<uint32_t>(5),

        // Tile size
        get_arg_val<uint32_t>(6),

        // Kernel ID
        get_arg_val<uint32_t>(7));
}
