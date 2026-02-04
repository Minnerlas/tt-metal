// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/debug/dprint.h"

static void read_from_dram(uint64_t* total, uint32_t dram_addr, uint32_t l1_addr, uint32_t sz) {
    noc_async_read(dram_addr, l1_addr, sz);
    *total += sz;
}

static uint32_t fill_mem_counter(uint32_t counter, uint8_t* buf, uint32_t size) {
    volatile uint32_t* b = (volatile uint32_t*)buf;
    for (uint32_t i = 0; i < size / sizeof(uint32_t); i++) {
        // DPRINT << i << " " << *counter << ENDL();
        b[i] = counter++;
    }

    return counter;
}

static int check_mem_counter(uint32_t cnt, uint8_t* buf, uint32_t size) {
    volatile uint32_t* b = (volatile uint32_t*)buf;
    for (uint32_t i = 0; i < size / sizeof(uint32_t); i++) {
        if (b[i] != cnt++) {
            return 1;
        }
    }

    return 0;
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
    constexpr auto in0_args = TensorAccessorArgs<0>();
    const auto in0 = TensorAccessor(in0_args, dram_buffer_src_addr, tile_size_bytes);

    constexpr auto out0_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    const auto out0 = TensorAccessor(out0_args, dram_buffer_dst_addr, tile_size_bytes);

    uint64_t readbytes = 0;

    // noc_async_read_tile(0, in0, l1_buffer_addr);
    // noc_async_read_barrier();

    uint64_t start = timestamp();
    DPRINT << start << " starting kernel: " << kernel_id << ENDL();

#define NUM 100
    uint64_t deltas[NUM] = {0};

    uint32_t controller = kernel_id;
    int noc = 0;

    uint8_t* const l1_buf = (uint8_t*)l1_buffer_addr;
    uint32_t counter = 0xdeadbeef + 231 * kernel_id;

    for (uint32_t j = 0; j < num_iter; j++) {
        for (uint64_t addr = 0; addr < ((4LL << 30) + 10); addr += tile_size_bytes) {
            uint64_t noc_addr = get_noc_addr_from_bank_id<true>(controller, addr, noc);

            uint32_t saved = counter;

            // DPRINT << "Filling memory in L1 (" << tile_size_bytes << " bytes)" << ENDL();
            counter = fill_mem_counter(counter, l1_buf, tile_size_bytes);
            // DPRINT << "Starting write to DRAM at " << addr << " (" << noc_addr << ")" << ENDL();
            if (check_mem_counter(saved, l1_buf, tile_size_bytes)) {
                DPRINT << "MISMATCH" << ENDL();
                goto end;
            }

            noc_async_write(l1_buffer_addr, noc_addr, tile_size_bytes, noc);
            noc_async_write_barrier();

            l1_buf[0] = 0;

            noc_async_read(noc_addr, l1_buffer_addr, tile_size_bytes, noc);
            noc_async_read_barrier();

            // DPRINT << "Checking the buffer" << ENDL();
            if (check_mem_counter(saved, l1_buf, tile_size_bytes)) {
                DPRINT << "MISMATCH" << ENDL();
                goto end;
            }
        }

        // uint64_t start = timestamp();
        // noc_async_read_barrier();
        // uint64_t delta = timestamp() - start;
        // if (j < NUM) {
        //     deltas[j] = delta;
        // }
    }
end:

    if (kernel_id == 0) {
        ((uint32_t*)l1_buffer_addr)[0] = 0x12344321;
        noc_async_write_tile(0, in0, l1_buffer_addr);
    } else {
        ((uint32_t*)l1_buffer_addr)[0] = 0xabcddcba;
        noc_async_write_tile(0, out0, l1_buffer_addr);
    }
    noc_async_write_barrier();

    // for (int i = 0; i < NUM; i++)
    // 	DPRINT << "read_from_dram " << deltas[i] << ENDL();

    // ((uint64_t*)l1_buffer_addr)[kernel_id] = readbytes;

    // noc_async_write_tile(0, out0, l1_buffer_addr);
    // noc_async_write_barrier();

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
