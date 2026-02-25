// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/debug/dprint.h"

#define WALL_CLOCK_L (*(volatile uint32_t*)0xFFB121F0)
#define WALL_CLOCK_H (*(volatile uint32_t*)0xFFB121F8)

#define NOC_ID_LOGICAL_P (*(volatile uint32_t*)0xFFB20148)

static uint64_t timestamp() {
    uint64_t low = WALL_CLOCK_L;
    asm volatile("" ::: "memory");
    uint64_t high = WALL_CLOCK_H;

    return (high << 32) | low;
}

/**
 * Any two RISC processors cannot use the same CMD_BUF
 * non_blocking APIs shouldn't be mixed with slow noc.h APIs
 * explicit flushes need to be used since the calls are non-blocking
 * */

void kernel_main() {
    std::uint64_t num_bytes = get_arg_val<uint32_t>(0);
    std::uint32_t loops = get_arg_val<uint32_t>(1);

    uint64_t start = timestamp();
    for (uint32_t i = 0; i < loops; i++) {
        eth_wait_for_bytes(num_bytes);
        eth_receiver_done();
    }
    float delta = (timestamp() - start) / 1.35e9;

    uint64_t totalrecv = num_bytes * loops;
    float speed = totalrecv / delta / (1 << 30);

    uint32_t id = NOC_ID_LOGICAL_P & 0xfff;
    int x = id & 0x3f;
    int y = (id >> 6) & 0x3f;

    totalrecv = totalrecv >> 20;
    DPRINT << "(" << x << "," << y << ") recieved " << totalrecv << "MB, took ";
    DPRINT << delta << " s";
    DPRINT << " at " << speed << "GB/s" << ENDL();
}
