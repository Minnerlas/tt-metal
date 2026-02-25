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

void kernel_main() {
    std::uint32_t local_eth_l1_src_addr = get_arg_val<uint32_t>(0);
    std::uint32_t remote_eth_l1_dst_addr = get_arg_val<uint32_t>(1);
    std::uint64_t num_bytes = get_arg_val<uint32_t>(2);
    std::uint32_t loops = get_arg_val<uint32_t>(3);

    constexpr uint32_t num_bytes_per_send = get_compile_time_arg_val(0);
    constexpr uint32_t num_bytes_per_send_word_size = get_compile_time_arg_val(1);

    for (uint64_t start = timestamp(), delta = (timestamp() - start) / 1.35e9; delta < 5;
         delta = (timestamp() - start) / 1.35e9);

    uint64_t start = timestamp();
    for (uint32_t i = 0; i < loops; i++) {
        eth_send_bytes(
            local_eth_l1_src_addr, remote_eth_l1_dst_addr, num_bytes, num_bytes_per_send, num_bytes_per_send_word_size);
        eth_wait_for_receiver_done();
    }
    float delta = (timestamp() - start) / 1.35e9;

    uint64_t totalsent = num_bytes * loops;
    float speed = totalsent / delta / (1 << 30);

    uint32_t id = NOC_ID_LOGICAL_P & 0xfff;
    int x = id & 0x3f;
    int y = (id >> 6) & 0x3f;

    totalsent = totalsent >> 20;
    DPRINT << "(" << x << "," << y << ") wrote " << totalsent << "MB, took ";
    DPRINT << delta << " s";
    DPRINT << " at " << speed << "GB/s" << ENDL();
}
