#ifndef _DRAM_COMMON_H
#define _DRAM_COMMON_H

#include <stdint.h>

#include "patterns/patterns.hpp"

static constexpr uint32_t DRAM_TEST_NOC_WORD_BYTES = 64;
static constexpr uint64_t DRAM_TEST_MAX_BANK_BYTES = 0xFF000000ULL;
static constexpr uint32_t DRAM_TEST_BYTES = 32u * 1024u * 1024u;

enum DramFailureKind : uint32_t {
    DRAM_FAILURE_NONE = 0,
    DRAM_FAILURE_WRITE = 1,
    DRAM_FAILURE_READ = 2,
};

struct DramBaseResult {
    uint32_t pattern_id;
    uint32_t pass_index;
    uint32_t repeat_index;
    uint32_t bank_id;
    uint32_t transfers;
    uint32_t words_checked;
    uint32_t failures;
    uint32_t first_fail_addr;
    uint32_t first_expected;
    uint32_t first_observed;

    uint32_t failure_kind;    // 0 none, 1 write, 2 read
    uint32_t readback_count;  // usually 5
    uint32_t readback_data[5];

    uint32_t suspected_write_failures;  // number of words classified as write errors
    uint32_t suspected_read_failures;   // number of words classified as read errors

    uint64_t prepare_ticks;
    uint64_t write_ticks;
    uint64_t read_ticks;
};

struct DramTestParameters {
    uint32_t bank_id;
    uint32_t bank_offset_lo;
    uint32_t bank_offset_hi;
    uint32_t total_bytes;
    uint32_t chunk_bytes;
    uint32_t pattern_id;
    uint32_t seed;
    uint32_t pass_index;
    uint32_t repeat_index;
    uint32_t result_l1_addr;
    uint32_t expect_l1_addr;
    uint32_t observe_l1_addr;
    uint32_t write_noc;
    uint32_t read_noc;
    uint32_t max_burst_len;
    uint32_t transfer_len_mode;
    uint32_t skip_writes;
    uint32_t skip_reads;
};

static inline uint64_t dram_test_bank_offset(const DramTestParameters& p) {
    return ((uint64_t)p.bank_offset_hi << 32) | (uint64_t)p.bank_offset_lo;
}

#endif /* _DRAM_COMMON_H */
