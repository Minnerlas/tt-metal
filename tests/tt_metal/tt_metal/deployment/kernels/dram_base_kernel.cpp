#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "patterns/patterns.hpp"
#include "common_dram.hpp"
#include "dram_utils.hpp"
#include "timestamp.hpp"

// ============================================================
// Error injection configuration
// ============================================================

// 0 = disabled, 1 = enabled
#define INSERT_WRITE_ERRORS 0
#define INSERT_READ_ERRORS 0

// Number of injected failing words of each kind
#define WRITE_ERROR_COUNT 20
#define READ_ERROR_COUNT 20

// Spread errors out so they are rare.
// Example: every 1 MiB / 4B = every 262144 words.
#define WRITE_ERROR_STRIDE_WORDS 262144u
#define READ_ERROR_STRIDE_WORDS 393216u

// Start offsets (in words) so write/read errors do not overlap
#define WRITE_ERROR_START_WORD 1024u
#define READ_ERROR_START_WORD 2048u

static inline uint32_t dram_read_word_from_bank(
    uint32_t bank_id, uint64_t bank_offset_base, uint32_t byte_offset, uint32_t noc_id, uint32_t scratch_l1_addr) {
    const uint32_t aligned_offset = byte_offset & ~0x1Fu;  // 32B align
    const uint32_t word_index = (byte_offset & 0x1Fu) / sizeof(uint32_t);

    uint64_t dram_noc_addr =
        get_noc_addr_from_bank_id<true>(bank_id, (uint32_t)(bank_offset_base + (uint64_t)aligned_offset), noc_id);

    noc_async_read(dram_noc_addr, scratch_l1_addr, 32);
    noc_async_read_barrier();

    volatile uint32_t* scratch_words = (volatile uint32_t*)scratch_l1_addr;
    return scratch_words[word_index];
}

static inline bool dram_should_inject_write_error(uint32_t global_word_index) {
#if INSERT_WRITE_ERRORS
    for (uint32_t n = 0; n < WRITE_ERROR_COUNT; ++n) {
        const uint32_t target = WRITE_ERROR_START_WORD + n * WRITE_ERROR_STRIDE_WORDS;
        if (global_word_index == target) {
            return true;
        }
    }
#endif
    return false;
}

static inline bool dram_should_inject_read_error(uint32_t global_word_index) {
#if INSERT_READ_ERRORS
    for (uint32_t n = 0; n < READ_ERROR_COUNT; ++n) {
        const uint32_t target = READ_ERROR_START_WORD + n * READ_ERROR_STRIDE_WORDS;
        if (global_word_index == target) {
            return true;
        }
    }
#endif
    return false;
}

[[maybe_unused]] static inline void dprint_top_2kb_of_bank_if_in_range(
    const DramTestParameters& p,
    uint64_t bank_offset_base,
    uint32_t offset,
    uint32_t transfer_bytes,
    uint32_t* expect_words,
    uint32_t* observe_words) {
    if (p.pattern_id != DRAM_PATTERN_CHECKERBOARD) {
        return;
    }
    if (p.bank_id != 0u) {
        return;
    }
    if (p.pass_index != 0u) {
        return;
    }
    if (p.repeat_index != 0u) {
        return;
    }

    constexpr uint64_t dump_size_bytes = 2048ull;
    constexpr uint64_t target_end = (uint64_t)DRAM_TEST_MAX_BANK_BYTES;
    constexpr uint64_t target_start = target_end - dump_size_bytes;

    const uint64_t chunk_start = bank_offset_base + (uint64_t)offset;
    const uint64_t chunk_end = chunk_start + (uint64_t)transfer_bytes;

    if (chunk_end <= target_start || chunk_start >= target_end) {
        return;
    }

    const uint64_t dump_from = (chunk_start < target_start) ? target_start : chunk_start;
    const uint64_t dump_to = (chunk_end > target_end) ? target_end : chunk_end;

    const uint32_t first_word = (uint32_t)((dump_from - chunk_start) / sizeof(uint32_t));
    const uint32_t last_word = (uint32_t)((dump_to - chunk_start) / sizeof(uint32_t));

    DPRINT << "=== DRAM top 2KB dump begin ===" << ENDL();
    DPRINT << "bank=" << p.bank_id << " pass=" << p.pass_index << " repeat=" << p.repeat_index << " chunk_start=0x"
           << HEX() << chunk_start << " chunk_end=0x" << chunk_end << " dump_from=0x" << dump_from << " dump_to=0x"
           << dump_to << ENDL();

    for (uint32_t i = first_word; i < last_word; ++i) {
        const uint64_t abs_addr = chunk_start + (uint64_t)i * sizeof(uint32_t);

        DPRINT << "addr=0x" << HEX() << abs_addr << " expected=0x" << expect_words[i] << " observed=0x"
               << observe_words[i] << ENDL();
    }

    DPRINT << "=== DRAM top 2KB dump end ===" << ENDL();
}

void kernel_main() {
    DramTestParameters p;
    p.bank_id = get_arg_val<uint32_t>(0);
    p.bank_offset_lo = get_arg_val<uint32_t>(1);
    p.bank_offset_hi = get_arg_val<uint32_t>(2);
    p.total_bytes = get_arg_val<uint32_t>(3);
    p.chunk_bytes = get_arg_val<uint32_t>(4);
    p.pattern_id = get_arg_val<uint32_t>(5);
    p.seed = get_arg_val<uint32_t>(6);
    p.pass_index = get_arg_val<uint32_t>(7);
    p.repeat_index = get_arg_val<uint32_t>(8);
    p.result_l1_addr = get_arg_val<uint32_t>(9);
    p.expect_l1_addr = get_arg_val<uint32_t>(10);
    p.observe_l1_addr = get_arg_val<uint32_t>(11);
    p.write_noc = get_arg_val<uint32_t>(12);
    p.read_noc = get_arg_val<uint32_t>(13);
    p.max_burst_len = get_arg_val<uint32_t>(14);
    p.transfer_len_mode = get_arg_val<uint32_t>(15);
    p.skip_writes = get_arg_val<uint32_t>(16);
    p.skip_reads = get_arg_val<uint32_t>(17);

    const uint64_t bank_offset_base = dram_test_bank_offset(p);

    volatile DramBaseResult* result = (volatile DramBaseResult*)p.result_l1_addr;

    result->pattern_id = p.pattern_id;
    result->pass_index = p.pass_index;
    result->repeat_index = p.repeat_index;
    result->bank_id = p.bank_id;
    result->transfers = 0u;
    result->words_checked = 0u;
    result->failures = 0u;
    result->first_fail_addr = 0xFFFFFFFFu;
    result->first_expected = 0u;
    result->first_observed = 0u;
    result->failure_kind = DRAM_FAILURE_NONE;
    result->readback_count = 0u;

    result->suspected_write_failures = 0u;
    result->suspected_read_failures = 0u;

    for (uint32_t i = 0; i < 5u; ++i) {
        result->readback_data[i] = 0u;
    }

    if ((p.chunk_bytes == 0u) || ((p.chunk_bytes & 0x3u) != 0u) || ((p.total_bytes & 0x3u) != 0u)) {
        result->failures = 1u;
        result->first_fail_addr = 0u;
        result->first_expected = 0u;
        result->first_observed = p.chunk_bytes;
        return;
    }

    uint32_t* expect_words = (uint32_t*)p.expect_l1_addr;
    uint32_t* observe_words = (uint32_t*)p.observe_l1_addr;

    uint64_t total_prepare_ticks = 0u;
    uint64_t total_write_ticks = 0u;
    uint64_t total_read_ticks = 0u;

    uint32_t rng_state = p.seed ^ p.pass_index;
    if (rng_state == 0u) {
        rng_state = 1u;
    }

    DramXoshiro128ppState xoshiro_state = dram_pattern_random_xoshiro128pp_init(p.seed ^ p.pass_index);

    const bool use_checkerboard_fastpath = (p.pattern_id == DRAM_PATTERN_CHECKERBOARD);

    const bool use_counter_fastpath = (p.pattern_id == DRAM_PATTERN_COUNTER);

    const bool use_address_fastpath = (p.pattern_id == DRAM_PATTERN_ADDRESS);

    const bool use_marching_one_bits_fastpath = (p.pattern_id == DRAM_PATTERN_MARCHING_ONE_BITS);

    const bool use_marching_zero_bits_fastpath = (p.pattern_id == DRAM_PATTERN_MARCHING_ZERO_BITS);

    if (use_checkerboard_fastpath) {
        const uint32_t precompute_word_count = p.chunk_bytes / sizeof(uint32_t);
        uint64_t t0 = timestamp();
        dram_pattern_checkerboard_fill_buffer(expect_words, precompute_word_count, p.pass_index);
        uint64_t t1 = timestamp();
        total_prepare_ticks += (t1 - t0);
    }

    for (uint32_t offset = 0; offset < p.total_bytes;) {
        uint32_t remaining_bytes = p.total_bytes - offset;
        uint32_t transfer_bytes = dram_choose_transfer_len(p, remaining_bytes, rng_state);

        const uint32_t word_count = transfer_bytes / sizeof(uint32_t);
        const uint32_t base_word_index = offset / sizeof(uint32_t);

        if (!use_checkerboard_fastpath) {
            uint64_t t0 = timestamp();

            if (use_counter_fastpath) {
                dram_pattern_counter_fill_buffer(expect_words, word_count, p.seed, base_word_index);

            } else if (use_address_fastpath) {
                dram_pattern_address_fill_buffer(expect_words, word_count, p.repeat_index, base_word_index);

            } else if (use_marching_one_bits_fastpath) {
                const uint32_t value = dram_pattern_marching_one_bits(p.pass_index);
                dram_pattern_constant_fill_buffer(expect_words, word_count, value);

            } else if (use_marching_zero_bits_fastpath) {
                const uint32_t value = dram_pattern_marching_zero_bits(p.pass_index);
                dram_pattern_constant_fill_buffer(expect_words, word_count, value);

            } else {
                for (uint32_t i = 0; i < word_count; ++i) {
                    if (p.pattern_id == DRAM_PATTERN_RANDOM) {
                        rng_state = dram_pattern_random_step(rng_state);
                        expect_words[i] = rng_state;
                    } else if (p.pattern_id == DRAM_PATTERN_RANDOM_XOSHIRO128PP) {
                        expect_words[i] = dram_pattern_random_xoshiro128pp_next(xoshiro_state);
                    } else {
                        expect_words[i] = dram_pattern_generate(
                            p.pattern_id, p.seed, p.pass_index, base_word_index + i, p.repeat_index);
                    }
                }
            }

            uint64_t t1 = timestamp();
            total_prepare_ticks += (t1 - t0);
        }

        if (!p.skip_writes) {
            uint64_t write_dram_noc_addr = get_noc_addr_from_bank_id<true>(
                p.bank_id, (uint32_t)(bank_offset_base + (uint64_t)offset), p.write_noc);

            uint64_t t0 = timestamp();
            noc_async_write(p.expect_l1_addr, write_dram_noc_addr, transfer_bytes);
            noc_async_write_barrier();

#if INSERT_WRITE_ERRORS
            for (uint32_t i = 0; i < word_count; ++i) {
                const uint32_t global_word_index = base_word_index + i;

                if (dram_should_inject_write_error(global_word_index)) {
                    uint32_t wrong_word = expect_words[i] ^ 0x00000001u;

                    uint64_t word_dram_noc_addr = get_noc_addr_from_bank_id<true>(
                        p.bank_id,
                        (uint32_t)(bank_offset_base + (uint64_t)offset + (uint64_t)i * sizeof(uint32_t)),
                        p.write_noc);

                    uint32_t* inject_word_ptr = (uint32_t*)p.observe_l1_addr;
                    inject_word_ptr[0] = wrong_word;

                    noc_async_write(p.observe_l1_addr, word_dram_noc_addr, sizeof(uint32_t));
                    noc_async_write_barrier();
                }
            }
#endif

            uint64_t t1 = timestamp();
            total_write_ticks += (t1 - t0);
        }

        if (!p.skip_reads) {
            uint64_t read_dram_noc_addr =
                get_noc_addr_from_bank_id<true>(p.bank_id, (uint32_t)(bank_offset_base + (uint64_t)offset), p.read_noc);

            uint64_t t0 = timestamp();
            noc_async_read(read_dram_noc_addr, p.observe_l1_addr, transfer_bytes);
            noc_async_read_barrier();
            uint64_t t1 = timestamp();
            total_read_ticks += (t1 - t0);

            /////////////////////////////////////////////////////////////////////////////////////////////////////////
            dprint_top_2kb_of_bank_if_in_range(
                p, bank_offset_base, offset, transfer_bytes, expect_words, observe_words);
            /////////////////////////////////////////////////////////////////////////////////////////////////////////

            for (uint32_t i = 0; i < word_count; ++i) {
                const uint32_t global_word_index = base_word_index + i;

                uint32_t expected = expect_words[i];
                uint32_t observed = observe_words[i];

#if INSERT_READ_ERRORS
                if (dram_should_inject_read_error(global_word_index)) {
                    observed ^= 0x00000001u;
                }
#endif

                result->words_checked++;

                if (observed != expected) {
                    const uint32_t fail_byte_offset = offset + i * sizeof(uint32_t);
                    const uint32_t scratch_l1_addr = p.observe_l1_addr;

                    uint32_t reread0 = dram_read_word_from_bank(
                        p.bank_id, bank_offset_base, fail_byte_offset, p.read_noc, scratch_l1_addr);
                    uint32_t reread1 = dram_read_word_from_bank(
                        p.bank_id, bank_offset_base, fail_byte_offset, p.read_noc, scratch_l1_addr);
                    uint32_t reread2 = dram_read_word_from_bank(
                        p.bank_id, bank_offset_base, fail_byte_offset, p.read_noc, scratch_l1_addr);
                    uint32_t reread3 = dram_read_word_from_bank(
                        p.bank_id, bank_offset_base, fail_byte_offset, p.read_noc, scratch_l1_addr);
                    uint32_t reread4 = dram_read_word_from_bank(
                        p.bank_id, bank_offset_base, fail_byte_offset, p.read_noc, scratch_l1_addr);

                    bool all_same =
                        (reread0 == reread1) && (reread0 == reread2) && (reread0 == reread3) && (reread0 == reread4);

                    uint32_t classified_kind = DRAM_FAILURE_READ;
                    if (all_same) {
                        if (reread0 == expected) {
                            classified_kind = DRAM_FAILURE_READ;
                        } else {
                            classified_kind = DRAM_FAILURE_WRITE;
                        }
                    } else {
                        classified_kind = DRAM_FAILURE_READ;
                    }

                    if (classified_kind == DRAM_FAILURE_WRITE) {
                        result->suspected_write_failures++;
                    } else {
                        result->suspected_read_failures++;
                    }

                    if (result->failures == 0u) {
                        result->first_fail_addr = fail_byte_offset;
                        result->first_expected = expected;
                        result->first_observed = observed;
                        result->failure_kind = classified_kind;

                        result->readback_count = 5u;
                        result->readback_data[0] = reread0;
                        result->readback_data[1] = reread1;
                        result->readback_data[2] = reread2;
                        result->readback_data[3] = reread3;
                        result->readback_data[4] = reread4;
                    }

                    result->failures++;
                }
            }
        }

        offset += transfer_bytes;
        result->transfers++;
    }

    result->prepare_ticks = total_prepare_ticks;
    result->write_ticks = total_write_ticks;
    result->read_ticks = total_read_ticks;
}
