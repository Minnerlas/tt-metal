#include "tt_metal/tt_metal/deployment/deployment_common.hpp"
#include "dram_base.hpp"

#include <gtest/gtest.h>
#include <tt-logger/tt-logger.hpp>

#include <atomic>
#include <csignal>

#include <thread>
#include <chrono>
#include <unistd.h>
#include <mutex>
#include <condition_variable>

#include <sstream>
#include <iostream>
#include <vector>

#include <iomanip>

namespace tt::tt_metal {

using namespace std;
using namespace tt;

static std::atomic<bool> g_stop_requested{false};
static std::atomic<bool> g_stop_message_printed{false};

static void handle_sigint(int) {
    g_stop_requested.store(true);

    if (!g_stop_message_printed.exchange(true)) {
        const char msg[] = "\nSIGINT received, waiting to finish current test...\n";
        write(2, msg, sizeof(msg) - 1);
    }
}

[[maybe_unused]] static uint64_t bytes_to_mb_floor(uint64_t bytes) { return bytes / (1024ull * 1024ull); }

static std::string format_bytes(uint64_t bytes) {
    std::ostringstream oss;

    const double KB = 1024.0;
    const double MB = 1024.0 * 1024.0;

    if (bytes < 1024ull) {
        oss << bytes << "B";
    } else if (bytes < 1024ull * 1024ull) {
        oss << std::fixed << std::setprecision(2) << (bytes / KB) << "KB";
    } else {
        oss << std::fixed << std::setprecision(2) << (bytes / MB) << "MB";
    }

    return oss.str();
}

static void print_subtest_status(
    uint32_t test_index,
    uint32_t total_tests,
    uint64_t subtest_index,
    uint64_t total_subtests,
    uint32_t mesh_x,
    uint32_t mesh_y,
    uint32_t bank_id,
    uint32_t pattern_id,
    double elapsed_ms,
    const DramRunSummary* summary = nullptr) {
    std::ostringstream oss;
    oss << "test " << test_index << "/" << total_tests << " subtest " << subtest_index << "/" << total_subtests
        << " mesh(" << mesh_x << "," << mesh_y << ")"
        << " bank:" << bank_id << " pattern:" << pattern_name(pattern_id) << " time:" << std::fixed
        << std::setprecision(2) << elapsed_ms << "ms";

    if (summary != nullptr && !summary->pass) {
        oss << " " << format_bytes(summary->suspected_write_error_bytes) << "/" << format_bytes(summary->checked_bytes)
            << " suspected write errors " << format_bytes(summary->suspected_read_error_bytes) << "/"
            << format_bytes(summary->checked_bytes) << " suspected read errors";
    }

    log_info(tt::LogTest, "{}", oss.str());
}

template <size_t N>
static uint64_t count_subtests_for_patterns(const uint32_t (&patterns)[N], uint32_t repeats) {
    uint64_t total = 0;
    for (uint32_t pattern_id : patterns) {
        total += static_cast<uint64_t>(repeats) * num_passes_for_pattern(pattern_id);
    }
    return total;
}

class [[maybe_unused]] Watchdog {
public:
    explicit Watchdog(std::chrono::seconds timeout) :
        test_finished(false), thread_([this, timeout]() {
            std::unique_lock<std::mutex> lock(mutex_);
            const bool finished_in_time = cv_.wait_for(lock, timeout, [this]() { return test_finished.load(); });

            if (!finished_in_time) {
                const char msg[] = "\nWatchdog timeout!\n";
                write(2, msg, sizeof(msg) - 1);
                std::raise(SIGINT);
            }
        }) {}

    ~Watchdog() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            test_finished = true;
        }
        cv_.notify_one();

        if (thread_.joinable()) {
            thread_.join();
        }
    }

private:
    std::atomic<bool> test_finished;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::thread thread_;
};

static std::vector<CoreCoord> get_worker_cores_for_deployment(IDevice* device) {
    std::vector<CoreCoord> cores;

    const auto grid = device->compute_with_storage_grid_size();
    for (uint32_t y = 0; y < grid.y; ++y) {
        for (uint32_t x = 0; x < grid.x; ++x) {
            cores.emplace_back(x, y);
        }
    }

    return cores;
}

[[maybe_unused]] static bool host_check_random_pattern_in_dram(
    IDevice* device,
    uint32_t bank_id,
    uint64_t bank_offset,
    uint32_t bytes_to_check,
    uint32_t seed,
    uint32_t pass_index) {
    TT_FATAL(bytes_to_check % sizeof(uint32_t) == 0, "bytes_to_check must be word aligned");

    const uint32_t word_count = bytes_to_check / sizeof(uint32_t);

    std::vector<uint32_t> observed;
    bool read_ok = detail::ReadFromDeviceDRAMChannel(
        device, static_cast<int>(bank_id), static_cast<uint32_t>(bank_offset), bytes_to_check, observed);

    TT_FATAL(read_ok, "ReadFromDeviceDRAMChannel failed");
    TT_FATAL(observed.size() == word_count, "Observed DRAM data size mismatch");

    uint32_t rng_state = seed ^ pass_index;
    if (rng_state == 0u) {
        rng_state = 1u;
    }

    uint32_t failures = 0;
    uint32_t first_word_index = 0;
    uint32_t first_expected = 0;
    uint32_t first_observed = 0;

    for (uint32_t i = 0; i < word_count; ++i) {
        rng_state = dram_pattern_random_step(rng_state);
        const uint32_t expected = rng_state;
        const uint32_t got = observed[i];

        if (got != expected) {
            if (failures == 0u) {
                first_word_index = i;
                first_expected = expected;
                first_observed = got;
            }
            failures++;
        }
    }

    if (failures == 0u) {
        log_info(
            tt::LogTest,
            "HOST DRAM RANDOM CHECK PASSED: bank={} bank_offset=0x{:08x} bytes_checked={} seed=0x{:08x} pass={}",
            bank_id,
            static_cast<uint32_t>(bank_offset),
            bytes_to_check,
            seed,
            pass_index);
        return true;
    }

    log_info(
        tt::LogTest,
        "HOST DRAM RANDOM CHECK FAILED: bank={} bank_offset=0x{:08x} bytes_checked={} seed=0x{:08x} pass={} "
        "failures={} first_word_index={} first_expected=0x{:08x} first_observed=0x{:08x}",
        bank_id,
        static_cast<uint32_t>(bank_offset),
        bytes_to_check,
        seed,
        pass_index,
        failures,
        first_word_index,
        first_expected,
        first_observed);

    return false;
}

[[maybe_unused]] static bool host_check_dram_read_write_roundtrip(
    IDevice* device, uint32_t bank_id, uint32_t bank_offset) {
    std::vector<uint32_t> write_data = {
        0x11223344u,
        0x55667788u,
        0xAABBCCDDu,
        0xDEADBEEFu,
    };

    detail::WriteToDeviceDRAMChannel(device, bank_id, bank_offset, write_data);

    std::vector<uint32_t> read_data;
    bool ok = detail::ReadFromDeviceDRAMChannel(
        device,
        static_cast<int>(bank_id),
        bank_offset,
        static_cast<uint32_t>(write_data.size() * sizeof(uint32_t)),
        read_data);

    TT_FATAL(ok, "ReadFromDeviceDRAMChannel failed");

    if (read_data.size() != write_data.size()) {
        log_info(
            tt::LogTest,
            "HOST DRAM ROUNDTRIP FAILED: size mismatch write_size={} read_size={}",
            write_data.size(),
            read_data.size());
        return false;
    }

    for (size_t i = 0; i < write_data.size(); ++i) {
        if (read_data[i] != write_data[i]) {
            log_info(
                tt::LogTest,
                "HOST DRAM ROUNDTRIP FAILED: index={} expected=0x{:08x} observed=0x{:08x}",
                i,
                write_data[i],
                read_data[i]);
            return false;
        }
    }

    log_info(tt::LogTest, "HOST DRAM ROUNDTRIP PASSED: bank={} offset=0x{:08x}", bank_id, bank_offset);

    return true;
}

[[maybe_unused]] bool host_check_random_pattern_sampling_32mb(
    IDevice* device, uint32_t bank_id, uint64_t bank_offset, uint32_t seed, uint32_t pass_index) {
    constexpr uint32_t sample_size_bytes = 4096;
    constexpr uint32_t total_check_bytes = 32 * 1024 * 1024;

    // proveravamo 8 tačaka kroz region
    constexpr int NUM_SAMPLES = 8;

    bool all_ok = true;

    for (int s = 0; s < NUM_SAMPLES; ++s) {
        uint32_t offset = (total_check_bytes / NUM_SAMPLES) * s;

        std::vector<uint32_t> observed(sample_size_bytes / sizeof(uint32_t));

        bool ok = detail::ReadFromDeviceDRAMChannel(
            device, bank_id, static_cast<uint32_t>(bank_offset + offset), sample_size_bytes, observed);

        if (!ok) {
            log_error(tt::LogTest, "DRAM read failed at sample {}", s);
            return false;
        }

        // --- REGENERATE EXPECTED LOCALLY ---
        uint32_t rng_state = seed ^ pass_index;
        if (rng_state == 0u) {
            rng_state = 1u;
        }

        // skip do ovog offset-a
        uint32_t start_word = offset / sizeof(uint32_t);

        for (uint32_t i = 0; i < start_word; ++i) {
            rng_state = dram_pattern_random_step(rng_state);
        }

        bool sample_ok = true;

        for (uint32_t i = 0; i < observed.size(); ++i) {
            rng_state = dram_pattern_random_step(rng_state);
            uint32_t expected = rng_state;

            if (observed[i] != expected) {
                log_error(
                    tt::LogTest,
                    "SAMPLE FAIL: sample={} word={} expected=0x{:08x} observed=0x{:08x}",
                    s,
                    i,
                    expected,
                    observed[i]);

                sample_ok = false;
                all_ok = false;
                break;
            }
        }

        if (sample_ok) {
            log_info(tt::LogTest, "Sample {} OK (offset=0x{:08x})", s, bank_offset + offset);
        }
    }

    if (all_ok) {
        log_info(tt::LogTest, "HOST 32MB RANDOM SAMPLING CHECK PASSED");
    } else {
        log_error(tt::LogTest, "HOST 32MB RANDOM SAMPLING CHECK FAILED");
    }

    return all_ok;
}

[[maybe_unused]] static uint32_t get_dram_test_bytes_from_env(uint32_t default_bytes) {
    const char* env = std::getenv("DRAM_TEST_MBYTES");

    uint64_t bytes = default_bytes;

    if (env != nullptr) {
        uint64_t value_mb = std::strtoull(env, nullptr, 0);

        TT_FATAL(value_mb > 0, "DRAM_TEST_BYTES must be > 0");

        bytes = value_mb * 1024ull * 1024ull;

        log_info(tt::LogTest, "Using DRAM_TEST_BYTES={} MB ({} bytes)", value_mb, bytes);
    }

    // STRICT (preporuka za CI)
    TT_FATAL(bytes <= DRAM_TEST_MAX_BANK_BYTES, "DRAM_TEST_BYTES too large: {} > {}", bytes, DRAM_TEST_MAX_BANK_BYTES);

    //    // DEV MODE (što ti sada radiš)
    //    if (bytes > DRAM_TEST_MAX_BANK_BYTES) {
    //        log_warning(tt::LogTest,
    //                    "Requested {} > max {}, clamping",
    //                    bytes,
    //                    DRAM_TEST_MAX_BANK_BYTES);
    //        bytes = DRAM_TEST_MAX_BANK_BYTES;
    //    }

    return static_cast<uint32_t>(bytes);
}

enum class [[maybe_unused]] DramNocMode { FIXED_0, FIXED_1, ALTERNATE };

[[maybe_unused]] static DramNocMode get_dram_noc_mode_from_env() {
    const char* env = std::getenv("DRAM_TEST_NOC_MODE");

    if (!env) {
        return DramNocMode::FIXED_0;  // default
    }

    std::string val(env);

    if (val == "0") {
        return DramNocMode::FIXED_0;
    }
    if (val == "1") {
        return DramNocMode::FIXED_1;
    }
    if (val == "alternate") {
        return DramNocMode::ALTERNATE;
    }

    TT_FATAL(false, "Invalid DRAM_TEST_NOC_MODE: {} (expected 0, 1, alternate)", val);
}

[[maybe_unused]] static uint32_t resolve_noc(DramNocMode mode, uint64_t pattern_index) {
    switch (mode) {
        case DramNocMode::FIXED_0: return 0;

        case DramNocMode::FIXED_1: return 1;

        case DramNocMode::ALTERNATE: return (pattern_index % 2 == 0) ? 0 : 1;
    }

    return 0;
}

[[maybe_unused]] static bool get_env_flag(const char* name, bool default_val = false) {
    const char* val = std::getenv(name);
    if (!val) {
        return default_val;
    }

    if (std::string(val) == "1" || std::string(val) == "true") {
        return true;
    }
    if (std::string(val) == "0" || std::string(val) == "false") {
        return false;
    }

    TT_FATAL(false, "Invalid value for {} (use 0/1)", name);
    return default_val;
}

[[maybe_unused]] static std::vector<CoreCoord> get_first_n_worker_cores(IDevice* device, size_t n) {
    const auto all_cores = get_worker_cores_for_deployment(device);
    TT_FATAL(all_cores.size() >= n, "Need at least {} worker cores, found {}", n, all_cores.size());

    return std::vector<CoreCoord>(all_cores.begin(), all_cores.begin() + n);
}

[[maybe_unused]] static uint32_t get_bank_id_for_core_in_all_controllers_test(size_t core_index, size_t total_cores) {
    constexpr size_t num_controllers = 8u;

    const size_t base_cores_per_controller = total_cores / num_controllers;
    const size_t remainder_cores = total_cores % num_controllers;

    size_t core_begin = 0;

    for (uint32_t bank_id = 0; bank_id < num_controllers; ++bank_id) {
        const size_t cores_in_this_controller = base_cores_per_controller + (bank_id < remainder_cores ? 1u : 0u);

        const size_t core_end = core_begin + cores_in_this_controller;

        if (core_index >= core_begin && core_index < core_end) {
            return bank_id;
        }

        core_begin = core_end;
    }

    TT_FATAL(false, "Invalid core_index={} for total_cores={}", core_index, total_cores);
}

[[maybe_unused]] const bool verbose = get_env_flag("DRAM_TEST_VERBOSE", false);

[[maybe_unused]] static void print_subtest_status_per_instance(
    uint32_t test_index,
    uint32_t total_tests,
    uint64_t subtest_index,
    uint64_t total_subtests,
    const DramPerCoreResult& per_core,
    double elapsed_ms) {
    DramRunSummary tmp{};
    tmp.pass = (per_core.result.failures == 0u);
    tmp.bank_id = per_core.result.bank_id;
    tmp.checked_bytes = static_cast<uint64_t>(per_core.result.words_checked) * sizeof(uint32_t);
    tmp.suspected_write_error_bytes =
        static_cast<uint64_t>(per_core.result.suspected_write_failures) * sizeof(uint32_t);
    tmp.suspected_read_error_bytes = static_cast<uint64_t>(per_core.result.suspected_read_failures) * sizeof(uint32_t);

    print_subtest_status(
        test_index,
        total_tests,
        subtest_index,
        total_subtests,
        per_core.core.x,
        per_core.core.y,
        per_core.result.bank_id,
        per_core.result.pattern_id,
        elapsed_ms,
        tmp.pass ? nullptr : &tmp);
}

[[maybe_unused]] static uint32_t get_dram_chunk_bytes_from_env(uint32_t default_bytes) {
    const char* env = std::getenv("DRAM_TEST_CHUNK_BYTES");

    uint64_t bytes = default_bytes;

    if (env != nullptr) {
        bytes = std::strtoull(env, nullptr, 0);

        TT_FATAL(bytes > 0, "DRAM_TEST_CHUNK_BYTES must be > 0");
        TT_FATAL(bytes % sizeof(uint32_t) == 0, "DRAM_TEST_CHUNK_BYTES must be word aligned");

        log_info(tt::LogTest, "Using DRAM_TEST_CHUNK_BYTES={} bytes", bytes);
    }

    // opciono: enforce neki max ako želiš
    TT_FATAL(bytes <= (1u << 20), "Chunk too large (>1MB)");  // možeš promeniti limit

    return static_cast<uint32_t>(bytes);
}

/*


TEST_F(UnitMeshCQProgramFixture, DramDeployment_Top2KBRefreshCheck) {
    g_stop_requested.store(false);
    g_stop_message_printed.store(false);

    bool all_pass = true;

    constexpr uint32_t seed = 0x12345678u;
    constexpr uint32_t repeat_index = 0u;
    constexpr uint32_t pass_index = 0u;

    constexpr uint32_t kTestIndex = 1u;
    constexpr uint32_t kTotalTests = 1u;
    constexpr uint64_t total_subtests = 2u;  // write phase + read phase

    uint64_t subtest_index = 0;

    const uint32_t chunk_bytes = get_dram_chunk_bytes_from_env(2048u);

    std::signal(SIGINT, handle_sigint);

    for (const auto& mesh_device : devices_) {
        if (g_stop_requested.load()) {
            break;
        }

        [[maybe_unused]] auto* const device = mesh_device->get_devices()[0];
        CoreCoord core = {0, 0};

        // Fiksiraj checkerboard i bank 0 za debug na vrhu memorije
        constexpr uint32_t bank_id = 0u;
        constexpr uint32_t pattern_id = DRAM_PATTERN_CHECKERBOARD;

        // -------------------------
        // FAZA 1: samo upis
        // -------------------------
        {
            DramDeploymentConfig cfg{
                .bank_id           = bank_id,
                .bank_offset       = DRAM_TEST_MAX_BANK_BYTES - chunk_bytes,
                .total_bytes       = chunk_bytes,
                .chunk_bytes       = chunk_bytes,
                .pattern_id        = pattern_id,
                .write_noc         = 0u,
                .read_noc          = 1u,
                .transfer_len_mode = 0u,
                .max_burst_len     = chunk_bytes,
                .skip_writes       = 0u,
                .skip_reads        = 0u,
            };

            auto subtest_start = std::chrono::steady_clock::now();

            DramRunSummary summary = run_dram_base_test(
                static_cast<MeshDispatchFixture*>(this),
                mesh_device,
                core,
                cfg,
                seed,
                pass_index,
                repeat_index,
                DataMovementProcessor::RISCV_0);

            auto subtest_end = std::chrono::steady_clock::now();
            double elapsed_ms =
                std::chrono::duration<double, std::milli>(subtest_end - subtest_start).count();

            ++subtest_index;

            print_subtest_status(
                kTestIndex,
                kTotalTests,
                subtest_index,
                total_subtests,
                core.x,
                core.y,
                cfg.bank_id,
                cfg.pattern_id,
                elapsed_ms,
                summary.pass ? nullptr : &summary);

            if (!summary.pass) {
                all_pass = false;
            }
        }

        if (!all_pass || g_stop_requested.load()) {
            break;
        }

        log_info(tt::LogTest, "Sleeping 10 seconds before readback...");
        //std::this_thread::sleep_for(std::chrono::seconds(1));////////////////////////////////////////////////////////////////////////////

        // -------------------------
        // FAZA 2: samo čitanje
        // -------------------------
        {
            DramDeploymentConfig cfg{
                .bank_id           = bank_id,
                .bank_offset       = DRAM_TEST_MAX_BANK_BYTES - chunk_bytes,
                .total_bytes       = chunk_bytes,
                .chunk_bytes       = chunk_bytes,
                .pattern_id        = pattern_id,
                .write_noc         = 0u,
                .read_noc          = 1u,
                .transfer_len_mode = 0u,
                .max_burst_len     = chunk_bytes,
                .skip_writes       = 1u,
                .skip_reads        = 0u,
            };

            auto subtest_start = std::chrono::steady_clock::now();

            DramRunSummary summary = run_dram_base_test(
                static_cast<MeshDispatchFixture*>(this),
                mesh_device,
                core,
                cfg,
                seed,
                pass_index,
                repeat_index,
                DataMovementProcessor::RISCV_0);

            auto subtest_end = std::chrono::steady_clock::now();
            double elapsed_ms =
                std::chrono::duration<double, std::milli>(subtest_end - subtest_start).count();

            ++subtest_index;

            print_subtest_status(
                kTestIndex,
                kTotalTests,
                subtest_index,
                total_subtests,
                core.x,
                core.y,
                cfg.bank_id,
                cfg.pattern_id,
                elapsed_ms,
                summary.pass ? nullptr : &summary);

            if (!summary.pass) {
                all_pass = false;
            }
        }
    }

    if (g_stop_requested.load()) {
        GTEST_SKIP() << "Test interrupted by user after current test finished.";
    }

    ASSERT_TRUE(all_pass);
}

//*/

TEST_F(UnitMeshCQProgramFixture, DramDeployment_SingleCoreSingleController) {
    g_stop_requested.store(false);
    g_stop_message_printed.store(false);

    bool all_pass = true;

    constexpr uint32_t repeats = 1u;
    constexpr uint32_t initial_seed = 0x12345678u;
    constexpr uint32_t advance_seed = 1u;
    constexpr bool stop_on_fail = false;

    static const uint32_t kDeploymentPatterns[] = {
        DRAM_PATTERN_COUNTER,
        DRAM_PATTERN_CHECKERBOARD,
        DRAM_PATTERN_ADDRESS,
        DRAM_PATTERN_MARCHING_ONES,
        DRAM_PATTERN_MARCHING_ZEROES,
        DRAM_PATTERN_MARCHING_ONE_BITS,
        DRAM_PATTERN_MARCHING_ZERO_BITS,
        DRAM_PATTERN_TOGGLE_BITS,
        DRAM_PATTERN_SATURATION,
        DRAM_PATTERN_REVERSIBLE_RANDOM,
        DRAM_PATTERN_RANDOM,
        DRAM_PATTERN_RANDOM_XOSHIRO128PP,
        DRAM_PATTERN_BYTEWISE_SSN,
    };

    constexpr uint32_t kTestIndex = 1u;
    constexpr uint32_t kTotalTests = 3u;

    const uint64_t total_subtests = count_subtests_for_patterns(kDeploymentPatterns, repeats);
    uint64_t subtest_index = 0;

    uint64_t pattern_toggle_index = 0;
    auto noc_mode = get_dram_noc_mode_from_env();

    const uint32_t chunk_bytes = get_dram_chunk_bytes_from_env(4096u);

    std::signal(SIGINT, handle_sigint);

    for (const auto& mesh_device : devices_) {
        if (g_stop_requested.load()) {
            break;
        }

        auto* const device = mesh_device->get_devices()[0];

        // Prvih 8 Tensix worker core-ova -> 8 DRAM kontrolera (0..7), po 1 core na svaki
        const auto selected_cores = get_first_n_worker_cores(device, 8);

        log_info(
            tt::LogTest, "Test 1 uses {} worker cores concurrently across 8 DRAM controllers", selected_cores.size());

        for (uint32_t pattern_id : kDeploymentPatterns) {
            if (g_stop_requested.load()) {
                break;
            }

            const uint32_t write_noc = resolve_noc(noc_mode, pattern_toggle_index);
            const uint32_t read_noc = resolve_noc(noc_mode, pattern_toggle_index + 1);

            bool pattern_pass = true;
            uint32_t seed = initial_seed;

            for (uint32_t repeat_index = 0; repeat_index < repeats; ++repeat_index) {
                if (g_stop_requested.load()) {
                    break;
                }

                const uint32_t num_passes = num_passes_for_pattern(pattern_id);

                for (uint32_t pass_index = 0; pass_index < num_passes; ++pass_index) {
                    if (g_stop_requested.load()) {
                        break;
                    }

                    auto subtest_start = std::chrono::steady_clock::now();

                    DramMultiInstanceSummary run = run_dram_eight_single_core_single_controller_test_verbose(
                        static_cast<MeshDispatchFixture*>(this),
                        mesh_device,
                        selected_cores,
                        0u,                                             // bank_offset
                        get_dram_test_bytes_from_env(DRAM_TEST_BYTES),  // bytes per controller
                        chunk_bytes,                                    // chunk_bytes
                        pattern_id,
                        write_noc,
                        read_noc,
                        0u,           // transfer_len_mode
                        chunk_bytes,  // max_burst_len
                        0u,           // skip_writes
                        0u,           // skip_reads
                        seed,
                        pass_index,
                        repeat_index,
                        DataMovementProcessor::RISCV_0);

                    auto subtest_end = std::chrono::steady_clock::now();
                    double elapsed_ms = std::chrono::duration<double, std::milli>(subtest_end - subtest_start).count();

                    ++subtest_index;

                    if (!verbose) {
                        print_subtest_status(
                            kTestIndex,
                            kTotalTests,
                            subtest_index,
                            total_subtests,
                            0,
                            0,
                            0u,
                            pattern_id,
                            elapsed_ms,
                            run.summary.pass ? nullptr : &run.summary);
                    } else {
                        for (const auto& per_core : run.per_core_results) {
                            print_subtest_status_per_instance(
                                kTestIndex, kTotalTests, subtest_index, total_subtests, per_core, elapsed_ms);
                        }
                    }

                    if (!run.summary.pass) {
                        pattern_pass = false;
                        all_pass = false;

                        if (stop_on_fail) {
                            break;
                        }
                    }
                }

                if (!pattern_pass && stop_on_fail) {
                    break;
                }

                seed += advance_seed;
            }

            ++pattern_toggle_index;
        }
    }

    if (g_stop_requested.load()) {
        GTEST_SKIP() << "Test interrupted by user after current test finished.";
    }

    ASSERT_TRUE(all_pass);
}

TEST_F(UnitMeshCQProgramFixture, DramDeployment_AllCoresEachSingleController) {
    g_stop_requested.store(false);
    g_stop_message_printed.store(false);

    bool all_pass = true;

    constexpr uint64_t controller_bank_offset = 0u;
    const uint32_t total_bytes_across_controller = get_dram_test_bytes_from_env(DRAM_TEST_BYTES);

    const uint32_t chunk_bytes = get_dram_chunk_bytes_from_env(4096u);

    constexpr uint32_t first_controller_bank_id = 0u;
    constexpr uint32_t last_controller_bank_id = 7u;

    constexpr uint32_t repeats = 1u;
    constexpr uint32_t initial_seed = 0x12345678u;
    constexpr uint32_t advance_seed = 1u;
    constexpr bool stop_on_fail = false;

    static const uint32_t kDeploymentPatterns[] = {
        DRAM_PATTERN_COUNTER,
        DRAM_PATTERN_CHECKERBOARD,
        DRAM_PATTERN_ADDRESS,
        DRAM_PATTERN_MARCHING_ONES,
        DRAM_PATTERN_MARCHING_ZEROES,
        DRAM_PATTERN_MARCHING_ONE_BITS,
        DRAM_PATTERN_MARCHING_ZERO_BITS,
        DRAM_PATTERN_TOGGLE_BITS,
        DRAM_PATTERN_SATURATION,
        DRAM_PATTERN_REVERSIBLE_RANDOM,
        DRAM_PATTERN_RANDOM,
        DRAM_PATTERN_RANDOM_XOSHIRO128PP,
        DRAM_PATTERN_BYTEWISE_SSN,
    };

    constexpr uint32_t kTestIndex = 2u;
    constexpr uint32_t kTotalTests = 3u;

    const uint64_t total_subtests = count_subtests_for_patterns(kDeploymentPatterns, repeats) *
                                    (last_controller_bank_id - first_controller_bank_id + 1u);

    uint64_t subtest_index = 0;

    uint64_t pattern_toggle_index = 0;
    auto noc_mode = get_dram_noc_mode_from_env();
    uint32_t write_noc = resolve_noc(noc_mode, pattern_toggle_index);
    uint32_t read_noc = resolve_noc(noc_mode, pattern_toggle_index + 1);

    std::signal(SIGINT, handle_sigint);

    for (const auto& mesh_device : devices_) {
        if (g_stop_requested.load()) {
            break;
        }

        [[maybe_unused]] auto* const device = mesh_device->get_devices()[0];
        const auto worker_cores = get_worker_cores_for_deployment(device);

        TT_FATAL(!worker_cores.empty(), "No worker cores found");

        for (uint32_t controller_bank_id = first_controller_bank_id; controller_bank_id <= last_controller_bank_id;
             ++controller_bank_id) {
            if (g_stop_requested.load()) {
                break;
            }

            for (uint32_t pattern_id : kDeploymentPatterns) {
                if (g_stop_requested.load()) {
                    break;
                }

                write_noc = resolve_noc(noc_mode, pattern_toggle_index);
                read_noc = resolve_noc(noc_mode, pattern_toggle_index + 1);

                bool pattern_pass = true;

                DramDeploymentConfig cfg{
                    .bank_id = controller_bank_id,
                    .bank_offset = controller_bank_offset,
                    .total_bytes = total_bytes_across_controller,
                    .chunk_bytes = chunk_bytes,
                    .pattern_id = pattern_id,
                    .write_noc = write_noc,
                    .read_noc = read_noc,
                    .transfer_len_mode = 0u,
                    .max_burst_len = chunk_bytes,
                    .skip_writes = 0u,
                    .skip_reads = 0u,
                };

                uint32_t seed = initial_seed;

                for (uint32_t repeat_index = 0; repeat_index < repeats; ++repeat_index) {
                    if (g_stop_requested.load()) {
                        break;
                    }

                    const uint32_t num_passes = num_passes_for_pattern(pattern_id);

                    for (uint32_t pass_index = 0; pass_index < num_passes; ++pass_index) {
                        if (g_stop_requested.load()) {
                            break;
                        }

                        auto subtest_start = std::chrono::steady_clock::now();

                        DramRunSummary summary = run_dram_multi_core_single_controller_test(
                            static_cast<MeshDispatchFixture*>(this),
                            mesh_device,
                            worker_cores,
                            cfg,
                            seed,
                            pass_index,
                            repeat_index,
                            DataMovementProcessor::RISCV_0);

                        auto subtest_end = std::chrono::steady_clock::now();
                        double elapsed_ms =
                            std::chrono::duration<double, std::milli>(subtest_end - subtest_start).count();

                        ++subtest_index;

                        if (!verbose) {
                            print_subtest_status(
                                kTestIndex,
                                kTotalTests,
                                subtest_index,
                                total_subtests,
                                0,
                                0,
                                cfg.bank_id,
                                pattern_id,
                                elapsed_ms,
                                summary.pass ? nullptr : &summary);
                        } else {
                            for (size_t inst_idx = 0; inst_idx < worker_cores.size(); ++inst_idx) {
                                const auto& core = worker_cores[inst_idx];

                                print_subtest_status(
                                    kTestIndex,
                                    kTotalTests,
                                    subtest_index,
                                    total_subtests,
                                    core.x,
                                    core.y,
                                    cfg.bank_id,
                                    pattern_id,
                                    elapsed_ms,
                                    nullptr);
                            }
                        }

                        if (!summary.pass) {
                            pattern_pass = false;
                            all_pass = false;

                            if (stop_on_fail) {
                                break;
                            }
                        }
                    }

                    if (!pattern_pass && stop_on_fail) {
                        break;
                    }

                    seed += advance_seed;
                }
                ++pattern_toggle_index;
            }
        }
    }

    if (g_stop_requested.load()) {
        GTEST_SKIP() << "Test interrupted by user after current test finished.";
    }

    ASSERT_TRUE(all_pass);
}

TEST_F(UnitMeshCQProgramFixture, DramDeployment_AllCoresAllControllers) {
    g_stop_requested.store(false);
    g_stop_message_printed.store(false);

    bool all_pass = true;

    const uint32_t total_bytes_per_controller = get_dram_test_bytes_from_env(DRAM_TEST_BYTES);

    const uint32_t chunk_bytes = get_dram_chunk_bytes_from_env(4096u);

    constexpr uint32_t repeats = 1u;
    constexpr uint32_t initial_seed = 0x12345678u;
    constexpr uint32_t advance_seed = 1u;
    constexpr bool stop_on_fail = false;

    static const uint32_t kDeploymentPatterns[] = {
        DRAM_PATTERN_COUNTER,
        DRAM_PATTERN_CHECKERBOARD,
        DRAM_PATTERN_ADDRESS,
        DRAM_PATTERN_MARCHING_ONES,
        DRAM_PATTERN_MARCHING_ZEROES,
        DRAM_PATTERN_MARCHING_ONE_BITS,
        DRAM_PATTERN_MARCHING_ZERO_BITS,
        DRAM_PATTERN_TOGGLE_BITS,
        DRAM_PATTERN_SATURATION,
        DRAM_PATTERN_REVERSIBLE_RANDOM,
        DRAM_PATTERN_RANDOM,
        DRAM_PATTERN_RANDOM_XOSHIRO128PP,
        DRAM_PATTERN_BYTEWISE_SSN,
    };

    constexpr uint32_t kTestIndex = 3u;
    constexpr uint32_t kTotalTests = 3u;

    const uint64_t total_subtests = count_subtests_for_patterns(kDeploymentPatterns, repeats);
    uint64_t subtest_index = 0;

    uint64_t pattern_toggle_index = 0;
    auto noc_mode = get_dram_noc_mode_from_env();
    uint32_t write_noc = resolve_noc(noc_mode, pattern_toggle_index);
    uint32_t read_noc = resolve_noc(noc_mode, pattern_toggle_index + 1);

    std::signal(SIGINT, handle_sigint);

    for (const auto& mesh_device : devices_) {
        if (g_stop_requested.load()) {
            break;
        }

        [[maybe_unused]] auto* const device = mesh_device->get_devices()[0];
        const auto worker_cores = get_worker_cores_for_deployment(device);

        TT_FATAL(!worker_cores.empty(), "No worker cores found");

        for (uint32_t pattern_id : kDeploymentPatterns) {
            if (g_stop_requested.load()) {
                break;
            }

            bool pattern_pass = true;
            uint32_t seed = initial_seed;

            for (uint32_t repeat_index = 0; repeat_index < repeats; ++repeat_index) {
                if (g_stop_requested.load()) {
                    break;
                }

                const uint32_t num_passes = num_passes_for_pattern(pattern_id);

                for (uint32_t pass_index = 0; pass_index < num_passes; ++pass_index) {
                    if (g_stop_requested.load()) {
                        break;
                    }

                    auto subtest_start = std::chrono::steady_clock::now();

                    write_noc = resolve_noc(noc_mode, pattern_toggle_index);
                    read_noc = resolve_noc(noc_mode, pattern_toggle_index + 1);

                    DramRunSummary summary = run_dram_multi_core_all_controllers_test(
                        static_cast<MeshDispatchFixture*>(this),
                        mesh_device,
                        worker_cores,
                        total_bytes_per_controller,
                        chunk_bytes,
                        pattern_id,
                        write_noc,  // write_noc
                        read_noc,   // read_noc
                        0u,         // transfer_len_mode
                        chunk_bytes,
                        0u,  // skip_writes
                        0u,  // skip_reads
                        seed,
                        pass_index,
                        repeat_index,
                        DataMovementProcessor::RISCV_0);

                    auto subtest_end = std::chrono::steady_clock::now();
                    double elapsed_ms = std::chrono::duration<double, std::milli>(subtest_end - subtest_start).count();

                    ++subtest_index;

                    if (!verbose) {
                        print_subtest_status(
                            kTestIndex,
                            kTotalTests,
                            subtest_index,
                            total_subtests,
                            0,
                            0,
                            0u,
                            pattern_id,
                            elapsed_ms,
                            summary.pass ? nullptr : &summary);
                    } else {
                        for (size_t inst_idx = 0; inst_idx < worker_cores.size(); ++inst_idx) {
                            const auto& core = worker_cores[inst_idx];

                            const uint32_t bank_id =
                                get_bank_id_for_core_in_all_controllers_test(inst_idx, worker_cores.size());

                            print_subtest_status(
                                kTestIndex,
                                kTotalTests,
                                subtest_index,
                                total_subtests,
                                core.x,
                                core.y,
                                bank_id,
                                pattern_id,
                                elapsed_ms,
                                nullptr);
                        }
                    }

                    if (!summary.pass) {
                        pattern_pass = false;
                        all_pass = false;

                        if (stop_on_fail) {
                            break;
                        }
                    }
                }

                if (!pattern_pass && stop_on_fail) {
                    break;
                }

                seed += advance_seed;
            }
            ++pattern_toggle_index;
        }
    }

    if (g_stop_requested.load()) {
        GTEST_SKIP() << "Test interrupted by user after current test finished.";
    }

    ASSERT_TRUE(all_pass);
}

}  // namespace tt::tt_metal
