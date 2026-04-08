#include "tt_metal/tt_metal/deployment/deployment_common.hpp"
#include "dram_base.hpp"

#include <gtest/gtest.h>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include <limits>

#include "command_queue_fixture.hpp"
#include "kernels/common_dram.hpp"

namespace tt::tt_metal {

using namespace std;
using namespace tt;

static inline const char* dram_failure_kind_name(uint32_t failure_kind) {
    switch (failure_kind) {
        case DRAM_FAILURE_WRITE: return "write";
        case DRAM_FAILURE_READ: return "read";
        case DRAM_FAILURE_NONE: return "none";
        default: return "unknown";
    }
}

static void log_dram_failure(IDevice* device, const CoreCoord& core, const DramBaseResult* result) {
    log_info(
        tt::LogTest,
        "Mismatch on device={} dram_controller={} core {} pattern={} repeat={} pass={}: failures={}, "
        "first_fail_classified_as={}, write_failures={}, read_failures={}",
        device->id(),
        result->bank_id,
        core,
        pattern_name(result->pattern_id),
        result->repeat_index,
        result->pass_index,
        result->failures,
        dram_failure_kind_name(result->failure_kind),
        result->suspected_write_failures,
        result->suspected_read_failures);
}

static inline void accumulate_result_into_summary(DramRunSummary& summary, const DramBaseResult* result) {
    summary.pass &= (result->failures == 0u);
    summary.bank_id = result->bank_id;
    summary.checked_bytes += static_cast<uint64_t>(result->words_checked) * sizeof(uint32_t);

    summary.suspected_write_error_bytes += static_cast<uint64_t>(result->suspected_write_failures) * sizeof(uint32_t);

    summary.suspected_read_error_bytes += static_cast<uint64_t>(result->suspected_read_failures) * sizeof(uint32_t);
}

DramRunSummary run_dram_base_test(
    MeshDispatchFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const CoreCoord& core,
    const DramDeploymentConfig& cfg,
    uint32_t seed,
    uint32_t pass_index,
    uint32_t repeat_index,
    DataMovementProcessor processor) {
    /* ====================== */
    auto* const device = mesh_device->get_devices()[0];

    TT_FATAL(cfg.bank_id < 8, "bank_id must not exceed the total number of controllers");
    TT_FATAL(cfg.total_bytes <= DRAM_TEST_MAX_BANK_BYTES, "total_bytes must be under (4GB-16MB)");
    TT_FATAL(cfg.chunk_bytes % sizeof(uint32_t) == 0, "chunk_bytes must be word aligned");
    TT_FATAL(cfg.total_bytes % sizeof(uint32_t) == 0, "total_bytes must be word aligned");

    struct l1_allocator alloc = new_tensix_allocator();

    const uint32_t result_l1_address = l1_alloc(alloc, sizeof(DramBaseResult));
    const uint32_t expect_l1_address = l1_alloc(alloc, cfg.chunk_bytes, DRAM_TEST_NOC_WORD_BYTES);
    const uint32_t observe_l1_address = l1_alloc(alloc, cfg.chunk_bytes, DRAM_TEST_NOC_WORD_BYTES);

    std::vector<uint32_t> zero_result(sizeof(DramBaseResult) / sizeof(uint32_t), 0u);
    MetalContext::instance().get_cluster().write_core(
        device->id(), device->worker_core_from_logical_core(core), zero_result, result_l1_address);

    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    distributed::MeshWorkload workload;
    tt_metal::Program program = tt_metal::Program();

    auto kernel_config = tt_metal::DataMovementConfig{
        .processor = processor,
        .noc = tt_metal::NOC::NOC_0,
    };

    auto kernel = tt_metal::CreateKernel(
        program, "tests/tt_metal/tt_metal/deployment/kernels/dram_base_kernel.cpp", core, kernel_config);

    DramTestParameters params{
        .bank_id = cfg.bank_id,
        .bank_offset_lo = static_cast<uint32_t>(cfg.bank_offset & 0xFFFFFFFFull),
        .bank_offset_hi = static_cast<uint32_t>((cfg.bank_offset >> 32) & 0xFFFFFFFFull),
        .total_bytes = cfg.total_bytes,
        .chunk_bytes = cfg.chunk_bytes,
        .pattern_id = cfg.pattern_id,
        .seed = seed,
        .pass_index = pass_index,
        .repeat_index = repeat_index,
        .result_l1_addr = result_l1_address,
        .expect_l1_addr = expect_l1_address,
        .observe_l1_addr = observe_l1_address,
        .write_noc = cfg.write_noc,
        .read_noc = cfg.read_noc,
        .max_burst_len = cfg.max_burst_len,
        .transfer_len_mode = cfg.transfer_len_mode,
        .skip_writes = cfg.skip_writes,
        .skip_reads = cfg.skip_reads,
    };

    tt_metal::SetRuntimeArgs(
        program,
        kernel,
        core,
        {
            params.bank_id,
            params.bank_offset_lo,
            params.bank_offset_hi,
            params.total_bytes,
            params.chunk_bytes,
            params.pattern_id,
            params.seed,
            params.pass_index,
            params.repeat_index,
            params.result_l1_addr,
            params.expect_l1_addr,
            params.observe_l1_addr,
            params.write_noc,
            params.read_noc,
            params.max_burst_len,
            params.transfer_len_mode,
            params.skip_writes,
            params.skip_reads,
        });

    workload.add_program(device_range, std::move(program));

    fixture->RunProgram(mesh_device, workload, true);
    fixture->FinishCommands(mesh_device);

    auto raw_result = MetalContext::instance().get_cluster().read_core(
        device->id(), device->worker_core_from_logical_core(core), result_l1_address, sizeof(DramBaseResult));

    const DramBaseResult* result = reinterpret_cast<const DramBaseResult*>(raw_result.data());

    DramRunSummary summary{};
    summary.pass = true;
    summary.bank_id = result->bank_id;
    summary.checked_bytes = 0;
    summary.suspected_write_error_bytes = 0;
    summary.suspected_read_error_bytes = 0;

    accumulate_result_into_summary(summary, result);

    if (result->failures > 0u) {
        log_dram_failure(device, core, result);
    }

    return summary;
}

DramRunSummary run_dram_multi_core_single_controller_test(
    tt::tt_metal::MeshDispatchFixture* fixture,
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    const std::vector<CoreCoord>& cores,
    const DramDeploymentConfig& cfg,
    uint32_t seed,
    uint32_t pass_index,
    uint32_t repeat_index,
    DataMovementProcessor processor) {
    auto* const device = mesh_device->get_devices()[0];

    TT_FATAL(!cores.empty(), "No cores provided");
    TT_FATAL(cfg.bank_id < 8, "bank_id must not exceed the total number of controllers");
    TT_FATAL(cfg.total_bytes <= DRAM_TEST_MAX_BANK_BYTES, "total_bytes must be under (4GB-16MB)");
    TT_FATAL(cfg.chunk_bytes % sizeof(uint32_t) == 0, "chunk_bytes must be word aligned");
    TT_FATAL(cfg.total_bytes % 4096u == 0, "total_bytes must be 4KB aligned for multi-core controller mode");

    const uint64_t total_bytes = cfg.total_bytes;
    const uint64_t bytes_per_core_base = (total_bytes / cores.size()) & ~0xFFFULL;

    TT_FATAL(bytes_per_core_base >= cfg.chunk_bytes, "bytes_per_core_base too small");
    TT_FATAL(bytes_per_core_base <= std::numeric_limits<uint32_t>::max(), "bytes_per_core_base must fit into uint32_t");

    const uint64_t covered_bytes = bytes_per_core_base * cores.size();
    const uint64_t remainder_bytes = total_bytes - covered_bytes;

    TT_FATAL((remainder_bytes & 0xFFFULL) == 0ULL, "remainder_bytes must stay 4KB aligned");

    struct l1_allocator alloc = new_tensix_allocator();

    const uint32_t result_l1_address = l1_alloc(alloc, sizeof(DramBaseResult));
    const uint32_t expect_l1_address = l1_alloc(alloc, cfg.chunk_bytes, DRAM_TEST_NOC_WORD_BYTES);
    const uint32_t observe_l1_address = l1_alloc(alloc, cfg.chunk_bytes, DRAM_TEST_NOC_WORD_BYTES);

    std::vector<uint32_t> zero_result(sizeof(DramBaseResult) / sizeof(uint32_t), 0u);
    for (const auto& core : cores) {
        MetalContext::instance().get_cluster().write_core(
            device->id(), device->worker_core_from_logical_core(core), zero_result, result_l1_address);
    }

    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    distributed::MeshWorkload workload;
    tt_metal::Program program = tt_metal::Program();

    auto kernel_config = tt_metal::DataMovementConfig{
        .processor = processor,
        .noc = tt_metal::NOC::NOC_0,
    };

    for (size_t i = 0; i < cores.size(); ++i) {
        const CoreCoord core = cores[i];
        const uint64_t bank_offset = cfg.bank_offset + i * bytes_per_core_base;

        uint64_t bytes_this_core = bytes_per_core_base;
        if (i == (cores.size() - 1)) {
            bytes_this_core += remainder_bytes;
        }

        TT_FATAL(bytes_this_core >= cfg.chunk_bytes, "bytes_this_core too small");
        TT_FATAL(bytes_this_core <= std::numeric_limits<uint32_t>::max(), "bytes_this_core must fit into uint32_t");

        auto kernel = tt_metal::CreateKernel(
            program, "tests/tt_metal/tt_metal/deployment/kernels/dram_base_kernel.cpp", core, kernel_config);

        tt_metal::SetRuntimeArgs(
            program,
            kernel,
            core,
            {
                cfg.bank_id,
                static_cast<uint32_t>(bank_offset & 0xFFFFFFFFull),
                static_cast<uint32_t>((bank_offset >> 32) & 0xFFFFFFFFull),
                static_cast<uint32_t>(bytes_this_core),
                cfg.chunk_bytes,
                cfg.pattern_id,
                seed,
                pass_index,
                repeat_index,
                result_l1_address,
                expect_l1_address,
                observe_l1_address,
                cfg.write_noc,
                cfg.read_noc,
                cfg.max_burst_len,
                cfg.transfer_len_mode,
                cfg.skip_writes,
                cfg.skip_reads,
            });
    }

    workload.add_program(device_range, std::move(program));

    fixture->RunProgram(mesh_device, workload, true);
    fixture->FinishCommands(mesh_device);

    DramRunSummary summary{};
    summary.pass = true;
    summary.bank_id = cfg.bank_id;
    summary.checked_bytes = 0;
    summary.suspected_write_error_bytes = 0;
    summary.suspected_read_error_bytes = 0;

    for (const auto& core : cores) {
        auto raw_result = MetalContext::instance().get_cluster().read_core(
            device->id(), device->worker_core_from_logical_core(core), result_l1_address, sizeof(DramBaseResult));

        const DramBaseResult* result = reinterpret_cast<const DramBaseResult*>(raw_result.data());

        accumulate_result_into_summary(summary, result);

        if (result->failures > 0u) {
            log_dram_failure(device, core, result);
        }
    }

    return summary;
}

DramRunSummary run_dram_multi_core_all_controllers_test(
    tt::tt_metal::MeshDispatchFixture* fixture,
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    const std::vector<CoreCoord>& cores,
    uint32_t total_bytes_per_controller,
    uint32_t chunk_bytes,
    uint32_t pattern_id,
    uint32_t write_noc,
    uint32_t read_noc,
    uint32_t transfer_len_mode,
    uint32_t max_burst_len,
    uint32_t skip_writes,
    uint32_t skip_reads,
    uint32_t seed,
    uint32_t pass_index,
    uint32_t repeat_index,
    DataMovementProcessor processor) {
    /* ========================= */
    auto* const device = mesh_device->get_devices()[0];

    constexpr uint32_t num_controllers = 8u;

    TT_FATAL(!cores.empty(), "No cores provided");
    TT_FATAL(
        total_bytes_per_controller <= DRAM_TEST_MAX_BANK_BYTES, "total_bytes_per_controller must be under (4GB-16MB)");
    TT_FATAL(chunk_bytes % sizeof(uint32_t) == 0, "chunk_bytes must be word aligned");
    TT_FATAL(total_bytes_per_controller % 4096u == 0, "total_bytes_per_controller must be 4KB aligned");

    struct l1_allocator alloc = new_tensix_allocator();

    const uint32_t result_l1_address = l1_alloc(alloc, sizeof(DramBaseResult));
    const uint32_t expect_l1_address = l1_alloc(alloc, chunk_bytes, DRAM_TEST_NOC_WORD_BYTES);
    const uint32_t observe_l1_address = l1_alloc(alloc, chunk_bytes, DRAM_TEST_NOC_WORD_BYTES);

    std::vector<uint32_t> zero_result(sizeof(DramBaseResult) / sizeof(uint32_t), 0u);
    for (const auto& core : cores) {
        MetalContext::instance().get_cluster().write_core(
            device->id(), device->worker_core_from_logical_core(core), zero_result, result_l1_address);
    }

    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    distributed::MeshWorkload workload;
    tt_metal::Program program = tt_metal::Program();

    auto kernel_config = tt_metal::DataMovementConfig{
        .processor = processor,
        .noc = tt_metal::NOC::NOC_0,
    };

    const size_t total_cores = cores.size();
    const size_t base_cores_per_controller = total_cores / num_controllers;
    const size_t remainder_cores = total_cores % num_controllers;

    size_t core_begin = 0;

    for (uint32_t bank_id = 0; bank_id < num_controllers; ++bank_id) {
        const size_t cores_in_this_controller = base_cores_per_controller + (bank_id < remainder_cores ? 1 : 0);

        if (cores_in_this_controller == 0) {
            continue;
        }

        const uint64_t bytes_per_core_base =
            (static_cast<uint64_t>(total_bytes_per_controller) / cores_in_this_controller) & ~0xFFFULL;

        TT_FATAL(bytes_per_core_base >= chunk_bytes, "bytes_per_core_base too small");
        TT_FATAL(
            bytes_per_core_base <= std::numeric_limits<uint32_t>::max(), "bytes_per_core_base must fit into uint32_t");

        const uint64_t covered_bytes = bytes_per_core_base * cores_in_this_controller;
        const uint64_t remainder_bytes = static_cast<uint64_t>(total_bytes_per_controller) - covered_bytes;

        TT_FATAL((remainder_bytes & 0xFFFULL) == 0ULL, "remainder_bytes must stay 4KB aligned");

        for (size_t local_idx = 0; local_idx < cores_in_this_controller; ++local_idx) {
            const size_t global_idx = core_begin + local_idx;
            const CoreCoord core = cores[global_idx];
            const uint64_t bank_offset = local_idx * bytes_per_core_base;

            uint64_t bytes_this_core = bytes_per_core_base;
            if (local_idx == (cores_in_this_controller - 1)) {
                bytes_this_core += remainder_bytes;
            }

            TT_FATAL(bytes_this_core >= chunk_bytes, "bytes_this_core too small");
            TT_FATAL(bytes_this_core <= std::numeric_limits<uint32_t>::max(), "bytes_this_core must fit into uint32_t");

            auto kernel = tt_metal::CreateKernel(
                program, "tests/tt_metal/tt_metal/deployment/kernels/dram_base_kernel.cpp", core, kernel_config);

            tt_metal::SetRuntimeArgs(
                program,
                kernel,
                core,
                {
                    bank_id,
                    static_cast<uint32_t>(bank_offset & 0xFFFFFFFFull),
                    static_cast<uint32_t>((bank_offset >> 32) & 0xFFFFFFFFull),
                    static_cast<uint32_t>(bytes_this_core),
                    chunk_bytes,
                    pattern_id,
                    seed,
                    pass_index,
                    repeat_index,
                    result_l1_address,
                    expect_l1_address,
                    observe_l1_address,
                    write_noc,
                    read_noc,
                    max_burst_len,
                    transfer_len_mode,
                    skip_writes,
                    skip_reads,
                });
        }

        core_begin += cores_in_this_controller;
    }

    workload.add_program(device_range, std::move(program));

    fixture->RunProgram(mesh_device, workload, true);
    fixture->FinishCommands(mesh_device);

    DramRunSummary summary{};
    summary.pass = true;
    summary.bank_id = 0;
    summary.checked_bytes = 0;
    summary.suspected_write_error_bytes = 0;
    summary.suspected_read_error_bytes = 0;

    for (const auto& core : cores) {
        auto raw_result = MetalContext::instance().get_cluster().read_core(
            device->id(), device->worker_core_from_logical_core(core), result_l1_address, sizeof(DramBaseResult));

        const DramBaseResult* result = reinterpret_cast<const DramBaseResult*>(raw_result.data());

        accumulate_result_into_summary(summary, result);

        if (result->failures > 0u) {
            log_dram_failure(device, core, result);
        }
    }

    return summary;
}

DramRunSummary run_dram_eight_single_core_single_controller_test(
    tt::tt_metal::MeshDispatchFixture* fixture,
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    const std::vector<CoreCoord>& cores,
    uint64_t bank_offset,
    uint32_t total_bytes_per_controller,
    uint32_t chunk_bytes,
    uint32_t pattern_id,
    uint32_t write_noc,
    uint32_t read_noc,
    uint32_t transfer_len_mode,
    uint32_t max_burst_len,
    uint32_t skip_writes,
    uint32_t skip_reads,
    uint32_t seed,
    uint32_t pass_index,
    uint32_t repeat_index,
    DataMovementProcessor processor) {
    /* ========================== */
    auto* const device = mesh_device->get_devices()[0];

    constexpr uint32_t num_controllers = 8u;

    TT_FATAL(!cores.empty(), "No cores provided");
    TT_FATAL(
        cores.size() <= num_controllers,
        "This helper supports at most {} cores, got {}",
        num_controllers,
        cores.size());
    TT_FATAL(
        total_bytes_per_controller <= DRAM_TEST_MAX_BANK_BYTES, "total_bytes_per_controller must be under (4GB-16MB)");
    TT_FATAL(chunk_bytes % sizeof(uint32_t) == 0, "chunk_bytes must be word aligned");
    TT_FATAL(total_bytes_per_controller % 4096u == 0, "total_bytes_per_controller must be 4KB aligned");
    TT_FATAL((bank_offset & 0xFFFULL) == 0ULL, "bank_offset must be 4KB aligned");
    TT_FATAL(
        bank_offset + static_cast<uint64_t>(total_bytes_per_controller) <= DRAM_TEST_MAX_BANK_BYTES,
        "bank_offset + total_bytes_per_controller exceeds DRAM_TEST_MAX_BANK_BYTES");
    TT_FATAL(bank_offset <= std::numeric_limits<uint64_t>::max(), "bank_offset out of range");

    struct l1_allocator alloc = new_tensix_allocator();

    const uint32_t result_l1_address = l1_alloc(alloc, sizeof(DramBaseResult));
    const uint32_t expect_l1_address = l1_alloc(alloc, chunk_bytes, DRAM_TEST_NOC_WORD_BYTES);
    const uint32_t observe_l1_address = l1_alloc(alloc, chunk_bytes, DRAM_TEST_NOC_WORD_BYTES);

    std::vector<uint32_t> zero_result(sizeof(DramBaseResult) / sizeof(uint32_t), 0u);
    for (const auto& core : cores) {
        MetalContext::instance().get_cluster().write_core(
            device->id(), device->worker_core_from_logical_core(core), zero_result, result_l1_address);
    }

    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    distributed::MeshWorkload workload;
    tt_metal::Program program = tt_metal::Program();

    auto kernel_config = tt_metal::DataMovementConfig{
        .processor = processor,
        .noc = tt_metal::NOC::NOC_0,
    };

    for (size_t inst_idx = 0; inst_idx < cores.size(); ++inst_idx) {
        const CoreCoord core = cores[inst_idx];
        const uint32_t bank_id = static_cast<uint32_t>(inst_idx);

        auto kernel = tt_metal::CreateKernel(
            program, "tests/tt_metal/tt_metal/deployment/kernels/dram_base_kernel.cpp", core, kernel_config);

        tt_metal::SetRuntimeArgs(
            program,
            kernel,
            core,
            {
                bank_id,
                static_cast<uint32_t>(bank_offset & 0xFFFFFFFFull),
                static_cast<uint32_t>((bank_offset >> 32) & 0xFFFFFFFFull),
                total_bytes_per_controller,
                chunk_bytes,
                pattern_id,
                seed,
                pass_index,
                repeat_index,
                result_l1_address,
                expect_l1_address,
                observe_l1_address,
                write_noc,
                read_noc,
                max_burst_len,
                transfer_len_mode,
                skip_writes,
                skip_reads,
            });
    }

    workload.add_program(device_range, std::move(program));

    fixture->RunProgram(mesh_device, workload, true);
    fixture->FinishCommands(mesh_device);

    DramRunSummary summary{};
    summary.pass = true;
    summary.bank_id = 0;
    summary.checked_bytes = 0;
    summary.suspected_write_error_bytes = 0;
    summary.suspected_read_error_bytes = 0;

    for (const auto& core : cores) {
        auto raw_result = MetalContext::instance().get_cluster().read_core(
            device->id(), device->worker_core_from_logical_core(core), result_l1_address, sizeof(DramBaseResult));

        const DramBaseResult* result = reinterpret_cast<const DramBaseResult*>(raw_result.data());

        accumulate_result_into_summary(summary, result);

        if (result->failures > 0u) {
            log_dram_failure(device, core, result);
        }
    }

    return summary;
}

struct [[maybe_unused]] DramPerCoreResult {
    CoreCoord core;
    DramBaseResult result;
};

struct [[maybe_unused]] DramMultiInstanceSummary {
    DramRunSummary summary;
    std::vector<DramPerCoreResult> per_core_results;
};

[[maybe_unused]] DramMultiInstanceSummary run_dram_eight_single_core_single_controller_test_verbose(
    tt::tt_metal::MeshDispatchFixture* fixture,
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    const std::vector<CoreCoord>& cores,
    uint64_t bank_offset,
    uint32_t total_bytes_per_controller,
    uint32_t chunk_bytes,
    uint32_t pattern_id,
    uint32_t write_noc,
    uint32_t read_noc,
    uint32_t transfer_len_mode,
    uint32_t max_burst_len,
    uint32_t skip_writes,
    uint32_t skip_reads,
    uint32_t seed,
    uint32_t pass_index,
    uint32_t repeat_index,
    DataMovementProcessor processor) {
    /* =============================== */
    auto* const device = mesh_device->get_devices()[0];

    constexpr uint32_t num_controllers = 8u;

    TT_FATAL(!cores.empty(), "No cores provided");
    TT_FATAL(
        cores.size() <= num_controllers,
        "This helper supports at most {} cores, got {}",
        num_controllers,
        cores.size());
    TT_FATAL(
        total_bytes_per_controller <= DRAM_TEST_MAX_BANK_BYTES, "total_bytes_per_controller must be under (4GB-16MB)");
    TT_FATAL(chunk_bytes % sizeof(uint32_t) == 0, "chunk_bytes must be word aligned");
    TT_FATAL(total_bytes_per_controller % 4096u == 0, "total_bytes_per_controller must be 4KB aligned");
    TT_FATAL((bank_offset & 0xFFFULL) == 0ULL, "bank_offset must be 4KB aligned");
    TT_FATAL(
        bank_offset + static_cast<uint64_t>(total_bytes_per_controller) <= DRAM_TEST_MAX_BANK_BYTES,
        "bank_offset + total_bytes_per_controller exceeds DRAM_TEST_MAX_BANK_BYTES");

    struct l1_allocator alloc = new_tensix_allocator();

    const uint32_t result_l1_address = l1_alloc(alloc, sizeof(DramBaseResult));
    const uint32_t expect_l1_address = l1_alloc(alloc, chunk_bytes, DRAM_TEST_NOC_WORD_BYTES);
    const uint32_t observe_l1_address = l1_alloc(alloc, chunk_bytes, DRAM_TEST_NOC_WORD_BYTES);

    std::vector<uint32_t> zero_result(sizeof(DramBaseResult) / sizeof(uint32_t), 0u);
    for (const auto& core : cores) {
        MetalContext::instance().get_cluster().write_core(
            device->id(), device->worker_core_from_logical_core(core), zero_result, result_l1_address);
    }

    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    distributed::MeshWorkload workload;
    tt_metal::Program program = tt_metal::Program();

    auto kernel_config = tt_metal::DataMovementConfig{
        .processor = processor,
        .noc = tt_metal::NOC::NOC_0,
    };

    for (size_t inst_idx = 0; inst_idx < cores.size(); ++inst_idx) {
        const CoreCoord core = cores[inst_idx];
        const uint32_t bank_id = static_cast<uint32_t>(inst_idx);

        auto kernel = tt_metal::CreateKernel(
            program, "tests/tt_metal/tt_metal/deployment/kernels/dram_base_kernel.cpp", core, kernel_config);

        tt_metal::SetRuntimeArgs(
            program,
            kernel,
            core,
            {
                bank_id,
                static_cast<uint32_t>(bank_offset & 0xFFFFFFFFull),
                static_cast<uint32_t>((bank_offset >> 32) & 0xFFFFFFFFull),
                total_bytes_per_controller,
                chunk_bytes,
                pattern_id,
                seed,
                pass_index,
                repeat_index,
                result_l1_address,
                expect_l1_address,
                observe_l1_address,
                write_noc,
                read_noc,
                max_burst_len,
                transfer_len_mode,
                skip_writes,
                skip_reads,
            });
    }

    workload.add_program(device_range, std::move(program));

    fixture->RunProgram(mesh_device, workload, true);
    fixture->FinishCommands(mesh_device);

    DramMultiInstanceSummary out{};
    out.summary.pass = true;
    out.summary.bank_id = 0;
    out.summary.checked_bytes = 0;
    out.summary.suspected_write_error_bytes = 0;
    out.summary.suspected_read_error_bytes = 0;

    out.per_core_results.reserve(cores.size());

    for (const auto& core : cores) {
        auto raw_result = MetalContext::instance().get_cluster().read_core(
            device->id(), device->worker_core_from_logical_core(core), result_l1_address, sizeof(DramBaseResult));

        const DramBaseResult* result = reinterpret_cast<const DramBaseResult*>(raw_result.data());

        accumulate_result_into_summary(out.summary, result);

        DramPerCoreResult per_core{};
        per_core.core = core;
        per_core.result = *result;
        out.per_core_results.push_back(per_core);

        if (result->failures > 0u) {
            log_dram_failure(device, core, result);
        }
    }

    return out;
}

}  // namespace tt::tt_metal
