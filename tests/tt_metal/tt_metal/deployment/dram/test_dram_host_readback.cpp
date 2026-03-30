// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/tt_metal/deployment/deployment_common.hpp"

#include "tt_metal/tt_metal/deployment/eth/common.hpp"  // TODO

#include <gtest/gtest.h>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include "tt_metal/test_utils/stimulus.hpp"
#include "command_queue_fixture.hpp"

#define BANDWIDTH_DRAM_COPY 35.0
#define BANDWIDTH_DRAM_COPY_SELF 15.0

namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace tt::test_utils;

template <typename FIXTURE>
static bool run_test_dram_host_readback(
    FIXTURE* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t src_bank) {
    /* ================= */
    auto* const device = mesh_device->get_devices()[0];

    uint32_t dram_start_addr = 0x500000;
    uint32_t dram_end_addr = 0xff000000u;
    TT_FATAL(dram_end_addr > dram_start_addr, "End address must be greater than start address");

    uint64_t total_size = dram_end_addr - dram_start_addr;

    uint32_t c = 123;
    size_t wordcount = total_size / sizeof(uint32_t);
    vector<uint32_t> inputs, zeros(wordcount);
    inputs.reserve(wordcount);
    for (long i = 0; i < wordcount; i++) {
        inputs.push_back(c++);
    }

    detail::WriteToDeviceDRAMChannel(device, src_bank, dram_start_addr, inputs);

    bool pass = true;
    pass &= dram_data_check(device, dram_start_addr, dram_end_addr, src_bank, inputs);

    return pass;
}

TEST_F(UnitMeshCQProgramFixture, TensixDeploymentDramHostReadback) {
    bool pass = true;

    SignalGuard g(SIGINT, handle_sigint);

    for (const auto& mesh_device : devices_) {
        auto* const device = mesh_device->get_devices()[0];
        log_info(tt::LogTest, "device id: {}", device->id());

        const int num_banks = device->num_dram_channels();
        for (int i = 0; i < num_banks; i++) {
            if (g_stop_requested.load()) {
                GTEST_SKIP() << "Test interrupted by user after current test finished.";
                return;
            }

            log_info(tt::LogTest, "  testing bank {}", i);
            pass &= run_test_dram_host_readback(this, mesh_device, i);
        }
    }

    ASSERT_TRUE(pass);
}

}  // namespace tt::tt_metal
