// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <cstdlib>
#include <tt-metalium/host_api.hpp>
#include <tt-logger/tt-logger.hpp>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <enchantum/enchantum.hpp>
#include <map>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <tt_stl/assert.hpp>
#include "command_queue_fixture.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include "device_fixture.hpp"
#include "mesh_dispatch_fixture.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include "jit_build/build.hpp"
#include <tt-metalium/kernel_types.hpp>
#include "llrt.hpp"
#include "mesh_device.hpp"
#include "multi_device_fixture.hpp"
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include "impl/context/metal_context.hpp"
#include "tt_memory.h"
#include "tt_metal/jit_build/build_env_manager.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include <umd/device/types/xy_pair.hpp>
#include "eth_test_common.hpp"

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {
constexpr std::int32_t WORD_SIZE = 16;  // 16 bytes per eth send packet

struct erisc_info_t {
    volatile uint32_t num_bytes;
    volatile uint32_t mode;
    volatile uint32_t reserved_0_;
    volatile uint32_t reserved_1_;
    volatile uint32_t bytes_done;
    volatile uint32_t reserverd_2_;
    volatile uint32_t reserverd_3_;
    volatile uint32_t reserverd_4_;
};
}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

using namespace tt;
using namespace tt::test_utils;
using namespace tt::tt_metal;

namespace unit_tests::erisc::direct_send {

template <typename FIXTURE>
static bool eth_direct_bandwidth_kernels(
    FIXTURE* fixture,
    const std::shared_ptr<distributed::MeshDevice>& sender_mesh_device,
    const std::shared_ptr<distributed::MeshDevice>& receiver_mesh_device,
    const size_t& byte_size,
    const size_t& src_eth_l1_byte_address,
    const size_t& dst_eth_l1_byte_address,
    const std::vector<CoreCoord> senders,
    const std::vector<CoreCoord> receivers,
    DataMovementProcessor processor = DataMovementProcessor::RISCV_0,
    uint32_t num_bytes_per_send = 16) {
    auto* const sender_device = sender_mesh_device->get_devices()[0];
    auto* const receiver_device = receiver_mesh_device->get_devices()[0];

    bool same_device = sender_mesh_device == receiver_mesh_device;

    // log_info(
    //     tt::LogTest,
    //     "Sending {} bytes from device {} eth core {} addr {} to device {} eth core {} addr {} processor {}",
    //     byte_size,
    //     sender_device->id(),
    //     eth_sender_core.str(),
    //     src_eth_l1_byte_address,
    //     receiver_device->id(),
    //     eth_receiver_core.str(),
    //     dst_eth_l1_byte_address,
    //     processor);

    auto inputs = generate_uniform_random_vector<uint32_t>(0, 100, byte_size / sizeof(uint32_t));

    for (const auto& sender : senders) {
        // Generate inputs
        tt::tt_metal::MetalContext::instance().get_cluster().write_core(
            sender_device->id(),
            sender_device->ethernet_core_from_logical_core(sender),
            inputs,
            src_eth_l1_byte_address);
    }

    for (const auto& receiver : receivers) {
        // Clear expected value at ethernet L1 address
        std::vector<uint32_t> all_zeros(inputs.size(), 0);
        tt::tt_metal::MetalContext::instance().get_cluster().write_core(
            receiver_device->id(),
            receiver_device->ethernet_core_from_logical_core(receiver),
            all_zeros,
            dst_eth_l1_byte_address);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Sender Device
    ////////////////////////////////////////////////////////////////////////////
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload sender_workload;
    tt_metal::Program sender_program = tt_metal::Program();

    auto ethernet_config = tt_metal::EthernetConfig{
        .noc = tt_metal::NOC::NOC_0,
        .processor = processor,
        .compile_args = {uint32_t(num_bytes_per_send), uint32_t(num_bytes_per_send >> 4)}};
    eth_test_common::set_arch_specific_eth_config(ethernet_config);

    uint32_t loops = 100000;

    for (int i = 0; i < senders.size(); i++) {
        const auto& eth_sender_core = senders[i];
        const uint32_t loopcount = loops * (i + 1);

        auto eth_sender_kernel = tt_metal::CreateKernel(
            sender_program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_l1_direct_send_bandwidth_total.cpp",
            eth_sender_core,
            ethernet_config);

        tt_metal::SetRuntimeArgs(
            sender_program,
            eth_sender_kernel,
            eth_sender_core,
            {
                (uint32_t)src_eth_l1_byte_address,
                (uint32_t)dst_eth_l1_byte_address,
                (uint32_t)byte_size,
                (uint32_t)loopcount,
            });
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Receiver Device
    ////////////////////////////////////////////////////////////////////////////
    distributed::MeshWorkload receiver_workload_;
    tt_metal::Program receiver_program_ = tt_metal::Program();

    distributed::MeshWorkload& receiver_workload = same_device ? sender_workload : receiver_workload_;
    tt_metal::Program& receiver_program = same_device ? sender_program : receiver_program_;

    for (int i = 0; i < receivers.size(); i++) {
        const auto& eth_receiver_core = receivers[i];
        const uint32_t loopcount = loops * (i + 1);

        auto eth_receiver_kernel = tt_metal::CreateKernel(
            receiver_program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_l1_direct_receive_bandwidth_total.cpp",
            eth_receiver_core,
            ethernet_config);

        tt_metal::SetRuntimeArgs(
            receiver_program,
            eth_receiver_kernel,
            eth_receiver_core,
            {
                (uint32_t)byte_size,
                (uint32_t)loopcount,
            });
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Programs
    ////////////////////////////////////////////////////////////////////////////
    sender_workload.add_program(device_range, std::move(sender_program));
    if (!same_device) {
        receiver_workload.add_program(device_range, std::move(receiver_program));
    }
    fixture->RunProgram(sender_mesh_device, sender_workload, true);
    if (!same_device) {
        fixture->RunProgram(receiver_mesh_device, receiver_workload, true);
    }

    fixture->FinishCommands(sender_mesh_device);
    if (!same_device) {
        fixture->FinishCommands(receiver_mesh_device);
    }

    bool pass = true;

    for (const auto& receiver : receivers) {
        auto readback_vec = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
            receiver_device->id(),
            receiver_device->ethernet_core_from_logical_core(receiver),
            dst_eth_l1_byte_address,
            byte_size);

        if (readback_vec != inputs) {
            std::cout << "Mismatch at Core: " << receiver.str() << std::endl;
            std::cout << readback_vec[0] << std::endl;
            pass = false;
        }
    }

    return pass;
}

}  // namespace unit_tests::erisc::direct_send

namespace tt::tt_metal {

TEST_F(UnitMeshCQProgramFixture, ActiveEthKernelsDirectSendAllConnectedChipsBandwidthTotal) {
    using namespace CMAKE_UNIQUE_NAMESPACE;

    std::vector<CoreCoord> senders;
    std::vector<CoreCoord> receivers;

    const size_t src_eth_l1_byte_address =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    const size_t src_eth_l1_byte_size =
        MetalContext::instance().hal().get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);

    const size_t dst_eth_l1_byte_address =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    const size_t dst_eth_l1_byte_size =
        MetalContext::instance().hal().get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    log_info(tt::LogTest, "src addr {}, src size {}", src_eth_l1_byte_address, src_eth_l1_byte_size);
    log_info(tt::LogTest, "dst addr {}, dst size {}", dst_eth_l1_byte_address, dst_eth_l1_byte_size);

    const auto num_eriscs = MetalContext::instance().hal().get_num_risc_processors(HalProgrammableCoreType::ACTIVE_ETH);
    for (const auto& sender_mesh_device : devices_) {
        auto* const sender_device = sender_mesh_device->get_devices()[0];
        for (const auto& receiver_mesh_device : devices_) {
            auto* const receiver_device = receiver_mesh_device->get_devices()[0];
            for (const auto& sender_core : sender_device->get_active_ethernet_cores(true)) {
                if (not tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_link_up(
                        sender_device->id(), sender_core)) {
                    continue;
                }
                auto [device_id, receiver_core] = sender_device->get_connected_ethernet_core(sender_core);
                if (receiver_device->id() != device_id) {
                    continue;
                }

                bool found = false;
                for (const auto& r : receivers) {
                    if (r == sender_core) {
                        found = true;
                        break;
                    }
                }
                if (found) {
                    continue;
                }

                senders.push_back(sender_core);
                receivers.push_back(receiver_core);
            }

            for (uint32_t erisc_idx = 0; erisc_idx < num_eriscs; erisc_idx++) {
                const auto processor = static_cast<DataMovementProcessor>(erisc_idx);
                ASSERT_TRUE(unit_tests::erisc::direct_send::eth_direct_bandwidth_kernels(
                    static_cast<MeshDispatchFixture*>(this),
                    sender_mesh_device,
                    receiver_mesh_device,
                    22000 * WORD_SIZE,
                    src_eth_l1_byte_address,
                    dst_eth_l1_byte_address,
                    senders,
                    receivers,
                    processor,
                    5500 * WORD_SIZE));
            }
        }
    }

    log_info(tt::LogTest, "senders: ");
    for (const auto& s : senders) {
        log_info(tt::LogTest, "{}", s);
    }

    log_info(tt::LogTest, "receivers: ");
    for (const auto& r : receivers) {
        log_info(tt::LogTest, "{}", r);
    }
}

}  // namespace tt::tt_metal
