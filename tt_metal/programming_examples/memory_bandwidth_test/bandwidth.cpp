// SPDX-FileCopyrightText: © 2023-2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <random>
#include <cstdint>

#include <fmt/ostream.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

// This example demonstrates a simple data copy from DRAM into L1(SRAM) and to
// another place in DRAM.  The general flow is as follows:
// 1. Initialize the device
// 2. Create the data movement kernel (fancy word of specialized subroutines)
// on core {0, 0} that will perform the copy
// 3. Create the buffer (both on DRAM And L1) and fill DRAM with data. Point
// the kernel to the buffers.
// 4. Execute the kernel
// 5. Read the data back from the buffer
// 6. Validate the data
// 7. Clean up the device. Exit

using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif
int main() {
    const uint64_t GiB = 1 << 30;
    const uint64_t MiB = 1 << 20;
    try {
        // Create a 1x1 mesh on device 0 (same API scales to multi-device
        // meshes)
        constexpr int device_id = 0;
        std::shared_ptr<MeshDevice> mesh_device = MeshDevice::create_unit_mesh(device_id);

        // Submit work via the mesh command queue: uploads/downloads and
        // program execution.
        MeshCommandQueue& cq = mesh_device->mesh_command_queue();

        const uint64_t num_dram_channels = 8;
        const uint32_t num_l1_buffers = 2;  // 256 / 8;
        constexpr uint32_t num_iter = 1024 / 8;
        constexpr uint32_t num_tiles = 1024;
        constexpr uint32_t elements_per_tile = MiB / 4 / sizeof(uint32_t);
        // constexpr uint32_t elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT * 8;
        constexpr uint32_t tile_size_bytes = sizeof(uint32_t) * elements_per_tile;
        constexpr uint32_t dram_buffer_size = tile_size_bytes * num_tiles;

        // Configure mesh buffers. Use single-tile page size so transfers
        // operate tile-by-tile.
        DeviceLocalBufferConfig dram_config{
            // Number of bytes when round-robin between banks. Usually this is
            // the same as the tile size for efficiency.
            .page_size = tile_size_bytes,
            // Type of buffer (DRAM or L1(SRAM))
            .buffer_type = tt::tt_metal::BufferType::DRAM,
        };

        DeviceLocalBufferConfig l1_config{
            .page_size = tile_size_bytes,
            // This time we allocate on L1
            .buffer_type = tt::tt_metal::BufferType::L1,
        };

        ReplicatedBufferConfig dram_buffer_config{
            // Size per device (replicated across mesh). Since we are
            // operating on a unit mesh this is the total size.
            .size = dram_buffer_size,
        };

        ReplicatedBufferConfig l1_buffer_config{
            .size = tile_size_bytes,
        };

        // Allocate the buffers (replicated across mesh;
        // on unit mesh ⇒ single device allocation)
        auto l1_buffer = MeshBuffer::create(l1_buffer_config, l1_config, mesh_device.get());
        std::vector<std::shared_ptr<MeshBuffer>> l1_buffers = {};

        for (int i = 0; i < num_l1_buffers; i++) {
            l1_buffers.push_back(MeshBuffer::create(l1_buffer_config, l1_config, mesh_device.get()));
        }

        auto input_dram_buffer = MeshBuffer::create(dram_buffer_config, dram_config, mesh_device.get());

        auto output_dram_buffer = MeshBuffer::create(dram_buffer_config, dram_config, mesh_device.get());

        // A program is a collection of kernels. Note that unlike OpenCL/CUDA
        // where every core must run the same kernel at a given time. Metalium
        // allows you to run different kernels on different cores
        // simultaneously.
        Program program = CreateProgram();

        // A MeshWorkload is a collection of programs that will be executed on
        // the mesh. Each workload is local to a single device. Here we create
        // a workload for our single-device mesh.
        MeshWorkload workload;
        MeshCoordinateRange device_range = MeshCoordinateRange(mesh_device->shape());

        // This example program will only use 1 Tensix core. So we set the core
        // to {0, 0} (the most top-left core).
        // constexpr CoreCoord core = {0, 0};
        auto corelist = mesh_device->get_optimal_dram_bank_to_logical_worker_assignment(NOC::NOC_0);
        // auto corelist = std::vector<CoreCoord>{
        //     {0, 1},
        //     {0, 4},
        //     {0, 6},
        //     {0, 9},
        //     {11, 0},
        //     {11, 3},
        //     {11, 6},
        //     {11, 9},
        // };
        if (num_dram_channels > corelist.size()) {
            exit(1);
        }
        auto cores = CoreRangeSet(std::span(corelist.data(), num_dram_channels));

        auto drams = mesh_device->get_optimal_dram_bank_to_logical_worker_assignment(NOC::NOC_0);
        fmt::print("{}\n", drams);

        // Create the data movement kernel. This kernel will be used to copy
        // data from DRAM to DRAM (see the `loopback_dram_copy.cpp` file for
        // the actual implementation). The kernel is created on the Tensix core
        // {0, 0} and uses the default NoC.
        std::vector<uint32_t> dram_copy_compile_time_args;

        TensorAccessorArgs(*input_dram_buffer->get_backing_buffer()).append_to(dram_copy_compile_time_args);

        TensorAccessorArgs(*output_dram_buffer->get_backing_buffer()).append_to(dram_copy_compile_time_args);

        KernelHandle dram_copy_kernel_id = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "memory_bandwidth_test/kernels/loopback_dram_copy.cpp",
            cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = dram_copy_compile_time_args,
            });

        // Initialize the input buffer with random data.
        std::vector<uint32_t> input_vec(elements_per_tile * num_tiles);
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> distribution(0.0f, 100.0f);
        for (auto& val : input_vec) {
            val = bfloat16(distribution(rng));
        }

        for (int i = 0; i < l1_buffers.size(); i++) {
            input_vec[i] = l1_buffers[i]->address();
        }

        // Upload the data from host to the device. The final argument is set
        // to false. This indicates to Metalium that the upload is
        // non-blocking, an upload will be launched, but the function will
        // return immediately, before the upload is complete. This is useful
        // for performance reasons, as it allows the host to continue while the
        // upload is in progress. Note that the host is responsible for
        // ensuring that the upload is complete before the memory holding the
        // data is freed.
        EnqueueWriteMeshBuffer(
            cq,
            input_dram_buffer,
            input_vec,
            /*blocking=*/false);

        for (uint32_t kernel_id = 0; kernel_id < num_dram_channels; kernel_id++) {
            // Set runtime arguments for the kernel.
            std::vector<uint32_t> runtime_args = {
                l1_buffer->address(),
                input_dram_buffer->address(),
                output_dram_buffer->address(),
                num_tiles,
                num_l1_buffers,
                num_iter,
                tile_size_bytes,
                kernel_id,
            };

            SetRuntimeArgs(program, dram_copy_kernel_id, corelist[kernel_id], runtime_args);
        }

        // Add the program to the workload for the mesh.
        workload.add_program(device_range, std::move(program));
        // Enqueue the workload for execution on the mesh (non-blocking) and
        // wait for completion before reading back.
        auto start = std::chrono::high_resolution_clock::now();
        distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
        distributed::Finish(cq);
        auto stop = std::chrono::high_resolution_clock::now();
        // NOTE: The above is equivalent to a blocking enqueue of the workload.

        // Read the result back from the shard at mesh coordinate {0,0}. Use
        // blocking=true to wait for completion.  The vector is automatically
        // resized to fit the data.
        std::vector<uint32_t> result_vec;
        distributed::EnqueueReadMeshBuffer(
            cq,
            result_vec,
            output_dram_buffer,
            /*blocking*/ true);

        // Compare the result with the input. The result should be the same as
        // the input.
        TT_FATAL(
            result_vec.size() == input_vec.size(),
            "Result vector size {} does not match input vector size {}",
            result_vec.size(),
            input_vec.size());

        // Close the device
        mesh_device->close();

        uint64_t bytes_read = num_dram_channels == 1
                                  ? result_vec[0] + ((uint64_t)result_vec[1] << 32)
                                  : num_dram_channels * num_iter * num_l1_buffers * num_tiles * tile_size_bytes;
        double duration = duration_cast<std::chrono::nanoseconds>(stop - start).count() / 1e9;
        fmt::print(" {} MiB bytes read in {:.5f}s\n", bytes_read / (float)MiB, duration);
        fmt::print("{:.4f} GiB/s\n", bytes_read / duration / GiB);

    } catch (const std::exception& e) {
        fmt::print(stderr, "Test failed with exception! what: {}\n", e.what());
        throw;
    }

    return 0;
}
