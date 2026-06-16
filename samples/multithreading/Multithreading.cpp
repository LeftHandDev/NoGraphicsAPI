// Headless multithreading demo and self-test.
//
// Phase 1 (correctness): N worker threads run concurrently against one
// device, each with its own LinearAllocator, GpuQueue, semaphore and
// pipeline (compiled concurrently from the same IR, stressing the
// static-sampler dedup). Every worker repeatedly fills an input buffer,
// records and submits its own command buffer, waits, and verifies the
// result on the CPU. Exercises concurrent: gpuMalloc/gpuFree, pipeline
// creation, command recording, submission and retirement.
//
// Phase 2 (scaling): the same total number of commands is recorded by one
// thread and then by N threads in parallel (each into its own command
// buffer), timing recording only. The API adds no locks on the recording
// path; measured scaling is driver-dependent (see docs/multithreading.md).
//
// Usage: multithreading [workers] [commandsPerBuffer]
//
// Note: on devices whose descriptor sizes require the patching path (e.g.
// lavapipe), concurrently *executing* submissions share the patch
// destination heaps — see docs/multithreading.md. This sample targets
// non-patching devices (RADV and other 32-byte-descriptor hardware).
//
// Run from the build/bin directory (loads shaders/multithreading/*.spv).

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <latch>
#include <thread>
#include <vector>

#include "Multithreading.h"
#include "Utilities.h"

namespace
{

    struct WorkerResult
    {
        bool pass = true;
        char message[128] = {};
    };

    double secondsSince(std::chrono::steady_clock::time_point t0)
    {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }

} // namespace

int main(int argc, char** argv)
{
    gpuCreateInstance();

    auto desc = gpuDeviceDesc(0);
    auto device = gpuCreateDevice(0);
    if (!device)
    {
        std::printf("no usable device\n");
        return 1;
    }

    const unsigned hw = std::thread::hardware_concurrency();
    unsigned workerCount = std::min(8u, hw > 1 ? hw : 2u);
    if (argc > 1)
        workerCount = static_cast<unsigned>(std::atoi(argv[1]));
    std::printf("device: %s, workers: %u\n\n", desc.name, workerCount);

    // Shared, set up once on the main thread, read-only afterwards: the IR
    // bytes, a 1x1 white texture and the descriptor heap that points at it.
    LinearAllocator mainAllocator(device);
    LinearAllocator<MEMORY_DESCRIPTOR> descriptorAllocator(device);
    auto mainQueue = gpuCreateQueue(device);

    auto ir = loadIR("shaders/multithreading/Multithreading.spv");

    GpuTextureDesc whiteDesc{
        .type = TEXTURE_2D,
        .dimensions = { 1, 1, 1 },
        .format = FORMAT_RGBA8_UNORM,
        .usage = static_cast<USAGE_FLAGS>(USAGE_SAMPLED | USAGE_TRANSFER_DST)
    };
    GpuTextureSizeAlign sizeAlign = gpuTextureSizeAlign(device, whiteDesc);
    void* whitePtr = gpuMalloc(device, sizeAlign.size, MEMORY_GPU);
    auto whiteTexture = gpuCreateTexture(device, whiteDesc, whitePtr);

    auto upload = mainAllocator.allocate<uint32_t>(1);
    upload.cpu[0] = 0xffffffffu;

    // Texture heaps must be allocated with MEMORY_DESCRIPTOR (descriptor-buffer
    // usage + alignment); a MEMORY_DEFAULT heap faults on devices that bind the
    // heap directly (descriptor size == 32 B).
    auto textureHeap = descriptorAllocator.allocate<GpuTextureDescriptor>(16);
    textureHeap.cpu[0] = gpuTextureViewDescriptor(whiteTexture, GpuViewDesc{ .format = FORMAT_RGBA8_UNORM });

    {
        auto semaphore = gpuCreateSemaphore(device, 0);
        auto cb = gpuStartCommandRecording(mainQueue);
        gpuCopyToTexture(cb, upload.gpu, whiteTexture);
        gpuSubmit(mainQueue, Span<GpuCommandBuffer>(&cb, 1), semaphore, 1);
        gpuWaitSemaphore(semaphore, 1);
        gpuDestroySemaphore(semaphore);
    }

    // ------------------------------------------------------------------
    // Phase 1: concurrent everything, CPU-verified results.
    // ------------------------------------------------------------------
    const uint32_t elements = 1u << 16;
    const int iterations = 8;

    std::vector<WorkerResult> results(workerCount);
    {
        std::latch start(workerCount);
        std::vector<std::thread> workers;
        for (unsigned w = 0; w < workerCount; w++)
        {
            workers.emplace_back([&, w]
                                 {
                WorkerResult& result = results[w];

                LinearAllocator allocator(device);
                auto queue = gpuCreateQueue(device);
                auto semaphore = gpuCreateSemaphore(device, 0);

                start.arrive_and_wait(); // maximize overlap, including pipeline creation
                auto pipeline = gpuCreateComputePipeline(device, ByteSpan(ir.data(), ir.size()));

                auto input = allocator.allocate<float>(elements);
                auto output = allocator.allocate<float>(elements);
                auto data = allocator.allocate<WorkerData>(1);

                for (int iter = 0; iter < iterations && result.pass; iter++)
                {
                    for (uint32_t i = 0; i < elements; i++)
                    {
                        input.cpu[i] = static_cast<float>((i + iter) % 1024) * 0.25f;
                    }
                    std::memset(output.cpu, 0, elements * sizeof(float));

                    data.cpu->count = elements;
                    data.cpu->workerId = w;
                    data.cpu->srcTexture = 0;
                    data.cpu->input = input.gpu;
                    data.cpu->output = output.gpu;

                    auto cb = gpuStartCommandRecording(queue);
                    gpuSetPipeline(cb, pipeline);
                    gpuSetActiveTextureHeapPtr(cb, textureHeap.gpu);
                    gpuDispatch(cb, data.gpu, { (elements + 63) / 64, 1, 1 });
                    gpuSubmit(queue, Span<GpuCommandBuffer>(&cb, 1), semaphore, static_cast<uint64_t>(iter) + 1);
                    gpuWaitSemaphore(semaphore, static_cast<uint64_t>(iter) + 1);

                    for (uint32_t i = 0; i < elements; i++)
                    {
                        const float fill = static_cast<float>((i + iter) % 1024) * 0.25f;
                        const float expected = fill + static_cast<float>(w);
                        if (input.cpu[i] != fill)
                        {
                            std::snprintf(result.message, sizeof(result.message),
                                          "worker %u iter %d: INPUT TRAMPLED input[%u]=%f (fill %f, output %f)",
                                          w, iter, i, input.cpu[i], fill, output.cpu[i]);
                            result.pass = false;
                            break;
                        }
                        if (output.cpu[i] != expected)
                        {
                            std::snprintf(result.message, sizeof(result.message),
                                          "worker %u iter %d: output[%u]=%f expected %f (input intact)",
                                          w, iter, i, output.cpu[i], expected);
                            result.pass = false;
                            break;
                        }
                    }
                }

                gpuFreePipeline(pipeline);
                gpuDestroySemaphore(semaphore);
                gpuDestroyQueue(queue);
                allocator.reset(); });
        }
        for (auto& t : workers)
        {
            t.join();
        }
    }

    bool pass = true;
    for (auto& result : results)
    {
        if (!result.pass)
        {
            std::printf("FAIL: %s\n", result.message);
            pass = false;
        }
    }
    std::printf("phase 1 (correctness, %u workers x %d iterations): %s\n", workerCount, iterations, pass ? "PASS" : "FAIL");

    // ------------------------------------------------------------------
    // Phase 2: recording scalability. Same total command count, recorded by
    // one thread vs by all workers in parallel. Recording time only.
    // ------------------------------------------------------------------
    const int commandsPerBuffer = argc > 2 ? std::atoi(argv[2]) : 2048;
    auto pipeline = gpuCreateComputePipeline(device, ByteSpan(ir.data(), ir.size()));
    auto scalingData = mainAllocator.allocate<WorkerData>(1);
    scalingData.cpu->count = 64;
    scalingData.cpu->workerId = 0;
    scalingData.cpu->srcTexture = 0;
    auto scratch = mainAllocator.allocate<float>(64);
    scalingData.cpu->input = scratch.gpu;
    scalingData.cpu->output = scratch.gpu;

    auto recordOne = [&](GpuQueue queue)
    {
        auto cb = gpuStartCommandRecording(queue);
        gpuSetPipeline(cb, pipeline);
        gpuSetActiveTextureHeapPtr(cb, textureHeap.gpu);
        for (int i = 0; i < commandsPerBuffer; i++)
        {
            gpuDispatch(cb, scalingData.gpu, { 1, 1, 1 });
            gpuBarrier(cb, STAGE_COMPUTE, STAGE_COMPUTE);
        }
        return cb;
    };

    auto scalingSemaphore = gpuCreateSemaphore(device, 0);
    uint64_t tick = 0;

    // Single-threaded reference: one thread records workerCount buffers.
    std::vector<GpuCommandBuffer> recorded(workerCount);
    auto t0 = std::chrono::steady_clock::now();
    for (unsigned w = 0; w < workerCount; w++)
    {
        recorded[w] = recordOne(mainQueue);
    }
    const double singleSeconds = secondsSince(t0);
    gpuSubmit(mainQueue, Span<GpuCommandBuffer>(recorded.data(), recorded.size()), scalingSemaphore, ++tick);
    gpuWaitSemaphore(scalingSemaphore, tick);

    // Parallel: every worker records one buffer concurrently.
    {
        std::latch start(workerCount);
        std::atomic<double> parallelSeconds = 0.0;
        std::vector<std::thread> workers;
        auto wall0 = std::chrono::steady_clock::now();
        for (unsigned w = 0; w < workerCount; w++)
        {
            workers.emplace_back([&, w]
                                 {
                start.arrive_and_wait();
                recorded[w] = recordOne(mainQueue); });
        }
        for (auto& t : workers)
        {
            t.join();
        }
        const double wallSeconds = secondsSince(wall0);
        gpuSubmit(mainQueue, Span<GpuCommandBuffer>(recorded.data(), recorded.size()), scalingSemaphore, ++tick);
        gpuWaitSemaphore(scalingSemaphore, tick);

        std::printf("phase 2 (recording %u x %d commands): 1 thread %.1f ms, %u threads %.1f ms, speedup %.2fx\n",
                    workerCount, commandsPerBuffer, singleSeconds * 1000.0, workerCount, wallSeconds * 1000.0,
                    singleSeconds / wallSeconds);
    }

    gpuDestroySemaphore(scalingSemaphore);
    gpuFreePipeline(pipeline);
    mainAllocator.reset();
    descriptorAllocator.reset();
    gpuDestroyTexture(whiteTexture);
    gpuFree(device, whitePtr);
    gpuDestroyQueue(mainQueue);
    gpuDestroyDevice(device);
    gpuDestroyInstance();
    return pass ? 0 : 1;
}
