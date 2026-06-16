// Headless sampler-declaration benchmark (scratch regression tool).
//
// Runs the identical tap loop (SamplerBench.slang) through the raw hardware
// default sampler, a STATIC_SAMPLER (spec-constant specialization) and an
// INLINE_SAMPLER (patched literal) and reports per-dispatch times — all three
// must be equal. Cross-checks: static and inline with identical state must
// dedup to the same sampler (byte-identical output), and a NEAREST/CLAMP
// static sampler must match a library-free Load-based reference exactly.
//
// Run from the build/bin directory (loads shaders/samplerbench/*.spv).

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "SamplerBench.h"
#include "Utilities.h"

namespace
{

    // Deterministic noise so the texture has no compressible structure.
    uint32_t hash32(uint32_t x)
    {
        x ^= x >> 16;
        x *= 0x7feb352du;
        x ^= x >> 15;
        x *= 0x846ca68bu;
        x ^= x >> 16;
        return x;
    }

    struct BatchTimer
    {
        GpuDevice device;
        GpuQueue queue;
        GpuSemaphore semaphore;
        uint64_t tick = 0;

        BatchTimer(GpuDevice device, GpuQueue queue)
            : device(device), queue(queue), semaphore(gpuCreateSemaphore(device, 0))
        {
        }
        ~BatchTimer()
        {
            gpuDestroySemaphore(semaphore);
        }

        void runBatch(GpuPipeline pipeline, void* heapGpu, void* dataGpu, uint3 grid, int dispatches)
        {
            auto cb = gpuStartCommandRecording(queue);
            for (int i = 0; i < dispatches; i++)
            {
                gpuSetPipeline(cb, pipeline);
                gpuSetActiveTextureHeapPtr(cb, heapGpu);
                gpuDispatch(cb, dataGpu, grid);
                gpuBarrier(cb, STAGE_COMPUTE, STAGE_COMPUTE);
            }
            gpuSubmit(queue, Span<GpuCommandBuffer>(&cb, 1), semaphore, ++tick);
            gpuWaitSemaphore(semaphore, tick);
        }

        // Returns milliseconds per dispatch (1 warmup batch, `repeats` timed ones).
        double time(GpuPipeline pipeline, void* heapGpu, void* dataGpu, uint3 grid, int dispatches, int repeats)
        {
            runBatch(pipeline, heapGpu, dataGpu, grid, dispatches);
            auto t0 = std::chrono::steady_clock::now();
            for (int rep = 0; rep < repeats; rep++)
                runBatch(pipeline, heapGpu, dataGpu, grid, dispatches);
            auto t1 = std::chrono::steady_clock::now();
            return std::chrono::duration<double, std::milli>(t1 - t0).count() / (dispatches * repeats);
        }
    };

    std::vector<uint8_t> readbackRGBA8(GpuDevice device, GpuQueue queue, GpuTexture texture, size_t bytes)
    {
        uint8_t* readback = static_cast<uint8_t*>(gpuMalloc(device, bytes, MEMORY_READBACK));
        auto semaphore = gpuCreateSemaphore(device, 0);
        auto cb = gpuStartCommandRecording(queue);
        gpuCopyFromTexture(cb, gpuHostToDevicePointer(device, readback), texture);
        gpuSubmit(queue, Span<GpuCommandBuffer>(&cb, 1), semaphore, 1);
        gpuWaitSemaphore(semaphore, 1);
        std::vector<uint8_t> result(readback, readback + bytes);
        gpuDestroySemaphore(semaphore);
        gpuFree(device, readback);
        return result;
    }

} // namespace

int main()
{
    gpuCreateInstance();

    auto desc = gpuDeviceDesc(0);
    auto device = gpuCreateDevice(0);
    if (!device)
    {
        std::printf("no usable device\n");
        return 1;
    }
    std::printf("device: %s (%s)\n\n", desc.name, desc.discrete ? "discrete" : "integrated");

    LinearAllocator allocator(device);
    LinearAllocator<MEMORY_DESCRIPTOR> descriptorAllocator(device);
    auto queue = gpuCreateQueue(device);

    auto benchIR = loadIR("shaders/samplerbench/Bench.spv");
    auto hwPipeline = gpuCreateComputePipeline(device, ByteSpan(benchIR.data(), benchIR.size()), "benchHardware");
    auto staticPipeline = gpuCreateComputePipeline(device, ByteSpan(benchIR.data(), benchIR.size()), "benchStatic");
    auto inlinePipeline = gpuCreateComputePipeline(device, ByteSpan(benchIR.data(), benchIR.size()), "benchInline");
    auto staticNearestPipeline = gpuCreateComputePipeline(device, ByteSpan(benchIR.data(), benchIR.size()), "benchStaticNearest");
    auto manualNearestPipeline = gpuCreateComputePipeline(device, ByteSpan(benchIR.data(), benchIR.size()), "benchManualNearest");

    const uint32_t width = 1024, height = 1024;
    const size_t textureBytes = static_cast<size_t>(width) * height * 4;

    auto textureHeap = descriptorAllocator.allocate<GpuTextureDescriptor>(16);

    auto upload = allocator.allocate<uint32_t>(width * height);
    for (uint32_t i = 0; i < width * height; i++)
        upload.cpu[i] = hash32(i);

    GpuTextureDesc srcDesc{
        .type = TEXTURE_2D,
        .dimensions = { width, height, 1 },
        .format = FORMAT_RGBA8_UNORM,
        .usage = static_cast<USAGE_FLAGS>(USAGE_SAMPLED | USAGE_TRANSFER_DST)
    };
    GpuTextureSizeAlign sizeAlign = gpuTextureSizeAlign(device, srcDesc);
    void* srcPtr = gpuMalloc(device, sizeAlign.size, MEMORY_GPU);
    auto srcTexture = gpuCreateTexture(device, srcDesc, srcPtr);

    GpuTextureDesc dstDesc{
        .type = TEXTURE_2D,
        .dimensions = { width, height, 1 },
        .format = FORMAT_RGBA8_UNORM,
        .usage = static_cast<USAGE_FLAGS>(USAGE_STORAGE | USAGE_TRANSFER_SRC)
    };
    void* dstPtr = gpuMalloc(device, sizeAlign.size, MEMORY_GPU);
    auto dstTexture = gpuCreateTexture(device, dstDesc, dstPtr);

    textureHeap.cpu[0] = gpuTextureViewDescriptor(srcTexture, GpuViewDesc{ .format = FORMAT_RGBA8_UNORM });
    textureHeap.cpu[1] = gpuRWTextureViewDescriptor(dstTexture, GpuViewDesc{ .format = FORMAT_RGBA8_UNORM });

    {
        auto semaphore = gpuCreateSemaphore(device, 0);
        auto cb = gpuStartCommandRecording(queue);
        gpuCopyToTexture(cb, upload.gpu, srcTexture);
        gpuSubmit(queue, Span<GpuCommandBuffer>(&cb, 1), semaphore, 1);
        gpuWaitSemaphore(semaphore, 1);
        gpuDestroySemaphore(semaphore);
    }

    auto data = allocator.allocate<SamplerBenchData>(1);
    data.cpu->imageSize = { width, height };
    data.cpu->srcTexture = 0;
    data.cpu->dstTexture = 1;

    const uint3 grid = { (width + 15) / 16, (height + 15) / 16, 1 };

    // Scoped so the timer's semaphore dies before the device teardown below.
    {
        BatchTimer timer(device, queue);

        auto compare = [&](GpuPipeline a, GpuPipeline b, const char* what, int allowed)
        {
            data.cpu->kernelRadius = 2;
            timer.runBatch(a, textureHeap.gpu, data.gpu, grid, 1);
            auto imageA = readbackRGBA8(device, queue, dstTexture, textureBytes);
            timer.runBatch(b, textureHeap.gpu, data.gpu, grid, 1);
            auto imageB = readbackRGBA8(device, queue, dstTexture, textureBytes);
            int maxDiff = 0;
            for (size_t i = 0; i < textureBytes; i++)
                maxDiff = std::max(maxDiff, std::abs(int(imageA[i]) - int(imageB[i])));
            std::printf("%-34s maxDiff=%d/255 %s\n", what, maxDiff, maxDiff <= allowed ? "(ok)" : "(BROKEN)");
            return maxDiff <= allowed;
        };

        bool ok = true;
        ok &= compare(staticPipeline, inlinePipeline, "static vs inline (same state):", 0);
        ok &= compare(staticNearestPipeline, manualNearestPipeline, "static NEAREST vs manual Load:", 0);
        std::printf("\n");

        std::printf("%-28s %12s %15s %15s %12s %12s\n", "config", "hw ms", "static ms", "inline ms", "static/hw", "inline/hw");
        for (int radius : { 0, 4, 16 })
        {
            data.cpu->kernelRadius = radius;
            const int taps = (2 * radius + 1) * (2 * radius + 1);
            // Keep each batch around the same total work so runtimes stay sane.
            const int dispatches = std::max(2, 512 / taps);
            const int repeats = 3;

            double hwMs = timer.time(hwPipeline, textureHeap.gpu, data.gpu, grid, dispatches, repeats);
            double staticMs = timer.time(staticPipeline, textureHeap.gpu, data.gpu, grid, dispatches, repeats);
            double inlineMs = timer.time(inlinePipeline, textureHeap.gpu, data.gpu, grid, dispatches, repeats);

            char label[64];
            std::snprintf(label, sizeof(label), "r=%-2d (%4d taps/pixel)", radius, taps);
            std::printf("%-28s %9.3f ms %12.3f ms %12.3f ms %11.2fx %11.2fx\n",
                        label, hwMs, staticMs, inlineMs, staticMs / hwMs, inlineMs / hwMs);
        }

        if (!ok)
        {
            std::printf("\nCROSS-CHECK FAILURE\n");
        }
    } // timer scope

    allocator.reset();
    descriptorAllocator.reset();
    gpuDestroyTexture(srcTexture);
    gpuDestroyTexture(dstTexture);
    gpuFree(device, srcPtr);
    gpuFree(device, dstPtr);
    gpuFreePipeline(hwPipeline);
    gpuFreePipeline(staticPipeline);
    gpuFreePipeline(inlinePipeline);
    gpuFreePipeline(staticNearestPipeline);
    gpuFreePipeline(manualNearestPipeline);
    gpuDestroyQueue(queue);
    gpuDestroyDevice(device);
    gpuDestroyInstance();
    return 0;
}
