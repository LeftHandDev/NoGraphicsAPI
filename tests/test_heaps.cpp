// Headless texture-heap checks (no golden image). On descriptor-patching
// devices (descriptor size != 32, e.g. lavapipe in CI) every
// gpuSetActiveTextureHeapPtr runs a patch dispatch, and every view descriptor
// takes a slot in a device-global array; on direct-bind devices the same checks
// must pass trivially.
#include "test_common.h"

#include "Utilities.h"          // LinearAllocator, loadIR
#include "shaders/HeapProbe.h" // HeapProbeData

#include <cmath>
#include <cstring>
#include <iostream>
#include <string>

namespace
{
    int failures = 0;

    void check(bool ok, const std::string& what)
    {
        std::cout << (ok ? "  ok   " : "  FAIL ") << what << "\n";
        if (!ok)
        {
            failures++;
        }
    }

    bool near(float4 a, float4 b)
    {
        return std::fabs(a.x - b.x) < 0.01f && std::fabs(a.y - b.y) < 0.01f &&
               std::fabs(a.z - b.z) < 0.01f && std::fabs(a.w - b.w) < 0.01f;
    }

    struct Texture
    {
        GpuTexture texture = nullptr;
        void* memory = nullptr;
    };

    Texture createTexel(GpuDevice device)
    {
        GpuTextureDesc desc = {
            .type = TEXTURE_2D,
            .dimensions = { 1, 1, 1 },
            .format = FORMAT_RGBA8_UNORM,
            .usage = static_cast<USAGE_FLAGS>(USAGE_SAMPLED | USAGE_TRANSFER_DST)
        };
        Texture t;
        t.memory = gpuMalloc(device, gpuTextureSizeAlign(device, desc).size, MEMORY_GPU);
        t.texture = gpuCreateTexture(device, desc, t.memory);
        return t;
    }

    void destroyTexel(GpuDevice device, Texture t)
    {
        gpuDestroyTexture(t.texture);
        gpuFree(device, t.memory);
    }
} // namespace

int main(int argc, char** argv)
{
    test::Args args = test::parseArgs(argc, argv);

    gpuCreateInstance();
    test::beginValidationCapture();

    auto device = gpuCreateDevice(args.device);
    if (!device)
    {
        std::cerr << "FAIL [heaps]: no suitable device at index " << args.device << "\n";
        return 1;
    }
    auto queue = gpuCreateQueue(device);
    LinearAllocator allocator(device);

    auto probeIR = loadIR(std::string(NGAPI_TEST_SHADER_DIR) + "/tests/HeapProbe.spv");
    auto probe = gpuCreateComputePipeline(device, ByteSpan(probeIR.data(), probeIR.size()));

    // Two 1x1 textures, red and green, each in slot 0 of its own heap.
    Texture red = createTexel(device);
    Texture green = createTexel(device);
    auto texels = allocator.allocate<uint32_t>(2);
    texels.cpu[0] = 0xff0000ffu; // RGBA8 red   (little-endian: R in the low byte)
    texels.cpu[1] = 0xff00ff00u; // RGBA8 green

    constexpr uint32_t HEAP_SIZE = 16;
    auto heapA = static_cast<GpuTextureDescriptor*>(gpuMalloc(device, sizeof(GpuTextureDescriptor) * HEAP_SIZE, MEMORY_DESCRIPTOR));
    auto heapB = static_cast<GpuTextureDescriptor*>(gpuMalloc(device, sizeof(GpuTextureDescriptor) * HEAP_SIZE, MEMORY_DESCRIPTOR));
    memset(heapA, 0, sizeof(GpuTextureDescriptor) * HEAP_SIZE);
    memset(heapB, 0, sizeof(GpuTextureDescriptor) * HEAP_SIZE);
    heapA[0] = gpuTextureViewDescriptor(red.texture, {});
    heapB[0] = gpuTextureViewDescriptor(green.texture, {});

    // Asking for a texture's descriptor repeatedly must not use up the
    // descriptor slots (each call used to take a new one, with no bounds
    // check, overflowing the slot array after 1024 calls).
    for (uint32_t i = 0; i < 4096; i++)
    {
        heapA[1] = gpuTextureViewDescriptor(red.texture, {});
    }

    // Destroyed textures must give their slots back.
    for (uint32_t i = 0; i < 1100; i++)
    {
        Texture scratch = createTexel(device);
        gpuTextureViewDescriptor(scratch.texture, {});
        destroyTexel(device, scratch);
    }

    auto results = allocator.allocate<float4>(3);
    auto probes = allocator.allocate<HeapProbeData>(3);
    probes.cpu[0] = { results.gpu + 0, 0 }; // heap A, slot 0: red
    probes.cpu[1] = { results.gpu + 1, 0 }; // heap B, slot 0: green
    probes.cpu[2] = { results.gpu + 2, 1 }; // heap A, slot 1: red again

    auto semaphore = gpuCreateSemaphore(device, 0);
    auto cb = gpuStartCommandRecording(queue);
    gpuCopyToTexture(cb, texels.gpu + 0, red.texture);
    gpuCopyToTexture(cb, texels.gpu + 1, green.texture);
    gpuBarrier(cb, STAGE_TRANSFER, STAGE_COMPUTE);

    // Bind heap A before any pipeline is set, then B in the same command
    // buffer: each dispatch must see its own heap.
    gpuSetActiveTextureHeapPtr(cb, gpuHostToDevicePointer(device, heapA));
    gpuSetPipeline(cb, probe);
    gpuDispatch(cb, probes.gpu + 0, { 1, 1, 1 });
    gpuSetActiveTextureHeapPtr(cb, gpuHostToDevicePointer(device, heapB));
    gpuDispatch(cb, probes.gpu + 1, { 1, 1, 1 });
    gpuSetActiveTextureHeapPtr(cb, gpuHostToDevicePointer(device, heapA));
    gpuDispatch(cb, probes.gpu + 2, { 1, 1, 1 });

    gpuSubmit(queue, Span<GpuCommandBuffer>(&cb, 1), semaphore, 1);
    gpuWaitSemaphore(semaphore, 1);

    const float4 redColor = { 1, 0, 0, 1 };
    const float4 greenColor = { 0, 1, 0, 1 };
    check(near(results.cpu[0], redColor), "first heap bind in a command buffer sees its own heap (A: red)");
    check(near(results.cpu[1], greenColor), "second heap bind in the same command buffer sees its own heap (B: green)");
    check(near(results.cpu[2], redColor), "rebinding the first heap after the second (A again: red)");
    check(!test::validationFailed(), "no validation messages");

    gpuDestroySemaphore(semaphore);
    gpuFree(device, heapA);
    gpuFree(device, heapB);
    destroyTexel(device, red);
    destroyTexel(device, green);
    gpuFreePipeline(probe);
    allocator.reset();
    gpuDestroyQueue(queue);
    gpuDestroyDevice(device);
    test::endValidationCapture();
    gpuDestroyInstance();

    if (test::validationFailed())
    {
        std::cerr << "FAIL [heaps]: Vulkan validation messages were emitted\n";
        return 1;
    }
    if (failures > 0)
    {
        std::cerr << "FAIL [heaps]: " << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "PASS [heaps]\n";
    return 0;
}
