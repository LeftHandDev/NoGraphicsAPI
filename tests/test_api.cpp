// Headless API contract checks. Unlike the rendering tests there is no golden
// image: each check exercises one behaviour of the API directly, and the test
// fails if any check does or if the validation layer reports anything.
#include "test_common.h"

#include <cstdint>
#include <iostream>

namespace
{
    int failures = 0;

    void check(bool ok, const char* what)
    {
        std::cout << (ok ? "  ok   " : "  FAIL ") << what << "\n";
        if (!ok)
        {
            failures++;
        }
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
        std::cerr << "FAIL [api]: no suitable device at index " << args.device << "\n";
        return 1;
    }
    auto queue = gpuCreateQueue(device);

    // Every STAGE x STAGE x HAZARD combination must record a valid barrier: an
    // access type paired with a stage that can't perform it is rejected by the
    // validation layer and synchronizes nothing.
    {
        const STAGE stages[] = {
            STAGE_TRANSFER,
            STAGE_COMPUTE,
            STAGE_RASTER_COLOR_OUT,
            STAGE_PIXEL_SHADER,
            STAGE_VERTEX_SHADER,
            STAGE_ACCELERATION_STRUCTURE_BUILD
        };
        const uint32_t allHazards = HAZARD_DRAW_ARGUMENTS | HAZARD_DESCRIPTORS | HAZARD_DEPTH_STENCIL | HAZARD_ACCELERATION_STRUCTURE;

        auto semaphore = gpuCreateSemaphore(device, 0);
        auto cb = gpuStartCommandRecording(queue);
        for (STAGE before : stages)
        {
            for (STAGE after : stages)
            {
                for (uint32_t hazards = 0; hazards <= allHazards; hazards++)
                {
                    gpuBarrier(cb, before, after, static_cast<HAZARD_FLAGS>(hazards));
                }
            }
        }
        gpuSubmit(queue, Span<GpuCommandBuffer>(&cb, 1), semaphore, 1);
        gpuWaitSemaphore(semaphore, 1);
        gpuDestroySemaphore(semaphore);
        check(!test::validationFailed(), "gpuBarrier: every STAGE x STAGE x HAZARD combination is valid");
    }

    // gpuFree(nullptr) must be a no-op. GPU-only allocations have no host
    // pointer, so a null free used to match (and free) the first of them.
    {
        constexpr uint32_t count = 64;
        constexpr size_t bytes = count * sizeof(uint32_t);
        void* gpuOnly = gpuMalloc(device, bytes, MEMORY_GPU);
        auto upload = static_cast<uint32_t*>(gpuMalloc(device, bytes));
        auto readback = static_cast<uint32_t*>(gpuMalloc(device, bytes, MEMORY_READBACK));
        for (uint32_t i = 0; i < count; i++)
        {
            upload[i] = 0xC0FFEE00u + i;
        }

        gpuFree(device, nullptr);

        // Round-trip through the GPU-only buffer; it must still exist.
        auto semaphore = gpuCreateSemaphore(device, 0);
        auto cb = gpuStartCommandRecording(queue);
        gpuMemCpy(cb, gpuOnly, gpuHostToDevicePointer(device, upload), bytes);
        gpuBarrier(cb, STAGE_TRANSFER, STAGE_TRANSFER);
        gpuMemCpy(cb, gpuHostToDevicePointer(device, readback), gpuOnly, bytes);
        gpuSubmit(queue, Span<GpuCommandBuffer>(&cb, 1), semaphore, 1);
        gpuWaitSemaphore(semaphore, 1);
        gpuDestroySemaphore(semaphore);

        bool intact = true;
        for (uint32_t i = 0; i < count; i++)
        {
            intact = intact && readback[i] == 0xC0FFEE00u + i;
        }
        check(intact && !test::validationFailed(), "gpuFree(nullptr) leaves other allocations alone");

        gpuFree(device, readback);
        gpuFree(device, upload);
        gpuFree(device, gpuOnly);
    }

    // gpuWaitSemaphore must report a timeout rather than return as if the work
    // had completed (it used to recycle the in-flight command pools, too).
    {
        auto semaphore = gpuCreateSemaphore(device, 0);
        check(gpuWaitSemaphore(semaphore, 1, 0) == RESULT_FAILURE, "gpuWaitSemaphore: an unsignaled value times out");

        auto cb = gpuStartCommandRecording(queue);
        gpuSubmit(queue, Span<GpuCommandBuffer>(&cb, 1), semaphore, 1);
        check(gpuWaitSemaphore(semaphore, 1) == RESULT_SUCCESS, "gpuWaitSemaphore: a signaled value succeeds");
        gpuDestroySemaphore(semaphore);
    }

    gpuDestroyQueue(queue);
    gpuDestroyDevice(device);
    test::endValidationCapture();
    gpuDestroyInstance();

    if (test::validationFailed())
    {
        std::cerr << "FAIL [api]: Vulkan validation messages were emitted\n";
        return 1;
    }
    if (failures > 0)
    {
        std::cerr << "FAIL [api]: " << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "PASS [api]\n";
    return 0;
}
