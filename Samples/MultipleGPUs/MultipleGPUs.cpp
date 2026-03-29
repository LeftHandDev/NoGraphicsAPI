#include "MultipleGPUs.h"

#include <iostream>
#include <vector>
#include <chrono>

void multipleGPUsSample()
{
    gpuCreateInstance();

    std::vector<std::pair<GpuDevice, GpuDeviceDesc>> devices;
    for (uint i = 0; i < gpuDeviceCount(); i++)
    {
        auto device = gpuCreateDevice(i);
        auto desc = gpuDeviceDesc(i);
        devices.push_back({device, desc});
    }

    // 1 MB round trip transfer test for each device
    for (uint i = 0; i < devices.size(); i++)
    {
        auto& [device, desc] = devices[i];

        const auto size = 1024 * 1024;
        auto upload = gpuMalloc(device, size, MEMORY_DEFAULT);
        auto gpu = gpuMalloc(device, size, MEMORY_GPU);
        auto readback = gpuMalloc(device, size, MEMORY_READBACK);

        auto queue = gpuCreateQueue(device);
        auto cmd = gpuStartCommandRecording(queue);
        auto semaphore = gpuCreateSemaphore(device, 0);

        auto start = std::chrono::high_resolution_clock::now();

        gpuMemCpy(cmd, gpu, gpuHostToDevicePointer(device, upload), size);
        gpuBarrier(cmd, STAGE_TRANSFER, STAGE_TRANSFER);
        gpuMemCpy(cmd, gpuHostToDevicePointer(device, readback), gpu, size);
        gpuSubmit(queue, Span<GpuCommandBuffer>(&cmd, 1), semaphore, 1);
        gpuWaitSemaphore(semaphore, 1);

        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "Device " << i << " (" << desc.name << "): " << duration.count() << " ms" << std::endl;

        gpuFree(device, upload);
        gpuFree(device, gpu);
        gpuFree(device, readback);
        gpuDestroySemaphore(semaphore);
        gpuDestroyQueue(queue);
    }

    for (auto& [device, desc] : devices)
    {
        gpuDestroyDevice(device);
    }

    gpuDestroyInstance();
}