# No Graphics API

A Vulkan+slang implementation of the simplified graphics API from Sebastian Aaltonen's blog post: [No Graphics API](https://www.sebastianaaltonen.com/blog/no-graphics-api).

## Instructions
Update VulkanSDK to 1.4.335 (install glm and SDL with the SDK)

## Simple Usage (No Surfaces)
### Common header
```c++
#include "../../Shaders/NoGraphicsAPI.h" // Must be included

struct alignas(16) Data
{
    float multiplier;

    float* input;
    float* output;
};
```
### Compute Shader (slang)
```c++
#include "Common.h"

[numthreads(16, 1, 1)]
void main(uint3 threadId: SV_DispatchThreadID, Data* data)
{
    data->output[threadId.x] = data->multiplier * data->input[threadId.x];
}
```
### CPU side
```c++
if (gpuCreateDevice() != RESULT_OK)
    return -1; // Required features not available

auto queue = gpuCreateQueue();
auto semaphore = gpuCreateSemaphore(0);

auto computeIR = loadIR("Compute.spv");
auto pipeline = gpuCreateComputePipeline(ByteSpan(computeIR.data(), computeIR.size()));

float* input = gpuMalloc<float>(16);
float* output = gpuMalloc<float>(16);

for (int i = 0; i < 16; i++)
    input[i] = static_cast<float>(i);

auto data = gpuMalloc<Data>();

data->multiplier = 2.f;
data->input = gpuHostToDevicePointer(input);
data->output = gpuHostToDevicePointer(output);

float* readback = gpuMalloc<float>(16, MEMORY_READBACK);

// GPU work
auto commandBuffer = gpuStartCommandRecording(queue);
gpuSetPipeline(commandBuffer, pipeline);
gpuDispatch(commandBuffer, gpuHostToDevicePointer(data), {1, 1, 1});
gpuBarrier(commandBuffer, STAGE_COMPUTE, STAGE_TRANSFER);
gpuMemCpy(commandBuffer, gpuHostToDevicePointer(readback), data->output, sizeof(float) * 16);
gpuSubmit(queue, Span<GpuCommandBuffer>(&commandBuffer, 1), semaphore, 1);
gpuWaitSemaphore(semaphore, 1);

for (int i = 0; i < 16; i++)
    std::cout << readback[i] << " ";

// Should output: 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30

// Cleanup
gpuFree(data);
gpuFree(input);
gpuFree(output);
gpuFree(readback);
gpuDestroySemaphore(semaphore);
gpuDestroyDevice();
```

## Dependencies
- VkBootstrap
- glm
- SDL
- stb_image & stb_image_write