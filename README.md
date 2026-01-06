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

## Windowed Usage

```c++
#include <SDL3/SDL.h>
#include "../../SDL_gpu.h"

#define FRAMES_IN_FLIGHT 2

int main()
{
    if (gpuCreateDevice() != RESULT_OK)
        return -1;

    auto window = SDL_CreateWindow("Example", 1920, 1080, SDL_WINDOW_GPU);
    auto surface = SDL_Gpu_CreateSurface(window);
    auto swapchain = gpuCreateSwapchain(surface, FRAMES_IN_FLIGHT);

    // Queue, semaphore creation...

    uint64_t nextFrame = 1;

    bool exit = false;
    while (!exit)
    {
        // SDL poll events...

        if (nextFrame > FRAMES_IN_FLIGHT)
            gpuWaitSemaphore(semaphore, nextFrame - FRAMES_IN_FLIGHT);

        auto commandBuffer = gpuStartCommandRecording(queue);
        auto image = gpuSwapchainImage(swapchain);

        // Render/copy to swapchain image...

        gpuSubmit(queue, Span<GpuCommandBuffer>(&commandBuffer, 1), semaphore, nextFrame);
        gpuPresent(swapchain, semaphore, nextFrame++);
    }

    // Cleanup
    // Semaphore, other objects...
    gpuDestroySwapchain(swapchain);
    SDL_Gpu_DestroySurface(surface);
    SDL_DestroyWindow(window);
    gpuDestroyDevice();
}
```
## Vertex + Pixel Shaders
Both structs for the vertex and pixel shader must be referenced in the function definitions.
### Common Header
```cpp
#include "../../Shaders/NoGraphicsAPI.h" // Must be included

struct alignas(16) VertexData
{
};

struct alignas(16) PixelData
{
}
```
### Vertex Shader
```cpp
struct VertexOut
{
};

VertexOut main(uint vertexId: SV_VertexID, VertexData *data, PixelData *_)
{
}
```
### Pixel Shader
```cpp
struct PixelIn
{
};

struct PixelOut
{
};

PixelOut main(PixelIn pixel, VertexData* _, PixelData* data)
{
}
```

## Dependencies
- VkBootstrap
- glm
- SDL
- stb_image & stb_image_write