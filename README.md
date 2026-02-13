# No Graphics API

A demo of the simplified graphics API from Sebastian Aaltonen's blog post [No Graphics API](https://www.sebastianaaltonen.com/blog/no-graphics-api), implemented in Vulkan. The entire API is not implemented, just enough to get some samples working.

The project started with the original header from the blog post, and built from there. The style and design attempts to match the blog post where possible.


## Samples

To see what the API looks like in practice, check out the [samples](https://github.com/LeftHandDev/NoGraphicsAPI/tree/main/Samples). For simple usage of the API, see below.

## Windowless Usage
### Common header
```c++
#include "Shaders/NoGraphicsAPI.h"

struct alignas(16) Data
{
    float multiplier;

    float* input;
    float* output;
};
```
### slang Compute Shader
```c++
[numthreads(16, 1, 1)]
void main(uint3 threadId: SV_DispatchThreadID, Data* data)
{
    data->output[threadId.x] = data->multiplier * data->input[threadId.x];
}
```
### CPU side
```c++
#include "NoGraphicsAPI.h"

int main()
{
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
}
```

## Window Usage

Include `SDL_gpu.h` to create a window and surface similar to when using Vulkan.

```c++
#include <SDL3/SDL.h>
#include "SDL_gpu.h"

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
## Graphics Pipeline Shaders
### Common Header
```cpp
#include "Shaders/NoGraphicsAPI.h"

struct alignas(16) VertexData
{
    float4x4 viewProjection;
    float4* vertices;
    float2* uvs;
};

struct alignas(16) PixelData
{
    uint texture;
};
```
### Vertex Shader
```cpp
struct VertexOut
{
    float4 position : SV_Position;
    float2 uv;
};

VertexOut main(uint vertexId: SV_VertexID, VertexData *data, PixelData *_)
{
    VertexOut out;
    out.position = mul(data->viewProjection, data->vertices[vertexId]);
    out.uv = data->uvs[vertexId];
    return out;
}
```
### Pixel Shader
```cpp
struct PixelIn
{
    float4 position : SV_Position;
    float2 uv;
};

struct PixelOut
{
    float4 color : SV_Target;
};

PixelOut main(PixelIn pixel, VertexData* _, PixelData* data)
{
    PixelOut out;
    out.color = textureHeap[data->texture].SampleLevel(samplerHeap[0], pixel.uv, 0);
    return out;
}
```

## Dependencies
- Included in the repo
    - [VkBootstrap](https://github.com/charles-lunarg/vk-bootstrap)
    - [stb_image & stb_image_write](https://github.com/nothings/stb/tree/master)

- [Vulkan SDK](https://vulkan.lunarg.com/sdk/home)
    - GLM
    - SDL3