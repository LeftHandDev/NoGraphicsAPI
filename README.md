# No Graphics API

A demo of the simplified graphics API from Sebastian Aaltonen's blog post [No Graphics API](https://www.sebastianaaltonen.com/blog/no-graphics-api), implemented in Vulkan. The entire API is not implemented, just enough to get some samples working.

The project started with the original header from the blog post, and built from there. The style and design attempts to match the blog post where possible.


## Samples

To see what the API looks like in practice, check out the [samples](https://github.com/LeftHandDev/NoGraphicsAPI/tree/main/samples). For simple usage of the API, see below.

## Building

Clone with submodules (or initialize them afterwards):

```sh
git clone --recursive <repo-url>
# or, in an existing clone:
git submodule update --init --recursive
```

Configure and build — CMake presets are provided:

```sh
cmake --preset linux   # GLFW backend (default)
cmake --build build
```

The per-sample executables (`compute`, `graphics`, `raytracing`, `multiplegpus`) are written to `build/bin/`, next to the compiled `shaders/` and `assets/` they load at runtime:

```sh
cd build/bin && ./raytracing
```

To use the SDL3 windowing backend instead of GLFW, configure with `cmake --preset linux-sdl3` (builds into `build/sdl3/`), or pass `-DNGA_WINDOW_BACKEND=SDL3`.

## Windowless Usage
### Common header
```c++
#include "NoGraphicsAPI.h"

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
#include "Compute.h" // Common header

int main()
{
    gpuCreateInstance();

    auto device = gpuCreateDevice(0);
    if (!device)
        return -1; // Required features not available

    auto queue = gpuCreateQueue(device);
    auto semaphore = gpuCreateSemaphore(device, 0);

    auto computeIR = loadIR("Compute.spv");
    auto pipeline = gpuCreateComputePipeline(device, ByteSpan(computeIR.data(), computeIR.size()));

    float* input = gpuMalloc<float>(device, 16);
    float* output = gpuMalloc<float>(device, 16);

    for (int i = 0; i < 16; i++)
        input[i] = static_cast<float>(i);

    auto data = gpuMalloc<Data>(device);

    data->multiplier = 2.f;
    data->input = gpuHostToDevicePointer(device, input);
    data->output = gpuHostToDevicePointer(device, output);

    float* readback = gpuMalloc<float>(device, 16, MEMORY_READBACK);

    // GPU work
    auto commandBuffer = gpuStartCommandRecording(queue);
    gpuSetPipeline(commandBuffer, pipeline);
    gpuDispatch(commandBuffer, gpuHostToDevicePointer(device, data), {1, 1, 1});
    gpuBarrier(commandBuffer, STAGE_COMPUTE, STAGE_TRANSFER);
    gpuMemCpy(commandBuffer, gpuHostToDevicePointer(device, readback), data->output, sizeof(float) * 16);
    gpuSubmit(queue, Span<GpuCommandBuffer>(&commandBuffer, 1), semaphore, 1);
    gpuWaitSemaphore(semaphore, 1);

    for (int i = 0; i < 16; i++)
        std::cout << readback[i] << " ";

    // Should output: 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30

    // Cleanup
    gpuFree(device, data);
    gpuFree(device, input);
    gpuFree(device, output);
    gpuFree(device, readback);
    gpuDestroySemaphore(semaphore);
    gpuDestroyQueue(queue);
    gpuDestroyDevice(device);
    gpuDestroyInstance();
}
```

## Window Usage

Include `window.h` to create a window and Vulkan surface through the selected windowing backend (GLFW or SDL3).

```c++
#include "window.h"

// include common header

#define FRAMES_IN_FLIGHT 2

int main()
{
    gpuCreateInstance();

    auto device = gpuCreateDevice(0);
    if (!device)
        return -1;

    auto window = nga::createWindow("Example", 1920, 1080);
    auto surface = nga::createSurface(window);
    auto swapchain = gpuCreateSwapchain(device, surface, FRAMES_IN_FLIGHT);

    // Queue, semaphore creation...

    uint64_t nextFrame = 1;

    while (!nga::shouldClose(window))
    {
        nga::pollEvents(window);

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
    nga::destroySurface(window, surface);
    nga::destroyWindow(window);
    gpuDestroyDevice(device);
    gpuDestroyInstance();
}
```
## Graphics Pipeline Shaders
### Common Header
```cpp
#include "NoGraphicsAPI.h"

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
## Raytracing

Raytracing pipelines and acceleration structures are not mentioned in the original header, so some liberties had to be taken in the API design. Since raytracing pipelines are not required to trace rays, for now they are not implemented.

To trace rays, simply create and build acceleration structures, and then pass the TLAS gpu pointer to a shader and use ray query. See the [Raytracing.cpp](https://github.com/LeftHandDev/NoGraphicsAPI/blob/main/Samples/Raytracing/Raytracing.cpp) sample for an example.

## Dependencies

Fetched as git submodules under `external/` and built from source — nothing to install:
- [GLFW](https://github.com/glfw/glfw) and [SDL3](https://github.com/libsdl-org/SDL) — windowing (pick one via `NGA_WINDOW_BACKEND`)
- [GLM](https://github.com/g-truc/glm) — math
- [vk-bootstrap](https://github.com/charles-lunarg/vk-bootstrap) — Vulkan instance/device setup

Vendored in the repo:
- [stb_image & stb_image_write](https://github.com/nothings/stb/tree/master)

Must be installed separately (the one prerequisite):
- [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) — Vulkan loader/headers and the `slangc` shader compiler