# No Graphics API

[![CI](https://github.com/LeftHandDev/NoGraphicsAPI/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/LeftHandDev/NoGraphicsAPI/actions/workflows/ci.yml)

A demo of the simplified graphics API from Sebastian Aaltonen's blog post [No Graphics API](https://www.sebastianaaltonen.com/blog/no-graphics-api), implemented in Vulkan. The entire API is not implemented, just enough to get some samples working.

The project started with the original header from the blog post, and built from there. The style and design attempts to match the blog post where possible.


## Samples

To see what the API looks like in practice, check out the [samples](https://github.com/LeftHandDev/NoGraphicsAPI/tree/main/samples). For simple usage of the API, see below.

## Using NGAPI in your project

The reusable library is the self-contained [`ngapi/`](ngapi/) folder. Add this
repo (or just that folder) as a submodule, a copy, or via FetchContent, then:

```cmake
add_subdirectory(path/to/NoGraphicsAPI)   # or just the ngapi/ folder, or include(<path>/ngapi/ngapi.cmake)
target_link_libraries(your_app PRIVATE ngapi::ngapi)
```

That's it — samples and tests build only when NGAPI is the top-level project,
the one library dependency (vk-bootstrap) resolves itself (submodule or
automatic download), and no shader compiler is required to build the library.
You only need CMake 3.22+, a C++20 compiler and Vulkan. See
[ngapi/README.md](ngapi/README.md) for the details.

## Building the samples

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

To use the SDL3 windowing backend instead of GLFW, configure with `cmake --preset linux-sdl3` (builds into `build/sdl3/`), or pass `-DNGAPI_WINDOW_BACKEND=SDL3`.

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

Include `window.h` to create a window and Vulkan surface through the selected windowing backend (GLFW or SDL3). Note that `window.h` is part of this repo's sample scaffolding ([`platform/`](platform/)), not the embeddable `ngapi/` library — in your own project, create the window and `VkSurfaceKHR` with your windowing library of choice and hand the surface to `gpuCreateSurface()`.

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

    auto window = ngapi::createWindow("Example", 1920, 1080);
    auto surface = ngapi::createSurface(window);
    auto swapchain = gpuCreateSwapchain(device, surface, FRAMES_IN_FLIGHT);

    // Queue, semaphore creation...

    uint64_t nextFrame = 1;

    while (!ngapi::shouldClose(window))
    {
        ngapi::pollEvents(window);

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
    ngapi::destroySurface(window, surface);
    ngapi::destroyWindow(window);
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
STATIC_SAMPLER(aniso16Wrap, LINEAR, LINEAR, LINEAR, WRAP, WRAP, 16)

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
    out.color = aniso16Wrap().Sample(textureHeap[data->texture], pixel.uv);
    return out;
}
```

## Samplers

Samplers are not API objects and there is no sampler API: a sampler is
declared next to the shader code that uses it, with the complete definition in
the declaration, and runs on a real hardware sampler the implementation
creates (and deduplicates) at pipeline creation. Two forms:

**Named, at global scope — the full state, every field required:**

```cpp
//             name         min     mag     mip     U     V     W     aniso  border             compare       bias minLod maxLod
STATIC_SAMPLER(aniso16Wrap, LINEAR, LINEAR, LINEAR, WRAP, WRAP, WRAP, 16,    TRANSPARENT_BLACK, COMPARE_NONE, 0.0, 0.0,   LOD_UNBOUNDED)

float4 color = aniso16Wrap().Sample(textureHeap[t], uv);          // implicit lod (pixel shaders)
float4 level = aniso16Wrap().SampleLevel(textureHeap[t], uv, 2);  // explicit lod
float shadow = aniso16Wrap().SampleCmp(textureHeap[s], uv, ref);  // needs a COMPARE_* op
```

**Inline, anywhere in shader code** (same fields minus the three lod values,
which stay at their defaults — bias 0, unclamped):

```cpp
HwSampler sampler = INLINE_SAMPLER(LINEAR, LINEAR, NEAREST, CLAMP, CLAMP, CLAMP, 1, TRANSPARENT_BLACK, COMPARE_NONE);
float4 color = sampler.SampleLevel(textureHeap[t], uv, 0);
```

The fields, in order:

- **minFilter / magFilter** — `NEAREST` or `LINEAR`; how texels blend when the
  texture is minified (smaller on screen) or magnified.
- **mipFilter** — `NEAREST` snaps to the closest mip level, `LINEAR`
  interpolates between two (trilinear).
- **addressU / addressV / addressW** — what happens outside [0,1] per axis:
  `WRAP` repeats, `CLAMP` extends the edge texel, `MIRROR` reflects,
  `BORDER` returns the border color, `MIRROR_CLAMP` reflects once then clamps.
- **maxAnisotropy** — 1 disables; up to 16 taps along the major axis for
  oblique views. Only applies through `Sample()`/`SampleCmp()` (implicit
  derivatives), not the explicit-lod variants.
- **borderColor** — `TRANSPARENT_BLACK`, `OPAQUE_BLACK` or `OPAQUE_WHITE`,
  used by `BORDER` addressing.
- **compare** — `COMPARE_NONE` for ordinary sampling, or a `COMPARE_*` op
  (`LESS`, `GREATER_EQUAL`, ...) to make a comparison sampler for the
  `SampleCmp*` methods (shadow maps).
- **lodBias** — added to the computed mip level; ±8 in 1/16 steps.
- **minLod / maxLod** — clamp the mip range; `LOD_UNBOUNDED` leaves the top
  end open. 1/16 steps, 0–15.9375.

How it works underneath: the declaration packs the state into a value the
implementation finds by scanning the shader's SPIR-V at pipeline creation —
a specialization constant's default for `STATIC_SAMPLER`, a tagged literal
for `INLINE_SAMPLER` — then creates the `VkSampler`, and rewires the
declaration to its heap slot. No sidecar metadata, no new API calls, and the
slot folds to a literal in the compiled pipeline. See
[`ngapi/include/Sampler.h`](ngapi/include/Sampler.h).
## Raytracing

Raytracing pipelines and acceleration structures are not mentioned in the original header, so some liberties had to be taken in the API design. Since raytracing pipelines are not required to trace rays, for now they are not implemented.

To trace rays, simply create and build acceleration structures, and then pass the TLAS gpu pointer to a shader and use ray query. See the [Raytracing.cpp](https://github.com/LeftHandDev/NoGraphicsAPI/blob/main/samples/raytracing/Raytracing.cpp) sample for an example.

## Dependencies

The library itself ([`ngapi/`](ngapi/)) depends only on Vulkan and
[vk-bootstrap](https://github.com/charles-lunarg/vk-bootstrap) (a submodule,
auto-downloaded if missing). The rest is samples/tests-only, fetched as git
submodules under `external/` and built from source — nothing to install:
- [GLFW](https://github.com/glfw/glfw) and [SDL3](https://github.com/libsdl-org/SDL) — windowing (pick one via `NGAPI_WINDOW_BACKEND`)
- [GLM](https://github.com/g-truc/glm) — math

Vendored in the repo:
- [stb_image & stb_image_write](https://github.com/nothings/stb/tree/master)

Must be installed separately (the one prerequisite):
- [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) — Vulkan loader/headers, plus the `slangc` shader compiler used by the samples and tests (the library itself does not need it)