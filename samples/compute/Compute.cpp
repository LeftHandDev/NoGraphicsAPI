#include "stb_image.h"
#include "stb_image_write.h"

#include <cstring>

#include "window.h"

#include "Compute.h"
#include "Utilities.h"

int main()
{
    gpuCreateInstance();
    auto device = gpuCreateDevice(0);

    const uint FRAMES_IN_FLIGHT = 2;

    auto window = ngapi::createWindow("Test Window", 1920, 1080);
    auto surface = ngapi::createSurface(window);

    LinearAllocator allocator(device);
    LinearAllocator<MEMORY_DESCRIPTOR> descriptorAllocator(device);

    auto swapchain = gpuCreateSwapchain(device, surface, FRAMES_IN_FLIGHT);
    auto swapchainDesc = gpuSwapchainDesc(swapchain);

    auto queue = gpuCreateQueue(device);
    auto semaphore = gpuCreateSemaphore(device, 0);

    auto computeIR = loadIR("shaders/compute/Compute.spv");
    auto pipeline = gpuCreateComputePipeline(device, ByteSpan(computeIR.data(), computeIR.size()));

    auto textureHeap = descriptorAllocator.allocate<GpuTextureDescriptor>(1024);

    // Load input image
    int width, height, channels;
    stbi_uc* inputImage = stbi_load("assets/Default.png", &width, &height, &channels, 4);

    auto upload = allocator.allocate<uint8_t>(width * height * 4);
    memcpy(upload.cpu, inputImage, width * height * 4);

    GpuTextureDesc textureDesc{
        .type = TEXTURE_2D,
        .dimensions = { static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1 },
        .format = FORMAT_RGBA8_UNORM,
        .usage = static_cast<USAGE_FLAGS>(USAGE_SAMPLED | USAGE_TRANSFER_DST)
    };

    GpuTextureSizeAlign textureSizeAlign = gpuTextureSizeAlign(device, textureDesc);
    void* texturePtr = gpuMalloc(device, textureSizeAlign.size, MEMORY_GPU);
    auto texture = gpuCreateTexture(device, textureDesc, texturePtr);

    GpuTextureDesc outputTextureDes{
        .type = TEXTURE_2D,
        .dimensions = { static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1 },
        .format = FORMAT_RGBA8_UNORM,
        .usage = static_cast<USAGE_FLAGS>(USAGE_STORAGE | USAGE_TRANSFER_SRC)
    };

    void* outputPtr = gpuMalloc(device, textureSizeAlign.size, MEMORY_GPU);
    auto outputTexture = gpuCreateTexture(device, outputTextureDes, outputPtr);

    textureHeap.cpu[0] = gpuTextureViewDescriptor(texture, GpuViewDesc{ .format = FORMAT_RGBA8_UNORM });
    textureHeap.cpu[1] = gpuRWTextureViewDescriptor(outputTexture, GpuViewDesc{ .format = FORMAT_RGBA8_UNORM });

    auto commandBuffer = gpuStartCommandRecording(queue);
    gpuCopyToTexture(commandBuffer, upload.gpu, texture);
    gpuSubmit(queue, Span<GpuCommandBuffer>(&commandBuffer, 1), semaphore, 1);
    gpuWaitSemaphore(semaphore, 1);

    auto data = allocator.allocate<ComputeData>(1);

    data.cpu->imageSize = { static_cast<uint32_t>(width), static_cast<uint32_t>(height) };
    data.cpu->srcTexture = 0;
    data.cpu->dstTexture = 1;

    // Attach a regular buffer: allocate an array of Tint structs, fill it on the
    // CPU, and hand its GPU pointer to the shader through the data struct. Nothing
    // is bound — the shader just follows `data->tints` to read whatever we point
    // it at, so the size and contents are entirely decided here at runtime.
    const uint32_t tintCount = 4;
    auto tints = allocator.allocate<Tint>(tintCount);
    tints.cpu[0] = { { 1.0f, 0.5f, 0.5f }, 0.6f }; // red band
    tints.cpu[1] = { { 0.5f, 1.0f, 0.5f }, 0.6f }; // green band
    tints.cpu[2] = { { 0.5f, 0.5f, 1.0f }, 0.6f }; // blue band
    tints.cpu[3] = { { 1.0f, 1.0f, 0.5f }, 0.6f }; // yellow band

    data.cpu->tints = tints.gpu;
    data.cpu->tintCount = tintCount;

    commandBuffer = gpuStartCommandRecording(queue);
    gpuSetPipeline(commandBuffer, pipeline);
    gpuSetActiveTextureHeapPtr(commandBuffer, textureHeap.gpu);
    gpuDispatch(commandBuffer, data.gpu, { static_cast<uint32_t>(width / 16), static_cast<uint32_t>(height / 16), 1 });

    gpuSubmit(queue, Span<GpuCommandBuffer>(&commandBuffer, 1), semaphore, 2);
    gpuWaitSemaphore(semaphore, 2);

    gpuDestroySemaphore(semaphore);
    semaphore = gpuCreateSemaphore(device, 0);

    uint64_t nextFrame = 1;

    while (!ngapi::shouldClose(window))
    {
        ngapi::pollEvents(window);

        if (nextFrame > FRAMES_IN_FLIGHT)
        {
            gpuWaitSemaphore(semaphore, nextFrame - FRAMES_IN_FLIGHT);
        }

        commandBuffer = gpuStartCommandRecording(queue);

        auto image = gpuSwapchainImage(swapchain);
        gpuBlitTexture(commandBuffer, image, outputTexture);
        gpuSubmit(queue, Span<GpuCommandBuffer>(&commandBuffer, 1), semaphore, nextFrame);
        gpuPresent(swapchain, semaphore, nextFrame++);
    }

    gpuWaitSemaphore(semaphore, nextFrame - 1);

    // Destroy the swapchain first: it drains all queues (including the present
    // queue) and releases the swapchain images, present semaphores and surface
    // before the device/instance are torn down. Destroying the device while
    // these children are still alive hangs drivers that honour object lifetimes.
    gpuDestroySwapchain(swapchain);
    ngapi::destroySurface(window, surface);
    ngapi::destroyWindow(window);

    stbi_image_free(inputImage);

    allocator.reset();
    descriptorAllocator.reset();

    gpuDestroySemaphore(semaphore);
    gpuDestroyTexture(texture);
    gpuDestroyTexture(outputTexture);
    gpuFreePipeline(pipeline);
    gpuFree(device, texturePtr);
    gpuFree(device, outputPtr);
    gpuDestroyQueue(queue);
    gpuDestroyDevice(device);
    gpuDestroyInstance();

    return 0;
}