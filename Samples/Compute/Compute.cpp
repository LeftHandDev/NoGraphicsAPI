#include "Compute.h"
#include "../../Utilities.h"
#include "../../External/stb_image.h"
#include "../../External/stb_image_write.h"

#include <SDL3/SDL.h>
#include "../../SDL_gpu.h"

void computeSample()
{
    gpuCreateDevice();

    const uint FRAMES_IN_FLIGHT = 2;

    auto window = SDL_CreateWindow("Test Window", 1920, 1080, SDL_WINDOW_GPU);
    auto surface = SDL_Gpu_CreateSurface(window);
    bool exit = false;

    auto swapchain = gpuCreateSwapchain(surface, FRAMES_IN_FLIGHT);
    auto swapchainDesc = gpuSwapchainDesc(swapchain);

    auto queue = gpuCreateQueue();
    auto semaphore = gpuCreateSemaphore(0);

    auto computeIR = loadIR("../Shaders/Compute/Compute.spv");
    auto pipeline = gpuCreateComputePipeline(ByteSpan(computeIR.data(), computeIR.size()));

    auto textureHeap = gpuMalloc<GpuTextureDescriptor>(1024);
    
    // Load input image
    int width, height, channels;
    stbi_uc* inputImage = stbi_load("Assets/NoGraphicsAPI.png", &width, &height, &channels, 4);

    auto upload = allocate<uint8_t>(width * height * 4);
    memcpy(upload.cpu, inputImage, width * height * 4);

    GpuTextureDesc textureDesc{
        .type = TEXTURE_2D,
        .dimensions = { static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1 },
        .format = FORMAT_RGBA8_UNORM,
        .usage = static_cast<USAGE_FLAGS>(USAGE_SAMPLED | USAGE_TRANSFER_DST)
    };

    GpuTextureSizeAlign textureSizeAlign = gpuTextureSizeAlign(textureDesc);
    void* texturePtr = gpuMalloc(textureSizeAlign.size, MEMORY_GPU);
    auto texture = gpuCreateTexture(textureDesc, texturePtr);

    GpuTextureDesc outputTextureDes{
        .type = TEXTURE_2D,
        .dimensions = { static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1 },
        .format = FORMAT_RGBA8_UNORM,
        .usage = static_cast<USAGE_FLAGS>(USAGE_STORAGE | USAGE_TRANSFER_SRC)
    };

    void* outputPtr = gpuMalloc(textureSizeAlign.size, MEMORY_GPU);
    auto outputTexture = gpuCreateTexture(outputTextureDes, outputPtr);

    textureHeap[0] = gpuTextureViewDescriptor(texture, GpuViewDesc{.format = FORMAT_RGBA8_UNORM });
    textureHeap[1] = gpuRWTextureViewDescriptor(outputTexture, GpuViewDesc{ .format = FORMAT_RGBA8_UNORM });

    auto commandBuffer = gpuStartCommandRecording(queue);
    gpuCopyToTexture(commandBuffer, upload.gpu, texture);
    gpuSubmit(queue, Span<GpuCommandBuffer>(&commandBuffer, 1), semaphore, 1);
    gpuWaitSemaphore(semaphore, 1);

    auto data = allocate<ComputeData>();

    data.cpu->imageSize = { static_cast<uint32_t>(width), static_cast<uint32_t>(height) };
    data.cpu->srcTexture = 0;
    data.cpu->dstTexture = 1;

    commandBuffer = gpuStartCommandRecording(queue);
    gpuSetPipeline(commandBuffer, pipeline);
    gpuSetActiveTextureHeapPtr(commandBuffer, gpuHostToDevicePointer(textureHeap));
    gpuDispatch(commandBuffer, data.gpu, { 
            static_cast<uint32_t>(width / 16), 
            static_cast<uint32_t>(height / 16), 
            1
    });

    gpuSubmit(queue, Span<GpuCommandBuffer>(&commandBuffer, 1), semaphore, 2);
    gpuWaitSemaphore(semaphore, 2);

    gpuDestroySemaphore(semaphore);
    semaphore = gpuCreateSemaphore(0);

    uint64_t nextFrame = 1;

    while (!exit)
    {
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            if (event.type == SDL_EVENT_QUIT)
            {
                exit = true;
                break;
            }
        }

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

    stbi_image_free(inputImage);
    gpuFree(textureHeap);
    gpuDestroySemaphore(semaphore);
    gpuDestroyTexture(texture);
    gpuDestroyTexture(outputTexture);
    gpuFreePipeline(pipeline);
    gpuFree(texturePtr);
    gpuFree(outputPtr);
    upload.free();
    data.free();
    gpuDestroyDevice();
}