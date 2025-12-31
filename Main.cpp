#include "NoGraphicsAPI.h"
#include "Shaders/Compute.h"
#include "Shaders/Blur.h"

#include "Common.h"

#define STB_IMAGE_IMPLEMENTATION
#include "External/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "External/stb_image_write.h"

#include "SDL_gpu.h"

#include <iostream>
#include <filesystem>
#include <fstream>

using namespace std;

template<typename T>
struct Allocation
{
    T* cpu;
    T* gpu;

    void free()
    {
        gpuFree(cpu);
    }
};

template<typename T>
Allocation<T> allocate(int count = 1)
{
    auto addr = gpuMalloc(sizeof(T) * count);
    return { 
        .cpu = static_cast<T*>(addr), 
        .gpu = static_cast<T*>(gpuHostToDevicePointer(addr)) 
    };
};

template<typename T>
T* gpuMalloc(int count = 1)
{
    return static_cast<T*>(::gpuMalloc(sizeof(T) * count));
}

void test_compute_shader()
{
    auto queue = gpuCreateQueue();
    auto semaphore = gpuCreateSemaphore(0);

    auto inputData = allocate<uint32_t>(64);
    for (uint32_t i = 0; i < 64; ++i) 
    {
        inputData.cpu[i] = i;
    }
    
    auto outputData = allocate<uint32_t>(64);

    auto data = allocate<Data>();
    data.cpu->color = { 2.0f, 0.0f, 0.0f, 1.0f };
    data.cpu->offset = 0;
    data.cpu->input = inputData.gpu;
    data.cpu->output = outputData.gpu;

    auto computeIR = Utilities::loadIR("../../../Shaders/Compute2.spv");
    auto pipeline = gpuCreateComputePipeline(ByteSpan(computeIR.data(), computeIR.size()));

    auto commandBuffer = gpuStartCommandRecording(queue);
    gpuSetPipeline(commandBuffer, pipeline);
    gpuDispatch(commandBuffer, data.gpu, {1, 1, 1});
    gpuSubmit(queue, Span<GpuCommandBuffer>(&commandBuffer, 1), semaphore, 1);
    gpuWaitSemaphore(semaphore, 1);

    auto readback = gpuMalloc(sizeof(uint32_t) * 64, MEMORY_READBACK);
    commandBuffer = gpuStartCommandRecording(queue);
    gpuMemCpy(commandBuffer, gpuHostToDevicePointer(readback), outputData.gpu, sizeof(uint32_t) * 64);
    gpuSubmit(queue, Span<GpuCommandBuffer>(&commandBuffer, 1), semaphore, 2);
    gpuWaitSemaphore(semaphore, 2);

    auto result = static_cast<uint32_t*>(readback);
    cout << "Compute shader output:" << endl;
    for (uint32_t i = 0; i < 64; ++i) 
    {
        cout << result[i] << " ";
    }
    cout << endl;

    inputData.free();
    outputData.free();
    data.free();
    gpuFree(readback);
    gpuDestroySemaphore(semaphore);
    gpuFreePipeline(pipeline);
}

void test_upload_download()
{
    auto queue = gpuCreateQueue();
    auto semaphore = gpuCreateSemaphore(0);

    uint4 data = uint4(1, 2, 3, 4);
    const auto size = sizeof(uint4);

    // Write some test data to gpu memory
    void* upload = gpuMalloc(size);
    memcpy(upload, &data, size);

    // Upload command buffer
    auto commandBuffer = gpuStartCommandRecording(queue);

    // Upload the data to a gpu buffer
    void* gpu = gpuMalloc(size, MEMORY_GPU);
    gpuMemCpy(commandBuffer, gpu, gpuHostToDevicePointer(upload), size);
    gpuSubmit(queue, Span<GpuCommandBuffer>(&commandBuffer, 1), semaphore, 1);
    gpuWaitSemaphore(semaphore, 1);

    // Download command buffer
    commandBuffer = gpuStartCommandRecording(queue);

    // Download the gpu-local buffer back to host-visible memory
    void* download = gpuMalloc(size, MEMORY_READBACK);
    gpuMemCpy(commandBuffer, gpuHostToDevicePointer(download), gpu, size);
    gpuSubmit(queue, Span<GpuCommandBuffer>(&commandBuffer, 1), semaphore, 2);
    gpuWaitSemaphore(semaphore, 2);

    cout << "Data copied to GPU and back." << endl;
    cout << "Downloaded data: ";
    uint4* downloadedData = static_cast<uint4*>(download);
    cout << downloadedData->x << ", " << downloadedData->y << ", " << downloadedData->z << ", " << downloadedData->w << endl;

    cout << "Original upload data: ";
    uint4* uploadedData = static_cast<uint4*>(upload);
    cout << uploadedData->x << ", " << uploadedData->y << ", " << uploadedData->z << ", " << uploadedData->w << endl;

    gpuFree(upload);
    gpuFree(gpu);
    gpuFree(download);
    gpuDestroySemaphore(semaphore);
}

void test_upload_download_barrier()
{
    auto queue = gpuCreateQueue();
    auto semaphore = gpuCreateSemaphore(0);

    auto upload = allocate<uint4>();
    void* gpu = gpuMalloc(sizeof(uint4), MEMORY_GPU);
    void* download = gpuMalloc(sizeof(uint4), MEMORY_READBACK);

    upload.cpu->x = 1;
    upload.cpu->y = 2;
    upload.cpu->z = 3;
    upload.cpu->w = 4;

    auto commandBuffer = gpuStartCommandRecording(queue);
    gpuMemCpy(commandBuffer, gpu, upload.gpu, sizeof(uint4));
    gpuBarrier(commandBuffer, STAGE_TRANSFER, STAGE_TRANSFER);
    gpuMemCpy(commandBuffer, gpuHostToDevicePointer(download), gpu, sizeof(uint4));
    gpuSubmit(queue, Span<GpuCommandBuffer>(&commandBuffer, 1), semaphore, 1);
    gpuWaitSemaphore(semaphore, 1);

    cout << "Data copied to GPU and back." << endl;
    cout << "Downloaded data: ";
    uint4* downloadedData = static_cast<uint4*>(download);
    cout << downloadedData->x << ", " << downloadedData->y << ", " << downloadedData->z << ", " << downloadedData->w << endl;

    cout << "Original upload data: ";
    cout << upload.cpu->x << ", " << upload.cpu->y << ", " << upload.cpu->z << ", " << upload.cpu->w << endl;

    upload.free();
    gpuFree(gpu);
    gpuFree(download);
    gpuDestroySemaphore(semaphore);
}

void test_upload_download_image()
{
    auto queue = gpuCreateQueue();
    auto semaphore = gpuCreateSemaphore(0);

    auto computeIR = Utilities::loadIR("../../../Shaders/Blur.spv");
    auto pipeline = gpuCreateComputePipeline(ByteSpan(computeIR.data(), computeIR.size()));

    // Load input image
    int width, height, channels;
    stbi_uc* inputImage = stbi_load("../../../Assets/NoGraphicsAPI.png", &width, &height, &channels, 4);
    size_t size = width * height * 4;

    auto upload = allocate<uint8_t>(size);
    memcpy(upload.cpu, inputImage, size);

    GpuTextureDesc textureDesc{
        .type = TEXTURE_2D,
        .dimensions = { static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1 },
        .format = FORMAT_RGBA8_UNORM,
        .usage = USAGE_SAMPLED
    };

    GpuTextureSizeAlign textureSizeAlign = gpuTextureSizeAlign(textureDesc);
    void* texturePtr = gpuMalloc(textureSizeAlign.size, textureSizeAlign.align, MEMORY_GPU);
    auto texture = gpuCreateTexture(textureDesc, texturePtr);

    auto commandBuffer = gpuStartCommandRecording(queue);
    gpuCopyToTexture(commandBuffer, upload.gpu, texture);
    gpuSubmit(queue, Span<GpuCommandBuffer>(&commandBuffer, 1), semaphore, 1);
    gpuWaitSemaphore(semaphore, 1);

    commandBuffer = gpuStartCommandRecording(queue);

    auto readback = gpuMalloc(textureSizeAlign.size, MEMORY_READBACK);
    gpuCopyFromTexture(commandBuffer, gpuHostToDevicePointer(readback), texture);

    gpuSubmit(queue, Span<GpuCommandBuffer>(&commandBuffer, 1), semaphore, 2);
    gpuWaitSemaphore(semaphore, 2);

    stbi_write_png("../../../Assets/Copied.png", width, height, 4, readback, width * 4);

    gpuDestroySemaphore(semaphore);
    gpuDestroyTexture(texture);
    gpuFreePipeline(pipeline);
    gpuFree(texturePtr);
    gpuFree(readback);
    upload.free();
}

void test_image_blur()
{
    auto queue = gpuCreateQueue();
    auto semaphore = gpuCreateSemaphore(0);

    auto computeIR = Utilities::loadIR("../../../Shaders/Blur.spv");
    auto pipeline = gpuCreateComputePipeline(ByteSpan(computeIR.data(), computeIR.size()));

    auto textureHeap = gpuMalloc<GpuTextureDescriptor>(1024);
    
    // Load input image
    int width, height, channels;
    stbi_uc* inputImage = stbi_load("../../../Assets/NoGraphicsAPI.png", &width, &height, &channels, 4);

    auto upload = allocate<uint8_t>(width * height * 4);
    memcpy(upload.cpu, inputImage, width * height * 4);

    GpuTextureDesc textureDesc{
        .type = TEXTURE_2D,
        .dimensions = { static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1 },
        .format = FORMAT_RGBA8_UNORM,
        .usage = USAGE_SAMPLED
    };

    GpuTextureSizeAlign textureSizeAlign = gpuTextureSizeAlign(textureDesc);
    void* texturePtr = gpuMalloc(textureSizeAlign.size, MEMORY_GPU);
    auto texture = gpuCreateTexture(textureDesc, texturePtr);

    GpuTextureDesc outputTextureDes{
        .type = TEXTURE_2D,
        .dimensions = { static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1 },
        .format = FORMAT_RGBA8_UNORM,
        .usage = USAGE_STORAGE
    };

    void* outputPtr = gpuMalloc(textureSizeAlign.size, MEMORY_GPU);
    auto outputTexture = gpuCreateTexture(outputTextureDes, outputPtr);

    textureHeap[0] = gpuTextureViewDescriptor(texture, GpuViewDesc{.format = FORMAT_RGBA8_UNORM });
    textureHeap[1] = gpuRWTextureViewDescriptor(outputTexture, GpuViewDesc{ .format = FORMAT_RGBA8_UNORM });

    auto test = textureHeap[1];

    auto commandBuffer = gpuStartCommandRecording(queue);
    gpuCopyToTexture(commandBuffer, upload.gpu, texture);
    gpuSubmit(queue, Span<GpuCommandBuffer>(&commandBuffer, 1), semaphore, 1);
    gpuWaitSemaphore(semaphore, 1);

    commandBuffer = gpuStartCommandRecording(queue);

    auto data = allocate<BlurData>();

    data.cpu->imageSize = { static_cast<uint32_t>(width), static_cast<uint32_t>(height) };
    data.cpu->srcTexture = 0;
    data.cpu->dstTexture = 1;

    gpuSetPipeline(commandBuffer, pipeline);
    gpuSetActiveTextureHeapPtr(commandBuffer, gpuHostToDevicePointer(textureHeap));
    gpuDispatch(commandBuffer, data.gpu, { 
            static_cast<uint32_t>(width / 16), 
            static_cast<uint32_t>(height / 16), 
            1
    });

    gpuSubmit(queue, Span<GpuCommandBuffer>(&commandBuffer, 1), semaphore, 2);
    gpuWaitSemaphore(semaphore, 2);


    commandBuffer = gpuStartCommandRecording(queue);
    auto readback = gpuMalloc(textureSizeAlign.size, MEMORY_READBACK);
    gpuCopyFromTexture(commandBuffer, gpuHostToDevicePointer(readback), outputTexture);
    gpuSubmit(queue, Span<GpuCommandBuffer>(&commandBuffer, 1), semaphore, 3);
    gpuWaitSemaphore(semaphore, 3);

    stbi_write_png("../../../Assets/BlurredOutput.png", width, height, 4, readback, width * 4);

    gpuDestroySemaphore(semaphore);
    gpuDestroyTexture(texture);
    gpuDestroyTexture(outputTexture);
    gpuFreePipeline(pipeline);
    gpuFree(texturePtr);
    gpuFree(outputPtr);
    gpuFree(readback);
    upload.free();
    data.free();
}

void test_sdl_window()
{
    auto window = SDL_CreateWindow("Test Window", 1920, 1080, SDL_WINDOW_GPU);
    auto surface = SDL_Gpu_CreateSurface(window);
    bool exit = false;

    const uint FRAMES_IN_FLIGHT = 2;

    auto queue = gpuCreateQueue();
    auto swapchain = gpuCreateSwapchain(surface, FRAMES_IN_FLIGHT);
    auto semaphore = gpuCreateSemaphore(0);
    uint64_t nextFrame = 1;

    // Load input image
    int width, height, channels;
    stbi_uc* inputImage = stbi_load("../../../Assets/NoGraphicsAPI.png", &width, &height, &channels, 4);
    auto upload = allocate<uint8_t>(width * height * 4);
    memcpy(upload.cpu, inputImage, width * height * 4);

    GpuTextureDesc textureDesc{
        .type = TEXTURE_2D,
        .dimensions = { static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1 },
        .format = FORMAT_RGBA8_UNORM,
        .usage = USAGE_SAMPLED
    };

    GpuTextureSizeAlign textureSizeAlign = gpuTextureSizeAlign(textureDesc);
    void* texturePtr = gpuMalloc(textureSizeAlign.size, MEMORY_GPU);
    auto texture = gpuCreateTexture(textureDesc, texturePtr);

    auto uploadSemaphore = gpuCreateSemaphore(0);
    auto commandBuffer = gpuStartCommandRecording(queue);
    gpuCopyToTexture(commandBuffer, upload.gpu, texture);
    gpuSubmit(queue, Span<GpuCommandBuffer>(&commandBuffer, 1), uploadSemaphore, 1);
    gpuWaitSemaphore(uploadSemaphore, 1);
    gpuDestroySemaphore(uploadSemaphore);

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
        gpuBlitTexture(commandBuffer, image, texture);
        gpuSubmit(queue, Span<GpuCommandBuffer>(&commandBuffer, 1), semaphore, nextFrame);
        gpuPresent(swapchain, semaphore, nextFrame++);
    }
    
    gpuDestroyTexture(texture);
    gpuFree(texturePtr);
    gpuDestroySemaphore(semaphore);
    upload.free();
    gpuDestroySwapchain(swapchain);
    SDL_Gpu_DestroySurface(surface);
    SDL_DestroyWindow(window);
}

int main()
{
    // test_upload_download();
    // test_upload_download_barrier();
    // test_compute_shader();
    // test_image_blur();
    // test_upload_download_image();

    test_sdl_window();

    return 0;
}