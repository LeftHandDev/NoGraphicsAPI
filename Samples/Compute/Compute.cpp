#include "Compute.h"
#include "../../Common.h"
#include "../../External/stb_image.h"
#include "../../External/stb_image_write.h"

void computeSample()
{
    auto queue = gpuCreateQueue();
    auto semaphore = gpuCreateSemaphore(0);

    auto computeIR = Utilities::loadIR("../../../Compute/Compute.spv");
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

    auto data = allocate<ComputeData>();

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