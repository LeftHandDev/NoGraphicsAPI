// Headless test mirroring the Compute sample: blur an input image with the
// compute shader (plus the Tint buffer), driven through a multi-frame
// submit/semaphore loop, then read the result back and compare to a golden.
#include "test_common.h"

#include "Utilities.h" // LinearAllocator, loadIR
#include "Compute.h"   // ComputeData, Tint

#include "stb_image.h"

#include <cstring>
#include <iostream>
#include <string>

int main(int argc, char** argv)
{
    test::Args args = test::parseArgs(argc, argv);

    gpuCreateInstance();
    test::beginValidationCapture();

    auto device = gpuCreateDevice(args.device);
    if (!device)
    {
        std::cerr << "FAIL [compute]: no suitable device at index " << args.device << "\n";
        return 1;
    }

    const uint FRAMES_IN_FLIGHT = 1; // serialize frames so cross-frame reads (history/accumulation) are race-free and deterministic
    auto queue = gpuCreateQueue(device);
    auto semaphore = gpuCreateSemaphore(device, 0);
    LinearAllocator allocator(device);

    auto computeIR = loadIR(std::string(NGAPI_TEST_SHADER_DIR) + "/compute/Compute.spv");
    auto pipeline = gpuCreateComputePipeline(device, ByteSpan(computeIR.data(), computeIR.size()));

    auto textureHeap = allocator.allocate<GpuTextureDescriptor>(1024);

    int width, height, channels;
    const std::string inputPath = std::string(NGAPI_TEST_ASSET_DIR) + "/Default.png";
    stbi_uc* inputImage = stbi_load(inputPath.c_str(), &width, &height, &channels, 4);
    if (!inputImage)
    {
        std::cerr << "FAIL [compute]: could not load " << inputPath << "\n";
        return 1;
    }

    auto upload = allocator.allocate<uint8_t>(width * height * 4);
    memcpy(upload.cpu, inputImage, width * height * 4);

    GpuTextureDesc textureDesc{
        .type = TEXTURE_2D,
        .dimensions = { static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1 },
        .format = FORMAT_RGBA8_UNORM,
        .usage = static_cast<USAGE_FLAGS>(USAGE_SAMPLED | USAGE_TRANSFER_DST)
    };
    GpuTextureSizeAlign sizeAlign = gpuTextureSizeAlign(device, textureDesc);
    void* texturePtr = gpuMalloc(device, sizeAlign.size, MEMORY_GPU);
    auto texture = gpuCreateTexture(device, textureDesc, texturePtr);

    GpuTextureDesc outputDesc{
        .type = TEXTURE_2D,
        .dimensions = { static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1 },
        .format = FORMAT_RGBA8_UNORM,
        .usage = static_cast<USAGE_FLAGS>(USAGE_STORAGE | USAGE_TRANSFER_SRC)
    };
    void* outputPtr = gpuMalloc(device, sizeAlign.size, MEMORY_GPU);
    auto outputTexture = gpuCreateTexture(device, outputDesc, outputPtr);

    textureHeap.cpu[0] = gpuTextureViewDescriptor(texture, GpuViewDesc{ .format = FORMAT_RGBA8_UNORM });
    textureHeap.cpu[1] = gpuRWTextureViewDescriptor(outputTexture, GpuViewDesc{ .format = FORMAT_RGBA8_UNORM });

    // Upload the input image once.
    auto commandBuffer = gpuStartCommandRecording(queue);
    gpuCopyToTexture(commandBuffer, upload.gpu, texture);
    gpuSubmit(queue, Span<GpuCommandBuffer>(&commandBuffer, 1), semaphore, 1);
    gpuWaitSemaphore(semaphore, 1);

    auto data = allocator.allocate<ComputeData>(1);
    data.cpu->imageSize = { static_cast<uint32_t>(width), static_cast<uint32_t>(height) };
    data.cpu->srcTexture = 0;
    data.cpu->dstTexture = 1;

    const uint32_t tintCount = 4;
    auto tints = allocator.allocate<Tint>(tintCount);
    tints.cpu[0] = { { 1.0f, 0.5f, 0.5f }, 0.6f };
    tints.cpu[1] = { { 0.5f, 1.0f, 0.5f }, 0.6f };
    tints.cpu[2] = { { 0.5f, 0.5f, 1.0f }, 0.6f };
    tints.cpu[3] = { { 1.0f, 1.0f, 0.5f }, 0.6f };
    data.cpu->tints = tints.gpu;
    data.cpu->tintCount = tintCount;

    // Fresh timeline for the frame loop (matches the sample structure).
    gpuDestroySemaphore(semaphore);
    semaphore = gpuCreateSemaphore(device, 0);

    // Ceil division so every output pixel is written (deterministic golden).
    const uint32_t groupsX = (static_cast<uint32_t>(width) + 15) / 16;
    const uint32_t groupsY = (static_cast<uint32_t>(height) + 15) / 16;

    uint64_t nextFrame = 1;
    for (uint32_t frame = 0; frame < args.frames; frame++)
    {
        if (nextFrame > FRAMES_IN_FLIGHT)
        {
            gpuWaitSemaphore(semaphore, nextFrame - FRAMES_IN_FLIGHT);
        }

        commandBuffer = gpuStartCommandRecording(queue);
        gpuSetPipeline(commandBuffer, pipeline);
        gpuSetActiveTextureHeapPtr(commandBuffer, textureHeap.gpu);
        gpuDispatch(commandBuffer, data.gpu, { groupsX, groupsY, 1 });
        gpuSubmit(queue, Span<GpuCommandBuffer>(&commandBuffer, 1), semaphore, nextFrame);
        nextFrame++;
    }
    gpuWaitSemaphore(semaphore, nextFrame - 1);

    test::Image actual = test::readbackRGBA8(device, queue, outputTexture, width, height);
    int rc = test::finalize(args, "compute", actual);

    stbi_image_free(inputImage);
    allocator.free();
    gpuDestroySemaphore(semaphore);
    gpuDestroyTexture(texture);
    gpuDestroyTexture(outputTexture);
    gpuFreePipeline(pipeline);
    gpuFree(device, texturePtr);
    gpuFree(device, outputPtr);
    gpuDestroyQueue(queue);
    gpuDestroyDevice(device);
    test::endValidationCapture();
    gpuDestroyInstance();

    if (test::validationFailed())
    {
        std::cerr << "FAIL [compute]: Vulkan validation messages were emitted\n";
        rc = 1;
    }
    return rc;
}
