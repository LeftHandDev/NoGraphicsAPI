// Headless test mirroring the Graphics sample: two rotating textured cubes with
// TAA, rendered for a fixed number of frames (frame-driven animation, no wall
// clock, so it is deterministic), with the TAA output blitted into a capture
// texture that is read back and compared to a golden.
#include "test_common.h"

#include "Utilities.h" // LinearAllocator, loadIR, getCube, haltonSequence
#include "Graphics.h"  // VertexData, PixelData, Instance
#include "TAA.h"       // TAAData

#include "stb_image.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cstring>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    test::Args args = test::parseArgs(argc, argv);

    gpuCreateInstance();
    test::beginValidationCapture();

    auto device = gpuCreateDevice(args.device);
    if (!device)
    {
        std::cerr << "FAIL [graphics]: no suitable device at index " << args.device << "\n";
        return 1;
    }

    const uint FRAMES_IN_FLIGHT = 1; // serialize frames so cross-frame reads (history/accumulation) are race-free and deterministic
    const uint32_t RENDER_W = 320;   // divisible by 16 for the TAA dispatch
    const uint32_t RENDER_H = 240;

    LinearAllocator allocator(device);

    int width, height, channels;
    const std::string inputPath = std::string(NGAPI_TEST_ASSET_DIR) + "/Default.png";
    stbi_uc* inputImage = stbi_load(inputPath.c_str(), &width, &height, &channels, 4);
    if (!inputImage)
    {
        std::cerr << "FAIL [graphics]: could not load " << inputPath << "\n";
        return 1;
    }
    auto upload = allocator.allocate<uint8_t>(width * height * 4);
    memcpy(upload.cpu, inputImage, width * height * 4);

    auto makeTexture = [&](GpuTextureDesc desc, void** ptrOut)
    {
        GpuTextureSizeAlign sa = gpuTextureSizeAlign(device, desc);
        *ptrOut = gpuMalloc(device, sa.size, MEMORY_GPU);
        return gpuCreateTexture(device, desc, *ptrOut);
    };

    void* texturePtr;
    auto texture = makeTexture({ .type = TEXTURE_2D, .dimensions = { (uint32_t)width, (uint32_t)height, 1 }, .format = FORMAT_RGBA8_UNORM, .usage = (USAGE_FLAGS)(USAGE_SAMPLED | USAGE_TRANSFER_DST) }, &texturePtr);

    void* depthPtr;
    GpuTextureDesc depthDesc{ .type = TEXTURE_2D, .dimensions = { RENDER_W, RENDER_H, 1 }, .format = FORMAT_D32_FLOAT, .usage = (USAGE_FLAGS)(USAGE_DEPTH_STENCIL_ATTACHMENT | USAGE_SAMPLED) };
    auto depthTexture = makeTexture(depthDesc, &depthPtr);

    void* historyPtr;
    GpuTextureDesc historyDesc{ .type = TEXTURE_2D, .dimensions = { RENDER_W, RENDER_H, 1 }, .format = FORMAT_RGBA8_UNORM, .usage = (USAGE_FLAGS)(USAGE_STORAGE | USAGE_SAMPLED | USAGE_TRANSFER_SRC | USAGE_TRANSFER_DST) };
    auto historyTexture = makeTexture(historyDesc, &historyPtr);

    void* rasterPtr;
    GpuTextureDesc rasterDesc{ .type = TEXTURE_2D, .dimensions = { RENDER_W, RENDER_H, 1 }, .format = FORMAT_RGBA8_UNORM, .usage = (USAGE_FLAGS)(USAGE_COLOR_ATTACHMENT | USAGE_SAMPLED) };
    auto rasterOutput = makeTexture(rasterDesc, &rasterPtr);

    void* taaPtr;
    GpuTextureDesc taaDesc{ .type = TEXTURE_2D, .dimensions = { RENDER_W, RENDER_H, 1 }, .format = FORMAT_RGBA8_UNORM, .usage = (USAGE_FLAGS)(USAGE_STORAGE | USAGE_TRANSFER_SRC) };
    auto taaOutput = makeTexture(taaDesc, &taaPtr);

    void* motionPtr;
    GpuTextureDesc motionDesc{ .type = TEXTURE_2D, .dimensions = { RENDER_W, RENDER_H, 1 }, .format = FORMAT_RGBA32_FLOAT, .usage = (USAGE_FLAGS)(USAGE_COLOR_ATTACHMENT | USAGE_SAMPLED) };
    auto motionVectors = makeTexture(motionDesc, &motionPtr);

    // Capture target the final image is blitted into (stands in for the swapchain).
    void* capturePtr;
    GpuTextureDesc captureDesc{ .type = TEXTURE_2D, .dimensions = { RENDER_W, RENDER_H, 1 }, .format = FORMAT_RGBA8_UNORM, .usage = (USAGE_FLAGS)(USAGE_TRANSFER_DST | USAGE_TRANSFER_SRC | USAGE_SAMPLED) };
    auto capture = makeTexture(captureDesc, &capturePtr);

    enum HeapIndices
    {
        INDEX_CUBE = 0,
        INDEX_CURRENT_FRAME = 1,
        INDEX_HISTORY = 2,
        INDEX_DEPTH = 3,
        INDEX_MOTION_VECTORS = 4,
        INDEX_TAA_OUTPUT = 5,
    };

    auto textureHeap = allocator.allocate<GpuTextureDescriptor>(1024);
    textureHeap.cpu[INDEX_CUBE] = gpuTextureViewDescriptor(texture, GpuViewDesc{ .format = FORMAT_RGBA8_UNORM });
    textureHeap.cpu[INDEX_CURRENT_FRAME] = gpuTextureViewDescriptor(rasterOutput, GpuViewDesc{ .format = FORMAT_RGBA8_UNORM });
    textureHeap.cpu[INDEX_HISTORY] = gpuTextureViewDescriptor(historyTexture, GpuViewDesc{ .format = FORMAT_RGBA8_UNORM });
    textureHeap.cpu[INDEX_DEPTH] = gpuTextureViewDescriptor(depthTexture, GpuViewDesc{ .format = FORMAT_D32_FLOAT });
    textureHeap.cpu[INDEX_MOTION_VECTORS] = gpuTextureViewDescriptor(motionVectors, GpuViewDesc{ .format = FORMAT_RGBA32_FLOAT });
    textureHeap.cpu[INDEX_TAA_OUTPUT] = gpuRWTextureViewDescriptor(taaOutput, GpuViewDesc{ .format = FORMAT_RGBA8_UNORM });

    ColorTarget colorTargets[2] = {};
    colorTargets[0].format = FORMAT_RGBA8_UNORM;
    colorTargets[1].format = FORMAT_RGBA32_FLOAT;
    GpuRasterDesc rasterPipeDesc = {
        .cull = CULL_CCW,
        .depthFormat = FORMAT_D32_FLOAT,
        .colorTargets = Span<ColorTarget>(colorTargets, 2)
    };

    auto vertexIR = loadIR(std::string(NGAPI_TEST_SHADER_DIR) + "/graphics/Vertex.spv");
    auto pixelIR = loadIR(std::string(NGAPI_TEST_SHADER_DIR) + "/graphics/Pixel.spv");
    auto pipeline = gpuCreateGraphicsPipeline(device, ByteSpan(vertexIR), ByteSpan(pixelIR), rasterPipeDesc);

    GpuDepthStencilDesc depthDescState = { .depthMode = (DEPTH_FLAGS)(DEPTH_READ | DEPTH_WRITE), .depthTest = OP_LESS };
    auto depthState = gpuCreateDepthStencilState(depthDescState);

    auto taaIR = loadIR(std::string(NGAPI_TEST_SHADER_DIR) + "/common/TAA.spv");
    auto taaPipeline = gpuCreateComputePipeline(device, ByteSpan(taaIR));

    std::vector<float3> cubeVertices, cubeNormals;
    std::vector<float2> cubeUVs;
    std::vector<uint32_t> cubeIndices;
    getCube(cubeVertices, cubeNormals, cubeUVs, cubeIndices);

    auto vertices = allocator.allocate<float3>(cubeVertices.size());
    memcpy(vertices.cpu, cubeVertices.data(), sizeof(float3) * cubeVertices.size());
    auto uvs = allocator.allocate<float2>(cubeUVs.size());
    memcpy(uvs.cpu, cubeUVs.data(), sizeof(float2) * cubeUVs.size());
    auto indices = allocator.allocate<uint32_t>(cubeIndices.size());
    memcpy(indices.cpu, cubeIndices.data(), sizeof(uint32_t) * cubeIndices.size());

    auto instances = allocator.allocate<Instance>(2);
    auto vertexData = allocator.allocate<VertexData>(1);
    auto pixelData = allocator.allocate<PixelData>(1);
    vertexData.cpu->vertices = vertices.gpu;
    vertexData.cpu->uvs = uvs.gpu;
    vertexData.cpu->instances = instances.gpu;
    pixelData.cpu->srcTexture = INDEX_CUBE;

    auto haltonSeq = haltonSequence();

    auto projection = glm::perspective(glm::radians(45.0f), float(RENDER_W) / float(RENDER_H), 0.1f, 100.0f);
    auto view = glm::lookAt(glm::vec3{ 0.0f, 3.0f, 5.0f }, glm::vec3{ 0.0f, 0.0f, 0.0f }, glm::vec3{ 0.0f, -1.0f, 0.0f });

    float xRotation = 0.0f, yRotation = 0.0f;
    glm::vec3 translation = glm::vec3(1.5f, 0.0f, 0.0f);
    auto instance0 = glm::rotate(glm::translate(glm::mat4(1.f), translation), xRotation, glm::vec3(1, 0, 0));
    auto instance1 = glm::rotate(glm::translate(glm::mat4(1.f), -translation), yRotation, glm::vec3(0, 1, 0));
    memcpy(&instances.cpu[0].model, &instance0, sizeof(float4x4));
    memcpy(&instances.cpu[1].model, &instance1, sizeof(float4x4));
    memcpy(&instances.cpu[0].prevModel, &instance0, sizeof(float4x4));
    memcpy(&instances.cpu[1].prevModel, &instance1, sizeof(float4x4));

    auto taaData = allocator.allocate<TAAData>(1);
    taaData.cpu->width = RENDER_W;
    taaData.cpu->height = RENDER_H;
    taaData.cpu->frame = 0;
    taaData.cpu->srcColor = INDEX_CURRENT_FRAME;
    taaData.cpu->srcHistory = INDEX_HISTORY;
    taaData.cpu->srcDepth = INDEX_DEPTH;
    taaData.cpu->srcMotionVectors = INDEX_MOTION_VECTORS;
    taaData.cpu->dstTexture = INDEX_TAA_OUTPUT;

    auto queue = gpuCreateQueue(device);
    auto semaphore = gpuCreateSemaphore(device, 0);

    // Clear the TAA history to zero so the first frame's accumulation is
    // deterministic (textures are otherwise uninitialized).
    auto zero = allocator.allocate<uint8_t>(RENDER_W * RENDER_H * 4);
    memset(zero.cpu, 0, RENDER_W * RENDER_H * 4);
    {
        auto clearCmd = gpuStartCommandRecording(queue);
        gpuCopyToTexture(clearCmd, zero.gpu, historyTexture);
        gpuSubmit(queue, Span<GpuCommandBuffer>(&clearCmd, 1), semaphore, 1);
        gpuWaitSemaphore(semaphore, 1);
        gpuDestroySemaphore(semaphore);
        semaphore = gpuCreateSemaphore(device, 0);
    }

    uint64_t nextFrame = 1;

    for (uint32_t frame = 0; frame < args.frames; frame++)
    {
        // The writes below mutate host-visible memory that in-flight frames
        // read (instance matrices, view projections, TAA constants), so the
        // previous frame must have fully completed first. Waiting after the
        // writes let a still-executing frame occasionally read its successor's
        // values — a rare nondeterministic edge shift in the output.
        if (nextFrame > FRAMES_IN_FLIGHT)
        {
            gpuWaitSemaphore(semaphore, nextFrame - FRAMES_IN_FLIGHT);
        }

        if (nextFrame > 1)
        {
            xRotation += 0.0001f;
            yRotation += 0.0001f;
            auto next0 = glm::translate(glm::mat4(1.f), translation) * glm::rotate(glm::mat4(1.f), xRotation, glm::vec3(1, 0, 0));
            auto next1 = glm::translate(glm::mat4(1.f), -translation) * glm::rotate(glm::mat4(1.f), yRotation, glm::vec3(0, 1, 0));
            memcpy(&instances.cpu[0].prevModel, &instances.cpu[0].model, sizeof(float4x4));
            memcpy(&instances.cpu[1].prevModel, &instances.cpu[1].model, sizeof(float4x4));
            memcpy(&instances.cpu[0].model, &next0, sizeof(float4x4));
            memcpy(&instances.cpu[1].model, &next1, sizeof(float4x4));
            taaData.cpu->frame++;
        }

        auto prevViewProjectionNj = projection * view;

        float jitterX = (haltonSeq[nextFrame % haltonSeq.size()].x - 0.5f) / RENDER_W;
        float jitterY = (haltonSeq[nextFrame % haltonSeq.size()].y - 0.5f) / RENDER_H;

        auto projectionWithJitter = projection;
        projectionWithJitter[2][0] += jitterX * 2.0f;
        projectionWithJitter[2][1] += jitterY * 2.0f;
        auto viewProjection = projectionWithJitter * view;
        auto viewProjectionNj = projection * view;
        memcpy(&vertexData.cpu->viewProjection, &viewProjection, sizeof(float4x4));
        memcpy(&vertexData.cpu->viewProjectionNj, &viewProjectionNj, sizeof(float4x4));
        memcpy(&vertexData.cpu->prevViewProjectionNj, &prevViewProjectionNj, sizeof(float4x4));
        taaData.cpu->jitter = { jitterX, jitterY };

        auto commandBuffer = gpuStartCommandRecording(queue);

        if (nextFrame == 1)
        {
            gpuCopyToTexture(commandBuffer, upload.gpu, texture);
            gpuBarrier(commandBuffer, STAGE_TRANSFER, STAGE_PIXEL_SHADER);
        }

        GpuTexture colorTargetsGpu[2] = { rasterOutput, motionVectors };
        GpuRenderPassDesc renderPassDesc = {
            .colorTargets = Span<GpuTexture>(colorTargetsGpu, 2),
            .depthStencilTarget = depthTexture
        };

        gpuSetPipeline(commandBuffer, pipeline);
        gpuSetActiveTextureHeapPtr(commandBuffer, textureHeap.gpu);
        gpuBeginRenderPass(commandBuffer, renderPassDesc);
        gpuSetDepthStencilState(commandBuffer, depthState);
        gpuDrawIndexedInstanced(commandBuffer, vertexData.gpu, pixelData.gpu, indices.gpu, 36, 2);
        gpuEndRenderPass(commandBuffer);

        gpuBarrier(commandBuffer, STAGE_RASTER_COLOR_OUT, STAGE_COMPUTE, HAZARD_DESCRIPTORS);
        gpuSetPipeline(commandBuffer, taaPipeline);
        gpuSetActiveTextureHeapPtr(commandBuffer, textureHeap.gpu);
        gpuDispatch(commandBuffer, taaData.gpu, { RENDER_W / 16, RENDER_H / 16, 1 });

        gpuBarrier(commandBuffer, STAGE_COMPUTE, STAGE_TRANSFER);
        gpuBlitTexture(commandBuffer, capture, taaOutput);
        gpuBlitTexture(commandBuffer, historyTexture, taaOutput);

        gpuSubmit(queue, Span<GpuCommandBuffer>(&commandBuffer, 1), semaphore, nextFrame);
        nextFrame++;
    }
    gpuWaitSemaphore(semaphore, nextFrame - 1);

    test::Image actual = test::readbackRGBA8(device, queue, capture, RENDER_W, RENDER_H);
    int rc = test::finalize(args, "graphics", actual);

    allocator.reset();
    stbi_image_free(inputImage);
    gpuDestroyTexture(texture);
    gpuFree(device, texturePtr);
    gpuDestroyTexture(depthTexture);
    gpuFree(device, depthPtr);
    gpuDestroyTexture(historyTexture);
    gpuFree(device, historyPtr);
    gpuDestroyTexture(rasterOutput);
    gpuFree(device, rasterPtr);
    gpuDestroyTexture(taaOutput);
    gpuFree(device, taaPtr);
    gpuDestroyTexture(motionVectors);
    gpuFree(device, motionPtr);
    gpuDestroyTexture(capture);
    gpuFree(device, capturePtr);
    gpuFreePipeline(pipeline);
    gpuFreePipeline(taaPipeline);
    gpuFreeDepthStencilState(depthState);
    gpuDestroySemaphore(semaphore);
    gpuDestroyQueue(queue);
    gpuDestroyDevice(device);
    test::endValidationCapture();
    gpuDestroyInstance();

    if (test::validationFailed())
    {
        std::cerr << "FAIL [graphics]: Vulkan validation messages were emitted\n";
        rc = 1;
    }
    return rc;
}
