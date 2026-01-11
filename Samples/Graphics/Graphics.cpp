#include "Graphics.h"
#include "../../Utilities.h"
#include "../../External/stb_image.h"
#include "../../External/stb_image_write.h"

#include <SDL3/SDL.h>
#include "../../SDL_gpu.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

std::vector<glm::vec2> haltonSequence(uint length = 16)
{
    std::vector<glm::vec2> sequence;
    sequence.reserve(length);
    for (uint i = 0; i < length; i++)
    {
        float x = 0.0f;
        float f = 1.0f;
        uint index = i + 1;
        while (index > 0)
        {
            f /= 2.0f;
            x += f * (index % 2);
            index /= 2;
        }

        float y = 0.0f;
        f = 1.0f;
        index = i + 1;
        while (index > 0)
        {
            f /= 3.0f;
            y += f * (index % 3);
            index /= 3;
        }

        sequence.push_back(glm::vec2(x, y));
    }
    return sequence;
}

void graphicsSample()
{
    gpuCreateDevice();

    const uint FRAMES_IN_FLIGHT = 2;

    auto window = SDL_CreateWindow("Test Window", 1920, 1080, SDL_WINDOW_GPU);
    auto surface = SDL_Gpu_CreateSurface(window);
    bool exit = false;

    int width, height, channels;
    stbi_set_flip_vertically_on_load(1);
    stbi_uc* inputImage = stbi_load("../../../Assets/Dice.png", &width, &height, &channels, 4);

    auto upload = allocate<uint8_t>(width * height * 4);
    memcpy(upload.cpu, inputImage, width * height * 4);

    // Cube texture
    GpuTextureDesc textureDesc{
        .type = TEXTURE_2D,
        .dimensions = { static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1 },
        .format = FORMAT_RGBA8_UNORM,
        .usage = static_cast<USAGE_FLAGS>(USAGE_SAMPLED | USAGE_TRANSFER_DST)
    };

    auto swapchain = gpuCreateSwapchain(surface, FRAMES_IN_FLIGHT);
    auto swapchainDesc = gpuSwapchainDesc(swapchain);

    GpuTextureSizeAlign textureSizeAlign = gpuTextureSizeAlign(textureDesc);
    void* texturePtr = gpuMalloc(textureSizeAlign.size, MEMORY_GPU);
    auto texture = gpuCreateTexture(textureDesc, texturePtr);

    // Depth texture
    GpuTextureDesc depthDesc = {
        .type = TEXTURE_2D,
        .dimensions = { swapchainDesc.dimensions.x, swapchainDesc.dimensions.y, 1 },
        .format = FORMAT_D32_FLOAT,
        .usage = static_cast<USAGE_FLAGS>(USAGE_DEPTH_STENCIL_ATTACHMENT | USAGE_SAMPLED)
    };
    
    GpuTextureSizeAlign depthSizeAlign = gpuTextureSizeAlign(depthDesc);
    void* depthPtr = gpuMalloc(depthSizeAlign.size, MEMORY_GPU);
    auto depthTexture = gpuCreateTexture(depthDesc, depthPtr);

    // History texture
    GpuTextureDesc historyTexture{
        .type = TEXTURE_2D,
        .dimensions = { swapchainDesc.dimensions.x, swapchainDesc.dimensions.y, 1 },
        .format = FORMAT_RGBA8_UNORM,
        .usage = static_cast<USAGE_FLAGS>(USAGE_STORAGE | USAGE_SAMPLED | USAGE_TRANSFER_SRC | USAGE_TRANSFER_DST)
    };

    GpuTextureSizeAlign historyTextureSizeAlign = gpuTextureSizeAlign(historyTexture);
    void* historyTexturePtr = gpuMalloc(historyTextureSizeAlign.size, MEMORY_GPU);
    auto historyTextureGpu = gpuCreateTexture(historyTexture, historyTexturePtr);

    // Raster output texture (separate from swapchain to avoid feedback loop in TAA)
    GpuTextureDesc rasterOutputDesc{
        .type = TEXTURE_2D,
        .dimensions = { swapchainDesc.dimensions.x, swapchainDesc.dimensions.y, 1 },
        .format = FORMAT_RGBA8_UNORM,
        .usage = static_cast<USAGE_FLAGS>(USAGE_COLOR_ATTACHMENT | USAGE_SAMPLED)
    };

    GpuTextureSizeAlign rasterOutputSizeAlign = gpuTextureSizeAlign(rasterOutputDesc);
    void* rasterOutputPtr = gpuMalloc(rasterOutputSizeAlign.size, MEMORY_GPU);
    auto rasterOutputGpu = gpuCreateTexture(rasterOutputDesc, rasterOutputPtr);

    // TAA output texture
    GpuTextureDesc taaOutput{
        .type = TEXTURE_2D,
        .dimensions = { swapchainDesc.dimensions.x, swapchainDesc.dimensions.y, 1 },
        .format = FORMAT_RGBA8_UNORM,
        .usage = static_cast<USAGE_FLAGS>(USAGE_STORAGE | USAGE_TRANSFER_SRC)
    };

    GpuTextureSizeAlign taaOutputSizeAlign = gpuTextureSizeAlign(taaOutput);
    void* taaOutputPtr = gpuMalloc(taaOutputSizeAlign.size, MEMORY_GPU);
    auto taaOutputGpu = gpuCreateTexture(taaOutput, taaOutputPtr);

    // Motion vectors
    GpuTextureDesc motionVectorsTexture{
        .type = TEXTURE_2D,
        .dimensions = { swapchainDesc.dimensions.x, swapchainDesc.dimensions.y, 1 },
        .format = FORMAT_RGBA32_FLOAT,
        .usage = static_cast<USAGE_FLAGS>(USAGE_COLOR_ATTACHMENT | USAGE_SAMPLED)
    };

    GpuTextureSizeAlign motionVectorsTextureSizeAlign = gpuTextureSizeAlign(motionVectorsTexture);
    void* motionVectorsTexturePtr = gpuMalloc(motionVectorsTextureSizeAlign.size, MEMORY_GPU);
    auto motionVectorsTextureGpu = gpuCreateTexture(motionVectorsTexture, motionVectorsTexturePtr);

    enum HeapIndices
    {
        INDEX_CUBE = 0,
        INDEX_CURRENT_FRAME = 1,
        INDEX_HISTORY = 2,
        INDEX_DEPTH = 3,
        INDEX_MOTION_VECTORS = 4,
        INDEX_TAA_OUTPUT = 5,
    };

    // Texture Heap
    auto textureHeap = allocate<GpuTextureDescriptor>(1024);
    textureHeap.cpu[HeapIndices::INDEX_CUBE] = gpuTextureViewDescriptor(texture, GpuViewDesc{.format = textureDesc.format });
    textureHeap.cpu[HeapIndices::INDEX_CURRENT_FRAME] = gpuTextureViewDescriptor(rasterOutputGpu, GpuViewDesc{ .format = rasterOutputDesc.format });
    textureHeap.cpu[HeapIndices::INDEX_HISTORY] = gpuTextureViewDescriptor(historyTextureGpu, GpuViewDesc{ .format = historyTexture.format });
    textureHeap.cpu[HeapIndices::INDEX_DEPTH] = gpuTextureViewDescriptor(depthTexture, GpuViewDesc{ .format = depthDesc.format });
    textureHeap.cpu[HeapIndices::INDEX_MOTION_VECTORS] = gpuTextureViewDescriptor(motionVectorsTextureGpu, GpuViewDesc{ .format = motionVectorsTexture.format });
    textureHeap.cpu[HeapIndices::INDEX_TAA_OUTPUT] = gpuRWTextureViewDescriptor(taaOutputGpu, GpuViewDesc{ .format = taaOutput.format });

    ColorTarget colorTargets[2] = {};
    colorTargets[0].format = rasterOutputDesc.format;
    colorTargets[1].format = motionVectorsTexture.format;

    GpuRasterDesc rasterDesc = {
        .cull = CULL_CW,
        .depthFormat = depthDesc.format,
        .colorTargets = Span<ColorTarget>(colorTargets, 2)
    };

    auto vertexIR = loadIR("../../../Samples/Graphics/Vertex.spv");
    auto pixelIR = loadIR("../../../Samples/Graphics/Pixel.spv");
    auto pipeline = gpuCreateGraphicsPipeline(
        ByteSpan(vertexIR),
        ByteSpan(pixelIR),
        rasterDesc
    );

    GpuDepthStencilDesc depthDescState = {
        .depthMode = static_cast<DEPTH_FLAGS>(DEPTH_READ | DEPTH_WRITE),
        .depthTest = OP_LESS,
    };
    GpuDepthStencilState depthState = gpuCreateDepthStencilState(depthDescState);

    auto taaIR = loadIR("../../../Samples/Graphics/TAA.spv");
    auto taaPipeline = gpuCreateComputePipeline(
        ByteSpan(taaIR)
    );

    std::vector<Vertex> verticesObj;
    std::vector<uint32_t> indicesObj;

    loadOBJ("../../../Assets/Cube.obj", verticesObj, indicesObj);

    auto vertices = allocate<Vertex>(static_cast<int>(verticesObj.size()));
    memcpy(vertices.cpu, verticesObj.data(), sizeof(Vertex) * verticesObj.size());

    auto indices = allocate<uint32_t>(static_cast<int>(indicesObj.size()));
    memcpy(indices.cpu, indicesObj.data(), sizeof(uint32_t) * indicesObj.size());

    auto instances = allocate<Instance>(2);

    auto vertexData = allocate<VertexData>();
    auto pixelData = allocate<PixelData>();
    vertexData.cpu->vertices = vertices.gpu;
    vertexData.cpu->instances = instances.gpu;
    pixelData.cpu->srcTexture = 0;

    auto haltonSeq = haltonSequence();

    auto projection = glm::perspective(glm::radians(45.0f), 1920.0f / 1080.0f, 0.1f, 100.0f);
    auto view = glm::lookAt(glm::vec3{ 0.0f, 3.0f, 5.0f }, glm::vec3{ 0.0f, 0.0f, 0.0f }, glm::vec3{ 0.0f, -1.0f, 0.0f });

    float xRotation = 0.0f;
    float yRotation = 0.0f;
    glm::vec3 translation = glm::vec3(1.5f, 0.0f, 0.0f);
    auto instance0 = glm::rotate(glm::translate(glm::mat4(1.f), translation), xRotation, glm::vec3(1.0f, 0.0f, 0.0f));
    auto instance1 = glm::rotate(glm::translate(glm::mat4(1.f), -translation), yRotation, glm::vec3(0.0f, 1.0f, 0.0f));
    memcpy(&instances.cpu[0].model, &instance0, sizeof(float4x4));
    memcpy(&instances.cpu[1].model, &instance1, sizeof(float4x4));
    memcpy(&instances.cpu[0].prevModel, &instance0, sizeof(float4x4));
    memcpy(&instances.cpu[1].prevModel, &instance1, sizeof(float4x4));

    // TAA data
    auto taaData = allocate<TAAData>();
    taaData.cpu->width = swapchainDesc.dimensions.x;
    taaData.cpu->height = swapchainDesc.dimensions.y;
    taaData.cpu->frame = 0;
    taaData.cpu->srcColor = HeapIndices::INDEX_CURRENT_FRAME;
    taaData.cpu->srcHistory = HeapIndices::INDEX_HISTORY;
    taaData.cpu->srcDepth = HeapIndices::INDEX_DEPTH;
    taaData.cpu->srcMotionVectors = HeapIndices::INDEX_MOTION_VECTORS;
    taaData.cpu->dstTexture = HeapIndices::INDEX_TAA_OUTPUT;

    auto queue = gpuCreateQueue();
    auto semaphore = gpuCreateSemaphore(0);
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

        // Update camera with jitter
        auto prevViewProjectionNj = projection * view;

        // update view matrix here for moving camera

        float jitterX = (haltonSeq[nextFrame % haltonSeq.size()].x - 0.5f) / swapchainDesc.dimensions.x;
        float jitterY = (haltonSeq[nextFrame % haltonSeq.size()].y - 0.5f) / swapchainDesc.dimensions.y;

        auto projectionWithJitter = projection;
        projectionWithJitter[2][0] += jitterX * 2.0f;
        projectionWithJitter[2][1] += jitterY * 2.0f;
        auto viewProjection = projectionWithJitter * view;
        auto viewProjectionNj = projection * view;
        memcpy(&vertexData.cpu->viewProjection, &viewProjection, sizeof(float4x4));
        memcpy(&vertexData.cpu->viewProjectionNj, &viewProjectionNj, sizeof(float4x4));
        memcpy(&vertexData.cpu->prevViewProjectionNj, &prevViewProjectionNj, sizeof(float4x4));

        // Pass jitter to TAA shader for unjittering
        taaData.cpu->jitter = { jitterX, jitterY };

        auto commandBuffer = gpuStartCommandRecording(queue);

        if (nextFrame == 1)
        {
            // First frame, copy texture data
            gpuCopyToTexture(commandBuffer, upload.gpu, texture);
            gpuBarrier(commandBuffer, STAGE_TRANSFER, STAGE_PIXEL_SHADER);
        }

        if (nextFrame > FRAMES_IN_FLIGHT)
        {
            gpuWaitSemaphore(semaphore, nextFrame - FRAMES_IN_FLIGHT);
        }

        auto image = gpuSwapchainImage(swapchain);

        GpuTexture colorTargets[2] = { rasterOutputGpu, motionVectorsTextureGpu };
        GpuRenderPassDesc renderPassDesc = {
            .colorTargets = Span<GpuTexture>(colorTargets, 2),
            .depthStencilTarget = depthTexture
        };

        // Raster pass
        gpuSetPipeline(commandBuffer, pipeline);
        gpuBeginRenderPass(commandBuffer, renderPassDesc);
        gpuSetDepthStencilState(commandBuffer, depthState);
        gpuSetActiveTextureHeapPtr(commandBuffer, textureHeap.gpu);
        gpuDrawIndexedInstanced(commandBuffer, vertexData.gpu, pixelData.gpu, indices.gpu, 36, 2);
        gpuEndRenderPass(commandBuffer);

        // TAA pass
        gpuBarrier(commandBuffer, STAGE_RASTER_COLOR_OUT, STAGE_COMPUTE, HAZARD_DESCRIPTORS);
        gpuSetPipeline(commandBuffer, taaPipeline);
        gpuSetActiveTextureHeapPtr(commandBuffer, textureHeap.gpu);
        gpuDispatch(commandBuffer, taaData.gpu, { swapchainDesc.dimensions.x / 16, swapchainDesc.dimensions.y / 16, 1 });

        // Blit taa output to swapchain and copy to history texture
        gpuBarrier(commandBuffer, STAGE_COMPUTE, STAGE_TRANSFER);
        gpuBlitTexture(commandBuffer, image, taaOutputGpu);
        gpuBlitTexture(commandBuffer, historyTextureGpu, taaOutputGpu);

        gpuSubmit(queue, Span<GpuCommandBuffer>(&commandBuffer, 1), semaphore, nextFrame);
        gpuPresent(swapchain, semaphore, nextFrame++);

        xRotation += 0.0001f;
        yRotation += 0.0001f;
        auto instance0 = glm::translate(glm::mat4(1.f), translation) * glm::rotate(glm::mat4(1.f), xRotation, glm::vec3(1.0f, 0.0f, 0.0f));
        auto instance1 = glm::translate(glm::mat4(1.f), -translation) * glm::rotate(glm::mat4(1.f), yRotation, glm::vec3(0.0f, 1.0f, 0.0f));
        
        // copy current model to previous model
        memcpy(&instances.cpu[0].prevModel, &instances.cpu[0].model, sizeof(float4x4));
        memcpy(&instances.cpu[1].prevModel, &instances.cpu[1].model, sizeof(float4x4));

        // update model matrices
        memcpy(&instances.cpu[0].model, &instance0, sizeof(float4x4));
        memcpy(&instances.cpu[1].model, &instance1, sizeof(float4x4));

        // Increment TAA frame counter
        taaData.cpu->frame++;
    }

    gpuDestroySemaphore(semaphore);
    gpuDestroySwapchain(swapchain);
    SDL_Gpu_DestroySurface(surface);
    SDL_DestroyWindow(window);

    gpuDestroyDevice();
}