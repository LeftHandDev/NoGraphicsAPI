#include "Graphics.h"
#include "../../Common.h"
#include "../../External/stb_image.h"
#include "../../External/stb_image_write.h"

#include <SDL3/SDL.h>
#include "../../SDL_gpu.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

void graphicsSample()
{
    const uint FRAMES_IN_FLIGHT = 2;

    auto window = SDL_CreateWindow("Test Window", 1920, 1080, SDL_WINDOW_GPU);
    auto surface = SDL_Gpu_CreateSurface(window);
    bool exit = false;


    int width, height, channels;
    stbi_set_flip_vertically_on_load(1);
    stbi_uc* inputImage = stbi_load("../../../Assets/Dice.png", &width, &height, &channels, 4);

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

    auto textureHeap = allocate<GpuTextureDescriptor>(1024);
    textureHeap.cpu[0] = gpuTextureViewDescriptor(texture, GpuViewDesc{.format = FORMAT_RGBA8_UNORM });


    auto swapchain = gpuCreateSwapchain(surface, FRAMES_IN_FLIGHT);
    auto swapchainDesc = gpuSwapchainDesc(swapchain);

    ColorTarget colorTarget = {};
    colorTarget.format = swapchainDesc.format;

    GpuRasterDesc rasterDesc = {
        .cull = CULL_CW,
        .colorTargets = Span<ColorTarget>(&colorTarget, 1)
    };

    auto vertexIR = Utilities::loadIR("../../../Samples/Graphics/Vertex.spv");
    auto pixelIR = Utilities::loadIR("../../../Samples/Graphics/Pixel.spv");

    auto pipeline = gpuCreateGraphicsPipeline(
        ByteSpan(vertexIR.data(), vertexIR.size()),
        ByteSpan(pixelIR.data(), pixelIR.size()),
        rasterDesc
    );

    std::vector<Vertex> verticesObj;
    std::vector<uint32_t> indicesObj;

    Utilities::loadOBJ("../../../Assets/Cube.obj", verticesObj, indicesObj);

    auto vertices = allocate<Vertex>(static_cast<int>(verticesObj.size()));
    memcpy(vertices.cpu, verticesObj.data(), sizeof(Vertex) * verticesObj.size());
    auto indices = allocate<uint32_t>(static_cast<int>(indicesObj.size()));
    memcpy(indices.cpu, indicesObj.data(), sizeof(uint32_t) * indicesObj.size());

    auto vertexData = allocate<VertexData>();
    auto pixelData = allocate<PixelData>();

    vertexData.cpu->vertices = vertices.gpu;
    pixelData.cpu->srcTexture = 0;

    auto projection = glm::perspective(glm::radians(45.0f), 1920.0f / 1080.0f, 0.1f, 100.0f);
    auto view = glm::lookAt(glm::vec3{ 3.0f, 3.0f, 3.0f }, glm::vec3{ 0.0f, 0.0f, 0.0f }, glm::vec3{ 0.0f, -1.0f, 0.0f });

    float yRotation = 0.0f;
    auto mvp = projection * view;
    memcpy(&vertexData.cpu->mvp, &mvp, sizeof(float4x4));

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

        auto commandBuffer = gpuStartCommandRecording(queue);

        if (nextFrame == 1)
        {
            // First frame, copy texture data
            gpuCopyToTexture(commandBuffer, upload.gpu, texture);
           // gpuBarrier(commandBuffer, STAGE_TRANSFER, STAGE_PIXEL_SHADER, HAZARD_DESCRIPTORS);
        }

        if (nextFrame > FRAMES_IN_FLIGHT)
        {
            gpuWaitSemaphore(semaphore, nextFrame - FRAMES_IN_FLIGHT);
        }

        auto image = gpuSwapchainImage(swapchain);
        GpuRenderPassDesc renderPassDesc = {
            .colorTargets = Span<GpuTexture>(&image, 1),
            .depthStencilTarget = nullptr
        };

        gpuSetPipeline(commandBuffer, pipeline);
        gpuBeginRenderPass(commandBuffer, renderPassDesc);
        gpuSetActiveTextureHeapPtr(commandBuffer, textureHeap.gpu);
        gpuDrawIndexedInstanced(commandBuffer, vertexData.gpu, pixelData.gpu, indices.gpu, 36, 1);
        gpuEndRenderPass(commandBuffer);
        gpuSubmit(queue, Span<GpuCommandBuffer>(&commandBuffer, 1), semaphore, nextFrame);
        gpuPresent(swapchain, semaphore, nextFrame++);

        mvp = projection * view * glm::rotate(glm::mat4(1.0f), yRotation += 0.0001f, glm::vec3{ 0.0f, 1.0f, 0.0f });
        memcpy(&vertexData.cpu->mvp, &mvp, sizeof(float4x4));
    }
    
    gpuDestroySemaphore(semaphore);
    gpuDestroySwapchain(swapchain);
    SDL_Gpu_DestroySurface(surface);
    SDL_DestroyWindow(window);
}