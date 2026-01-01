#include "Graphics.h"
#include "../../Common.h"
#include "../../External/stb_image.h"
#include "../../External/stb_image_write.h"

#include "../../SDL_gpu.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// Gets the size requirements for a obj file
// void getObjectSizeRequirements(const std::filesystem::path& path, size_t vertices, size_t indiecs, size_t uvs, size_t normals)
// {

// }

void graphicsSample()
{
    const uint FRAMES_IN_FLIGHT = 2;

    auto window = SDL_CreateWindow("Test Window", 1920, 1080, SDL_WINDOW_GPU);
    auto surface = SDL_Gpu_CreateSurface(window);
    bool exit = false;


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

    auto textureHeap = allocate<GpuTextureDescriptor>(1024);
    textureHeap.cpu[0] = gpuTextureViewDescriptor(texture, GpuViewDesc{.format = FORMAT_RGBA8_UNORM });


    auto swapchain = gpuCreateSwapchain(surface, FRAMES_IN_FLIGHT);
    auto swapchainDesc = gpuSwapchainDesc(swapchain);

    ColorTarget colorTarget = {};
    colorTarget.format = swapchainDesc.format;

    GpuRasterDesc rasterDesc = {
        .colorTargets = Span<ColorTarget>(&colorTarget, 1)
    };

    auto vertexIR = Utilities::loadIR("../../../Samples/Graphics/Vertex.spv");
    auto pixelIR = Utilities::loadIR("../../../Samples/Graphics/Pixel.spv");

    auto pipeline = gpuCreateGraphicsPipeline(
        ByteSpan(vertexIR.data(), vertexIR.size()),
        ByteSpan(pixelIR.data(), pixelIR.size()),
        rasterDesc
    );

    auto vertices = allocate<Vertex>(8);
    auto indices = allocate<uint3>(12);
    
    vertices.cpu[0].position = float4{-1.0f, -1.0f, -1.0f, 1.0f};
    vertices.cpu[1].position = float4{ 1.0f, -1.0f, -1.0f, 1.0f };
    vertices.cpu[2].position = float4{ 1.0f,  1.0f, -1.0f, 1.0f};
    vertices.cpu[3].position = float4{-1.0f,  1.0f, -1.0f, 1.0f };
    vertices.cpu[4].position = float4{-1.0f, -1.0f,  1.0f, 1.0f};
    vertices.cpu[5].position = float4{ 1.0f, -1.0f,  1.0f, 1.0f };
    vertices.cpu[6].position = float4{ 1.0f,  1.0f,  1.0f, 1.0f};
    vertices.cpu[7].position = float4{-1.0f,  1.0f,  1.0f, 1.0f };

    vertices.cpu[0].uv = float2{0.0f, 0.0f};
    vertices.cpu[1].uv = float2{1.0f, 0.0f};
    vertices.cpu[2].uv = float2{1.0f, 1.0f};
    vertices.cpu[3].uv = float2{0.0f, 1.0f};
    vertices.cpu[4].uv = float2{0.0f, 0.0f};
    vertices.cpu[5].uv = float2{1.0f, 0.0f};
    vertices.cpu[6].uv = float2{1.0f, 1.0f};
    vertices.cpu[7].uv = float2{0.0f, 1.0f};

    indices.cpu[0] = uint3{0,1,2};
    indices.cpu[1] = uint3{2,3,0};
    indices.cpu[2] = uint3{4,5,6};
    indices.cpu[3] = uint3{6,7,4};
    indices.cpu[4] = uint3{0,4,7};
    indices.cpu[5] = uint3{7,3,0};
    indices.cpu[6] = uint3{1,5,6};
    indices.cpu[7] = uint3{6,2,1};
    indices.cpu[8] = uint3{3,2,6};
    indices.cpu[9] = uint3{6,7,3};
    indices.cpu[10] = uint3{0,1,5};
    indices.cpu[11] = uint3{5,4,0};

    auto vertexData = allocate<VertexData>();
    auto pixelData = allocate<PixelData>();

    vertexData.cpu->vertices = vertices.gpu;
    pixelData.cpu->srcTexture = 0;

    auto projection = glm::perspective(glm::radians(45.0f), 1920.0f / 1080.0f, 0.1f, 100.0f);
    auto view = glm::lookAt(glm::vec3{ 3.0f, 3.0f, 3.0f }, glm::vec3{ 0.0f, 0.0f, 0.0f }, glm::vec3{ 0.0f, -1.0f, 0.0f });

    float yRotation = 0.0f;
    auto model = glm::rotate(glm::mat4(1.0f), yRotation, glm::vec3{ 0.0f, 1.0f, 0.0f });

    auto mvp = projection * view * model;
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

        if (nextFrame == 1)
        {
            // First frame, copy texture data
            auto commandBuffer = gpuStartCommandRecording(queue);
            auto uploadSemaphore = gpuCreateSemaphore(0);
            gpuCopyToTexture(commandBuffer, upload.gpu, texture);
            gpuSubmit(queue, Span<GpuCommandBuffer>(&commandBuffer, 1), uploadSemaphore, 1);
            gpuWaitSemaphore(uploadSemaphore, 1);
            gpuDestroySemaphore(uploadSemaphore);
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

        auto commandBuffer = gpuStartCommandRecording(queue);
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