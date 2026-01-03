#include "Raytracing.h"
#include "../../Utilities.h"
#include "../../External/stb_image.h"
#include "../../External/stb_image_write.h"

#include <SDL3/SDL.h>
#include "../../SDL_gpu.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

void raytracingSample()
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

    auto swapchain = gpuCreateSwapchain(surface, FRAMES_IN_FLIGHT);
    auto swapchainDesc = gpuSwapchainDesc(swapchain);

    GpuTextureDesc textureDesc{
        .type = TEXTURE_2D,
        .dimensions = { static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1 },
        .format = FORMAT_RGBA8_UNORM,
        .usage = static_cast<USAGE_FLAGS>(USAGE_SAMPLED | USAGE_TRANSFER_DST)
    };

    GpuTextureSizeAlign textureSizeAlign = gpuTextureSizeAlign(textureDesc);
    void* texturePtr = gpuMalloc(textureSizeAlign.size, MEMORY_GPU);
    auto texture = gpuCreateTexture(textureDesc, texturePtr);

    GpuTextureDesc outputTextureDesc{
        .type = TEXTURE_2D,
        .dimensions = { static_cast<uint32_t>(swapchainDesc.dimensions.x), static_cast<uint32_t>(swapchainDesc.dimensions.y), 1 },
        .format = FORMAT_RGBA8_UNORM,
        .usage = static_cast<USAGE_FLAGS>(USAGE_STORAGE | USAGE_TRANSFER_SRC)
    };

    GpuTextureSizeAlign outputTextureSizeAlign = gpuTextureSizeAlign(outputTextureDesc);
    void* outputTexturePtr = gpuMalloc(outputTextureSizeAlign.size, MEMORY_GPU);
    auto outputTexture = gpuCreateTexture(outputTextureDesc, outputTexturePtr);

    auto textureHeap = allocate<GpuTextureDescriptor>(1024);
    textureHeap.cpu[0] = gpuTextureViewDescriptor(texture, GpuViewDesc{.format = FORMAT_RGBA8_UNORM });
    textureHeap.cpu[1] = gpuRWTextureViewDescriptor(outputTexture, GpuViewDesc{ .format = FORMAT_RGBA8_UNORM });

    ColorTarget colorTarget = {};
    colorTarget.format = swapchainDesc.format;

    GpuRasterDesc rasterDesc = {
        .cull = CULL_CW,
        .colorTargets = Span<ColorTarget>(&colorTarget, 1)
    };

    auto computeIR = loadIR("../../../Samples/Raytracing/Raytracing.spv");
    auto pipeline = gpuCreateComputePipeline(ByteSpan(computeIR.data(), computeIR.size()));

    auto mesh = createMesh("../../../Assets/Cube.obj");
    auto meshData = allocate<MeshData>(); // only allocates the struct, not the geometry data
    mesh.allocate(meshData.cpu);
    mesh.load(meshData.cpu);

    GpuAccelerationStructureTrianglesDesc triangleDesc = {
        .vertexDataGpu = gpuHostToDevicePointer(meshData.cpu->vertices),
        .vertexCount = static_cast<uint32_t>(mesh.vertices.size()),
        .vertexStride = sizeof(float4),
        .vertexFormat = FORMAT_RGB32_FLOAT, // w component unused
        .indexDataGpu = gpuHostToDevicePointer(meshData.cpu->indices),
        .indexType = INDEX_TYPE_UINT32,
        .transformDataGpu = nullptr // optional
    };

    GpuAccelerationStructureDesc blasDesc = {
        .type = TYPE_BOTTOM_LEVEL,
        .blasDesc = {
            .type = GEOMETRY_TYPE_TRIANGLES,
            .triangles = Span<GpuAccelerationStructureTrianglesDesc>(&triangleDesc, 1)
        },
        .buildRange = new GpuAccelerationStructureBuildRange{
            .primitiveCount = static_cast<uint32_t>(mesh.indices.size() / 3),
            .primitiveOffset = 0,
            .firstVertex = 0,
            .transformOffset = 0
        }
    };

    auto blasSize = gpuAccelerationStructureSizes(blasDesc);
    void* blasPtr = gpuMalloc(blasSize.size, MEMORY_GPU);
    auto blas = gpuCreateAccelerationStructure(blasDesc, blasPtr, blasSize.size);

    auto instance = gpuMalloc<GpuAccelerationStructureInstanceDesc>();
    instance->transform = float3x4{
        {1,0,0,0},
        {0,1,0,0},
        {0,0,1,0}
    };
    instance->instanceID = 0;
    instance->instanceMask = 0xFF;
    instance->hitGroupIndex = 0;
    instance->flags = 0; // Or VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR etc.
    instance->blasAddress = blasPtr;
    
    GpuAccelerationStructureDesc tlasDesc = {
        .type = TYPE_TOP_LEVEL,
        .tlasDesc = {
            .arrayOfPointers = false,
            .instancesGpu = gpuHostToDevicePointer(instance)
        },
        .buildRange = new GpuAccelerationStructureBuildRange{
            .primitiveCount = 1,
            .primitiveOffset = 0,
            .firstVertex = 0,
            .transformOffset = 0
        }
    };

    auto tlasSize = gpuAccelerationStructureSizes(tlasDesc);
    void* tlasPtr = gpuMalloc(tlasSize.size, MEMORY_GPU);
    auto tlas = gpuCreateAccelerationStructure(tlasDesc, tlasPtr, tlasSize.size);

    auto scratchSize = std::max(blasSize.buildScratchSize, tlasSize.buildScratchSize);
    void* scratchPtr = gpuMalloc(scratchSize, MEMORY_GPU);

    auto camera = glm::vec4{ 0.0f, 3.0f, 5.0f, 1.f };
    auto projection = glm::perspective(glm::radians(45.0f), 1920.0f / 1080.0f, 0.1f, 100.0f);
    auto view = glm::lookAt(glm::vec3(camera), glm::vec3{ 0.0f, 0.0f, 0.0f }, glm::vec3{ 0.0f, -1.0f, 0.0f });
    auto invViewProjection = glm::inverse(projection * view);

    auto viewProjection = projection * view;

    auto raytracingData = allocate<RaytracingData>();
    memcpy(&raytracingData.cpu->cameraPosition, &camera, sizeof(float4));
    memcpy(&raytracingData.cpu->invViewProjection, &invViewProjection, sizeof(float4x4));
    raytracingData.cpu->meshes = meshData.gpu;
    raytracingData.cpu->tlas = tlasPtr;
    raytracingData.cpu->dstTexture = 1;

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
            gpuBarrier(commandBuffer, STAGE_TRANSFER, STAGE_COMPUTE, HAZARD_DESCRIPTORS);

            // Build acceleration structures
            gpuBuildAccelerationStructures(commandBuffer, Span<GpuAccelerationStructure>(&blas, 1), scratchPtr, MODE_BUILD);
            gpuBarrier(commandBuffer, STAGE_ACCELERATION_STRUCTURE_BUILD, STAGE_ACCELERATION_STRUCTURE_BUILD, HAZARD_ACCELERATION_STRUCTURE);
            gpuBuildAccelerationStructures(commandBuffer, Span<GpuAccelerationStructure>(&tlas, 1), scratchPtr, MODE_BUILD);
            gpuBarrier(commandBuffer, STAGE_ACCELERATION_STRUCTURE_BUILD, STAGE_COMPUTE, HAZARD_ACCELERATION_STRUCTURE);
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
        gpuSetActiveTextureHeapPtr(commandBuffer, textureHeap.gpu);
        
        // inline raytracing
        gpuDispatch(commandBuffer, raytracingData.gpu, { (uint32_t)swapchainDesc.dimensions.x / 8, (uint32_t)swapchainDesc.dimensions.y / 8, 1 });
        gpuBarrier(commandBuffer, STAGE_COMPUTE, STAGE_TRANSFER);
        
        // copy to swapchain image
        gpuBlitTexture(commandBuffer, image, outputTexture);

        gpuSubmit(queue, Span<GpuCommandBuffer>(&commandBuffer, 1), semaphore, nextFrame);
        gpuPresent(swapchain, semaphore, nextFrame++);
    }
    
    gpuDestroySemaphore(semaphore);
    gpuDestroySwapchain(swapchain);
    SDL_Gpu_DestroySurface(surface);
    SDL_DestroyWindow(window);

    gpuDestroyDevice();
}