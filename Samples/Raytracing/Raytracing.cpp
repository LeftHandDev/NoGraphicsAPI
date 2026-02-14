#include "Raytracing.h"
#include "../../Utilities.h"
#include "../../External/stb_image.h"
#include "../../External/stb_image_write.h"

#include <SDL3/SDL.h>
#include "../../SDL_gpu.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>

#include <random>

void raytracingSample()
{
    gpuCreateDevice();

    const uint FRAMES_IN_FLIGHT = 2;

    auto window = SDL_CreateWindow("Test Window", 1920, 1080, SDL_WINDOW_GPU);
    auto surface = SDL_Gpu_CreateSurface(window);
    bool exit = false;

    int width, height, channels;
    stbi_uc *inputImage = stbi_load("Assets/NoGraphicsAPI.png", &width, &height, &channels, 4);

    auto upload = allocate<uint8_t>(width * height * 4);
    memcpy(upload.cpu, inputImage, width * height * 4);

    auto swapchain = gpuCreateSwapchain(surface, FRAMES_IN_FLIGHT);
    auto swapchainDesc = gpuSwapchainDesc(swapchain);

    GpuTextureDesc textureDesc{
        .type = TEXTURE_2D,
        .dimensions = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1},
        .format = FORMAT_RGBA8_UNORM,
        .usage = static_cast<USAGE_FLAGS>(USAGE_SAMPLED | USAGE_TRANSFER_DST)};

    GpuTextureSizeAlign textureSizeAlign = gpuTextureSizeAlign(textureDesc);
    void *texturePtr = gpuMalloc(textureSizeAlign.size, MEMORY_GPU);
    auto texture = gpuCreateTexture(textureDesc, texturePtr);

    GpuTextureDesc outputTextureDesc{
        .type = TEXTURE_2D,
        .dimensions = {static_cast<uint32_t>(swapchainDesc.dimensions.x), static_cast<uint32_t>(swapchainDesc.dimensions.y), 1},
        .format = FORMAT_RGBA32_FLOAT,
        .usage = static_cast<USAGE_FLAGS>(USAGE_STORAGE | USAGE_TRANSFER_SRC)};

    GpuTextureSizeAlign outputTextureSizeAlign = gpuTextureSizeAlign(outputTextureDesc);
    void *outputTexturePtr = gpuMalloc(outputTextureSizeAlign.size, MEMORY_GPU);
    auto outputTexture = gpuCreateTexture(outputTextureDesc, outputTexturePtr);

    auto textureHeap = allocate<GpuTextureDescriptor>(1024);
    textureHeap.cpu[0] = gpuTextureViewDescriptor(texture, GpuViewDesc{.format = FORMAT_RGBA8_UNORM});
    textureHeap.cpu[1] = gpuRWTextureViewDescriptor(outputTexture, GpuViewDesc{.format = FORMAT_RGBA32_FLOAT});

    ColorTarget colorTarget = {};
    colorTarget.format = swapchainDesc.format;

    GpuRasterDesc rasterDesc = {
        .cull = CULL_CW,
        .colorTargets = Span<ColorTarget>(&colorTarget, 1)};

    auto referenceIR = loadIR("../Shaders/Raytracing/Raytracing.spv");
    auto referencePipeline = gpuCreateComputePipeline(ByteSpan(referenceIR));

    auto restirIR = loadIR("../Shaders/Raytracing/ReSTIR.spv");
    auto restirPipeline = gpuCreateComputePipeline(ByteSpan(restirIR));

    size_t allocatorSize = 2 * 1024 * 1024; // 2 MB
    LinearAllocator allocator(allocatorSize);
    auto raytracingData = allocator.allocate<RaytracingData>();

    uint32_t numLights = 100;
    auto lightData = allocator.allocate<LightData>(numLights);

    glm::vec3 cameraPos(0, 0, -5);   
    glm::vec3 prevCameraPos = cameraPos;

    {
        std::random_device rd;
        std::mt19937 gen(rd());
        float offset = static_cast<float>(numLights);
        std::uniform_real_distribution<float> dis(-offset, offset);

        // put one light at the camera so we can always see something
        lightData.cpu[0].position = {cameraPos.x, cameraPos.y, cameraPos.z, 1};
        lightData.cpu[0].color = {1, 1, 1, 1};
        lightData.cpu[0].intensity = 10.0f;

        // Randomly position lights in the scene
        for (int i = 1; i < numLights; ++i)
        {
            lightData.cpu[i].position = {dis(gen), dis(gen), dis(gen), 1};
            lightData.cpu[i].color = {1, 1, 1, 1};
            lightData.cpu[i].intensity = 10.0f;
        }
    }

    auto camDataAlloc = allocator.allocate<CameraData>();
 
    auto setCamera = [&] {
        lightData.cpu[0].position = {cameraPos.x, cameraPos.y, cameraPos.z, 1}; // first light follows the camera, so we can always see something
        camDataAlloc.cpu->position = {cameraPos.x, cameraPos.y, cameraPos.z, 1};
        auto view = glm::lookAt(cameraPos, glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
        auto projection = glm::perspective(glm::radians(60.0f), swapchainDesc.dimensions.x / static_cast<float>(swapchainDesc.dimensions.y), 0.1f, 100.0f);
        auto invViewProjection = glm::inverse(projection * view);
        memcpy(&camDataAlloc.cpu->invViewProjection, &invViewProjection, sizeof(float4x4));

        view = glm::lookAt(prevCameraPos, glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
        auto viewProjection = projection * view;
        memcpy(&camDataAlloc.cpu->prevViewProjection, &viewProjection, sizeof(float4x4));

        prevCameraPos = cameraPos;
    };

    setCamera();

    std::vector<float3> vertices;
    std::vector<float3> normals;
    std::vector<float2> uvs;
    std::vector<uint32_t> indices;
    getCube(vertices, normals, uvs, indices);

    auto vertexBuffer = allocator.allocate<float3>(vertices.size());
    auto normalBuffer = allocator.allocate<float3>(normals.size());
    auto uvBuffer = allocator.allocate<float2>(uvs.size());
    auto indexBuffer = allocator.allocate<uint32_t>(indices.size());
    memcpy(vertexBuffer.cpu, vertices.data(), vertices.size() * sizeof(float3));
    memcpy(normalBuffer.cpu, normals.data(), normals.size() * sizeof(float3));
    memcpy(uvBuffer.cpu, uvs.data(), uvs.size() * sizeof(float2));
    memcpy(indexBuffer.cpu, indices.data(), indices.size() * sizeof(uint32_t));

    // Setup mesh/primitive data for shader access
    auto primitiveData = allocator.allocate<PrimitiveData>();
    primitiveData.cpu->indices = indexBuffer.gpu;
    primitiveData.cpu->vertices = vertexBuffer.gpu;
    primitiveData.cpu->uvs = uvBuffer.gpu;
    primitiveData.cpu->normals = normalBuffer.gpu;
    primitiveData.cpu->texture = 0;

    auto meshData = allocator.allocate<MeshData>();
    meshData.cpu->primitives = primitiveData.gpu;

    const uint32_t cubeCount = 20;
    auto instanceToMesh = allocator.allocate<uint32_t>(cubeCount);
    for (uint32_t i = 0; i < cubeCount; ++i)
    {
        // All instances use the same mesh in this scenario
        instanceToMesh.cpu[i] = 0;
    }

    GpuAccelerationStructureTrianglesDesc trianglesDesc = {
        .vertexDataGpu = vertexBuffer.gpu,
        .vertexCount = static_cast<uint32_t>(vertices.size()),
        .vertexStride = sizeof(float3),
        .vertexFormat = FORMAT_RGB32_FLOAT,
        .indexDataGpu = indexBuffer.gpu,
        .indexType = INDEX_TYPE_UINT32,
        .transformDataGpu = nullptr};

    GpuAccelerationStructureBlasDesc blasDesc = {
        .type = GEOMETRY_TYPE_TRIANGLES,
        .triangles = Span<GpuAccelerationStructureTrianglesDesc>(&trianglesDesc, 1)};

    GpuAccelerationStructureBuildRange blasBuildRange = {
        .primitiveCount = static_cast<uint32_t>(indices.size() / 3),
        .primitiveOffset = 0,
        .firstVertex = 0,
        .transformOffset = 0};

    GpuAccelerationStructureDesc blasASDesc = {
        .type = TYPE_BOTTOM_LEVEL,
        .blasDesc = blasDesc,
        .buildRanges = Span<GpuAccelerationStructureBuildRange>(&blasBuildRange, 1)};

    auto blasSize = gpuAccelerationStructureSizes(blasASDesc);
    void *blasPtr = gpuMalloc(blasSize.size, MEMORY_GPU);
    auto blas = gpuCreateAccelerationStructure(blasASDesc, blasPtr, blasSize.size);

    auto instances = gpuMalloc<GpuAccelerationStructureInstanceDesc>(cubeCount);
    const float scale = 0.5f;

    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-2.5f, 2.5f);

        // Randomly place cubes in the scene
        for (size_t i = 0; i < cubeCount; ++i)
        {
            // glm random rotation
            glm::quat rotation = glm::quat(glm::vec3(dis(gen), dis(gen), dis(gen) - 5.0f));

            glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(dis(gen), dis(gen), dis(gen))) * glm::toMat4(rotation) * glm::scale(glm::mat4(1.0f), glm::vec3(scale));

            instances[i].transform = float3x4{
                {model[0][0], model[1][0], model[2][0], model[3][0]},
                {model[0][1], model[1][1], model[2][1], model[3][1]},
                {model[0][2], model[1][2], model[2][2], model[3][2]}};
            instances[i].instanceID = static_cast<uint32_t>(i);
            instances[i].instanceMask = 0xFF;
            instances[i].hitGroupIndex = 0;
            instances[i].flags = 0;
            instances[i].blasAddress = blasPtr;
        }
    }

    GpuAccelerationStructureBuildRange tlasBuildRange = {
        .primitiveCount = static_cast<uint32_t>(cubeCount),
        .primitiveOffset = 0,
        .firstVertex = 0,
        .transformOffset = 0};

    GpuAccelerationStructureDesc tlasDesc = {
        .type = TYPE_TOP_LEVEL,
        .tlasDesc = {
            .arrayOfPointers = false,
            .instancesGpu = gpuHostToDevicePointer(instances)},
        .buildRanges = Span<GpuAccelerationStructureBuildRange>(&tlasBuildRange, 1)};

    auto tlasSize = gpuAccelerationStructureSizes(tlasDesc);
    void *tlasPtr = gpuMalloc(tlasSize.size, MEMORY_GPU);
    auto tlas = gpuCreateAccelerationStructure(tlasDesc, tlasPtr, tlasSize.size);

    size_t scratchSize = std::max(blasSize.buildScratchSize, tlasSize.buildScratchSize);
    void *scratchPtr = gpuMalloc(scratchSize, MEMORY_GPU);

    // ReSTIR reservoirs
    auto reservoirs = allocate<Reservoir>(swapchainDesc.dimensions.x * swapchainDesc.dimensions.y);
    auto history = allocate<Reservoir>(swapchainDesc.dimensions.x * swapchainDesc.dimensions.y);

    raytracingData.cpu->camData = camDataAlloc.gpu;
    raytracingData.cpu->tlas = tlasPtr;
    raytracingData.cpu->instanceToMesh = instanceToMesh.gpu;
    raytracingData.cpu->meshes = meshData.gpu;
    raytracingData.cpu->lights = lightData.gpu;
    raytracingData.cpu->reservoirs = reservoirs.gpu;
    raytracingData.cpu->history = history.gpu;
    raytracingData.cpu->numLights = numLights;
    raytracingData.cpu->dstTexture = 1;
    raytracingData.cpu->frame = 0;
    raytracingData.cpu->M = 8;

    auto queue = gpuCreateQueue();
    auto semaphore = gpuCreateSemaphore(0);
    uint64_t nextFrame = 1;

    auto pipeline = referencePipeline;

    glm::vec3 velocity = glm::vec3(0.f);
    float velocityScale = 0.001f;

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
            else if (event.type == SDL_EVENT_KEY_DOWN) // if holding "A" key, set accumulate to 1
            {
                if (event.key.key == SDLK_A)
                {
                    raytracingData.cpu->accumulate = 1;
                }
                else if (event.key.key == SDLK_R)
                {
                    if (pipeline == restirPipeline)
                    {
                        // Reset history reservoir when switching back to reference pipeline
                        memset(history.cpu, 0, sizeof(Reservoir) * swapchainDesc.dimensions.x * swapchainDesc.dimensions.y);
                        pipeline = referencePipeline;
                    }
                    else
                    {
                        pipeline = restirPipeline;
                    }
                }
                else if (event.key.key == SDLK_LEFT)
                {
                    velocity.x = -velocityScale;
                }
                else if (event.key.key == SDLK_RIGHT)
                {
                    velocity.x = velocityScale;
                }
                else if (event.key.key == SDLK_UP)
                {
                    velocity.y = velocityScale;
                }
                else if (event.key.key == SDLK_DOWN)
                {
                    velocity.y = -velocityScale;
                }
            }
            else if (event.type == SDL_EVENT_KEY_UP)
            {
                if (event.key.key == SDLK_A)
                {
                    raytracingData.cpu->accumulate = 0;
                    raytracingData.cpu->accumulatedFrames = 0;
                }
                else if (event.key.key == SDLK_LEFT || event.key.key == SDLK_RIGHT)
                {
                    velocity.x = 0.f;
                }
                else if (event.key.key == SDLK_UP || event.key.key == SDLK_DOWN)
                {
                    velocity.y = 0.f;
                }
            }
        }

        cameraPos += velocity;
        setCamera();
        auto commandBuffer = gpuStartCommandRecording(queue);

        if (nextFrame == 1)
        {
            // First frame, copy texture data
            gpuCopyToTexture(commandBuffer, upload.gpu, texture);
            gpuBarrier(commandBuffer, STAGE_TRANSFER, STAGE_COMPUTE, HAZARD_DESCRIPTORS);

            gpuBuildAccelerationStructures(commandBuffer, Span<GpuAccelerationStructure>(&blas, 1), scratchPtr, MODE_BUILD);
            gpuBarrier(commandBuffer, STAGE_ACCELERATION_STRUCTURE_BUILD, STAGE_ACCELERATION_STRUCTURE_BUILD, HAZARD_ACCELERATION_STRUCTURE);
            gpuBuildAccelerationStructures(commandBuffer, Span<GpuAccelerationStructure>(&tlas, 1), scratchPtr, MODE_BUILD);
            gpuBarrier(commandBuffer, STAGE_ACCELERATION_STRUCTURE_BUILD, STAGE_COMPUTE, HAZARD_ACCELERATION_STRUCTURE);
        }
        else
        {
            gpuBuildAccelerationStructures(commandBuffer, Span<GpuAccelerationStructure>(&tlas, 1), scratchPtr, MODE_UPDATE);
            gpuBarrier(commandBuffer, STAGE_ACCELERATION_STRUCTURE_BUILD, STAGE_COMPUTE, HAZARD_ACCELERATION_STRUCTURE);
        }

        if (nextFrame > FRAMES_IN_FLIGHT)
        {
            gpuWaitSemaphore(semaphore, nextFrame - FRAMES_IN_FLIGHT);
        }

        auto image = gpuSwapchainImage(swapchain);
        GpuRenderPassDesc renderPassDesc = {
            .colorTargets = Span<GpuTexture>(&image, 1),
            .depthStencilTarget = nullptr};

        gpuSetPipeline(commandBuffer, pipeline);
        gpuSetActiveTextureHeapPtr(commandBuffer, textureHeap.gpu);

        // inline raytracing
        gpuDispatch(commandBuffer, raytracingData.gpu, {(uint32_t)swapchainDesc.dimensions.x / 8, (uint32_t)swapchainDesc.dimensions.y / 8, 1});
        gpuBarrier(commandBuffer, STAGE_COMPUTE, STAGE_TRANSFER);

        // copy to swapchain image
        gpuBlitTexture(commandBuffer, image, outputTexture);

        gpuSubmit(queue, Span<GpuCommandBuffer>(&commandBuffer, 1), semaphore, nextFrame);
        gpuPresent(swapchain, semaphore, nextFrame++);

        raytracingData.cpu->frame++;
        if (raytracingData.cpu->accumulate == 1)
        {
            raytracingData.cpu->accumulatedFrames++;
        }
        else
        {
            raytracingData.cpu->accumulatedFrames = 0;
        }
    }

    gpuWaitSemaphore(semaphore, nextFrame - 1);

    stbi_image_free(inputImage);
    upload.free();
    allocator.free();
    gpuDestroyTexture(texture);
    gpuFree(texturePtr);
    gpuDestroyTexture(outputTexture);
    gpuFree(outputTexturePtr);
    textureHeap.free();
    gpuFreePipeline(pipeline);
    gpuDestroyAccelerationStructure(blas);
    gpuFree(blasPtr);
    gpuDestroyAccelerationStructure(tlas);
    gpuFree(tlasPtr);
    gpuFree(instances);
    gpuFree(scratchPtr);
    gpuDestroySemaphore(semaphore);
    gpuDestroySwapchain(swapchain);
    SDL_Gpu_DestroySurface(surface);
    SDL_DestroyWindow(window);

    gpuDestroyDevice();
}