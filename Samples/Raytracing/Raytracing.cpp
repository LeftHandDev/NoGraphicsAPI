#include "../../External/stb_image.h"
#include "../../External/stb_image_write.h"

#include <SDL3/SDL.h>
#include "../../SDL_gpu.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>

#include <random>
#include <string>

#include "Raytracing.h"
#include "../Common/Utilities.h"
#include "../Common/Text.h"

static std::string getModeText(bool reference, bool spatial, bool temporal)
{
    if (reference)
        return "Reference [R: RIS]";

    std::string text = "RIS [R: Reference";
    text += spatial  ? " | S: Spatial On"  : " | S: Spatial Off";
    text += temporal ? " | T: Temporal On" : " | T: Temporal Off";
    text += "]";
    return text;
}

void raytracingSample()
{
    gpuCreateDevice();

    const uint FRAMES_IN_FLIGHT = 2;

    if (!SDL_Init(SDL_INIT_VIDEO))
    {
        return;
    }

    SDL_DisplayID display = SDL_GetPrimaryDisplay();
    const SDL_DisplayMode *displayMode = SDL_GetCurrentDisplayMode(display);
    auto window = SDL_CreateWindow("Test Window", displayMode->w, displayMode->h, SDL_WINDOW_GPU | SDL_WINDOW_BORDERLESS);
    auto surface = SDL_Gpu_CreateSurface(window);
    bool exit = false;

    int width, height, channels;
    stbi_uc *inputImage = stbi_load("Assets/Default.png", &width, &height, &channels, 4);

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

    // output texture
    GpuTextureDesc outputTextureDesc{
        .type = TEXTURE_2D,
        .dimensions = {static_cast<uint32_t>(swapchainDesc.dimensions.x), static_cast<uint32_t>(swapchainDesc.dimensions.y), 1},
        .format = FORMAT_RGBA32_FLOAT,
        .usage = static_cast<USAGE_FLAGS>(USAGE_STORAGE | USAGE_TRANSFER_SRC | USAGE_COLOR_ATTACHMENT)};

    GpuTextureSizeAlign outputTextureSizeAlign = gpuTextureSizeAlign(outputTextureDesc);
    void *outputTexturePtr = gpuMalloc(outputTextureSizeAlign.size, MEMORY_GPU);
    auto outputTexture = gpuCreateTexture(outputTextureDesc, outputTexturePtr);

    // albedo texture
    GpuTextureDesc albedoTextureDesc{
        .type = TEXTURE_2D,
        .dimensions = {static_cast<uint32_t>(swapchainDesc.dimensions.x), static_cast<uint32_t>(swapchainDesc.dimensions.y), 1},
        .format = FORMAT_RGBA32_FLOAT,
        .usage = static_cast<USAGE_FLAGS>(USAGE_STORAGE | USAGE_SAMPLED)};

    GpuTextureSizeAlign albedoTextureSizeAlign = gpuTextureSizeAlign(albedoTextureDesc);
    void *albedoTexturePtr = gpuMalloc(albedoTextureSizeAlign.size, MEMORY_GPU);
    auto albedoTexture = gpuCreateTexture(albedoTextureDesc, albedoTexturePtr);

    // normals
     GpuTextureDesc normalsTextureDesc{
        .type = TEXTURE_2D,
        .dimensions = {static_cast<uint32_t>(swapchainDesc.dimensions.x), static_cast<uint32_t>(swapchainDesc.dimensions.y), 1},
        .format = FORMAT_RGBA32_FLOAT,
        .usage = static_cast<USAGE_FLAGS>(USAGE_STORAGE | USAGE_SAMPLED)};

    GpuTextureSizeAlign normalsTextureSizeAlign = gpuTextureSizeAlign(normalsTextureDesc);
    void *normalsTexturePtr = gpuMalloc(normalsTextureSizeAlign.size, MEMORY_GPU);
    auto normalsTexture = gpuCreateTexture(normalsTextureDesc, normalsTexturePtr);

    // motion vectors
    GpuTextureDesc motionVectorsTextureDesc{
        .type = TEXTURE_2D,
        .dimensions = {static_cast<uint32_t>(swapchainDesc.dimensions.x), static_cast<uint32_t>(swapchainDesc.dimensions.y), 1},
        .format = FORMAT_RGBA32_FLOAT,
        .usage = static_cast<USAGE_FLAGS>(USAGE_STORAGE | USAGE_SAMPLED)};

    GpuTextureSizeAlign motionVectorsTextureSizeAlign = gpuTextureSizeAlign(motionVectorsTextureDesc);
    void *motionVectorsTexturePtr = gpuMalloc(motionVectorsTextureSizeAlign.size, MEMORY_GPU);
    auto motionVectorsTexture = gpuCreateTexture(motionVectorsTextureDesc, motionVectorsTexturePtr);

    enum HeapIndices
    {
        INDEX_CUBE = 0,
        INDEX_CURRENT_FRAME = 1,
        INDEX_ALBEDO = 2,
        INDEX_NORMALS = 3,
        INDEX_MOTION_VECTORS = 4,
    };

    auto textureHeap = allocate<GpuTextureDescriptor>(1024);
    textureHeap.cpu[INDEX_CUBE] = gpuTextureViewDescriptor(texture, GpuViewDesc{.format = FORMAT_RGBA8_UNORM});
    textureHeap.cpu[INDEX_CURRENT_FRAME] = gpuRWTextureViewDescriptor(outputTexture, GpuViewDesc{.format = FORMAT_RGBA32_FLOAT});
    textureHeap.cpu[INDEX_ALBEDO] = gpuRWTextureViewDescriptor(albedoTexture, GpuViewDesc{.format = FORMAT_RGBA32_FLOAT});
    textureHeap.cpu[INDEX_NORMALS] = gpuRWTextureViewDescriptor(normalsTexture, GpuViewDesc{.format = FORMAT_RGBA32_FLOAT});
    textureHeap.cpu[INDEX_MOTION_VECTORS] = gpuRWTextureViewDescriptor(motionVectorsTexture, GpuViewDesc{.format = FORMAT_RGBA32_FLOAT});

    ColorTarget colorTarget = {};
    colorTarget.format = swapchainDesc.format;

    auto referenceIR = loadIR("../Shaders/Raytracing/Raytracing.spv");
    auto referencePipeline = gpuCreateComputePipeline(ByteSpan(referenceIR));

    auto risIR = loadIR("../Shaders/Raytracing/RIS.spv");
    auto risPipeline = gpuCreateComputePipeline(ByteSpan(risIR));

    auto reuseIR = loadIR("../Shaders/Raytracing/Reuse.spv");
    auto reusePipeline = gpuCreateComputePipeline(ByteSpan(reuseIR));

    auto shadeIR = loadIR("../Shaders/Raytracing/Shade.spv");
    auto shadePipeline = gpuCreateComputePipeline(ByteSpan(shadeIR));

    size_t allocatorSize = 2 * 1024 * 1024; // 2 MB
    LinearAllocator allocator(allocatorSize);
    auto rtDataRingBufffer = allocator.allocate<RaytracingData>(FRAMES_IN_FLIGHT);
    RaytracingData raytracingData = {};

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

    auto camDataAlloc = allocator.allocate<CameraData>(FRAMES_IN_FLIGHT);

    auto setCamera = [&](size_t frameIndex) {
        camDataAlloc.cpu[frameIndex].position = {cameraPos.x, cameraPos.y, cameraPos.z, 1};
        auto view = glm::lookAt(cameraPos, glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
        auto projection = glm::perspective(glm::radians(60.0f), swapchainDesc.dimensions.x / static_cast<float>(swapchainDesc.dimensions.y), 0.1f, 100.0f);
        auto invViewProjection = glm::inverse(projection * view);
        memcpy(&camDataAlloc.cpu[frameIndex].invViewProjection, &invViewProjection, sizeof(float4x4));

        view = glm::lookAt(prevCameraPos, glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
        auto viewProjection = projection * view;
        memcpy(&camDataAlloc.cpu[frameIndex].prevViewProjection, &viewProjection, sizeof(float4x4));

        prevCameraPos = cameraPos;
    };

    for (size_t i = 0; i < FRAMES_IN_FLIGHT; ++i)
    {
        setCamera(i);
    }

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

    // ReSTIR buffers
    auto pixelSample = gpuMalloc<Sample>(swapchainDesc.dimensions.x * swapchainDesc.dimensions.y, MEMORY_GPU);
    auto prevPixelSample = gpuMalloc<Sample>(swapchainDesc.dimensions.x * swapchainDesc.dimensions.y, MEMORY_GPU);

    raytracingData.camData = camDataAlloc.gpu;
    raytracingData.tlas = tlasPtr;
    raytracingData.instanceToMesh = instanceToMesh.gpu;
    raytracingData.meshes = meshData.gpu;
    raytracingData.lights = lightData.gpu;
    raytracingData.pixelSample = pixelSample;
    raytracingData.prevPixelSample = prevPixelSample;
    raytracingData.numLights = numLights;
    raytracingData.albedo = INDEX_ALBEDO;
    raytracingData.normals = INDEX_NORMALS;
    raytracingData.motionVectors = INDEX_MOTION_VECTORS; 
    raytracingData.dstTexture = INDEX_CURRENT_FRAME;
    raytracingData.frame = 0;
    raytracingData.M = 8;

    auto queue = gpuCreateQueue();
    auto semaphore = gpuCreateSemaphore(0);
    uint64_t nextFrame = 1;

    bool reference = true;

    glm::vec3 velocity = glm::vec3(0.f);
    float velocityScale = 5.f;
    float delta = 0.016f; // ~60 FPS
    auto timestamp = std::chrono::high_resolution_clock::now();

    TextRenderer* textRenderer = new TextRenderer(outputTextureDesc);

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
                    if (raytracingData.accumulate == 0)
                    {
                        raytracingData.accumulate = 1;
                        raytracingData.frame = 0;
                        raytracingData.accumulatedFrames = 0;
                    }
                }
                else if (event.key.key == SDLK_S)
                {
                    raytracingData.spatial = raytracingData.spatial == 0 ? 1 : 0;
                    raytracingData.frame = 0;
                    raytracingData.accumulatedFrames = 0;
                }
                else if (event.key.key == SDLK_T)
                {
                    raytracingData.temporal = raytracingData.temporal == 0 ? 1 : 0;
                    raytracingData.frame = 0;
                    raytracingData.accumulatedFrames = 0;
                }
                else if (event.key.key == SDLK_R)
                {
                    reference = !reference;
                    raytracingData.frame = 0;
                    raytracingData.accumulatedFrames = 0;
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
                else if (event.key.key == SDLK_ESCAPE)
                {
                    exit = true;
                    break;
                }
            }
            else if (event.type == SDL_EVENT_KEY_UP)
            {
                if (event.key.key == SDLK_A)
                {
                    raytracingData.accumulate = 0;
                    raytracingData.accumulatedFrames = 0;
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

        auto offset = (nextFrame - 1) % FRAMES_IN_FLIGHT;

        cameraPos += velocity * delta;
        setCamera(offset);
        raytracingData.camData = camDataAlloc.gpu + offset;
        rtDataRingBufffer.cpu[offset] = raytracingData;

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


        if (reference)
        {
            gpuSetPipeline(commandBuffer, referencePipeline);
            gpuSetActiveTextureHeapPtr(commandBuffer, textureHeap.gpu);
            gpuDispatch(commandBuffer, rtDataRingBufffer.gpu + offset, {(uint32_t)swapchainDesc.dimensions.x / 8, (uint32_t)swapchainDesc.dimensions.y / 8, 1});
        }
        else // ReSTIR
        {
            gpuSetPipeline(commandBuffer, risPipeline);
            gpuSetActiveTextureHeapPtr(commandBuffer, textureHeap.gpu);
            gpuDispatch(commandBuffer, rtDataRingBufffer.gpu + offset, {(uint32_t)swapchainDesc.dimensions.x / 8, (uint32_t)swapchainDesc.dimensions.y / 8, 1});
            gpuBarrier(commandBuffer, STAGE_COMPUTE, STAGE_COMPUTE);

            gpuSetPipeline(commandBuffer, reusePipeline);
            gpuDispatch(commandBuffer, rtDataRingBufffer.gpu + offset, {(uint32_t)swapchainDesc.dimensions.x / 8, (uint32_t)swapchainDesc.dimensions.y / 8, 1});
            gpuBarrier(commandBuffer, STAGE_COMPUTE, STAGE_COMPUTE);

            gpuSetPipeline(commandBuffer, shadePipeline);
            gpuDispatch(commandBuffer, rtDataRingBufffer.gpu + offset, {(uint32_t)swapchainDesc.dimensions.x / 8, (uint32_t)swapchainDesc.dimensions.y / 8, 1});
        }
        gpuBarrier(commandBuffer, STAGE_COMPUTE, STAGE_TRANSFER);

        // copy to swapchain image
        gpuBlitTexture(commandBuffer, image, outputTexture);

        // Render text to swapchain image
        auto modeText = getModeText(reference, raytracingData.spatial, raytracingData.temporal);
        textRenderer->renderText(commandBuffer, image,
            modeText.c_str(), 10.0f, 10.0f, 1.0f, float3(1, 1, 1));

        gpuSubmit(queue, Span<GpuCommandBuffer>(&commandBuffer, 1), semaphore, nextFrame);
        gpuPresent(swapchain, semaphore, nextFrame++);

        // Swap reservoir buffers
        auto pixelSample = raytracingData.pixelSample;
        raytracingData.pixelSample = raytracingData.prevPixelSample;
        raytracingData.prevPixelSample = pixelSample;

        raytracingData.frame = nextFrame;
        if (raytracingData.accumulate == 1)
        {
            raytracingData.accumulatedFrames++;
        }
        else
        {
            raytracingData.accumulatedFrames = 0;
        }

        // update delta time and timestamp
        auto now = std::chrono::high_resolution_clock::now();
        delta = std::chrono::duration<float>(now - timestamp).count();
        timestamp = now;
    }

    gpuWaitSemaphore(semaphore, nextFrame - 1);

    delete textRenderer;
    stbi_image_free(inputImage);
    upload.free();
    allocator.free();
    gpuDestroyTexture(texture);
    gpuFree(texturePtr);
    gpuDestroyTexture(outputTexture);
    gpuFree(outputTexturePtr);
    textureHeap.free();
    gpuFreePipeline(referencePipeline);
    gpuFreePipeline(risPipeline);
    gpuFreePipeline(reusePipeline);
    gpuFreePipeline(shadePipeline);
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
    SDL_Quit();

    gpuDestroyDevice();
}