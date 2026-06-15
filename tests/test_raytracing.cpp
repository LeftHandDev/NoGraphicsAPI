// Headless test mirroring the Raytracing sample (reference path): builds BLAS/TLAS
// for a fixed-seed scene of cubes + lights, accumulates a fixed number of frames
// of ray-queried direct lighting, and blits the float output into an RGBA8 capture
// texture that is read back and compared to a golden. Deterministic: fixed RNG
// seeds, static camera, no wall-clock.
#include "test_common.h"

#include "Utilities.h"  // LinearAllocator, loadIR, getCube, haltonSequence
#include "Raytracing.h" // RaytracingData, CameraData, LightData, PrimitiveData, MeshData, Sample

#include "stb_image.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <random>
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
        std::cerr << "FAIL [raytracing]: no suitable device at index " << args.device << "\n";
        return 1;
    }

    const uint FRAMES_IN_FLIGHT = 1; // serialize frames so cross-frame reads (history/accumulation) are race-free and deterministic
    const uint32_t RENDER_W = 320;   // divisible by 8 for the ray-tracing dispatch
    const uint32_t RENDER_H = 240;
    const uint32_t SEED = 1337; // fixed so the scene is reproducible

    LinearAllocator allocator(device);
    LinearAllocator<MEMORY_DESCRIPTOR> descriptorAllocator(device);

    int width, height, channels;
    const std::string inputPath = std::string(NGAPI_TEST_ASSET_DIR) + "/Default.png";
    stbi_uc* inputImage = stbi_load(inputPath.c_str(), &width, &height, &channels, 4);
    if (!inputImage)
    {
        std::cerr << "FAIL [raytracing]: could not load " << inputPath << "\n";
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

    void* outputPtr;
    auto outputTexture = makeTexture({ .type = TEXTURE_2D, .dimensions = { RENDER_W, RENDER_H, 1 }, .format = FORMAT_RGBA32_FLOAT, .usage = (USAGE_FLAGS)(USAGE_STORAGE | USAGE_TRANSFER_SRC | USAGE_TRANSFER_DST | USAGE_COLOR_ATTACHMENT | USAGE_SAMPLED) }, &outputPtr);

    void* albedoPtr;
    auto albedoTexture = makeTexture({ .type = TEXTURE_2D, .dimensions = { RENDER_W, RENDER_H, 1 }, .format = FORMAT_RGBA16_FLOAT, .usage = (USAGE_FLAGS)(USAGE_STORAGE | USAGE_SAMPLED) }, &albedoPtr);

    void* normalsPtr;
    auto normalsTexture = makeTexture({ .type = TEXTURE_2D, .dimensions = { RENDER_W, RENDER_H, 1 }, .format = FORMAT_RGBA16_FLOAT, .usage = (USAGE_FLAGS)(USAGE_STORAGE | USAGE_SAMPLED) }, &normalsPtr);

    void* motionPtr;
    auto motionVectorsTexture = makeTexture({ .type = TEXTURE_2D, .dimensions = { RENDER_W, RENDER_H, 1 }, .format = FORMAT_RGBA16_FLOAT, .usage = (USAGE_FLAGS)(USAGE_STORAGE | USAGE_SAMPLED) }, &motionPtr);

    void* capturePtr;
    auto capture = makeTexture({ .type = TEXTURE_2D, .dimensions = { RENDER_W, RENDER_H, 1 }, .format = FORMAT_RGBA8_UNORM, .usage = (USAGE_FLAGS)(USAGE_TRANSFER_DST | USAGE_TRANSFER_SRC | USAGE_SAMPLED) }, &capturePtr);

    enum HeapIndices
    {
        INDEX_CUBE = 0,
        INDEX_CURRENT_FRAME = 1,
        INDEX_ALBEDO = 2,
        INDEX_NORMALS = 3,
        INDEX_MOTION_VECTORS = 4,
        INDEX_OUTPUT_SAMPLED = 7,
    };

    auto textureHeap = descriptorAllocator.allocate<GpuTextureDescriptor>(1024);
    textureHeap.cpu[INDEX_CUBE] = gpuTextureViewDescriptor(texture, GpuViewDesc{ .format = FORMAT_RGBA8_UNORM });
    textureHeap.cpu[INDEX_CURRENT_FRAME] = gpuRWTextureViewDescriptor(outputTexture, GpuViewDesc{ .format = FORMAT_RGBA32_FLOAT });
    textureHeap.cpu[INDEX_ALBEDO] = gpuRWTextureViewDescriptor(albedoTexture, GpuViewDesc{ .format = FORMAT_RGBA16_FLOAT });
    textureHeap.cpu[INDEX_NORMALS] = gpuRWTextureViewDescriptor(normalsTexture, GpuViewDesc{ .format = FORMAT_RGBA16_FLOAT });
    textureHeap.cpu[INDEX_MOTION_VECTORS] = gpuRWTextureViewDescriptor(motionVectorsTexture, GpuViewDesc{ .format = FORMAT_RGBA16_FLOAT });

    auto referenceIR = loadIR(std::string(NGAPI_TEST_SHADER_DIR) + "/raytracing/Raytracing.spv");
    auto referencePipeline = gpuCreateComputePipeline(device, ByteSpan(referenceIR));

    auto rtDataRing = allocator.allocate<RaytracingData>(FRAMES_IN_FLIGHT);
    RaytracingData raytracingData = {};

    uint32_t numLights = 100;
    auto lightData = allocator.allocate<LightData>(numLights);

    glm::vec3 cameraPos(0, 0, -5);
    glm::vec3 prevCameraPos = cameraPos;

    {
        std::mt19937 gen(SEED);
        float offset = static_cast<float>(numLights);
        std::uniform_real_distribution<float> dis(-offset, offset);
        lightData.cpu[0].position = { cameraPos.x, cameraPos.y, cameraPos.z, 1 };
        lightData.cpu[0].color = { 1, 1, 1, 1 };
        lightData.cpu[0].intensity = 10.0f;
        for (uint32_t i = 1; i < numLights; ++i)
        {
            lightData.cpu[i].position = { dis(gen), dis(gen), dis(gen), 1 };
            lightData.cpu[i].color = { 1, 1, 1, 1 };
            lightData.cpu[i].intensity = 10.0f;
        }
    }

    auto camDataAlloc = allocator.allocate<CameraData>(FRAMES_IN_FLIGHT);
    auto setCamera = [&](size_t frameIndex)
    {
        camDataAlloc.cpu[frameIndex].position = { cameraPos.x, cameraPos.y, cameraPos.z, 1 };
        auto view = glm::lookAt(cameraPos, glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
        auto projection = glm::perspective(glm::radians(60.0f), float(RENDER_W) / float(RENDER_H), 0.1f, 100.0f);
        auto invViewProjection = glm::inverse(projection * view);
        memcpy(&camDataAlloc.cpu[frameIndex].invViewProjection, &invViewProjection, sizeof(float4x4));
        auto viewProjection = projection * view;
        memcpy(&camDataAlloc.cpu[frameIndex].viewProjection, &viewProjection, sizeof(float4x4));
        auto prevView = glm::lookAt(prevCameraPos, glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
        auto prevViewProjection = projection * prevView;
        memcpy(&camDataAlloc.cpu[frameIndex].prevViewProjection, &prevViewProjection, sizeof(float4x4));
        prevCameraPos = cameraPos;
    };
    for (size_t i = 0; i < FRAMES_IN_FLIGHT; ++i)
        setCamera(i);

    std::vector<float3> vertices, normals;
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

    auto primitiveData = allocator.allocate<PrimitiveData>();
    primitiveData.cpu->indices = indexBuffer.gpu;
    primitiveData.cpu->vertices = vertexBuffer.gpu;
    primitiveData.cpu->uvs = uvBuffer.gpu;
    primitiveData.cpu->normals = normalBuffer.gpu;
    primitiveData.cpu->texture = INDEX_CUBE;

    auto meshData = allocator.allocate<MeshData>();
    meshData.cpu->primitives = primitiveData.gpu;

    const uint32_t cubeCount = 20;
    auto instanceToMesh = allocator.allocate<uint32_t>(cubeCount);
    for (uint32_t i = 0; i < cubeCount; ++i)
        instanceToMesh.cpu[i] = 0;

    GpuAccelerationStructureTrianglesDesc trianglesDesc = {
        .vertexDataGpu = vertexBuffer.gpu,
        .vertexCount = static_cast<uint32_t>(vertices.size()),
        .vertexStride = sizeof(float3),
        .vertexFormat = FORMAT_RGB32_FLOAT,
        .indexDataGpu = indexBuffer.gpu,
        .indexType = INDEX_TYPE_UINT32,
        .transformDataGpu = nullptr
    };
    GpuAccelerationStructureBlasDesc blasDesc = { .type = GEOMETRY_TYPE_TRIANGLES, .triangles = Span<GpuAccelerationStructureTrianglesDesc>(&trianglesDesc, 1) };
    GpuAccelerationStructureBuildRange blasBuildRange = { .primitiveCount = static_cast<uint32_t>(indices.size() / 3) };
    GpuAccelerationStructureDesc blasASDesc = { .type = TYPE_BOTTOM_LEVEL, .blasDesc = blasDesc, .buildRanges = Span<GpuAccelerationStructureBuildRange>(&blasBuildRange, 1) };

    auto blasSize = gpuAccelerationStructureSizes(device, blasASDesc);
    void* blasPtr = gpuMalloc(device, blasSize.size, MEMORY_GPU);
    auto blas = gpuCreateAccelerationStructure(device, blasASDesc, blasPtr, blasSize.size);

    auto instances = allocator.allocate<GpuAccelerationStructureInstanceDesc>(cubeCount);
    const float scale = 0.5f;
    {
        std::mt19937 gen(SEED + 1);
        std::uniform_real_distribution<float> dis(-2.5f, 2.5f);
        for (uint32_t i = 0; i < cubeCount; ++i)
        {
            glm::quat rotation = glm::quat(glm::vec3(dis(gen), dis(gen), dis(gen) - 5.0f));
            glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(dis(gen), dis(gen), dis(gen))) * glm::toMat4(rotation) * glm::scale(glm::mat4(1.0f), glm::vec3(scale));
            instances.cpu[i].transform = float3x4{
                { model[0][0], model[1][0], model[2][0], model[3][0] },
                { model[0][1], model[1][1], model[2][1], model[3][1] },
                { model[0][2], model[1][2], model[2][2], model[3][2] }
            };
            instances.cpu[i].instanceID = i;
            instances.cpu[i].instanceMask = 0xFF;
            instances.cpu[i].hitGroupIndex = 0;
            instances.cpu[i].flags = 0;
            instances.cpu[i].blasAddress = blasPtr;
        }
    }

    GpuAccelerationStructureBuildRange tlasBuildRange = { .primitiveCount = cubeCount };
    GpuAccelerationStructureDesc tlasDesc = {
        .type = TYPE_TOP_LEVEL,
        .tlasDesc = { .arrayOfPointers = false, .instancesGpu = instances.gpu },
        .buildRanges = Span<GpuAccelerationStructureBuildRange>(&tlasBuildRange, 1)
    };
    auto tlasSize = gpuAccelerationStructureSizes(device, tlasDesc);
    void* tlasPtr = gpuMalloc(device, tlasSize.size, MEMORY_GPU);
    auto tlas = gpuCreateAccelerationStructure(device, tlasDesc, tlasPtr, tlasSize.size);

    size_t scratchSize = std::max(blasSize.buildScratchSize, tlasSize.buildScratchSize);
    void* scratchPtr = gpuMalloc(device, scratchSize, MEMORY_GPU);

    void* pixelSample = gpuMalloc(device, sizeof(Sample) * RENDER_W * RENDER_H, MEMORY_GPU);
    void* prevPixelSample = gpuMalloc(device, sizeof(Sample) * RENDER_W * RENDER_H, MEMORY_GPU);

    raytracingData.camData = camDataAlloc.gpu;
    raytracingData.tlas = tlasPtr;
    raytracingData.instanceToMesh = instanceToMesh.gpu;
    raytracingData.meshes = meshData.gpu;
    raytracingData.lights = lightData.gpu;
    raytracingData.pixelSample = reinterpret_cast<Sample*>(pixelSample);
    raytracingData.prevPixelSample = reinterpret_cast<Sample*>(prevPixelSample);
    raytracingData.numLights = numLights;
    raytracingData.albedo = INDEX_ALBEDO;
    raytracingData.normals = INDEX_NORMALS;
    raytracingData.motionVectors = INDEX_MOTION_VECTORS;
    raytracingData.dstTexture = INDEX_CURRENT_FRAME;
    raytracingData.frame = 0;
    raytracingData.M = 8;
    raytracingData.accumulate = 1; // accumulate across frames (also exercises the lifecycle)
    raytracingData.accumulatedFrames = 0;

    auto queue = gpuCreateQueue(device);
    auto semaphore = gpuCreateSemaphore(device, 0);

    // Clear the accumulation target to zero. The shader's accumulate path reads
    // outputTexture (color = prevColor * accumulatedFrames + ...); uninitialized
    // float memory can be NaN, and NaN * 0 = NaN, so frame 0 would be garbage.
    auto zero = allocator.allocate<uint8_t>(RENDER_W * RENDER_H * 16);
    memset(zero.cpu, 0, RENDER_W * RENDER_H * 16);
    {
        auto clearCmd = gpuStartCommandRecording(queue);
        gpuCopyToTexture(clearCmd, zero.gpu, outputTexture);
        gpuSubmit(queue, Span<GpuCommandBuffer>(&clearCmd, 1), semaphore, 1);
        gpuWaitSemaphore(semaphore, 1);
        gpuDestroySemaphore(semaphore);
        semaphore = gpuCreateSemaphore(device, 0);
    }

    uint64_t nextFrame = 1;

    for (uint32_t frame = 0; frame < args.frames; frame++)
    {
        auto offset = (nextFrame - 1) % FRAMES_IN_FLIGHT;
        setCamera(offset);
        raytracingData.camData = camDataAlloc.gpu + offset;
        rtDataRing.cpu[offset] = raytracingData;

        auto commandBuffer = gpuStartCommandRecording(queue);
        if (nextFrame == 1)
        {
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

        gpuSetPipeline(commandBuffer, referencePipeline);
        gpuSetActiveTextureHeapPtr(commandBuffer, textureHeap.gpu);
        gpuDispatch(commandBuffer, rtDataRing.gpu + offset, { RENDER_W / 8, RENDER_H / 8, 1 });

        gpuBarrier(commandBuffer, STAGE_COMPUTE, STAGE_TRANSFER);
        gpuBlitTexture(commandBuffer, capture, outputTexture);

        gpuSubmit(queue, Span<GpuCommandBuffer>(&commandBuffer, 1), semaphore, nextFrame);
        nextFrame++;

        auto swap = raytracingData.pixelSample;
        raytracingData.pixelSample = raytracingData.prevPixelSample;
        raytracingData.prevPixelSample = swap;
        raytracingData.frame = nextFrame;
        raytracingData.accumulatedFrames++;
    }
    gpuWaitSemaphore(semaphore, nextFrame - 1);

    test::Image actual = test::readbackRGBA8(device, queue, capture, RENDER_W, RENDER_H);
    int rc = test::finalize(args, "raytracing", actual);

    allocator.reset();
    descriptorAllocator.reset();
    stbi_image_free(inputImage);
    gpuDestroyTexture(texture);
    gpuFree(device, texturePtr);
    gpuDestroyTexture(outputTexture);
    gpuFree(device, outputPtr);
    gpuDestroyTexture(albedoTexture);
    gpuFree(device, albedoPtr);
    gpuDestroyTexture(normalsTexture);
    gpuFree(device, normalsPtr);
    gpuDestroyTexture(motionVectorsTexture);
    gpuFree(device, motionPtr);
    gpuDestroyTexture(capture);
    gpuFree(device, capturePtr);
    gpuFreePipeline(referencePipeline);
    gpuDestroyAccelerationStructure(blas);
    gpuFree(device, blasPtr);
    gpuDestroyAccelerationStructure(tlas);
    gpuFree(device, tlasPtr);
    gpuFree(device, scratchPtr);
    gpuFree(device, pixelSample);
    gpuFree(device, prevPixelSample);
    gpuDestroySemaphore(semaphore);
    gpuDestroyQueue(queue);
    gpuDestroyDevice(device);
    test::endValidationCapture();
    gpuDestroyInstance();

    if (test::validationFailed())
    {
        std::cerr << "FAIL [raytracing]: Vulkan validation messages were emitted\n";
        rc = 1;
    }
    return rc;
}
