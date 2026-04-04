#include "Utilities.h"

#include "../../External/stb_image.h"

std::vector<uint8_t> loadIR(const std::filesystem::path &path)
{
    std::ifstream file { path, std::ios::binary | std::ios::ate };
    if (!file.is_open())
    {
        return {};
    }
    auto size = file.tellg();
    std::vector<uint8_t> buffer(size);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), size);
    return buffer;
}

void getCube(std::vector<float3> &vertices, std::vector<float3> &normals, std::vector<float2> &uvs, std::vector<uint32_t> &indices)
{
    // 24 vertices: 4 per face × 6 faces, each with correct face normal
    vertices = {
        // Front face (z = -1)
        { -1.0f, -1.0f, -1.0f },
        {  1.0f, -1.0f, -1.0f },
        {  1.0f,  1.0f, -1.0f },
        { -1.0f,  1.0f, -1.0f },
        // Back face (z = +1)
        {  1.0f, -1.0f,  1.0f },
        { -1.0f, -1.0f,  1.0f },
        { -1.0f,  1.0f,  1.0f },
        {  1.0f,  1.0f,  1.0f },
        // Left face (x = -1)
        { -1.0f, -1.0f,  1.0f },
        { -1.0f, -1.0f, -1.0f },
        { -1.0f,  1.0f, -1.0f },
        { -1.0f,  1.0f,  1.0f },
        // Right face (x = +1)
        {  1.0f, -1.0f, -1.0f },
        {  1.0f, -1.0f,  1.0f },
        {  1.0f,  1.0f,  1.0f },
        {  1.0f,  1.0f, -1.0f },
        // Bottom face (y = -1)
        { -1.0f, -1.0f,  1.0f },
        {  1.0f, -1.0f,  1.0f },
        {  1.0f, -1.0f, -1.0f },
        { -1.0f, -1.0f, -1.0f },
        // Top face (y = +1)
        { -1.0f,  1.0f, -1.0f },
        {  1.0f,  1.0f, -1.0f },
        {  1.0f,  1.0f,  1.0f },
        { -1.0f,  1.0f,  1.0f },
    };

    normals = {
        // Front face
        {  0.0f,  0.0f, -1.0f },
        {  0.0f,  0.0f, -1.0f },
        {  0.0f,  0.0f, -1.0f },
        {  0.0f,  0.0f, -1.0f },
        // Back face
        {  0.0f,  0.0f,  1.0f },
        {  0.0f,  0.0f,  1.0f },
        {  0.0f,  0.0f,  1.0f },
        {  0.0f,  0.0f,  1.0f },
        // Left face
        { -1.0f,  0.0f,  0.0f },
        { -1.0f,  0.0f,  0.0f },
        { -1.0f,  0.0f,  0.0f },
        { -1.0f,  0.0f,  0.0f },
        // Right face
        {  1.0f,  0.0f,  0.0f },
        {  1.0f,  0.0f,  0.0f },
        {  1.0f,  0.0f,  0.0f },
        {  1.0f,  0.0f,  0.0f },
        // Bottom face
        {  0.0f, -1.0f,  0.0f },
        {  0.0f, -1.0f,  0.0f },
        {  0.0f, -1.0f,  0.0f },
        {  0.0f, -1.0f,  0.0f },
        // Top face
        {  0.0f,  1.0f,  0.0f },
        {  0.0f,  1.0f,  0.0f },
        {  0.0f,  1.0f,  0.0f },
        {  0.0f,  1.0f,  0.0f },
    };

    uvs = {
        // Front face
        { 0.0f, 1.0f }, { 1.0f, 1.0f }, { 1.0f, 0.0f }, { 0.0f, 0.0f },
        // Back face
        { 0.0f, 1.0f }, { 1.0f, 1.0f }, { 1.0f, 0.0f }, { 0.0f, 0.0f },
        // Left face
        { 0.0f, 1.0f }, { 1.0f, 1.0f }, { 1.0f, 0.0f }, { 0.0f, 0.0f },
        // Right face
        { 0.0f, 1.0f }, { 1.0f, 1.0f }, { 1.0f, 0.0f }, { 0.0f, 0.0f },
        // Bottom face
        { 0.0f, 1.0f }, { 1.0f, 1.0f }, { 1.0f, 0.0f }, { 0.0f, 0.0f },
        // Top face
        { 0.0f, 1.0f }, { 1.0f, 1.0f }, { 1.0f, 0.0f }, { 0.0f, 0.0f },
    };

    indices = {
         0,  1,  2,  2,  3,  0,  // Front
         4,  5,  6,  6,  7,  4,  // Back
         8,  9, 10, 10, 11,  8,  // Left
        12, 13, 14, 14, 15, 12,  // Right
        16, 17, 18, 18, 19, 16,  // Bottom
        20, 21, 22, 22, 23, 20,  // Top
    };
}

std::vector<glm::vec2> haltonSequence(uint length)
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

TextRenderer::TextRenderer(GpuDevice gpuDevice, GpuTextureDesc textureDesc)
    : device(gpuDevice), targetDesc(textureDesc)
{
    allocator = new LinearAllocator(device);

    auto textIRVertex = loadIR("../Shaders/Common/TextVertex.spv");
    auto textIRPixel = loadIR("../Shaders/Common/TextPixel.spv");

    ColorTarget colorTarget = {};
    colorTarget.format = textureDesc.format;

    GpuRasterDesc rasterDesc = {
        .cull = CULL_NONE,
        .colorTargets = Span<ColorTarget>(&colorTarget, 1)
    };

    pipeline = gpuCreateGraphicsPipeline(device, ByteSpan(textIRVertex), ByteSpan(textIRPixel), rasterDesc);

    std::vector<uint32_t> indices = { 0, 1, 2, 2, 3, 0 };
    
    vertexData = allocator->allocate<TextVertexData>();
    pixelData = allocator->allocate<TextPixelData>();
    indexData = allocator->allocate<uint32_t>(6);
    textData = allocator->allocate<uint8_t>(maxTextLength);

    memcpy(indexData.cpu, indices.data(), sizeof(uint32_t) * 6);

    int atlasChannels;
    stbi_uc* atlasData = stbi_load("./Assets/Atlas.png", &atlasWidth, &atlasHeight, &atlasChannels, 4);
    if (atlasData)
    {
        auto atlasUpload = allocator->allocate<uint8_t>(atlasWidth * atlasHeight * 4);
        memcpy(atlasUpload.cpu, atlasData, atlasWidth * atlasHeight * 4);

        GpuTextureDesc atlasDesc{
            .type = TEXTURE_2D,
            .dimensions = {static_cast<uint32_t>(atlasWidth), static_cast<uint32_t>(atlasHeight), 1},
            .format = FORMAT_RGBA8_UNORM,
            .usage = static_cast<USAGE_FLAGS>(USAGE_SAMPLED | USAGE_TRANSFER_DST)};

        GpuTextureSizeAlign atlasSizeAlign = gpuTextureSizeAlign(device, atlasDesc);
        atlasPtr = gpuMalloc(device, atlasSizeAlign.size, MEMORY_GPU);
        atlas = gpuCreateTexture(device, atlasDesc, atlasPtr);

        auto semaphore = gpuCreateSemaphore(device, 0);
        auto queue = gpuCreateQueue(device);
        auto cmd = gpuStartCommandRecording(queue);
        
        gpuCopyToTexture(cmd, atlasUpload.gpu, atlas);

        gpuSubmit(queue, Span<GpuCommandBuffer>(&cmd, 1), semaphore, 1);
        gpuWaitSemaphore(semaphore, 1);

        gpuDestroySemaphore(semaphore);
        stbi_image_free(atlasData);

        textureHeap = allocator->allocate<GpuTextureDescriptor>(1024);
        textureHeap.cpu[0] = gpuTextureViewDescriptor(atlas, GpuViewDesc{.format = FORMAT_RGBA8_UNORM});
    }
}

TextRenderer::~TextRenderer()
{
    gpuFreePipeline(pipeline);
    gpuDestroyTexture(atlas);
    gpuFree(device, atlasPtr);
    allocator->free();
    delete allocator;
}

void TextRenderer::renderText(GpuCommandBuffer cmd, GpuTexture target, const std::string &text, float x, float y, float scale, float3 color)
{
    if (offset + text.size() > maxTextLength)
    {
        offset = 0;
    }

    memcpy(textData.cpu + offset, text.data(), std::min(text.size(), static_cast<size_t>(maxTextLength - offset)));

    vertexData.cpu->width = targetDesc.dimensions.x;
    vertexData.cpu->height = targetDesc.dimensions.y;
    vertexData.cpu->textWidth = atlasWidth / 256;
    vertexData.cpu->textHeight = atlasHeight;
    vertexData.cpu->atlasWidth = atlasWidth;
    vertexData.cpu->atlasHeight = atlasHeight;
    vertexData.cpu->text = textData.gpu + offset;

    offset += text.size();

    pixelData.cpu->atlas = 0;

    GpuTexture colorTargets[] = { target };
    GpuRenderPassDesc renderPassDesc = {
        .colorTargets = Span<GpuTexture>(colorTargets, 1),
        .loadOp = LOAD_OP_LOAD
    };

    gpuSetPipeline(cmd, pipeline);
    gpuSetActiveTextureHeapPtr(cmd, textureHeap.gpu);
    gpuBeginRenderPass(cmd, renderPassDesc);
    gpuDrawIndexedInstanced(cmd, vertexData.gpu, pixelData.gpu, indexData.gpu, 6, static_cast<uint32_t>(text.size()));
    gpuEndRenderPass(cmd);
}
