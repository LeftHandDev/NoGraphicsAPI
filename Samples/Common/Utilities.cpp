#include "Utilities.h"

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

TextRenderer::TextRenderer(GpuTexture textureTarget, GpuTextureDesc textureDesc)
    : target(textureTarget), targetDesc(textureDesc)
{
    auto textIRVertex = loadIR("../Shaders/Common/TextVertex.spv");
    auto textIRPixel = loadIR("../Shaders/Common/TextPixel.spv");

    ColorTarget colorTarget = {};
    colorTarget.format = textureDesc.format;

    GpuRasterDesc rasterDesc = {
        .cull = CULL_NONE,
        .colorTargets = Span<ColorTarget>(&colorTarget, 1)
    };

    pipeline = gpuCreateGraphicsPipeline(ByteSpan(textIRVertex), ByteSpan(textIRPixel), rasterDesc);

    std::vector<uint32_t> indices = { 0, 1, 2, 2, 3, 0 };
    

    vertexData = allocate<TextVertexData>();
    pixelData = allocate<TextPixelData>();
    indexData = allocate<uint32_t>(6);
    textData = allocate<uint8_t>(maxTextLength);

    memcpy(indexData.cpu, indices.data(), sizeof(uint32_t) * 6);

}

TextRenderer::~TextRenderer()
{
    gpuFreePipeline(pipeline);
    vertexData.free();
    pixelData.free();
    indexData.free();
    textData.free();
}

void TextRenderer::renderText(GpuCommandBuffer cmd, const std::string &text, float x, float y, float scale, float3 color)
{
    if (offset + text.size() > maxTextLength)
    {
        offset = 0;
    }

    memcpy(textData.cpu + offset, text.data(), std::min(text.size(), static_cast<size_t>(maxTextLength - offset)));
    offset += text.size();

    vertexData.cpu->width = targetDesc.dimensions.x;
    vertexData.cpu->height = targetDesc.dimensions.y;
    vertexData.cpu->textWidth = 16;
    vertexData.cpu->textHeight = 16;
    vertexData.cpu->text = textData.gpu + offset;

    pixelData.cpu->atlas = 0;

    gpuSetPipeline(cmd, pipeline);
    gpuDrawIndexedInstanced(cmd, vertexData.gpu, pixelData.gpu, nullptr, 6, static_cast<uint32_t>(text.size()));
}
