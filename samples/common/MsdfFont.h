#ifndef MSDF_FONT_H
#define MSDF_FONT_H

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "NoGraphicsAPI.h"
#include "MsdfShared.h"

// MTSDF text rendering on top of NoGraphicsAPI.
//
// Two pieces:
//   * MsdfFont          - a baked font: the atlas (resident in a texture heap)
//                         plus glyph metrics from msdf-atlas-gen's JSON. Its
//                         measure()/lineHeight()/ascent() are pure-CPU sizing
//                         helpers a layout library (e.g. Clay) calls during
//                         layout -- they need no GPU work and no live frame.
//   * MsdfTextRenderer  - owns the pipeline + atlas heap; batches glyph quads
//                         from any number of addText() calls and draws them in
//                         a single instanced draw.
//
// Build a batch once (clear -> addText... -> upload) and render() it each frame;
// rebuild only when the text changes.

// A baked MTSDF font. Created via MsdfTextRenderer::loadFont(), which registers
// the atlas into the renderer's texture heap.
class MsdfFont
{
public:
    struct Glyph
    {
        float advance = 0.0f; // pen advance, em units (1 em == one pixelSize)
        // Quad bounds relative to the pen origin, em units, +y up (msdf convention).
        float planeLeft = 0.0f, planeBottom = 0.0f, planeRight = 0.0f, planeTop = 0.0f;
        // Atlas rect in texels, +y up (the JSON's yOrigin == "bottom").
        float atlasLeft = 0.0f, atlasBottom = 0.0f, atlasRight = 0.0f, atlasTop = 0.0f;
        bool hasGeometry = false; // false for whitespace (advance only, no quad)
    };

    // Loads metrics from jsonPath and the atlas image from pngPath, uploads the
    // atlas into heapCpu[heapSlot] (the descriptor heap the renderer owns), and
    // records that heap index. The atlas upload is synchronous. Throws
    // std::runtime_error on failure.
    MsdfFont(GpuDevice device, const std::string& jsonPath, const std::string& pngPath,
             GpuTextureDescriptor* heapCpu, uint32_t heapSlot);
    ~MsdfFont();

    MsdfFont(const MsdfFont&) = delete;
    MsdfFont& operator=(const MsdfFont&) = delete;

    // --- sizing helpers (CPU only; intended for layout / Clay) --------------
    // pixelSize is the em size in pixels (the "font size"). letterSpacing adds
    // extra pixels after each glyph. Both honor '\n' for multi-line strings.
    //
    // Returns { width, height } in pixels: width is the widest line's pen
    // advance; height is lineCount * lineHeight(pixelSize). This is exactly the
    // shape a Clay_MeasureTextFunction needs -- see measureFunctionExample() in
    // the .cpp for a ready-to-adapt binding.
    float2 measure(std::string_view utf8, float pixelSize, float letterSpacing = 0.0f) const;

    float lineHeight(float pixelSize) const; // baseline-to-baseline, pixels
    float ascent(float pixelSize) const;     // baseline to top, pixels
    float descent(float pixelSize) const;    // baseline to bottom, pixels (positive depth)

    // --- accessors used by the renderer -------------------------------------
    const Glyph* glyph(uint32_t codepoint) const;
    uint32_t atlasIndex() const
    {
        return heapIndex;
    }
    float distanceRange() const
    {
        return pxRange;
    }
    float2 atlasSize() const
    {
        return { static_cast<float>(atlasW), static_cast<float>(atlasH) };
    }

private:
    GpuDevice device;
    GpuTexture atlas = nullptr;
    void* atlasPtr = nullptr;
    uint32_t heapIndex = 0;

    int atlasW = 0, atlasH = 0;
    float pxRange = 0.0f;           // distanceRange, atlas texels
    float metricsLineHeight = 0.0f; // em
    float metricsAscender = 0.0f;   // em
    float metricsDescender = 0.0f;  // em (negative)

    std::unordered_map<uint32_t, Glyph> glyphs;
};

// Batches and draws MTSDF text. All addText() calls between clear() and the next
// clear() must use the same font (single atlas per draw); upload() pushes the
// batch to the GPU and render() issues one instanced draw of every glyph.
class MsdfTextRenderer
{
public:
    // colorTargetFormat must match the texture render() draws into. shaderDir is
    // the directory the compiled .spv live in (samples run from build/bin so the
    // default "shaders" works; tests pass NGAPI_TEST_SHADER_DIR).
    MsdfTextRenderer(GpuDevice device, FORMAT colorTargetFormat,
                     const std::string& shaderDir = "shaders",
                     uint32_t maxGlyphs = 16384, uint32_t heapCapacity = 256);
    ~MsdfTextRenderer();

    MsdfTextRenderer(const MsdfTextRenderer&) = delete;
    MsdfTextRenderer& operator=(const MsdfTextRenderer&) = delete;

    // Loads a font and registers its atlas in the next free heap slot.
    MsdfFont* loadFont(const std::string& jsonPath, const std::string& pngPath);

    void clear(); // begin a new batch

    // Appends a (possibly multi-line) string. (x, y) is the pen origin in target
    // pixels: x is the left edge, y is the baseline of the first line. Pass a
    // non-zero outlineWidthPx (with an outlineColor) to draw an outline.
    void addText(const MsdfFont* font, std::string_view utf8, float x, float y,
                 float pixelSize, float4 color, float letterSpacing = 0.0f,
                 float4 outlineColor = { 0.0f, 0.0f, 0.0f, 0.0f }, float outlineWidthPx = 0.0f);

    void upload(); // push the accumulated batch to the GPU (call once after addText)

    // Records one render pass into target (sized targetW x targetH) and draws the
    // whole batch. LOAD_OP_CLEAR clears to opaque black first; LOAD_OP_LOAD
    // composites over existing contents.
    void render(GpuCommandBuffer cmd, GpuTexture target, uint32_t targetW, uint32_t targetH,
                LOAD_OP loadOp = LOAD_OP_LOAD);

    uint32_t glyphCount() const
    {
        return count;
    }

private:
    GpuDevice device;
    GpuPipeline pipeline = nullptr;
    GpuDepthStencilState depthStencilState = nullptr;

    // Texture heap holding the atlas descriptors. Allocated with
    // MEMORY_DESCRIPTOR so it satisfies the descriptor-buffer usage and
    // alignment that gpuSetActiveTextureHeapPtr requires on direct-bind devices.
    void* heapAlloc = nullptr;
    GpuTextureDescriptor* heapCpu = nullptr;
    void* heapGpu = nullptr;
    uint32_t heapCapacity = 0;
    uint32_t nextHeapSlot = 0;

    // Glyph instance buffer + the two push-constant blocks. Each buffer is
    // host-visible (cpu) with a matching device address (gpu) for the draw.
    void* instAlloc = nullptr;
    MsdfGlyphInstance* instancesCpu = nullptr;
    MsdfGlyphInstance* instancesGpu = nullptr;
    void* vertexAlloc = nullptr;
    MsdfVertexData* vertexDataCpu = nullptr;
    MsdfVertexData* vertexDataGpu = nullptr;
    void* pixelAlloc = nullptr;
    MsdfPixelData* pixelDataCpu = nullptr;
    MsdfPixelData* pixelDataGpu = nullptr;
    void* indexAlloc = nullptr;
    uint32_t* indexDataCpu = nullptr;
    uint32_t* indexDataGpu = nullptr;

    uint32_t maxGlyphs = 0;
    uint32_t count = 0;                  // glyphs in the current batch
    const MsdfFont* batchFont = nullptr; // font of the current batch (sets pixelData)

    std::vector<std::unique_ptr<MsdfFont>> fonts;
};

#endif // MSDF_FONT_H
