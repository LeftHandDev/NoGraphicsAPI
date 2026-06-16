#include "MsdfFont.h"

#include "Utilities.h" // loadIR

#include "stb_image.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

// ===========================================================================
// Minimal JSON reader
// ===========================================================================
// msdf-atlas-gen emits a small, well-structured JSON (objects, arrays, numbers,
// strings, booleans). Rather than pull in a dependency we parse just that subset
// here. Numbers are stored as double; objects preserve insertion order in a
// small vector (the files are tiny, so linear key lookup is fine).
namespace
{
    struct JsonValue
    {
        enum class Type
        {
            Null,
            Bool,
            Number,
            String,
            Array,
            Object
        };

        Type type = Type::Null;
        double number = 0.0;
        bool boolean = false;
        std::string string;
        std::vector<JsonValue> array;
        std::vector<std::pair<std::string, JsonValue>> object;

        const JsonValue* find(std::string_view key) const
        {
            for (const auto& kv : object)
            {
                if (kv.first == key)
                {
                    return &kv.second;
                }
            }
            return nullptr;
        }

        double num(double fallback = 0.0) const
        {
            return type == Type::Number ? number : fallback;
        }
    };

    class JsonParser
    {
    public:
        explicit JsonParser(const std::string& text) : p(text.data()), end(text.data() + text.size())
        {
        }

        JsonValue parse()
        {
            skipWs();
            JsonValue v = parseValue();
            return v;
        }

        bool ok() const
        {
            return good;
        }

    private:
        const char* p;
        const char* end;
        bool good = true;

        void skipWs()
        {
            while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r'))
            {
                p++;
            }
        }

        JsonValue parseValue()
        {
            skipWs();
            if (p >= end)
            {
                good = false;
                return {};
            }
            switch (*p)
            {
            case '{':
                return parseObject();
            case '[':
                return parseArray();
            case '"':
            {
                JsonValue v;
                v.type = JsonValue::Type::String;
                v.string = parseString();
                return v;
            }
            case 't':
            case 'f':
                return parseBool();
            case 'n':
                p = std::min(end, p + 4); // null
                return {};
            default:
                return parseNumber();
            }
        }

        std::string parseString()
        {
            std::string out;
            p++; // opening quote
            while (p < end && *p != '"')
            {
                if (*p == '\\')
                {
                    p++;
                    if (p >= end)
                    {
                        break;
                    }
                    switch (*p)
                    {
                    case 'n':
                        out += '\n';
                        break;
                    case 't':
                        out += '\t';
                        break;
                    case 'r':
                        out += '\r';
                        break;
                    case 'b':
                        out += '\b';
                        break;
                    case 'f':
                        out += '\f';
                        break;
                    case '"':
                        out += '"';
                        break;
                    case '\\':
                        out += '\\';
                        break;
                    case '/':
                        out += '/';
                        break;
                    case 'u':
                        // Not needed by the msdf metrics; skip the 4 hex digits.
                        for (int i = 0; i < 4 && p + 1 < end; i++)
                        {
                            p++;
                        }
                        out += '?';
                        break;
                    default:
                        out += *p;
                        break;
                    }
                    p++;
                }
                else
                {
                    out += *p++;
                }
            }
            if (p < end)
            {
                p++; // closing quote
            }
            return out;
        }

        JsonValue parseNumber()
        {
            const char* start = p;
            while (p < end && (*p == '-' || *p == '+' || *p == '.' || *p == 'e' || *p == 'E' || (*p >= '0' && *p <= '9')))
            {
                p++;
            }
            JsonValue v;
            v.type = JsonValue::Type::Number;
            v.number = std::strtod(std::string(start, p).c_str(), nullptr);
            return v;
        }

        JsonValue parseBool()
        {
            JsonValue v;
            v.type = JsonValue::Type::Bool;
            if (*p == 't')
            {
                v.boolean = true;
                p = std::min(end, p + 4);
            }
            else
            {
                v.boolean = false;
                p = std::min(end, p + 5);
            }
            return v;
        }

        JsonValue parseArray()
        {
            JsonValue v;
            v.type = JsonValue::Type::Array;
            p++; // [
            skipWs();
            if (p < end && *p == ']')
            {
                p++;
                return v;
            }
            while (p < end)
            {
                v.array.push_back(parseValue());
                skipWs();
                if (p < end && *p == ',')
                {
                    p++;
                    continue;
                }
                if (p < end && *p == ']')
                {
                    p++;
                    break;
                }
                good = false;
                break;
            }
            return v;
        }

        JsonValue parseObject()
        {
            JsonValue v;
            v.type = JsonValue::Type::Object;
            p++; // {
            skipWs();
            if (p < end && *p == '}')
            {
                p++;
                return v;
            }
            while (p < end)
            {
                skipWs();
                if (p >= end || *p != '"')
                {
                    good = false;
                    break;
                }
                std::string key = parseString();
                skipWs();
                if (p < end && *p == ':')
                {
                    p++;
                }
                else
                {
                    good = false;
                    break;
                }
                v.object.emplace_back(std::move(key), parseValue());
                skipWs();
                if (p < end && *p == ',')
                {
                    p++;
                    continue;
                }
                if (p < end && *p == '}')
                {
                    p++;
                    break;
                }
                good = false;
                break;
            }
            return v;
        }
    };

    // Decode one UTF-8 codepoint starting at s[i], advancing i past it. Returns
    // U+FFFD on malformed input. (ASCII fast-paths to the byte value.)
    uint32_t decodeUtf8(std::string_view s, size_t& i)
    {
        unsigned char c = static_cast<unsigned char>(s[i]);
        if (c < 0x80)
        {
            i++;
            return c;
        }
        uint32_t cp;
        int extra;
        if ((c >> 5) == 0x6)
        {
            cp = c & 0x1f;
            extra = 1;
        }
        else if ((c >> 4) == 0xe)
        {
            cp = c & 0x0f;
            extra = 2;
        }
        else if ((c >> 3) == 0x1e)
        {
            cp = c & 0x07;
            extra = 3;
        }
        else
        {
            i++;
            return 0xFFFD;
        }
        i++;
        for (int k = 0; k < extra; k++)
        {
            if (i >= s.size() || (static_cast<unsigned char>(s[i]) & 0xc0) != 0x80)
            {
                return 0xFFFD;
            }
            cp = (cp << 6) | (static_cast<unsigned char>(s[i]) & 0x3f);
            i++;
        }
        return cp;
    }

    std::string readFile(const std::string& path)
    {
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open())
        {
            return {};
        }
        std::ostringstream ss;
        ss << f.rdbuf();
        return ss.str();
    }
} // namespace

// ---------------------------------------------------------------------------
// Example: binding measure() to Clay's text-measurement callback. Clay isn't a
// dependency of this repo, so this lives in a comment; drop it next to your Clay
// setup and call Clay_SetMeasureTextFunction(measureText, &fonts).
//
//   // fonts: your own array indexed by Clay's config->fontId, each entry an MsdfFont*.
//   Clay_Dimensions measureText(Clay_StringSlice text, Clay_TextElementConfig* config, void* userData) {
//       auto* fonts = static_cast<std::vector<MsdfFont*>*>(userData);
//       const MsdfFont* font = (*fonts)[config->fontId];
//       float2 d = font->measure({ text.chars, (size_t)text.length },
//                                (float)config->fontSize, (float)config->letterSpacing);
//       return Clay_Dimensions{ d.x, d.y };
//   }
//
// Clay does its own word wrapping by calling this per run, so measure() needs
// only single-line (plus '\n') handling, which it provides.
// ---------------------------------------------------------------------------

// ===========================================================================
// MsdfFont
// ===========================================================================
MsdfFont::MsdfFont(GpuDevice gpuDevice, const std::string& jsonPath, const std::string& pngPath,
                   GpuTextureDescriptor* heapCpu, uint32_t heapSlot)
    : device(gpuDevice)
{
    // --- parse the metrics JSON ---
    std::string json = readFile(jsonPath);
    if (json.empty())
    {
        throw std::runtime_error("MsdfFont: could not read JSON '" + jsonPath + "'");
    }
    JsonParser parser(json);
    JsonValue root = parser.parse();

    const JsonValue* atlasNode = root.find("atlas");
    const JsonValue* metricsNode = root.find("metrics");
    const JsonValue* glyphsNode = root.find("glyphs");
    if (!atlasNode || !metricsNode || !glyphsNode || glyphsNode->type != JsonValue::Type::Array)
    {
        throw std::runtime_error("MsdfFont: malformed JSON '" + jsonPath + "'");
    }

    pxRange = static_cast<float>(atlasNode->find("distanceRange") ? atlasNode->find("distanceRange")->num(4.0) : 4.0);
    atlasW = static_cast<int>(atlasNode->find("width") ? atlasNode->find("width")->num() : 0);
    atlasH = static_cast<int>(atlasNode->find("height") ? atlasNode->find("height")->num() : 0);

    // Metrics are in em units when emSize == 1; normalize by emSize so advances
    // and plane bounds are always per-em (a no-op for the typical emSize == 1).
    double emSize = metricsNode->find("emSize") ? metricsNode->find("emSize")->num(1.0) : 1.0;
    if (emSize == 0.0)
    {
        emSize = 1.0;
    }
    metricsLineHeight = static_cast<float>((metricsNode->find("lineHeight") ? metricsNode->find("lineHeight")->num(1.2) : 1.2) / emSize);
    metricsAscender = static_cast<float>((metricsNode->find("ascender") ? metricsNode->find("ascender")->num() : 0.0) / emSize);
    metricsDescender = static_cast<float>((metricsNode->find("descender") ? metricsNode->find("descender")->num() : 0.0) / emSize);

    for (const JsonValue& g : glyphsNode->array)
    {
        const JsonValue* uni = g.find("unicode");
        if (!uni)
        {
            continue;
        }
        Glyph glyph;
        glyph.advance = static_cast<float>((g.find("advance") ? g.find("advance")->num() : 0.0) / emSize);

        if (const JsonValue* pb = g.find("planeBounds"))
        {
            glyph.hasGeometry = true;
            glyph.planeLeft = static_cast<float>((pb->find("left") ? pb->find("left")->num() : 0.0) / emSize);
            glyph.planeBottom = static_cast<float>((pb->find("bottom") ? pb->find("bottom")->num() : 0.0) / emSize);
            glyph.planeRight = static_cast<float>((pb->find("right") ? pb->find("right")->num() : 0.0) / emSize);
            glyph.planeTop = static_cast<float>((pb->find("top") ? pb->find("top")->num() : 0.0) / emSize);
        }
        if (const JsonValue* ab = g.find("atlasBounds"))
        {
            glyph.atlasLeft = static_cast<float>(ab->find("left") ? ab->find("left")->num() : 0.0);
            glyph.atlasBottom = static_cast<float>(ab->find("bottom") ? ab->find("bottom")->num() : 0.0);
            glyph.atlasRight = static_cast<float>(ab->find("right") ? ab->find("right")->num() : 0.0);
            glyph.atlasTop = static_cast<float>(ab->find("top") ? ab->find("top")->num() : 0.0);
        }
        glyphs.emplace(static_cast<uint32_t>(uni->num()), glyph);
    }

    // --- upload the atlas image into the texture heap ---
    int imgW = 0, imgH = 0, imgChannels = 0;
    stbi_uc* image = stbi_load(pngPath.c_str(), &imgW, &imgH, &imgChannels, 4);
    if (!image)
    {
        throw std::runtime_error("MsdfFont: could not read atlas image '" + pngPath + "'");
    }
    if (atlasW == 0 || atlasH == 0)
    {
        atlasW = imgW;
        atlasH = imgH;
    }

    const size_t bytes = static_cast<size_t>(imgW) * imgH * 4;
    void* uploadCpu = gpuMalloc(device, bytes);
    void* uploadGpu = gpuHostToDevicePointer(device, uploadCpu);
    memcpy(uploadCpu, image, bytes);
    stbi_image_free(image);

    // RGBA8_UNORM (not sRGB): MSDF stores raw distances, so the values must come
    // back un-gamma-corrected.
    GpuTextureDesc atlasDesc{
        .type = TEXTURE_2D,
        .dimensions = { static_cast<uint32_t>(imgW), static_cast<uint32_t>(imgH), 1 },
        .format = FORMAT_RGBA8_UNORM,
        .usage = static_cast<USAGE_FLAGS>(USAGE_SAMPLED | USAGE_TRANSFER_DST)
    };
    GpuTextureSizeAlign sizeAlign = gpuTextureSizeAlign(device, atlasDesc);
    atlasPtr = gpuMalloc(device, sizeAlign.size, MEMORY_GPU);
    atlas = gpuCreateTexture(device, atlasDesc, atlasPtr);

    GpuQueue queue = gpuCreateQueue(device);
    GpuSemaphore semaphore = gpuCreateSemaphore(device, 0);
    GpuCommandBuffer cmd = gpuStartCommandRecording(queue);
    gpuCopyToTexture(cmd, uploadGpu, atlas);
    gpuBarrier(cmd, STAGE_TRANSFER, STAGE_PIXEL_SHADER);
    gpuSubmit(queue, Span<GpuCommandBuffer>(&cmd, 1), semaphore, 1);
    gpuWaitSemaphore(semaphore, 1);
    gpuDestroySemaphore(semaphore);
    gpuDestroyQueue(queue);
    gpuFree(device, uploadCpu);

    heapCpu[heapSlot] = gpuTextureViewDescriptor(atlas, GpuViewDesc{ .format = FORMAT_RGBA8_UNORM });
    heapIndex = heapSlot;
}

MsdfFont::~MsdfFont()
{
    if (atlas)
    {
        gpuDestroyTexture(atlas);
    }
    if (atlasPtr)
    {
        gpuFree(device, atlasPtr);
    }
}

const MsdfFont::Glyph* MsdfFont::glyph(uint32_t codepoint) const
{
    auto it = glyphs.find(codepoint);
    return it == glyphs.end() ? nullptr : &it->second;
}

float MsdfFont::lineHeight(float pixelSize) const
{
    return metricsLineHeight * pixelSize;
}
float MsdfFont::ascent(float pixelSize) const
{
    return metricsAscender * pixelSize;
}
float MsdfFont::descent(float pixelSize) const
{
    return -metricsDescender * pixelSize;
}

float2 MsdfFont::measure(std::string_view utf8, float pixelSize, float letterSpacing) const
{
    float maxWidth = 0.0f;
    float lineWidth = 0.0f;
    int lines = 1;

    size_t i = 0;
    while (i < utf8.size())
    {
        uint32_t cp = decodeUtf8(utf8, i);
        if (cp == '\n')
        {
            maxWidth = std::max(maxWidth, lineWidth);
            lineWidth = 0.0f;
            lines++;
            continue;
        }
        const Glyph* g = glyph(cp);
        if (!g)
        {
            g = glyph(0x20); // fall back to the space advance for unknown glyphs
        }
        if (g)
        {
            lineWidth += g->advance * pixelSize + letterSpacing;
        }
    }
    maxWidth = std::max(maxWidth, lineWidth);
    return { maxWidth, lines * lineHeight(pixelSize) };
}

// ===========================================================================
// MsdfTextRenderer
// ===========================================================================
MsdfTextRenderer::MsdfTextRenderer(GpuDevice gpuDevice, FORMAT colorTargetFormat,
                                   const std::string& shaderDir, uint32_t maxGlyphsIn, uint32_t heapCapacityIn)
    : device(gpuDevice), heapCapacity(heapCapacityIn), maxGlyphs(maxGlyphsIn)
{
    std::vector<uint8_t> vertexIR = loadIR(shaderDir + "/common/MsdfVertex.spv");
    std::vector<uint8_t> pixelIR = loadIR(shaderDir + "/common/MsdfPixel.spv");

    // Straight-alpha compositing: out = src.rgb*src.a + dst.rgb*(1-src.a); the
    // alpha channel accumulates coverage the same way.
    GpuBlendDesc blend{
        .colorOp = BLEND_ADD,
        .srcColorFactor = FACTOR_SRC_ALPHA,
        .dstColorFactor = FACTOR_ONE_MINUS_SRC_ALPHA,
        .alphaOp = BLEND_ADD,
        .srcAlphaFactor = FACTOR_ONE,
        .dstAlphaFactor = FACTOR_ONE_MINUS_SRC_ALPHA,
        .colorWriteMask = 0xf
    };
    ColorTarget colorTarget{ .format = colorTargetFormat };
    GpuRasterDesc rasterDesc{
        .cull = CULL_NONE,
        .colorTargets = Span<ColorTarget>(&colorTarget, 1),
        .blendState = &blend
    };
    pipeline = gpuCreateGraphicsPipeline(device, ByteSpan(vertexIR), ByteSpan(pixelIR), rasterDesc);

    // The text pass has no depth/stencil target, but depth/stencil enable is
    // dynamic pipeline state, so a disabled state must be bound before drawing.
    depthStencilState = gpuCreateDepthStencilState(GpuDepthStencilDesc{});

    heapAlloc = gpuMalloc(device, sizeof(GpuTextureDescriptor) * heapCapacity, MEMORY_DESCRIPTOR);
    heapCpu = static_cast<GpuTextureDescriptor*>(heapAlloc);
    heapGpu = gpuHostToDevicePointer(device, heapAlloc);

    instAlloc = gpuMalloc(device, sizeof(MsdfGlyphInstance) * maxGlyphs);
    instancesCpu = static_cast<MsdfGlyphInstance*>(instAlloc);
    instancesGpu = static_cast<MsdfGlyphInstance*>(gpuHostToDevicePointer(device, instAlloc));

    vertexAlloc = gpuMalloc(device, sizeof(MsdfVertexData));
    vertexDataCpu = static_cast<MsdfVertexData*>(vertexAlloc);
    vertexDataGpu = static_cast<MsdfVertexData*>(gpuHostToDevicePointer(device, vertexAlloc));

    pixelAlloc = gpuMalloc(device, sizeof(MsdfPixelData));
    pixelDataCpu = static_cast<MsdfPixelData*>(pixelAlloc);
    pixelDataGpu = static_cast<MsdfPixelData*>(gpuHostToDevicePointer(device, pixelAlloc));

    indexAlloc = gpuMalloc(device, sizeof(uint32_t) * 6);
    indexDataCpu = static_cast<uint32_t*>(indexAlloc);
    indexDataGpu = static_cast<uint32_t*>(gpuHostToDevicePointer(device, indexAlloc));
    const uint32_t indices[6] = { 0, 1, 2, 2, 3, 0 };
    memcpy(indexDataCpu, indices, sizeof(indices));
}

MsdfTextRenderer::~MsdfTextRenderer()
{
    fonts.clear(); // destroy fonts (and their atlases) before freeing the heap

    if (pipeline)
    {
        gpuFreePipeline(pipeline);
    }
    if (depthStencilState)
    {
        gpuFreeDepthStencilState(depthStencilState);
    }
    gpuFree(device, heapAlloc);
    gpuFree(device, instAlloc);
    gpuFree(device, vertexAlloc);
    gpuFree(device, pixelAlloc);
    gpuFree(device, indexAlloc);
}

MsdfFont* MsdfTextRenderer::loadFont(const std::string& jsonPath, const std::string& pngPath)
{
    uint32_t slot = nextHeapSlot++;
    fonts.push_back(std::make_unique<MsdfFont>(device, jsonPath, pngPath, heapCpu, slot));
    return fonts.back().get();
}

void MsdfTextRenderer::clear()
{
    count = 0;
    batchFont = nullptr;
}

void MsdfTextRenderer::addText(const MsdfFont* font, std::string_view utf8, float x, float y,
                               float pixelSize, float4 color, float letterSpacing,
                               float4 outlineColor, float outlineWidthPx)
{
    if (!batchFont)
    {
        batchFont = font;
    }

    const float W = font->atlasSize().x;
    const float H = font->atlasSize().y;

    float penX = x;
    float baselineY = y;

    size_t i = 0;
    while (i < utf8.size())
    {
        uint32_t cp = decodeUtf8(utf8, i);
        if (cp == '\n')
        {
            penX = x;
            baselineY += font->lineHeight(pixelSize);
            continue;
        }
        const MsdfFont::Glyph* g = font->glyph(cp);
        if (!g)
        {
            const MsdfFont::Glyph* space = font->glyph(0x20);
            if (space)
            {
                penX += space->advance * pixelSize + letterSpacing;
            }
            continue;
        }

        if (g->hasGeometry && count < maxGlyphs)
        {
            MsdfGlyphInstance inst{};
            // plane bounds are baseline-relative, +y up; screen +y is down.
            inst.posMin = { penX + g->planeLeft * pixelSize, baselineY - g->planeTop * pixelSize };
            inst.posMax = { penX + g->planeRight * pixelSize, baselineY - g->planeBottom * pixelSize };
            // atlas bounds are texels with +y up (yOrigin "bottom"); flip V for a
            // top-left image origin.
            inst.uvMin = { g->atlasLeft / W, (H - g->atlasTop) / H };
            inst.uvMax = { g->atlasRight / W, (H - g->atlasBottom) / H };
            inst.color = color;
            inst.outlineColor = outlineColor;
            inst.outlineWidth = outlineWidthPx;
            instancesCpu[count++] = inst;
        }

        penX += g->advance * pixelSize + letterSpacing;
    }
}

void MsdfTextRenderer::upload()
{
    vertexDataCpu->glyphs = instancesGpu;
    if (batchFont)
    {
        pixelDataCpu->atlas = batchFont->atlasIndex();
        pixelDataCpu->distanceRange = batchFont->distanceRange();
        pixelDataCpu->atlasSize = batchFont->atlasSize();
    }
}

void MsdfTextRenderer::render(GpuCommandBuffer cmd, GpuTexture target, uint32_t targetW, uint32_t targetH,
                              LOAD_OP loadOp)
{
    vertexDataCpu->targetSize = { static_cast<float>(targetW), static_cast<float>(targetH) };

    GpuTexture targets[] = { target };
    GpuRenderPassDesc renderPassDesc{
        .colorTargets = Span<GpuTexture>(targets, 1),
        .loadOp = loadOp
    };

    gpuSetPipeline(cmd, pipeline);
    gpuSetDepthStencilState(cmd, depthStencilState);
    gpuSetActiveTextureHeapPtr(cmd, heapGpu);
    gpuBeginRenderPass(cmd, renderPassDesc);
    if (count > 0)
    {
        gpuDrawIndexedInstanced(cmd, vertexDataGpu, pixelDataGpu, indexDataGpu, 6, count);
    }
    gpuEndRenderPass(cmd);
}
