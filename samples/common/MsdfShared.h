#ifndef SAMPLES_SHADER_MSDF_SHARED_H
#define SAMPLES_SHADER_MSDF_SHARED_H

#include "NoGraphicsAPI.h"

// GPU data shared verbatim between the C++ renderer and the MSDF shaders, so
// field order/offsets must match on both sides (mirrors samples/common/Text.h).
//
// One instanced quad is drawn per glyph. Positions are in render-target pixels
// with a top-left origin and +y pointing down (matching the pixel->NDC mapping
// in MsdfVertex.slang and Vulkan clip space). UVs are normalized atlas
// coordinates. Colors are straight (non-premultiplied) RGBA in [0, 1].
struct alignas(16) MsdfGlyphInstance
{
    float2 posMin;       // glyph quad top-left, target pixels
    float2 posMax;       // glyph quad bottom-right, target pixels
    float2 uvMin;        // atlas UV at posMin
    float2 uvMax;        // atlas UV at posMax
    float4 color;        // fill color, straight-alpha RGBA
    float4 outlineColor; // outline color, used when outlineWidth > 0
    float outlineWidth;  // outline half-width in screen pixels (0 = no outline)
    float _pad0;
    float _pad1;
    float _pad2;
};

struct alignas(16) MsdfVertexData
{
    float2 targetSize;          // render-target size in pixels (pixels -> NDC)
    float2 _pad;
    MsdfGlyphInstance* glyphs;  // device pointer to the instance array
};

struct alignas(16) MsdfPixelData
{
    uint atlas;          // texture-heap index of the MTSDF atlas
    float distanceRange; // msdf distance range, in atlas texels (the bake's -pxrange)
    float2 atlasSize;    // atlas dimensions in texels
};

#endif // SAMPLES_SHADER_MSDF_SHARED_H
