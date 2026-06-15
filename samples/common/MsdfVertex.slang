#include "MsdfShared.h"

// One instanced quad per glyph. vertexId 0=top-left, 1=top-right, 2=bottom-right,
// 3=bottom-left (drawn with indices {0,1,2, 2,3,0}).
static float2 quadCorners[4] = {
    float2(0.0, 0.0),
    float2(1.0, 0.0),
    float2(1.0, 1.0),
    float2(0.0, 1.0)
};

struct VertexOut
{
    float4 position : SV_Position;
    float2 uv;
    float4 color;
    float4 outlineColor;
    float outlineWidth;
};

VertexOut main(uint vertexId : SV_VertexID, uint instanceId : SV_InstanceID, MsdfVertexData* data, MsdfPixelData* _)
{
    MsdfGlyphInstance glyph = data->glyphs[instanceId];
    float2 t = quadCorners[vertexId];

    // Pixel (0,0) is the top-left of the target; +y goes down. Vulkan clip space
    // also has +y down, so map straight to NDC with no flip.
    float2 posPx = lerp(glyph.posMin, glyph.posMax, t);
    float2 ndc = posPx / data->targetSize * 2.0 - 1.0;

    VertexOut output;
    output.position = float4(ndc, 0.0, 1.0);
    output.uv = lerp(glyph.uvMin, glyph.uvMax, t);
    output.color = glyph.color;
    output.outlineColor = glyph.outlineColor;
    output.outlineWidth = glyph.outlineWidth;
    return output;
}
