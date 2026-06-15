#include "MsdfShared.h"

// MTSDF text. The atlas stores, per texel, the signed distances to the nearest
// glyph edges spread across R/G/B (multi-channel SDF); the median of the three
// reconstructs a distance that keeps sharp corners that a single channel rounds
// off. The alpha channel holds a plain true-distance SDF (the "T" in MTSDF),
// available for effects; we key antialiasing off the median here.
//
// Linear filtering is mandatory (bilinear reconstruction is the antialiasing);
// clamp addressing avoids bleeding at the atlas edges. No mips — sample LOD 0.
STATIC_SAMPLER(msdfSampler, LINEAR, LINEAR, LINEAR, CLAMP, CLAMP, CLAMP, 1, TRANSPARENT_BLACK, COMPARE_NONE, 0.0, 0.0, LOD_UNBOUNDED)

struct PixelIn
{
    float4 position : SV_Position;
    float2 uv;
    float4 color;
    float4 outlineColor;
    float outlineWidth;
};

float median(float3 c)
{
    return max(min(c.r, c.g), min(max(c.r, c.g), c.b));
}

float4 main(PixelIn input, MsdfVertexData* _, MsdfPixelData* data)
{
    float4 texel = msdfSampler().Sample(textureHeap[data->atlas], input.uv);
    float signedDistance = median(texel.rgb);

    // Convert the atlas-texel distance range into screen pixels using screen-space
    // derivatives, so the edge stays ~1px-soft at any scale/rotation (Chlumsky).
    // The 1.0 floor keeps thin strokes from vanishing when text gets tiny.
    float2 unitRange = float2(data->distanceRange, data->distanceRange) / data->atlasSize;
    float2 screenTexSize = float2(1.0, 1.0) / fwidth(input.uv);
    float screenPxRange = max(0.5 * dot(unitRange, screenTexSize), 1.0);

    float screenPxDistance = screenPxRange * (signedDistance - 0.5);
    float fill = clamp(screenPxDistance + 0.5, 0.0, 1.0);

    if (input.outlineWidth > 0.0)
    {
        // Extend coverage outward by outlineWidth screen pixels for the outline,
        // then place the fill color on top of the outline color.
        float outer = clamp(screenPxDistance + 0.5 + input.outlineWidth, 0.0, 1.0);
        if (outer <= 0.0)
        {
            discard;
        }
        float3 rgb = lerp(input.outlineColor.rgb, input.color.rgb, fill);
        return float4(rgb, outer * input.color.a);
    }

    if (fill <= 0.0)
    {
        discard;
    }
    return float4(input.color.rgb, input.color.a * fill);
}
