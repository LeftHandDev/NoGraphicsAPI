#include "Text.h"

STATIC_SAMPLER(linearWrap, LINEAR, LINEAR, NEAREST, WRAP, WRAP, WRAP, 1, TRANSPARENT_BLACK, COMPARE_NONE, 0.0, 0.0, LOD_UNBOUNDED)

struct PixelIn
{
    float4 position : SV_Position;
    float2 uv;
};

float4 main(PixelIn inPixel, TextVertexData* _, TextPixelData* data)
{
    float4 color = linearWrap().SampleLevel(textureHeap[data->atlas], inPixel.uv, 0);
    if (color.w == 0.0)
    {
        discard;
    }
    return color;
}