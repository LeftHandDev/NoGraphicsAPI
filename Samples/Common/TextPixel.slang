#include "Text.h"

struct PixelIn
{
    float4 position : SV_Position;
    float2 uv;
};

float4 main(PixelIn inPixel, TextVertexData* _, TextPixelData* data)
{
    float4 color = textureHeap[data->atlas].SampleLevel(samplerHeap[0], inPixel.uv, 0);
    if (color.w == 0.0)
    {
        discard;
    }
    return color;
}