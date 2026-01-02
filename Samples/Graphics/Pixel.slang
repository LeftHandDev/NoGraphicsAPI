#include "Graphics.h"

struct PixelIn
{
    float4 position : SV_Position;
    float2 uv;
};

struct PixelOut
{
    float4 color : SV_Target;
};

PixelOut main(PixelIn inPixel, VertexData* _, PixelData* data)
{
    PixelOut outPixel;
    outPixel.color = textureHeap[data->srcTexture].SampleLevel(samplerHeap[0], inPixel.uv, 0);
    return outPixel;
}