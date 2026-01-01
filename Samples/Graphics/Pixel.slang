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
    outPixel.color = textureHeap[data->srcTexture].Sample(samplerHeap[0], inPixel.uv);
    return outPixel;
}