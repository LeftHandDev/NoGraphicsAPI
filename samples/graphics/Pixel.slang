#include "Graphics.h"

// The full sampler definition lives in the declaration: trilinear-ready
// filtering, wrap addressing and 16x anisotropy. Sampled below with
// Sample() (implicit derivatives) so the anisotropic footprint applies.
STATIC_SAMPLER(aniso16Wrap, LINEAR, LINEAR, LINEAR, WRAP, WRAP, WRAP, 16, TRANSPARENT_BLACK, COMPARE_NONE, 0.0, 0.0, LOD_UNBOUNDED)

struct PixelIn
{
    float4 position : SV_Position;
    float2 motionVectors;
    float2 uv;
};

struct PixelOut
{
    float4 color : SV_Target;
    float4 motionVectors : SV_Target1;
};

PixelOut main(PixelIn inPixel, VertexData* _, PixelData* data)
{
    PixelOut outPixel;
    outPixel.color = aniso16Wrap().Sample(textureHeap[data->srcTexture], inPixel.uv);
    outPixel.motionVectors = float4(inPixel.motionVectors, 0.0, 0.0);
    return outPixel;
}