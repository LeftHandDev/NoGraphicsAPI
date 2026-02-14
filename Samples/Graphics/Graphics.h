#ifndef SAMPLES_SHADER_GRAPHICS_H
#define SAMPLES_SHADER_GRAPHICS_H

#include "../../NoGraphicsAPI.h"

void graphicsSample();

struct alignas(16) Instance
{
    float4x4 model;
    float4x4 prevModel;
};

struct alignas(16) VertexData
{
    float4x4 viewProjection;
    float4x4 viewProjectionNj;
    float4x4 prevViewProjectionNj;
    float3* vertices;
    float2* uvs;
    Instance* instances;
};

struct alignas(16) PixelData
{
    uint srcTexture;
};

#endif // SHADERS_GRAPHICS_H