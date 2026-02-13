#ifndef SAMPLES_SHADER_GRAPHICS_H
#define SAMPLES_SHADER_GRAPHICS_H

#include "../../Shaders/NoGraphicsAPI.h"

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

struct alignas(16) TAAData
{
    uint width;
    uint height;
    uint frame;
    uint srcColor;
    uint srcHistory;
    uint srcDepth;
    uint srcMotionVectors;
    uint dstTexture;
    float2 jitter;
};

#endif // SHADERS_GRAPHICS_H