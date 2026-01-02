#ifndef SHADERS_GRAPHICS_H
#define SHADERS_GRAPHICS_H

#include "../../Shaders/common.h"

void graphicsSample();

struct Vertex
{
    float4 vertex;
    float2 uv;
};

struct alignas(16) Instance
{
    float4x4 model;
};

struct alignas(16) VertexData
{
    float4x4 viewProjection;
    Vertex* vertices;
    Instance* instances;
};

struct alignas(16) PixelData
{
    uint srcTexture;
};

#endif // SHADERS_GRAPHICS_H