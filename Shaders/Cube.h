#ifndef SHADERS_CUBE_H
#define SHADERS_CUBE_H

#include "Common.h"

struct Vertex
{
    float4 position;
};

struct alignas(16) VertexData
{
    float4x4 mvp;
    Vertex* vertices;
};

struct alignas(16) PixelData
{
    float4 color;
};

#endif // SHADERS_CUBE_H