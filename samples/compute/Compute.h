#ifndef SAMPLES_SHADER_COMPUTE_H
#define SAMPLES_SHADER_COMPUTE_H

#include "NoGraphicsAPI.h"

// A plain data struct kept in its own GPU buffer. ComputeData below holds a
// pointer to an array of these; the shader follows that pointer to read them.
struct alignas(16) Tint
{
    float3 color;
    float strength;
};

struct alignas(16) ComputeData
{
    uint2 imageSize;
    uint srcTexture;
    uint dstTexture;

    // A regular buffer attached by pointer: the shader dereferences `tints`
    // directly (a GPU address), no descriptor binding involved.
    Tint* tints;
    uint tintCount;
};

#endif // SAMPLES_SHADER_COMPUTE_H