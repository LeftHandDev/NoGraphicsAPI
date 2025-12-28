#ifndef SHADERS_COMPUTE_H
#define SHADERS_COMPUTE_H

#include "Common.h"

struct alignas(16) Data
{
    float4 color;
    uint offset;
    const uint* input;
    uint* output;
};

#endif // SHADERS_COMPUTE_H