#ifndef SHADERS_COMPUTE_H
#define SHADERS_COMPUTE_H

#include "../../Shaders/Common.h"

void computeSample();

struct alignas(16) ComputeData
{
    uint2 imageSize;
    uint srcTexture;
    uint dstTexture;
};

#endif // SHADERS_COMPUTE_H