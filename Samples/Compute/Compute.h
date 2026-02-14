#ifndef SAMPLES_SHADER_COMPUTE_H
#define SAMPLES_SHADER_COMPUTE_H

#include "../../NoGraphicsAPI.h"

void computeSample();

struct alignas(16) ComputeData
{
    uint2 imageSize;
    uint srcTexture;
    uint dstTexture;
};

#endif // SAMPLES_SHADER_COMPUTE_H