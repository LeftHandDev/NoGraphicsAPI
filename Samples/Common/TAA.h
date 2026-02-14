#ifndef SAMPLES_SHADER_TAA_H
#define SAMPLES_SHADER_TAA_H

#include "../../Shaders/NoGraphicsAPI.h"

struct alignas(16) TAAData
{
    uint width;
    uint height;
    uint frame;
    uint srcColor;
    uint srcHistory;
    uint srcDepth; // Currently unused
    uint srcMotionVectors;
    uint dstTexture;
    float2 jitter; // Currently unused
};

#endif // SAMPLES_SHADER_TAA_H