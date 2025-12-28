#ifndef SHADERS_BLUR_H
#define SHADERS_BLUR_H

#include "Common.h"

struct alignas(16) BlurData
{
    uint2 imageSize;
    uint srcTexture;
    uint dstTexture;
};

#endif // SHADERS_BLUR_H