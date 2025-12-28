#ifndef SHADERS_COMMON_H
#define SHADERS_COMMON_H

#ifdef __cplusplus

#include "NoGraphicsAPI.h"

#else
#define alignas(x) // do nothing in shader

const Sampler2D<float4> textureHeap[];
const RWTexture2D<float4> rwTextureHeap[];

#endif

#endif // SHADERS_COMMON_H