#ifndef SHADERS_COMMON_H
#define SHADERS_COMMON_H

#ifdef __cplusplus

#include "NoGraphicsAPI.h"

#else
#define alignas(x) // do nothing in shader

[[vk::binding(0, 0)]]
Texture2D<float4> textureHeap[];

[[vk::binding(0, 1)]]
RWTexture2D<float4> rwTextureHeap[];

[[vk::binding(0, 2)]]
SamplerState samplerHeap[];

#endif

#endif // SHADERS_COMMON_H