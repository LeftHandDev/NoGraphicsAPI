#ifndef NO_GRAPHICS_API_COMMON_H
#define NO_GRAPHICS_API_COMMON_H

#ifdef __cplusplus

#include "../NoGraphicsAPI.h"

#define AccelerationStructure void*

#else
#define alignas(x) // do nothing in shader

#define AccelerationStructure uint64_t

[[vk::binding(0, 0)]]
Texture2D<float4> textureHeap[];

[[vk::binding(0, 1)]]
RWTexture2D<float4> rwTextureHeap[];

[[vk::binding(0, 2)]]
SamplerState samplerHeap[];

#endif

#endif // NO_GRAPHICS_API_COMMON_H