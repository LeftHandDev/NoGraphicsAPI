#ifndef NO_GRAPHICS_API_H
#define NO_GRAPHICS_API_H

#ifdef __cplusplus

#include "NoGraphicsAPI_Impl.h"

#define AccelerationStructure void*

#else
#define alignas(x) // do nothing in shader

#define AccelerationStructure uint64_t

[[vk::binding(0, 0)]]
Texture2D<float4> textureHeap[];

[[vk::binding(0, 1)]]
RWTexture2D<float4> rwTextureHeap[];

#endif

// Software samplers: plain-data sampler state with shader-code filtering.
#include "Sampler.h"

#endif // NO_GRAPHICS_API_H