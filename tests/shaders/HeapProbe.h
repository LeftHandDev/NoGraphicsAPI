#ifndef TESTS_SHADERS_HEAP_PROBE_H
#define TESTS_SHADERS_HEAP_PROBE_H

#include "NoGraphicsAPI.h"

// Reads texel (0, 0) of textureHeap[index] into *result.
struct alignas(16) HeapProbeData
{
    float4* result;
    uint index;
};

#endif // TESTS_SHADERS_HEAP_PROBE_H
