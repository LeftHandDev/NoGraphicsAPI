#ifndef SAMPLES_SHADER_RAYTRACING_H
#define SAMPLES_SHADER_RAYTRACING_H

#include "../../Shaders/NoGraphicsAPI.h"

void raytracingSample();

struct alignas(16) PrimitiveData
{
    uint32_t* indices;
    float3* vertices;
    float2* uvs;
    float3* normals;
    uint texture;
    uint padding[3];
};

struct alignas(16) MeshData
{
    PrimitiveData *primitives;
};

struct alignas(16) LightData
{
    float4 position;
    float4 color;
    float intensity;
    float3 padding;
};

struct alignas(16) CameraData
{
    float4x4 invViewProjection;
    float4 position;
};

struct alignas(16) Reservoir
{
    int path; // light index
    float w;
    int M;
    int padding;
};

struct alignas(16) RaytracingData
{
    CameraData* camData;
    AccelerationStructure tlas;
    uint32_t* instanceToMesh;
    MeshData* meshes;
    LightData* lights;
    Reservoir* reservoirs;
    Reservoir* history;
    uint frame;
    uint accumulate; // 0 reset, 1 accumulate
    uint accumulatedFrames;
    uint numLights;
    uint dstTexture;
    uint M; // RIS
};

#ifdef __cplusplus
#else
uint xxhash32(uint3 p)
{
    const uint PRIME32_2 = 2246822519U;
    const uint PRIME32_3 = 3266489917U;
    const uint PRIME32_4 = 668265263U;
    const uint PRIME32_5 = 374761393U;

    uint h32 = p.z + PRIME32_5 + p.x * PRIME32_3;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> 15));
    h32 += p.y * PRIME32_3;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> 15));
    h32 = PRIME32_2 * (h32 ^ (h32 >> 15));
    h32 = PRIME32_3 * (h32 ^ (h32 >> 13));
    return h32 ^ (h32 >> 16);
}

uint pcg(inout uint state)
{
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float randomFloat(inout uint state)
{
    return float(pcg(state)) / 4294967295.0;
}

float shadowRay(RaytracingAccelerationStructure tlas, LightData light, float3 x1, float3 x2)
{
    RayQuery<RAY_FLAG_NONE> shadowRay;
    RayDesc ray;
    ray.Origin = x1;
    ray.Direction = normalize(x2 - x1);
    ray.TMin = 0.001;
    ray.TMax = length(x2 - x1);
    shadowRay.TraceRayInline(tlas, RAY_FLAG_NONE, 0xFF, ray);

    while (shadowRay.Proceed())
    {
        // Handle candidate hits if needed (e.g., alpha testing)
    }

    if (shadowRay.CommittedStatus() == COMMITTED_NOTHING)
    {
        return 1.0;
    }
    else
    {
        return 0.0;
    }
}
#endif

#endif // SHADERS_RAYTRACING_H