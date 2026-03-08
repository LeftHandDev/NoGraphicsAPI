#ifndef SAMPLES_SHADER_RAYTRACING_H
#define SAMPLES_SHADER_RAYTRACING_H

#include "../../NoGraphicsAPI.h"
#include "Random.h"

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
    float4x4 prevViewProjection;
    float4 position;
};

struct alignas(16) Path
{
    float3 x1 = float3(0.0, 0.0, 0.0); // x0 = origin (omitted), x1 = hit position, x2 = light position
    uint padding;
    float3 x2 = float3(0.0, 0.0, 0.0);
    int light = -1;
};

struct alignas(16) Sample
{
    Path x;
    float w = 0;
    int padding[3];
};

struct alignas(16) Reservoir
{
    Path sampleOut;
    float w = 0;
    int padding[3];

#ifndef __cplusplus
    [mutating] void addSample(Path x, float W, inout uint state)
    {
        w += W;
        if (randomFloat(state) < (W / w))
        {
            sampleOut = x;
        }
    }
#endif
};

struct alignas(16) RaytracingData
{
    CameraData* camData;
    AccelerationStructure tlas;
    uint32_t* instanceToMesh;
    MeshData* meshes;
    LightData* lights;
    Sample* pixelSample;
    Sample* prevPixelSample;
    uint frame;
    uint accumulate; // 0 reset, 1 accumulate
    uint accumulatedFrames;
    uint numLights;
    uint albedo;
    uint normals;
    uint motionVectors; // not actual motion vectors, but a map to the history texture
    uint dstTexture;
    uint M; // RIS
    uint spatial = 0;
    uint temporal = 0;
};

#ifdef __cplusplus
#else

float V(Path x, RaytracingAccelerationStructure tlas)
{
    RayQuery<RAY_FLAG_NONE> shadowRay;
    RayDesc ray;
    ray.Origin = x.x1;
    ray.Direction = normalize(x.x2 - x.x1);
    ray.TMin = 0.001;
    ray.TMax = distance(x.x2, x.x1);
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

float BRDF()
{
    return 1.0 / 3.14159265;
}

float3 Le(Path x, LightData* lights)
{
    if (x.light == -1)
    {
        return float3(0.0, 0.0, 0.0);
    }
    
    LightData light = lights[x.light];
    return light.color.xyz * light.intensity;
}

float G(Path x, float3 normal)
{
    float dist = distance(x.x1, x.x2);
    if (dist < 1e-6)
    {
        return 0.0;
    }
    return max(dot(normal, normalize(x.x2 - x.x1)), 0.0) / (dist * dist);
}

float p(Path x, float3 albedo, float3 normal, LightData* lights)
{ 
    return length(albedo * BRDF() * Le(x, lights) * G(x, normal));
}

float pdf(int numLights)
{
    return 1.0 / float(numLights);
}

float W(int numLights)
{
    return float(numLights);
}

uint2 randomPixelInNeighborhood(uint2 pixel, uint radius, uint2 imgSize, inout uint state)
{
    int x = int(pixel.x) + int(randomInt(state) % (2 * radius + 1)) - int(radius);
    int y = int(pixel.y) + int(randomInt(state) % (2 * radius + 1)) - int(radius);
    x = clamp(x, 0, int(imgSize.x) - 1);
    y = clamp(y, 0, int(imgSize.y) - 1);
    return uint2(x, y);
}

#endif

#endif // SHADERS_RAYTRACING_H