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
    PrimitiveData* primitives;
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

struct alignas(16) RaytracingData
{
    CameraData* camData;
    AccelerationStructure tlas;
    uint32_t* instanceToMesh;
    MeshData* meshes;
    LightData* lights;
    uint frame;
    uint accumulate; // 0 reset, 1 accumulate
    uint accumulatedFrames;
    uint numLights;
    uint dstTexture;
};

#endif // SHADERS_RAYTRACING_H