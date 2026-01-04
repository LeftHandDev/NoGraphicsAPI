#ifndef SHADERS_RAYTRACING_H
#define SHADERS_RAYTRACING_H

#include "../../Shaders/common.h"

void raytracingSample();

struct alignas(16) MeshData
{
    uint32_t* indices;
    float4* vertices;
    float2* uvs;
    uint32_t* uvIndices;
    float3* normals;
    uint32_t* normalIndices;
    uint texture;
};

struct alignas(16) LightData
{
    float4 position;
    float4 color;
    float intensity;
};

struct alignas(16) RaytracingData
{
    float4x4 invViewProjection;
    float4 cameraPosition;
    AccelerationStructure tlas;
    MeshData* meshes;
    LightData* lights;
    uint numLights;
    uint dstTexture;
};

#endif // SHADERS_RAYTRACING_H