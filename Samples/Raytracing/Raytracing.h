#ifndef SAMPLES_SHADER_RAYTRACING_H
#define SAMPLES_SHADER_RAYTRACING_H
    
#include "../../Shaders/NoGraphicsAPI.h"

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
    uint padding;
};

struct alignas(16) LightData
{
    float4 position;
    float4 color;
    float intensity;
    float3 padding;
};

struct alignas(16) RaytracingData
{
    float4x4 invViewProjection;
    float4 cameraPosition;
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