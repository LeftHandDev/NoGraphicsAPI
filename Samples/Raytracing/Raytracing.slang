#include "Raytracing.h"

[numthreads(8, 8, 1)]
void main(uint3 threadId: SV_DispatchThreadID, RaytracingData *data)
{
    RWTexture2D<float4> outputTexture = rwTextureHeap[data->dstTexture];

    uint2 imgSize;
    outputTexture.GetDimensions(imgSize.x, imgSize.y);

    if (threadId.x >= imgSize.x || threadId.y >= imgSize.y)
        return;

    float2 uv = (float2(threadId.xy) + 0.5) / float2(imgSize);

    RaytracingAccelerationStructure tlas = RaytracingAccelerationStructure(data->tlas);

    float2 ndc = uv * 2.0 - 1.0;
    // ndc.y = -ndc.y;

    float4 clip = float4(ndc, 0.0, 1.0);
    float4 world = mul(data->invViewProjection, clip);
    world.xyz /= world.w;

    RayDesc ray;
    ray.Origin = data->cameraPosition.xyz;
    ray.Direction = normalize(world.xyz - ray.Origin);
    ray.TMin = 0.001;
    ray.TMax = 10000.0;

    RayQuery<RAY_FLAG_NONE> rayQuery;
    rayQuery.TraceRayInline(tlas, RAY_FLAG_NONE, 0xFF, ray);

    float4 color = float4(0.0, 0.0, 0.0, 1.0);

    // while (rayQuery.Proceed())
    // {
    // }

    if (rayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        MeshData mesh = data->meshes[rayQuery.CommittedInstanceID()];
        float2 barycentrics = rayQuery.CommittedTriangleBarycentrics();
        uint primitiveIndex = rayQuery.CommittedPrimitiveIndex();

        uint uvIndex0 = mesh.uvIndices[primitiveIndex * 3 + 0];
        uint uvIndex1 = mesh.uvIndices[primitiveIndex * 3 + 1];
        uint uvIndex2 = mesh.uvIndices[primitiveIndex * 3 + 2];

        float2 uv0 = mesh.uvs[uvIndex0];
        float2 uv1 = mesh.uvs[uvIndex1];
        float2 uv2 = mesh.uvs[uvIndex2];

        float2 hitUV = uv0 * (1.0 - barycentrics.x - barycentrics.y) + uv1 * barycentrics.x + uv2 * barycentrics.y;

        Texture2D<float4> tex = textureHeap[NonUniformResourceIndex(mesh.texture)];
        SamplerState sampler = samplerHeap[0];
        color = tex.SampleLevel(sampler, hitUV, 0);
    }

    outputTexture[threadId.xy] = color;
}