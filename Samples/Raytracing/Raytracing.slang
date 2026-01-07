#include "Raytracing.h"

// Visibility function
float V(RaytracingAccelerationStructure tlas, LightData light, float3 x0, float3 x1)
{
    RayQuery<RAY_FLAG_NONE> shadowRay;
    RayDesc ray;
    ray.Origin = x0;
    ray.Direction = normalize(x1 - x0);
    ray.TMin = 0.001;
    ray.TMax = length(x1 - x0);
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

    while (rayQuery.Proceed())
    {
        // Handle candidate hits if needed (e.g., alpha testing)
    }

    if (rayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        uint32_t meshId = data->instanceToMesh[rayQuery.CommittedInstanceID()];
        MeshData mesh = data->meshes[meshId];
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
        float4 albedo = tex.SampleLevel(sampler, hitUV, 0);

        for (uint i = 0; i < data->numLights; i++)
        {
            LightData light = data->lights[i];
            float3 hitT = rayQuery.WorldRayDirection() * rayQuery.CommittedRayT() + rayQuery.WorldRayOrigin();
            float3 lightDir = normalize(light.position.xyz - hitT);

            uint normalIndex0 = mesh.normalIndices[primitiveIndex * 3 + 0];
            uint normalIndex1 = mesh.normalIndices[primitiveIndex * 3 + 1];
            uint normalIndex2 = mesh.normalIndices[primitiveIndex * 3 + 2];

            float3 normal0 = mesh.normals[normalIndex0];
            float3 normal1 = mesh.normals[normalIndex1];
            float3 normal2 = mesh.normals[normalIndex2];

            float3 normal = normalize(normal0 * (1.0 - barycentrics.x - barycentrics.y) + normal1 * barycentrics.x + normal2 * barycentrics.y);
            normal = normalize(mul(rayQuery.CommittedWorldToObject4x3(), normal)).xyz;

            float NdotL = max(dot(normal, lightDir), 0.0);
            float3 Li = (albedo.xyz * light.color.xyz * light.intensity * NdotL) / pow(distance(light.position.xyz, hitT), 2.0);
            color.xyz += Li * V(tlas, light, hitT + normal * 0.001, light.position.xyz);
        }

        color.xyz /= float(data->numLights);
    }

    outputTexture[threadId.xy] = color;
}