#include "Raytracing.h"

void addSample(Reservoir *res, uint resId, uint path, float w, int M, inout uint state)
{
    res[resId].w += w;
    res[resId].M += M;
    if (randomFloat(state) < (w / res[resId].w))
    {
        res[resId].path = path;
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
    float4 world = mul(data->camData->invViewProjection, clip);
    world.xyz /= world.w;

    RayDesc ray;
    ray.Origin = data->camData->position.xyz;
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
        PrimitiveData primitive = mesh.primitives[rayQuery.CommittedGeometryIndex()];
        float2 barycentrics = rayQuery.CommittedTriangleBarycentrics();
        uint primitiveIndex = rayQuery.CommittedPrimitiveIndex();

        uint index0 = primitive.indices[primitiveIndex * 3 + 0];
        uint index1 = primitive.indices[primitiveIndex * 3 + 1];
        uint index2 = primitive.indices[primitiveIndex * 3 + 2];

        float2 uv0 = primitive.uvs[index0];
        float2 uv1 = primitive.uvs[index1];
        float2 uv2 = primitive.uvs[index2];

        float2 hitUV = uv0 * (1.0 - barycentrics.x - barycentrics.y) + uv1 * barycentrics.x + uv2 * barycentrics.y;

        Texture2D<float4> tex = textureHeap[NonUniformResourceIndex(primitive.texture)];
        SamplerState sampler = samplerHeap[0];
        float4 albedo = tex.SampleLevel(sampler, hitUV, 0);

        float3 normal0 = primitive.normals[index0];
        float3 normal1 = primitive.normals[index1];
        float3 normal2 = primitive.normals[index2];

        float3 normal = normalize(normal0 * (1.0 - barycentrics.x - barycentrics.y) + normal1 * barycentrics.x + normal2 * barycentrics.y);
        normal = normalize(mul(rayQuery.CommittedWorldToObject4x3(), normal)).xyz;

        // Skip lighting if no lights
        if (data->numLights == 0)
        {
            color.xyz = albedo.xyz;
        }
        else
        {
            uint seed = xxhash32(uint3(threadId.xy, data->frame));
            uint resId = threadId.y * imgSize.x + threadId.x;
            float3 hitT = rayQuery.WorldRayDirection() * rayQuery.CommittedRayT() + rayQuery.WorldRayOrigin();

            float4 prevClip = mul(data->camData->prevViewProjection, float4(hitT, 1.0));
            float2 prevUV = prevClip.xy / prevClip.w * 0.5 + 0.5;
            int2 prevPixel = int2(prevUV * float2(imgSize));
            uint prevResId = prevPixel.y * imgSize.x + prevPixel.x;

            bool validHistory = prevPixel.x >= 0 && prevPixel.x < imgSize.x && prevPixel.y >= 0 && prevPixel.y < imgSize.y;

            // Reset reservoir for this frame
            data->reservoirs[resId].w = 0.0;
            data->reservoirs[resId].path = 0;
            data->reservoirs[resId].M = 0;

            // RIS
            for (int i = 0; i < data->M; ++i)
            {
                uint lightIndex = pcg(seed) % data->numLights;

                LightData light = data->lights[lightIndex];
                float3 lightDir = normalize(light.position.xyz - hitT);
                float NoL = max(dot(normal, lightDir), 0.0);
                float3 Li = (albedo.xyz * light.color.xyz * light.intensity * NoL) / pow(distance(light.position.xyz, hitT), 2.0);
                float pdf = 1.0 / float(data->numLights);
                float w = length(Li) / pdf;

                addSample(data->reservoirs, resId, lightIndex, w, 1, seed);
            }

            if (validHistory)
            {
                // Temporal pass
                {
                    uint lightIndex = data->reservoirs[resId].path;
                    LightData light = data->lights[lightIndex];
                    float3 lightDir = normalize(light.position.xyz - hitT);
                    float NoL = max(dot(normal, lightDir), 0.0);
                    float3 Li = (albedo.xyz * light.color.xyz * light.intensity * NoL) / pow(distance(light.position.xyz, hitT), 2.0);

                    float V = shadowRay(tlas, light, hitT + normal * 0.001, light.position.xyz);

                    int prevPath = data->history[prevResId].path;
                    float prevW = data->history[prevResId].w;
                    int prevM = data->history[prevResId].M;

                    const int historyLength = 20;
                    int prevMClamped = min(prevM, historyLength * data->M);
                    float bias = float(prevMClamped) / float(prevM + 1);

                    data->history[prevResId].w = 0.0;
                    data->history[prevResId].path = 0;
                    data->history[prevResId].M = 0;

                    addSample(data->history, prevResId, prevPath, prevW * bias, prevMClamped, seed);
                    addSample(data->history, prevResId, lightIndex, data->reservoirs[resId].w * V, data->reservoirs[resId].M, seed);
                }

                // Shade pass
                {
                    uint lightIndex = data->history[prevResId].path;
                    LightData light = data->lights[lightIndex];
                    float3 lightDir = normalize(light.position.xyz - hitT);
                    float NoL = max(dot(normal, lightDir), 0.0);
                    float3 Li = (albedo.xyz * light.color.xyz * light.intensity * NoL) / pow(distance(light.position.xyz, hitT), 2.0);
                    float W = (1.0 / float(data->history[prevResId].M)) * (1.0 / length(Li)) * data->history[prevResId].w;

                    color.xyz = Li * W;
                }
            }
            else // Invalid history, use current reservoir
            {
                uint lightIndex = data->reservoirs[resId].path;
                LightData light = data->lights[lightIndex];
                float3 lightDir = normalize(light.position.xyz - hitT);
                float NoL = max(dot(normal, lightDir), 0.0);
                float3 Li = (albedo.xyz * light.color.xyz * light.intensity * NoL) / pow(distance(light.position.xyz, hitT), 2.0);

                float V = shadowRay(tlas, light, hitT + normal * 0.001, light.position.xyz);

                float W = (1.0 / float(data->reservoirs[resId].M)) * (1.0 / length(Li)) * data->reservoirs[resId].w;

                color.xyz = Li * W;
            }
        }
    }

    if (data->accumulate == 1)
    {
        float4 prevColor = outputTexture[threadId.xy];
        color = (prevColor * data->accumulatedFrames + color) / (data->accumulatedFrames + 1);
    }

    outputTexture[threadId.xy] = color;
}
