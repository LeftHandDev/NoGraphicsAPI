#include "Raytracing.h"

[numthreads(8, 8, 1)]
void RIS(uint3 threadId: SV_DispatchThreadID, RaytracingData *data)
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

    float3 debug = float3(0.0, 0.0, 0.0);

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
        float3 albedo = tex.SampleLevel(sampler, hitUV, 0).xyz;

        float3 normal0 = primitive.normals[index0];
        float3 normal1 = primitive.normals[index1];
        float3 normal2 = primitive.normals[index2];

        float3 normal = normalize(normal0 * (1.0 - barycentrics.x - barycentrics.y) + normal1 * barycentrics.x + normal2 * barycentrics.y);
        normal = normalize(mul(rayQuery.CommittedWorldToObject4x3(), normal).xyz);

        uint seed = xxhash32(uint3(threadId.xy, data->frame));
        float3 hitT = rayQuery.WorldRayDirection() * rayQuery.CommittedRayT() + rayQuery.WorldRayOrigin();

        float4 prevClip = mul(data->camData->prevViewProjection, float4(hitT, 1.0));
        float2 prevUV = prevClip.xy / prevClip.w * 0.5 + 0.5;
        int2 prevPixel = int2(prevUV * float2(imgSize));

        RWTexture2D<float4> albedoTexture = rwTextureHeap[data->albedo];
        albedoTexture[threadId.xy] = float4(albedo, 1.0);

        RWTexture2D<float4> normalsTexture = rwTextureHeap[data->normals];
        normalsTexture[threadId.xy] = float4(normal, 1.0);

        RWTexture2D<float4> motionVectorsTexture = rwTextureHeap[data->motionVectors];
        motionVectorsTexture[threadId.xy] = float4(threadId.xy - prevPixel, 0.0, 0.0);

        // RIS reservoir
        Reservoir r;

        // MIS weight
        float m = 1.0 / data->M;

        // RIS
        for (int i = 0; i < data->M; ++i)
        {
            uint lightIndex = pcg(seed) % data->numLights;
            LightData light = data->lights[lightIndex];

            Path x;
            x.x1 = hitT + normal * 0.001; // Offset ray origin to avoid self-intersection
            x.x2 = light.position.xyz;
            x.light = lightIndex;

            float w = m * p(x, albedo, normal, data->lights) / pdf(data->numLights);

            r.addSample(x, w, seed);
        }

        uint q = imgSize.x * threadId.y + threadId.x;
        Sample s;

        Path y = r.sampleOut;
        if (y.light != -1)
        {
            float w = 0.0;
            float p_hat = p(y, albedo, normal, data->lights);
            if (V(y, tlas) > 0.0 && p_hat > 0.0)
            {
                w = r.w / p_hat;
            }

            s.x = y;
            s.w = w;

            // debug = albedo * BRDF() * Le(y, data->lights) * G(y, normal) * w;
        }

        data->pixelSample[q] = s;
    }
    else
    {
        // No hit, write default values
        RWTexture2D<float4> albedoTexture = rwTextureHeap[data->albedo];
        albedoTexture[threadId.xy] = float4(0.0, 0.0, 0.0, 0.0);

        RWTexture2D<float4> normalsTexture = rwTextureHeap[data->normals];
        normalsTexture[threadId.xy] = float4(0.0, 0.0, 0.0, 0.0);

        RWTexture2D<float4> motionVectorsTexture = rwTextureHeap[data->motionVectors];
        motionVectorsTexture[threadId.xy] = float4(0.0, 0.0, 0.0, 0.0);

        Sample s;
        uint q = imgSize.x * threadId.y + threadId.x;
        data->pixelSample[q] = s;
    }

    // if (data->accumulate == 1)
    // {
    //     float3 prevColor = outputTexture[threadId.xy].xyz;
    //     debug = (prevColor * data->accumulatedFrames + debug) / (data->accumulatedFrames + 1);
    // }

    // outputTexture[threadId.xy] = float4(debug, 1.0);
}

[numthreads(8, 8, 1)]
void Reuse(uint3 threadId: SV_DispatchThreadID, RaytracingData *data)
{
    RWTexture2D<float4> outputTexture = rwTextureHeap[data->dstTexture];

    uint2 imgSize;
    outputTexture.GetDimensions(imgSize.x, imgSize.y);

    if (threadId.x >= imgSize.x || threadId.y >= imgSize.y)
        return;

    // Add output texture size to the threadId so we don't reuse the same random numbers from the RIS path
    uint seed = xxhash32(uint3(threadId.xy + imgSize, data->frame));

    float3 albedo = rwTextureHeap[data->albedo][threadId.xy].xyz;
    float3 normal = rwTextureHeap[data->normals][threadId.xy].xyz;
    int2 motionVector = int4(rwTextureHeap[data->motionVectors][threadId.xy]).xy;
    int2 prevThreadId = int2(threadId.xy) - motionVector;
    bool valid = prevThreadId.x >= 0 && prevThreadId.x < imgSize.x && prevThreadId.y >= 0 && prevThreadId.y < imgSize.y;

    if (length(albedo) == 0.0 || length(normal) == 0.0)
    {
        // No hit in RIS pass, skip reuse
        return;
    }

    // Spatial
    if (data->spatial == 1)
    {
        Reservoir r;

        uint q = imgSize.x * threadId.y + threadId.x;
        const uint k = 8;
        float sum = 0.0;
        Sample neighbors[k];

        float m = 1.0 / k;

        for (int i = 0; i < k - 1; ++i)
        {
            uint2 neighbor = randomPixelInNeighborhood(threadId.xy, k, imgSize, seed);
            uint Q = imgSize.x * neighbor.y + neighbor.x;
            Sample s = data->pixelSample[Q];
            
            // if (s.x.light != -1)
            // {
            //     sum += p(s.x, albedo, normal, data->lights);
            // }

            neighbors[i] = s;
        }

        Sample cur = data->pixelSample[q];
        float w = m * p(cur.x, albedo, normal, data->lights) * cur.w;
        r.addSample(cur.x, w, seed);

        for (int i = 0; i < k - 1; ++i)
        {
            Sample s = neighbors[i];
            float w = m * p(s.x, albedo, normal, data->lights) * s.w;
            r.addSample(s.x, w, seed);
        }

        Sample y;
        y.x = r.sampleOut;

        float p_hat = p(y.x, albedo, normal, data->lights);

        if (p_hat > 0.0)
        {
            y.w = r.w / p_hat;
            data->pixelSample[q] = y;
        }
    }

    // Temporal
    if (data->temporal == 1 && valid && data->frame > 0)
    {
        // Index into samples
        uint q = imgSize.x * threadId.y + threadId.x;
        uint Q = imgSize.x * prevThreadId.y + prevThreadId.x;

        Reservoir r;
        float m;
        float w;

        Sample cur = data->pixelSample[q];
        Sample prev = data->prevPixelSample[Q];

        float pCur = p(cur.x, albedo, normal, data->lights);
        float pPrev = p(prev.x, albedo, normal, data->lights);

        // MIS denominator
        float d = (data->M * (data->frame + 1));

        m = pCur / d;
        w = m * pCur * cur.w;
        r.addSample(cur.x, cur.w / d, seed);

        m = pPrev / d;
        w = m * pPrev * prev.w;
        r.addSample(prev.x, prev.w / d, seed);

        Sample y;
        y.x = r.sampleOut;
        y.w = r.w / p(y.x, albedo, normal, data->lights);

        data->pixelSample[q] = y;
    }
}

[numthreads(8, 8, 1)]
void Shade(uint3 threadId: SV_DispatchThreadID, RaytracingData *data)
{
    RWTexture2D<float4> outputTexture = rwTextureHeap[data->dstTexture];

    uint2 imgSize;
    outputTexture.GetDimensions(imgSize.x, imgSize.y);

    if (threadId.x >= imgSize.x || threadId.y >= imgSize.y)
        return;

    float3 albedo = rwTextureHeap[data->albedo][threadId.xy].xyz;
    float3 normal = rwTextureHeap[data->normals][threadId.xy].xyz;

    uint q = imgSize.x * threadId.y + threadId.x;
    Sample y = data->pixelSample[q];

    float4 color;

    if (y.x.light == -1)
    {
        color = float4(0.f, 0.f, 0.f, 1.f);
    }
    else 
    {
        color = float4((albedo * BRDF() * Le(y.x, data->lights) * G(y.x, normal)) * y.w, 1.0);
    }

    if (data->accumulate == 1)
    {
        float4 prevColor = outputTexture[threadId.xy];
        color = (prevColor * data->accumulatedFrames + color) / (data->accumulatedFrames + 1);
    }

    if (isnan(length(color)))
    {
        color = float4(0.f, 0.f, 1.f, 1.f);
    }

    outputTexture[threadId.xy] = color;
}