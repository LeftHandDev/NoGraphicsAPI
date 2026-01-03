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

    float4 color = float4(0.0, 0.0, 0.0, 1.0); // default: miss (black)

    while (rayQuery.Proceed())
    {
        // Handle candidate hits if needed (e.g., alpha testing)
    }

    if (rayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        float2 barycentrics = rayQuery.CommittedTriangleBarycentrics();
        color = float4(barycentrics, 1.0 - barycentrics.x - barycentrics.y, 1.0);
    }

    outputTexture[threadId.xy] = color;
}