#include "TAA.h"

[numthreads(16, 16, 1)]
void main(uint3 threadId : SV_DispatchThreadID, TAAData* data)
{
    uint width = data->width;
    uint height = data->height;

    if (threadId.x >= width || threadId.y >= height)
        return;

    float2 uv = float2(threadId.x + 0.5, threadId.y + 0.5) / float2(width, height);
    float2 texelSize = 1.0 / float2(width, height);

    Texture2D<float4> colorTexture = textureHeap[data->srcColor];

    float4 color = colorTexture.SampleLevel(samplerHeap[0], uv, 0);
    float2 motionVectors = textureHeap[data->srcMotionVectors].SampleLevel(samplerHeap[0], uv, 0).xy;

    // Sample 3x3 neighborhood to find min/max bounds
    float4 neighborMin = color;
    float4 neighborMax = color;
    
    for (int y = -1; y <= 1; y++)
    {
        for (int x = -1; x <= 1; x++)
        {
            float4 neighbor = colorTexture.SampleLevel(samplerHeap[0], uv + float2(x, y) * texelSize, 0);
            neighborMin = min(neighborMin, neighbor);
            neighborMax = max(neighborMax, neighbor);
        }
    }

    // Reproject UVs using motion vectors (subtract because motion = current - prev, scale by 0.5 for NDC to UV)
    float2 reprojectedUV = uv - motionVectors * 0.5;
    float4 historyColor = textureHeap[data->srcHistory].SampleLevel(samplerHeap[0], reprojectedUV, 0);

    // Clamp history to neighborhood bounds
    historyColor = clamp(historyColor, neighborMin, neighborMax);

    bool validUV = reprojectedUV.x >= 0.0 && reprojectedUV.x <= 1.0 &&
                   reprojectedUV.y >= 0.0 && reprojectedUV.y <= 1.0;

    float blendFactor = 0.05;
    
    if (data->frame == 0 || !validUV)
    {
        blendFactor = 1.0; 
    }

    float4 finalColor = lerp(historyColor, color, blendFactor);

    RWTexture2D<float4> dstTexture = rwTextureHeap[data->dstTexture];
    dstTexture[threadId.xy] = finalColor;
}