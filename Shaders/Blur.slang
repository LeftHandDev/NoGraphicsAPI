#include "Blur.h"

[numthreads(16, 16, 1)]
void main(uint3 threadId: SV_DispatchThreadID, BlurData* data)
{
    int2 imgSize = int2(data->imageSize.x, data->imageSize.y);
    int2 pixelCoord = int2(threadId.x, threadId.y);

    if (pixelCoord.x >= imgSize.x || pixelCoord.y >= imgSize.y)
        return;


    Sampler2D<float4> srcTexture = textureHeap[data->srcTexture];
    RWTexture2D<float4> dstTexture = rwTextureHeap[data->dstTexture];

    float4 color = float4(0.0, 0.0, 0.0, 0.0);
    const int kernelRadius = 16;

    for (int ky = -kernelRadius; ky <= kernelRadius; ky++)
    {
        for (int kx = -kernelRadius; kx <= kernelRadius; kx++)
        {
            int2 sampleCoord = pixelCoord + int2(kx, ky);
            sampleCoord = clamp(sampleCoord, int2(0, 0), imgSize - int2(1, 1));
            float2 uv = (float2(sampleCoord.x, sampleCoord.y) + 0.5) / float2(imgSize.x, imgSize.y);
            color += srcTexture.SampleLevel(uv, 0);
        }
    }
    
    float2 uv = (float2(pixelCoord.x, pixelCoord.y) + 0.5) / float2(imgSize.x, imgSize.y);

    dstTexture[pixelCoord] = color / pow(kernelRadius * 2 + 1, 2);
}