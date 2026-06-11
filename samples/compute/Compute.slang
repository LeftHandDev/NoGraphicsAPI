#include "Compute.h"

[numthreads(16, 16, 1)] void main(uint3 threadId : SV_DispatchThreadID, ComputeData* data)
{
    int2 imgSize = int2(data->imageSize.x, data->imageSize.y);
    int2 pixelCoord = int2(threadId.x, threadId.y);

    if (pixelCoord.x >= imgSize.x || pixelCoord.y >= imgSize.y)
        return;

    Texture2D<float4> srcTexture = textureHeap[data->srcTexture];
    RWTexture2D<float4> dstTexture = rwTextureHeap[data->dstTexture];

    // The sampler is declared inline, right where it is used — no API object,
    // no descriptor management; it still runs on a real hardware sampler.
    HwSampler sampler = INLINE_SAMPLER(LINEAR, LINEAR, NEAREST, CLAMP, CLAMP, CLAMP, 1, TRANSPARENT_BLACK, COMPARE_NONE);

    float4 color = float4(0.0, 0.0, 0.0, 0.0);
    const int kernelRadius = 16;

    for (int ky = -kernelRadius; ky <= kernelRadius; ky++)
    {
        for (int kx = -kernelRadius; kx <= kernelRadius; kx++)
        {
            int2 sampleCoord = pixelCoord + int2(kx, ky);
            sampleCoord = clamp(sampleCoord, int2(0, 0), imgSize - int2(1, 1));
            float2 uv = (float2(sampleCoord.x, sampleCoord.y) + 0.5) / float2(imgSize.x, imgSize.y);
            color += sampler.SampleLevel(srcTexture, uv, 0);
        }
    }

    color = color / pow(kernelRadius * 2 + 1, 2);

    // Follow the pointer attached on the CPU side and read the Tint buffer. The
    // image is split into `tintCount` vertical bands; each band uses one entry.
    uint band = min((uint)pixelCoord.x * data->tintCount / (uint)imgSize.x, data->tintCount - 1);
    Tint tint = data->tints[band];
    color.rgb = lerp(color.rgb, color.rgb * tint.color, tint.strength);

    dstTexture[pixelCoord] = color;
}