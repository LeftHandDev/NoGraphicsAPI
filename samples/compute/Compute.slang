#include "Compute.h"

// The blur's sampler state is fixed, so declare it next to the shader code as
// a static sampler — it runs on a real hardware sampler at full rate. For
// sampler state only known at runtime, use the plain-data Sampler instead
// (see Sampler.h); it is the same declaration style with shader-code filtering.
STATIC_SAMPLER(linearClamp, LINEAR, LINEAR, NEAREST, CLAMP, CLAMP)

[numthreads(16, 16, 1)] void main(uint3 threadId : SV_DispatchThreadID, ComputeData* data)
{
    int2 imgSize = int2(data->imageSize.x, data->imageSize.y);
    int2 pixelCoord = int2(threadId.x, threadId.y);

    if (pixelCoord.x >= imgSize.x || pixelCoord.y >= imgSize.y)
        return;

    Texture2D<float4> srcTexture = textureHeap[data->srcTexture];
    RWTexture2D<float4> dstTexture = rwTextureHeap[data->dstTexture];
    HwSampler sampler = linearClamp();

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