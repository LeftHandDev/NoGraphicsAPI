#include "SamplerBench.h"

// Identical tap loops; the only difference is how the sampler is declared —
// the raw hardware default sampler at internal heap slot 0 (ground truth),
// a STATIC_SAMPLER (64-bit spec constant, specialized at pipeline creation),
// or an INLINE_SAMPLER (tagged 32-bit literal, patched at pipeline creation).
// All three must time identically; the cross-check entries prove the
// resolved samplers are the declared ones.

float2 tapUV(int2 pixelCoord, int kx, int ky, float2 invSize)
{
    return (float2(pixelCoord) + 0.5 + float2(kx, ky) * 0.71) * invSize;
}

STATIC_SAMPLER(staticLinearWrap, LINEAR, LINEAR, NEAREST, WRAP, WRAP, WRAP, 1, TRANSPARENT_BLACK, COMPARE_NONE, 0.0, 0.0, LOD_UNBOUNDED)
STATIC_SAMPLER(staticNearestClamp, NEAREST, NEAREST, NEAREST, CLAMP, CLAMP, CLAMP, 1, TRANSPARENT_BLACK, COMPARE_NONE, 0.0, 0.0, LOD_UNBOUNDED)

[numthreads(16, 16, 1)] void benchHardware(uint3 threadId : SV_DispatchThreadID, SamplerBenchData* data)
{
    int2 imgSize = int2(data->imageSize.x, data->imageSize.y);
    int2 pixelCoord = int2(threadId.x, threadId.y);
    if (pixelCoord.x >= imgSize.x || pixelCoord.y >= imgSize.y)
        return;

    Texture2D<float4> srcTexture = textureHeap[data->srcTexture];
    SamplerState sampler = ngapiSamplerHeap[0]; // internal heap: raw reference row only
    float2 invSize = 1.0 / float2(imgSize);

    int r = data->kernelRadius;
    float4 acc = float4(0, 0, 0, 0);
    for (int ky = -r; ky <= r; ky++)
        for (int kx = -r; kx <= r; kx++)
            acc += srcTexture.SampleLevel(sampler, tapUV(pixelCoord, kx, ky, invSize), 0);

    int taps = (2 * r + 1) * (2 * r + 1);
    rwTextureHeap[data->dstTexture][pixelCoord] = acc / taps;
}

    [numthreads(16, 16, 1)] void benchStatic(uint3 threadId : SV_DispatchThreadID, SamplerBenchData* data)
{
    int2 imgSize = int2(data->imageSize.x, data->imageSize.y);
    int2 pixelCoord = int2(threadId.x, threadId.y);
    if (pixelCoord.x >= imgSize.x || pixelCoord.y >= imgSize.y)
        return;

    Texture2D<float4> srcTexture = textureHeap[data->srcTexture];
    HwSampler sampler = staticLinearWrap();
    float2 invSize = 1.0 / float2(imgSize);

    int r = data->kernelRadius;
    float4 acc = float4(0, 0, 0, 0);
    for (int ky = -r; ky <= r; ky++)
        for (int kx = -r; kx <= r; kx++)
            acc += sampler.SampleLevel(srcTexture, tapUV(pixelCoord, kx, ky, invSize), 0);

    int taps = (2 * r + 1) * (2 * r + 1);
    rwTextureHeap[data->dstTexture][pixelCoord] = acc / taps;
}

[numthreads(16, 16, 1)] void benchInline(uint3 threadId : SV_DispatchThreadID, SamplerBenchData* data)
{
    int2 imgSize = int2(data->imageSize.x, data->imageSize.y);
    int2 pixelCoord = int2(threadId.x, threadId.y);
    if (pixelCoord.x >= imgSize.x || pixelCoord.y >= imgSize.y)
        return;

    Texture2D<float4> srcTexture = textureHeap[data->srcTexture];
    HwSampler sampler = INLINE_SAMPLER(LINEAR, LINEAR, NEAREST, WRAP, WRAP, WRAP, 1, TRANSPARENT_BLACK, COMPARE_NONE);
    float2 invSize = 1.0 / float2(imgSize);

    int r = data->kernelRadius;
    float4 acc = float4(0, 0, 0, 0);
    for (int ky = -r; ky <= r; ky++)
        for (int kx = -r; kx <= r; kx++)
            acc += sampler.SampleLevel(srcTexture, tapUV(pixelCoord, kx, ky, invSize), 0);

    int taps = (2 * r + 1) * (2 * r + 1);
    rwTextureHeap[data->dstTexture][pixelCoord] = acc / taps;
}

    [numthreads(16, 16, 1)] void benchStaticNearest(uint3 threadId : SV_DispatchThreadID, SamplerBenchData* data)
{
    int2 imgSize = int2(data->imageSize.x, data->imageSize.y);
    int2 pixelCoord = int2(threadId.x, threadId.y);
    if (pixelCoord.x >= imgSize.x || pixelCoord.y >= imgSize.y)
        return;

    Texture2D<float4> srcTexture = textureHeap[data->srcTexture];
    HwSampler sampler = staticNearestClamp();
    float2 invSize = 1.0 / float2(imgSize);

    int r = data->kernelRadius;
    float4 acc = float4(0, 0, 0, 0);
    for (int ky = -r; ky <= r; ky++)
        for (int kx = -r; kx <= r; kx++)
            acc += sampler.SampleLevel(srcTexture, tapUV(pixelCoord, kx, ky, invSize), 0);

    int taps = (2 * r + 1) * (2 * r + 1);
    rwTextureHeap[data->dstTexture][pixelCoord] = acc / taps;
}

// Library-free nearest/clamp reference using raw Loads: the ground truth the
// NEAREST/CLAMP static sampler must match byte-for-byte.
[numthreads(16, 16, 1)] void benchManualNearest(uint3 threadId : SV_DispatchThreadID, SamplerBenchData* data)
{
    int2 imgSize = int2(data->imageSize.x, data->imageSize.y);
    int2 pixelCoord = int2(threadId.x, threadId.y);
    if (pixelCoord.x >= imgSize.x || pixelCoord.y >= imgSize.y)
        return;

    Texture2D<float4> srcTexture = textureHeap[data->srcTexture];
    float2 invSize = 1.0 / float2(imgSize);

    int r = data->kernelRadius;
    float4 acc = float4(0, 0, 0, 0);
    for (int ky = -r; ky <= r; ky++)
    {
        for (int kx = -r; kx <= r; kx++)
        {
            float2 uv = tapUV(pixelCoord, kx, ky, invSize);
            int2 coord = clamp(int2(floor(uv * float2(imgSize))), int2(0, 0), imgSize - int2(1, 1));
            acc += srcTexture.Load(int3(coord, 0));
        }
    }

    int taps = (2 * r + 1) * (2 * r + 1);
    rwTextureHeap[data->dstTexture][pixelCoord] = acc / taps;
}
