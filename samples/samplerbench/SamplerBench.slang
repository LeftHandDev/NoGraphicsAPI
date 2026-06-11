#include "SamplerBench.h"

// Two entry points with the identical tap loop; the only difference is who
// does the bilinear filtering — the hardware sampler (samplerHeap[0], the
// implementation's LINEAR/WRAP default) or the software Sampler (same
// settings via its defaults). Tap offsets are deliberately non-integer so the
// bilinear weights are real work in both paths, and the accumulated result is
// written out so nothing can be dead-code eliminated.

float2 tapUV(int2 pixelCoord, int kx, int ky, float2 invSize)
{
    return (float2(pixelCoord) + 0.5 + float2(kx, ky) * 0.71) * invSize;
}

// --- static samplers (STATIC_SAMPLER from Sampler.h) ------------------------
// Declared next to the shader code; the implementation creates the matching
// hardware samplers at pipeline creation and specializes the heap slots in.
// benchStatic matches the hardware row's sampler state for the timing
// comparison; the NEAREST/CLAMP pair below exists to prove specialization
// happened (its output must match the equivalent software sampler exactly,
// which cannot happen if the shader still reads a stale heap slot).

STATIC_SAMPLER(staticLinearWrap, LINEAR, LINEAR, NEAREST, WRAP, WRAP)
STATIC_SAMPLER(staticNearestClamp, NEAREST, NEAREST, NEAREST, CLAMP, CLAMP)

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

[numthreads(16, 16, 1)] void benchSoftwareNearest(uint3 threadId : SV_DispatchThreadID, SamplerBenchData* data)
{
    int2 imgSize = int2(data->imageSize.x, data->imageSize.y);
    int2 pixelCoord = int2(threadId.x, threadId.y);
    if (pixelCoord.x >= imgSize.x || pixelCoord.y >= imgSize.y)
        return;

    Texture2D<float4> srcTexture = textureHeap[data->srcTexture];
    Sampler sampler = {};
    sampler.minFilter = NEAREST;
    sampler.magFilter = NEAREST;
    sampler.addressU = CLAMP;
    sampler.addressV = CLAMP;
    float2 invSize = 1.0 / float2(imgSize);

    int r = data->kernelRadius;
    float4 acc = float4(0, 0, 0, 0);
    for (int ky = -r; ky <= r; ky++)
        for (int kx = -r; kx <= r; kx++)
            acc += sampler.SampleLevel(srcTexture, tapUV(pixelCoord, kx, ky, invSize), 0);

    int taps = (2 * r + 1) * (2 * r + 1);
    rwTextureHeap[data->dstTexture][pixelCoord] = acc / taps;
}

    [numthreads(16, 16, 1)] void benchHardware(uint3 threadId : SV_DispatchThreadID, SamplerBenchData* data)
{
    int2 imgSize = int2(data->imageSize.x, data->imageSize.y);
    int2 pixelCoord = int2(threadId.x, threadId.y);
    if (pixelCoord.x >= imgSize.x || pixelCoord.y >= imgSize.y)
        return;

    Texture2D<float4> srcTexture = textureHeap[data->srcTexture];
    SamplerState sampler = samplerHeap[0];
    float2 invSize = 1.0 / float2(imgSize);

    int r = data->kernelRadius;
    float4 acc = float4(0, 0, 0, 0);
    for (int ky = -r; ky <= r; ky++)
        for (int kx = -r; kx <= r; kx++)
            acc += srcTexture.SampleLevel(sampler, tapUV(pixelCoord, kx, ky, invSize), 0);

    int taps = (2 * r + 1) * (2 * r + 1);
    rwTextureHeap[data->dstTexture][pixelCoord] = acc / taps;
}

[numthreads(16, 16, 1)] void benchSoftware(uint3 threadId : SV_DispatchThreadID, SamplerBenchData* data)
{
    int2 imgSize = int2(data->imageSize.x, data->imageSize.y);
    int2 pixelCoord = int2(threadId.x, threadId.y);
    if (pixelCoord.x >= imgSize.x || pixelCoord.y >= imgSize.y)
        return;

    Texture2D<float4> srcTexture = textureHeap[data->srcTexture];
    Sampler sampler = {}; // defaults: LINEAR min/mag, WRAP — matches the hardware path
    float2 invSize = 1.0 / float2(imgSize);

    int r = data->kernelRadius;
    float4 acc = float4(0, 0, 0, 0);
    for (int ky = -r; ky <= r; ky++)
        for (int kx = -r; kx <= r; kx++)
            acc += sampler.SampleLevel(srcTexture, tapUV(pixelCoord, kx, ky, invSize), 0);

    int taps = (2 * r + 1) * (2 * r + 1);
    rwTextureHeap[data->dstTexture][pixelCoord] = acc / taps;
}
