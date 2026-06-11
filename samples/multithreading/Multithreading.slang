#include "Multithreading.h"

// Sampling a 1x1 white texture through a static sampler contributes exactly
// 1.0 — and routes every worker's concurrently-created pipeline through the
// sampler-dedup machinery, which is part of what the sample stress-tests.
STATIC_SAMPLER(linearClamp, LINEAR, LINEAR, NEAREST, CLAMP, CLAMP, CLAMP, 1, TRANSPARENT_BLACK, COMPARE_NONE, 0.0, 0.0, LOD_UNBOUNDED)

[numthreads(64, 1, 1)] void main(uint3 threadId : SV_DispatchThreadID, WorkerData* data)
{
    if (threadId.x >= data->count)
        return;

    float white = linearClamp().SampleLevel(textureHeap[data->srcTexture], float2(0.5, 0.5), 0).x;
    data->output[threadId.x] = data->input[threadId.x] + float(data->workerId) * white;
}
