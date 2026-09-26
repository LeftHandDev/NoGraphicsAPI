#include "HeapProbe.h"

[numthreads(1, 1, 1)] void main(uint3 threadId : SV_DispatchThreadID, HeapProbeData* data)
{
    *data->result = textureHeap[data->index].Load(int3(0, 0, 0));
}
