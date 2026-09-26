#include "./PatchDescriptors.h"

// The parameters are push constants (by value, not a pointer like other NGAPI
// shaders take): each gpuSetActiveTextureHeapPtr records its own copy.
[numthreads(16, 1, 1)] void main(uint3 threadId : SV_DispatchThreadID, uniform PatchDescriptorsData params)
{
    PatchDescriptorsData* data = &params;
    uint index = threadId.x;
    if (index >= data->numDescriptors)
        return;

    Descriptor descriptor = data->descriptors[index];

    if (descriptor.type == 0)
    {
        for (uint i = 0; i < data->descriptorSize; i++)
        {
            data->dstDescriptors[index * data->descriptorSize + i] = data->srcDescriptors[descriptor.offset + i];
        }
    }
    else if (descriptor.type == 1)
    {
        for (uint i = 0; i < data->rwDescriptorSize; i++)
        {
            data->rwDstDescriptors[index * data->rwDescriptorSize + i] = data->rwSrcDescriptors[descriptor.offset + i];
        }
    }
}