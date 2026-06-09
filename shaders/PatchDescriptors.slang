#include "./PatchDescriptors.h"

[numthreads(16, 1, 1)]
void main(uint3 threadId: SV_DispatchThreadID, PatchDescriptorsData* data)
{
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