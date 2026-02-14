#ifndef SHADERS_PATCH_DESCRIPTORS_H
#define SHADERS_PATCH_DESCRIPTORS_H

#include "../NoGraphicsAPI.h"

// When the descriptor size on the device is not 32 bytes, we write this descriptor to the user's descriptor heap
struct alignas(16) Descriptor
{
    uint64_t offset; // offset to real descriptor data
    uint64_t type; // 0 = read descriptor, 1 = read/write descriptor
    uint64_t padding[2]; // unused
};

struct alignas(16) PatchDescriptorsData
{
    uint numDescriptors;
    uint descriptorSize;
    uint rwDescriptorSize;
    Descriptor* descriptors; // the heap that the user has bound, which we need to patch
    uint8_t* srcDescriptors; // the offset in Descriptor points to this buffer (when read type)
    uint8_t* rwSrcDescriptors; // the offset in Descriptor points to this buffer (when read/write type)
    uint8_t* dstDescriptors; // the read heap that will actually be used
    uint8_t* rwDstDescriptors;  // the read/write heap that will actually be used
};


#endif // SHADERS_PATCH_DESCRIPTORS_H