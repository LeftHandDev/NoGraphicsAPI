#ifndef TESTS_LAYOUT_MISMATCH_H
#define TESTS_LAYOUT_MISMATCH_H

#include "NoGraphicsAPI.h"

// Deliberately laid out differently in C++ than on the GPU: the layout check
// in compile_shader must reject Mismatch.slang (see tests/CMakeLists.txt).

// alignas(16) pads Item to 16 bytes in C++ only; the shader indexes Item
// arrays with its GPU stride of 12.
struct alignas(16) Item
{
    float3 position;
};

// bool is 1 byte in C++ but 4 on the GPU.
struct Flags
{
    uint count;
    bool enabled;
    uint after;
};

// The alignas(16) Item member starts at byte 16 in C++, at byte 4 on the GPU.
struct Shifted
{
    uint first;
    Item item;
};

struct MismatchData
{
    Item* items;
    Flags* flags;
    Shifted* shifted;
};

#endif // TESTS_LAYOUT_MISMATCH_H
