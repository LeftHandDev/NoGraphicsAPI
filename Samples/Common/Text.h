#ifndef SAMPLES_SHADER_TEXT_H
#define SAMPLES_SHADER_TEXT_H

#include "../../NoGraphicsAPI.h"

struct alignas(16) TextVertexData
{
    uint width;
    uint height;
    uint textWidth;
    uint textHeight;
    uint8_t* text; // per instance text data, 1 byte per character (ASCII)
};

struct alignas(16) TextPixelData
{
    uint atlas;
};

#endif // SAMPLES_SHADER_TEXT_H