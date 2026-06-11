#ifndef SAMPLES_SHADER_MULTITHREADING_H
#define SAMPLES_SHADER_MULTITHREADING_H

#include "NoGraphicsAPI.h"

struct alignas(16) WorkerData
{
    uint count;
    uint workerId;
    uint srcTexture;

    float* input;
    float* output;
};

#endif // SAMPLES_SHADER_MULTITHREADING_H
