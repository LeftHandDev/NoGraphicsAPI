#ifndef SAMPLES_TENSOR_COMMON_H
#define SAMPLES_TENSOR_COMMON_H

#include "../../NoGraphicsAPI.h"

struct alignas(16) TensorData
{
    uint64_t n;
    float *x;
    float *y;
    float *z;
};

struct alignas(16) TensorTransposeData
{
    uint64_t n;
    uint64_t w;
    uint64_t h;
    float *x;
    float *y;
};

#endif