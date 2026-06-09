#ifndef SAMPLES_TENSOR_COMMON_H
#define SAMPLES_TENSOR_COMMON_H

#include "NoGraphicsAPI.h"

struct alignas(16) TensorData
{
    uint64_t n;
    float* x;
    float* y;
    float* z;
};

struct alignas(16) TensorTransposeData
{
    uint64_t n;
    uint64_t r;
    uint64_t c;
    float* x;
    float* y;
};

struct alignas(16) TensorMatMulData
{
    uint64_t n;
    uint64_t a;
    uint64_t b;
    uint64_t c;
    float* x;
    float* y;
    float* z;
};

#endif