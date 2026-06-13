#ifndef SAMPLES_TENSOR_COMMON_H
#define SAMPLES_TENSOR_COMMON_H

#include "NoGraphicsAPI.h"

struct alignas(16) TensorData
{
    uint64_t n; // number of elements in x and z
    uint64_t m; // number of elements in y (m <= n)
    float* x;   // input
    float* y;   // input
    float* z;   // output
};

struct alignas(16) TensorTransposeData
{
    uint64_t n; // number of elements in x and y
    uint64_t r; // number of rows
    uint64_t c; // number of columns
    float* x;   // input
    float* y;   // output
};

struct alignas(16) TensorMatMulData
{
    uint64_t n; // number of elements in z
    uint64_t a; // number of rows in x
    uint64_t b; // number of columns in x, number of rows in y
    uint64_t c; // number of columns in y
    float* x;   // input
    float* y;   // input
    float* z;   // output
};

#endif