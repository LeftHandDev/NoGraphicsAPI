#ifndef SAMPLES_TENSOR_COMMON_H
#define SAMPLES_TENSOR_COMMON_H

#include "NoGraphicsAPI.h"

struct alignas(16) TensorData
{
    uint64_t n; // number of elements in x and z
    uint64_t m; // number of elements in y (m <= n)
    float a;    // leaky relu alpha
    uint8_t* x; // input
    uint8_t* y; // input (grad in _relu_backard)
    uint8_t* z; // output
};

struct alignas(16) TensorTransposeData
{
    uint64_t n; // number of elements in x and y
    uint r;     // number of rows
    uint c;     // number of columns
    uint8_t* x; // input
    uint8_t* y; // output
};

struct alignas(16) TensorMatMulData
{
    uint64_t n; // number of elements in z
    uint a;     // number of rows in x
    uint b;     // number of columns in x, number of rows in y
    uint c;     // number of columns in y
    uint8_t* x; // input
    uint8_t* y; // input
    uint8_t* z; // output
};

struct alignas(16) TensorAffineData
{
    uint64_t n; // number of elements in w
    uint a;     // number of rows in x
    uint b;     // number of columns in x, number of rows in y
    uint c;     // number of columns in y
    uint8_t* x; // input (activations)
    uint8_t* y; // input (weights)
    uint8_t* z; // input (biases)
    uint8_t* w; // output
};

struct alignas(16) TensorAdamData
{
    uint64_t n; // number of elements
    float b1;
    float b2;
    float b1t;
    float b2t;
    uint8_t* grad;       // input
    uint8_t* mean;       // input/output
    uint8_t* variance;   // input/output
    uint8_t* adjustment; // output
};

#endif