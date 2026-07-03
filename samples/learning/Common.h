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

struct alignas(16) TensorPermuteData
{
    uint64_t n;  // number of elements in x and y
    uint64_t m;  // number of elements in s and t
    uint32_t* s; // input shape
    uint32_t* t; // permutation of s
    uint8_t* x;  // input
    uint8_t* y;  // output
};

struct alignas(16) TensorUnfoldData
{
    uint64_t n; // forward: H*W*C*k*k output elements; backward: H*W*C input elements
    uint h;     // image height
    uint w;     // image width
    uint c;     // channels
    uint k;     // kernel size
    uint pad;   // zero-pad radius
    uint8_t* x; // input  (forward: image (H,W,C); backward: grad_out (H,W,C,k,k))
    uint8_t* y; // output (forward: neighborhood (H,W,C,k,k); backward: grad_in (H,W,C))
};

struct alignas(16) TensorReduceData
{
    uint64_t n;     // number of output elements
    uint64_t outer; // product of dims before the axis
    uint64_t axis;  // size of the reduced (sum) / expanded (broadcast) axis
    uint64_t inner; // product of dims after the axis
    uint8_t* x;     // input
    uint8_t* y;     // output
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