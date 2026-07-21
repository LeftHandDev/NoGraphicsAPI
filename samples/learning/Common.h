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
    uint64_t n;  // forward: H*W*C*k*k output elements; backward: H*W*C input elements
    uint h;      // image height
    uint w;      // image width
    uint c;      // channels
    uint k;      // kernel size
    uint pad;    // zero-pad radius
    uint mode;   // padding mode: 0 = zero, 1 = reflect
    uint stride; // window stride
    uint oh;     // output height = (h + 2*pad - k) / stride + 1
    uint ow;     // output width  = (w + 2*pad - k) / stride + 1
    uint8_t* x;  // input  (forward: image (H,W,C); backward: grad_out (OH,OW,C,k,k))
    uint8_t* y;  // output (forward: neighborhood (OH,OW,C,k,k); backward: grad_in (H,W,C))
};

struct alignas(16) TensorReduceData
{
    uint64_t n;     // number of output elements
    uint64_t outer; // product of dims before the axis
    uint64_t axis;  // size of the reduced (sum) / expanded (broadcast) axis
    uint64_t inner; // product of dims after the axis
    uint8_t* x;     // input
    uint8_t* y;     // output
    uint8_t* z;     // aux input (reduce_max backward: grad_out)
};

struct alignas(16) TensorReduceBlockData
{
    uint64_t outer;  // segments before the reduced axis
    uint64_t axis;   // current length of the reduced axis (this pass's input)
    uint64_t inner;  // segments after the reduced axis
    uint64_t groups; // output axis length = ceil(axis / block size)
    uint8_t* x;      // input  (outer, axis,   inner)
    uint8_t* y;      // output (outer, groups, inner)
};

struct alignas(16) TensorConcatData
{
    uint64_t n;      // elements this dispatch touches (= outer * axis * inner)
    uint64_t outer;  // product of dims before the concat axis
    uint64_t inner;  // product of dims after the concat axis
    uint64_t axis;   // this slab's size along the axis
    uint64_t total;  // combined axis size (a_axis + b_axis)
    uint64_t offset; // start of this slab within the combined axis
    uint8_t* x;      // forward: a slab input; backward: grad_out (wide)
    uint8_t* y;      // forward: combined output (wide); backward: grad_in (slab)
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
    uint64_t n;      // number of elements in z
    uint a;          // number of rows in x
    uint b;          // number of columns in x, number of rows in y
    uint c;          // number of columns in y
    uint splits;     // split-K: contraction partitions (partials are (splits, a, c))
    uint transposeA; // split-K: if set, x is stored (b, a) and read transposed (avoids a materialized mT)
    uint8_t* x;      // input
    uint8_t* y;      // input
    uint8_t* z;      // output
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