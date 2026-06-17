#include "Common.h"
#include "../common/Random.h"

static const float PI = 3.1415926535897932384626433832795f;

float _rand_uniform(uint t, uint seed)
{
    uint hash = xxhash32(uint3(t, seed, 1));
    return randomFloat(hash);
}

void _add_impl<T : IArithmetic>(uint t, TensorData* data)
{
    if (t >= data->n)
    {
        return;
    }
    T* x = (T*)data->x;
    T* y = (T*)data->y;
    T* z = (T*)data->z;
    z[t] = x[t] + y[t % data->m];
}

void _sub_impl<T : IArithmetic>(uint t , TensorData* data)
{
    if (t >= data->n)
    {
        return;
    }
    T* x = (T*)data->x;
    T* y = (T*)data->y;
    T* z = (T*)data->z;
    z[t] = x[t] - y[t % data->m];
}

void _mul_impl<T : IArithmetic>(uint t, TensorData* data)
{
    if (t >= data->n)
    {
        return;
    }
    T* x = (T*)data->x;
    T* y = (T*)data->y;
    T* z = (T*)data->z;
    z[t] = x[t] * y[t % data->m];
}

void _div_impl<T : IArithmetic>(uint t, TensorData* data)
{
    if (t >= data->n)
    {
        return;
    }
    T* x = (T*)data->x;
    T* y = (T*)data->y;
    T* z = (T*)data->z;
    z[t] = x[t] / y[t % data->m];
}

void _dot_impl_f32(uint t, TensorData* data)
{
    if (t >= data->n)
    {
        return;
    }
    float* x = (float*)data->x;
    float* y = (float*)data->y;
    float* z = (float*)data->z;

    float result = 0.f;
    for (int i = 0; i < data->n; i++)
    {
        result = result + (x[i] * y[i]);
    }
    z[0] = result;
}

void _dot_impl_f16(uint t, TensorData* data)
{
    if (t >= data->n)
    {
        return;
    }
    half* x = (half*)data->x;
    half* y = (half*)data->y;
    half* z = (half*)data->z;

    float result = 0.f;
    for (int i = 0; i < data->n; i++)
    {
        result = result + (float(x[i]) * float(y[i]));
    }
    z[0] = half(result);
}

void _mT_impl<T : IArithmetic>(uint t, TensorTransposeData *data)
{
    if (t >= data->n)
    {
        return;
    }
    T* x = (T*)data->x;
    T* y = (T*)data->y;

    uint r = t / data->c;
    uint c = t % data->c;

    y[c * data->r + r] = x[r * data->c + c];
}

[numthreads(32, 1, 1)] 
void _affine(uint3 g : SV_GroupID, TensorAffineData* data)
{
    half* x = (half*)data->x;
    half* y = (half*)data->y;
    float* z = (float*)data->z;
    half* w = (half*)data->w;  

    uint tiles = data->c / 16;

    uint z_row = g.x / tiles;
    uint z_col = g.x % tiles;

    var c = linalg::CoopMat<float, MemoryScope.Subgroup, 16, 16, linalg::CoopMatMatrixUse.MatrixAccumulator>
                .Load<linalg::CoopMatMatrixLayout.RowMajor>(z, z_col * 16, 0);

    for (uint i = 0; i < data->b; i += 16)
    {
        let a = linalg::CoopMat<half, MemoryScope.Subgroup, 16, 16, linalg::CoopMatMatrixUse.MatrixA>
            .Load<linalg::CoopMatMatrixLayout.RowMajor>(x, z_row * 16 * data->b + i, data->b);
        let b = linalg::CoopMat<half, MemoryScope.Subgroup, 16, 16, linalg::CoopMatMatrixUse.MatrixB>
            .Load<linalg::CoopMatMatrixLayout.RowMajor>(y, i * data->c + z_col * 16, data->c);
        c = linalg::coopMatMulAdd<float, false>(a, b, c);
    }

    var ch = linalg::CoopMat<half, MemoryScope.Subgroup, 16, 16, linalg::CoopMatMatrixUse.MatrixAccumulator>(c);
    ch.Store<linalg::CoopMatMatrixLayout.RowMajor>(w, z_row * 16 * data->c + z_col * 16, data->c);
}

void _matmul_impl_f32(uint t, TensorMatMulData* data)
{
    if (t >= data->n)
    {
        return;
    }

    float* x = (float*)data->x;
    float* y = (float*)data->y;
    float* z = (float*)data->z;

    uint x_row = t / data->c;
    uint y_col = t % data->c;

    float d = 0.f;
    for (int i = 0; i < data->b; i++)
    {
        d = d + (x[x_row * data->b + i] * y[data->c * i + y_col]);
    }

    z[t] = d;
}

void _matmul_impl_f16(uint t, TensorMatMulData* data)
{
    if (t >= data->n)
    {
        return;
    }

    half* x = (half*)data->x;
    half* y = (half*)data->y;
    half* z = (half*)data->z;

    uint x_row = t / data->c;
    uint y_col = t % data->c;

    float d = 0.f;
    for (int i = 0; i < data->b; i++)
    {
        d = d + (float(x[x_row * data->b + i]) * float(y[data->c * i + y_col]));
    }

    z[t] = half(d);
}

void _pow_impl_f32(uint t, TensorData* data)
{
    if (t >= data->n)
    {
        return;
    }    
    float* x = (float*)data->x;
    float* y = (float*)data->y;
    float* z = (float*)data->z;

    z[t] = pow(x[t], y[t % data->m]);
}

void _pow_impl_f16(uint t, TensorData* data)
{
    if (t >= data->n)
    {
        return;
    }    
    half* x = (half*)data->x;
    half* y = (half*)data->y;
    half* z = (half*)data->z;

    z[t] = pow(x[t], y[t % data->m]);
}

void _log_impl_f32(uint t, TensorData* data)
{
    if (t >= data->n)
    {
        return;
    }
    float* x = (float*)data->x;
    float* z = (float*)data->z;
    z[t] = log(x[t]);
}

void _log_impl_f16(uint t, TensorData* data)
{
    if (t >= data->n)
    {
        return;
    }
    half* x = (half*)data->x;
    half* z = (half*)data->z;
    z[t] = log(x[t]);
}

void _tanh_impl_f32(uint t, TensorData* data)
{
    if (t >= data->n)
    {
        return;
    }
    float* x = (float*)data->x;
    float* z = (float*)data->z;
    z[t] = tanh(x[t]);
}

void _tanh_impl_f16(uint t, TensorData* data)
{
    if (t >= data->n)
    {
        return;
    }
    half* x = (half*)data->x;
    half* z = (half*)data->z;
    z[t] = tanh(x[t]);
}

void _cosh_impl_f32(uint t, TensorData* data)
{
    if (t >= data->n)
    {
        return;
    }
    float* x = (float*)data->x;
    float* z = (float*)data->z;
    z[t] = cosh(x[t]);
}

void _cosh_impl_f16(uint t, TensorData* data)
{
    if (t >= data->n)
    {
        return;
    }
    half* x = (half*)data->x;
    half* z = (half*)data->z;
    z[t] = cosh(x[t]);
}

void _relu_impl_f32(uint t, TensorData* data)
{
    if (t >= data->n)
    {
        return;
    }
    float* x = (float*)data->x;
    float* z = (float*)data->z;
    float v = x[t];
    z[t] = v > 0.f ? v : v * data->a;
}

void _relu_impl_f16(uint t, TensorData* data)
{
    if (t >= data->n)
    {
        return;
    }
    half* x = (half*)data->x;
    half* z = (half*)data->z;
    half v = x[t];
    z[t] = v > 0.h ? v : v * half(data->a);
}

void _relu_backward_impl_f32(uint t, TensorData* data)
{
    if (t >= data->n)
    {
        return;
    }
    float* x = (float*)data->x;
    float* y = (float*)data->y;
    float* z = (float*)data->z;
    z[t] = x[t] > 0.f ? y[t] : y[t] * data->a;
}

void _relu_backward_impl_f16(uint t, TensorData* data)
{
    if (t >= data->n)
    {
        return;
    }
    half* x = (half*)data->x;
    half* y = (half*)data->y;
    half* z = (half*)data->z;
    z[t] = x[t] > 0.h ? y[t] : y[t] * half(data->a);
}

void _gelu_impl_f32(uint t, TensorData* data)
{
    if (t >= data->n)
    {
        return;
    }
    float* x = (float*)data->x;
    float* z = (float*)data->z;
    float v = x[t];

    z[t] = 0.5f * v * (1.f + tanh(sqrt(2.f / PI) * (v + 0.044715 * v * v * v)));
}

void _gelu_impl_f16(uint t, TensorData* data)
{
    if (t >= data->n)
    {
        return;
    }
    half* x = (half*)data->x;
    half* z = (half*)data->z;
    float v = x[t];

    z[t] = half(0.5f * v * (1.f + tanh(sqrt(2.f / PI) * (v + 0.044715 * v * v * v))));
}

void _gelu_backward_impl_f32(uint t, TensorData* data)
{
    if (t >= data->n)
    {
        return;
    }
    float* x = (float*)data->x;
    float* y = (float*)data->y;
    float* z = (float*)data->z;
    float v = x[t];
    float g = y[t];

    float u = 0.797885f * v + 0.0356774f * v * v * v;
    z[t] = g * (0.5f + (0.398942f * v + 0.0535161f * v * v * v) * (1.f / (cosh(u) * cosh(u))) + 0.5f * tanh(u));
}

void _gelu_backward_impl_f16(uint t, TensorData* data)
{
    if (t >= data->n)
    {
        return;
    }
    half* x = (half*)data->x;
    half* y = (half*)data->y;
    half* z = (half*)data->z;
    float v = x[t];
    float g = y[t];

    float u = 0.797885f * v + 0.0356774f * v * v * v;
    z[t] = half(g * (0.5f + (0.398942f * v + 0.0535161f * v * v * v) * (1.f / (cosh(u) * cosh(u))) + 0.5f * tanh(u)));
}

void _adam_impl_f32(uint t, TensorAdamData* data)
{
    if (t >= data->n)
    {
        return;
    }
    float* grad = (float*)data->grad;
    float* _mean = (float*)data->mean;
    float* _variance = (float*)data->variance;
    float* adjustment = (float*)data->adjustment;

    _mean[t] = lerp(grad[t], _mean[t], data->b1);
    _variance[t] = lerp(grad[t] * grad[t], _variance[t], data->b2);

    float mean = _mean[t] / (1.f - data->b1t);
    float variance = _variance[t] / (1.f - data->b2t);

    adjustment[t] = mean / (sqrt(variance) + 1e-10);
}

void _adam_impl_f16(uint t, TensorAdamData* data)
{
    if (t >= data->n)
    {
        return;
    }
    half* grad = (half*)data->grad;
    half* _mean = (half*)data->mean;
    half* _variance = (half*)data->variance;
    half* adjustment = (half*)data->adjustment;

    float g = float(grad[t]);
    float meanState = lerp(g, float(_mean[t]), data->b1);
    float varState = lerp(g * g, float(_variance[t]), data->b2);
    _mean[t] = half(meanState);
    _variance[t] = half(varState);

    float mean = meanState / (1.f - data->b1t);
    float variance = varState / (1.f - data->b2t);

    adjustment[t] = half(mean / (sqrt(variance) + 1e-10));
}

[numthreads(64, 1, 1)]
void _fp32_fp16(uint t : SV_DispatchThreadID, TensorData* data)
{
    if (t >= data->n)
    {
        return;
    }
    half* x = (half*)data->x;
    float* z = (float*)data->z;
    z[t] = x[t];
}

[numthreads(64, 1, 1)]
void _fp16_fp32(uint t : SV_DispatchThreadID, TensorData* data)
{
    if (t >= data->n)
    {
        return;
    }
    float* x = (float*)data->x;
    half* z = (half*)data->z;
    z[t] = (half)x[t];
}

[numthreads(64, 1, 1)] void _rand_f32(uint t : SV_DispatchThreadID, TensorData* data)
{
    if (t >= data->n)
    {
        return;
    }
    float* z = (float*)data->z;
    z[t] = _rand_uniform(t, (uint)data->m);
}

[numthreads(64, 1, 1)] void _rand_f16(uint t : SV_DispatchThreadID, TensorData* data)
{
    if (t >= data->n)
    {
        return;
    }
    half* z = (half*)data->z;
    z[t] = (half)_rand_uniform(t, (uint)data->m);
}

[numthreads(64, 1, 1)] void _add_f32(uint t : SV_DispatchThreadID, TensorData* data){ _add_impl<float>(t, data); }
[numthreads(64, 1, 1)] void _add_f16(uint t : SV_DispatchThreadID, TensorData* data){ _add_impl<half>(t, data); }

[numthreads(64, 1, 1)] void _sub_f32(uint t : SV_DispatchThreadID, TensorData* data){ _sub_impl<float>(t, data); }
[numthreads(64, 1, 1)] void _sub_f16(uint t : SV_DispatchThreadID, TensorData* data){ _sub_impl<half>(t, data); }

[numthreads(64, 1, 1)] void _mul_f32(uint t : SV_DispatchThreadID, TensorData* data){ _mul_impl<float>(t, data); }
[numthreads(64, 1, 1)] void _mul_f16(uint t : SV_DispatchThreadID, TensorData* data){ _mul_impl<half>(t, data); }

[numthreads(64, 1, 1)] void _div_f32(uint t : SV_DispatchThreadID, TensorData* data){ _div_impl<float>(t, data); }
[numthreads(64, 1, 1)] void _div_f16(uint t : SV_DispatchThreadID, TensorData* data){ _div_impl<half>(t, data); }

[numthreads(64, 1, 1)] void _mT_f32(uint t : SV_DispatchThreadID, TensorTransposeData* data){ _mT_impl<float>(t, data); }
[numthreads(64, 1, 1)] void _mT_f16(uint t : SV_DispatchThreadID, TensorTransposeData* data){ _mT_impl<half>(t, data); }

[numthreads(64, 1, 1)] void _matmul_f32(uint t : SV_DispatchThreadID, TensorMatMulData* data){ _matmul_impl_f32(t, data); }
[numthreads(64, 1, 1)] void _matmul_f16(uint t : SV_DispatchThreadID, TensorMatMulData* data){ _matmul_impl_f16(t, data); }

[numthreads(1, 1, 1)] void _dot_f32(uint t : SV_DispatchThreadID, TensorData* data){ _dot_impl_f32(t, data); }
[numthreads(1, 1, 1)] void _dot_f16(uint t : SV_DispatchThreadID, TensorData* data){ _dot_impl_f16(t, data); }

[numthreads(64, 1, 1)] void _pow_f32(uint t : SV_DispatchThreadID, TensorData* data){ _pow_impl_f32(t, data); }
[numthreads(64, 1, 1)] void _pow_f16(uint t : SV_DispatchThreadID, TensorData* data){ _pow_impl_f16(t, data); }

[numthreads(64, 1, 1)] void _log_f32(uint t : SV_DispatchThreadID, TensorData* data){ _log_impl_f32(t, data); }
[numthreads(64, 1, 1)] void _log_f16(uint t : SV_DispatchThreadID, TensorData* data){ _log_impl_f16(t, data); }

[numthreads(64, 1, 1)] void _cosh_f32(uint t : SV_DispatchThreadID, TensorData* data){ _cosh_impl_f32(t, data); }
[numthreads(64, 1, 1)] void _cosh_f16(uint t : SV_DispatchThreadID, TensorData* data){ _cosh_impl_f16(t, data); }

[numthreads(64, 1, 1)] void _tanh_f32(uint t : SV_DispatchThreadID, TensorData* data){ _tanh_impl_f32(t, data); }
[numthreads(64, 1, 1)] void _tanh_f16(uint t : SV_DispatchThreadID, TensorData* data){ _tanh_impl_f16(t, data); }

[numthreads(64, 1, 1)] void _relu_f32(uint t : SV_DispatchThreadID, TensorData* data){ _relu_impl_f32(t, data); }
[numthreads(64, 1, 1)] void _relu_f16(uint t : SV_DispatchThreadID, TensorData* data){ _relu_impl_f16(t, data); }

[numthreads(64, 1, 1)] void _relu_backward_f32(uint t : SV_DispatchThreadID, TensorData* data){ _relu_backward_impl_f32(t, data); }
[numthreads(64, 1, 1)] void _relu_backward_f16(uint t : SV_DispatchThreadID, TensorData* data){ _relu_backward_impl_f16(t, data); }

[numthreads(64, 1, 1)] void _gelu_f32(uint t : SV_DispatchThreadID, TensorData* data){ _gelu_impl_f32(t, data); }
[numthreads(64, 1, 1)] void _gelu_f16(uint t : SV_DispatchThreadID, TensorData* data){ _gelu_impl_f16(t, data); }

[numthreads(64, 1, 1)] void _gelu_backward_f32(uint t : SV_DispatchThreadID, TensorData* data){ _gelu_backward_impl_f32(t, data); }
[numthreads(64, 1, 1)] void _gelu_backward_f16(uint t : SV_DispatchThreadID, TensorData* data){ _gelu_backward_impl_f16(t, data); }

[numthreads(64, 1, 1)] void _adam_f32(uint t : SV_DispatchThreadID, TensorAdamData* data){ _adam_impl_f32(t, data); }
[numthreads(64, 1, 1)] void _adam_f16(uint t : SV_DispatchThreadID, TensorAdamData* data){ _adam_impl_f16(t, data); }