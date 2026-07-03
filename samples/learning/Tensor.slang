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

void _permute_impl<T : IArithmetic>(uint t, TensorPermuteData *data)
{
    if (t >= data->n)
    {
        return;
    }
    T* x = (T*)data->x;
    T* y = (T*)data->y;

    int m = (int)data->m;

    uint64_t dst = 0;
    uint64_t dst_stride = 1;
    for (int i = m - 1; i >= 0; i--)
    {
        uint a = data->t[i]; // input axis

        uint64_t src_stride = 1;
        for (uint k = a + 1; k < m; k++)
        {
            src_stride *= data->s[k];
        }

        uint coord = uint((t / src_stride) % data->s[a]);
        dst += uint64_t(coord) * dst_stride;
        dst_stride *= data->s[a];
    }

    y[dst] = x[t];
}

// reflect a coordinate back into [0, n) without repeating the edge pixel
// (mirror boundary: -1 -> 1, n -> n-2). handles arbitrary out-of-range values.
int _reflect(int p, int n)
{
    if (n <= 1)
    {
        return 0;
    }
    int period = 2 * (n - 1);
    int m = p % period;
    if (m < 0)
    {
        m += period;
    }
    if (m >= n)
    {
        m = period - m;
    }
    return m;
}

void _unfold_impl<T : IArithmetic>(uint t, TensorUnfoldData *data)
{
    if (t >= data->n)
    {
        return;
    }
    T* x = (T*)data->x; // (H, W, C)
    T* y = (T*)data->y; // (H, W, C, k, k)

    uint k = data->k;

    // decode output index t over (H, W, C, k, k)
    uint dx = t % k;
    uint r1 = t / k;
    uint dy = r1 % k;
    uint r2 = r1 / k;
    uint c = r2 % data->c;
    uint r3 = r2 / data->c;
    uint col = r3 % data->w;
    uint row = r3 / data->w;

    int sy = int(row) + int(dy) - int(data->pad);
    int sx = int(col) + int(dx) - int(data->pad);

    bool oob = (sy < 0 || sy >= int(data->h) || sx < 0 || sx >= int(data->w));
    if (oob)
    {
        if (data->mode == 1u) // reflect (no edge repeat)
        {
            sy = _reflect(sy, int(data->h));
            sx = _reflect(sx, int(data->w));
        }
        else // zero padding: out-of-bounds neighbours contribute 0
        {
            y[t] = T(0);
            return;
        }
    }

    uint src = (uint(sy) * data->w + uint(sx)) * data->c + c;
    y[t] = x[src];
}

// accumulate grad_in[t] by gathering every output window that read this input
// pixel. In reflect mode a boundary pixel is additionally read via mirrored
// out-of-bounds neighbours, so we sum over direct + reflected source candidates.
void _unfold_backward_impl_f32(uint t, TensorUnfoldData *data)
{
    if (t >= data->n)
    {
        return;
    }
    float* x = (float*)data->x; // grad_out (H, W, C, k, k)
    float* y = (float*)data->y; // grad_in  (H, W, C)

    uint k = data->k;

    // decode input index t over (H, W, C)
    uint c = t % data->c;
    uint r1 = t / data->c;
    uint col = r1 % data->w;
    uint row = r1 / data->w;

    int H = int(data->h);
    int W = int(data->w);
    int pad = int(data->pad);
    int i = int(row);
    int j = int(col);

    float acc = 0.f;

    if (data->mode == 1u) // reflect
    {
        // source coords on each axis that map (directly or by mirroring) to i / j
        int cy[3];
        bool cy_ok[3];
        cy[0] = i;               cy_ok[0] = true;          // in-bounds source
        cy[1] = -i;              cy_ok[1] = (i > 0);       // top reflection (source < 0)
        cy[2] = 2 * (H - 1) - i; cy_ok[2] = (i < H - 1);   // bottom reflection (source >= H)
        int cx[3];
        bool cx_ok[3];
        cx[0] = j;               cx_ok[0] = true;
        cx[1] = -j;              cx_ok[1] = (j > 0);
        cx[2] = 2 * (W - 1) - j; cx_ok[2] = (j < W - 1);

        for (uint dy = 0; dy < k; dy++)
        {
            for (uint dx = 0; dx < k; dx++)
            {
                for (int a = 0; a < 3; a++)
                {
                    if (!cy_ok[a])
                    {
                        continue;
                    }
                    int oy = cy[a] - int(dy) + pad;
                    if (oy < 0 || oy >= H)
                    {
                        continue;
                    }
                    for (int b = 0; b < 3; b++)
                    {
                        if (!cx_ok[b])
                        {
                            continue;
                        }
                        int ox = cx[b] - int(dx) + pad;
                        if (ox < 0 || ox >= W)
                        {
                            continue;
                        }
                        uint go = (((uint(oy) * data->w + uint(ox)) * data->c + c) * k + dy) * k + dx;
                        acc += x[go];
                    }
                }
            }
        }
    }
    else // zero padding
    {
        for (uint dy = 0; dy < k; dy++)
        {
            for (uint dx = 0; dx < k; dx++)
            {
                int oy = i - int(dy) + pad;
                int ox = j - int(dx) + pad;
                if (oy < 0 || oy >= H || ox < 0 || ox >= W)
                {
                    continue;
                }
                uint go = (((uint(oy) * data->w + uint(ox)) * data->c + c) * k + dy) * k + dx;
                acc += x[go];
            }
        }
    }

    y[t] = acc;
}

void _unfold_backward_impl_f16(uint t, TensorUnfoldData *data)
{
    if (t >= data->n)
    {
        return;
    }
    half* x = (half*)data->x; // grad_out (H, W, C, k, k)
    half* y = (half*)data->y; // grad_in  (H, W, C)

    uint k = data->k;

    // decode input index t over (H, W, C)
    uint c = t % data->c;
    uint r1 = t / data->c;
    uint col = r1 % data->w;
    uint row = r1 / data->w;

    int H = int(data->h);
    int W = int(data->w);
    int pad = int(data->pad);
    int i = int(row);
    int j = int(col);

    float acc = 0.f;

    if (data->mode == 1u) // reflect
    {
        // source coords on each axis that map (directly or by mirroring) to i / j
        int cy[3];
        bool cy_ok[3];
        cy[0] = i;               cy_ok[0] = true;          // in-bounds source
        cy[1] = -i;              cy_ok[1] = (i > 0);       // top reflection (source < 0)
        cy[2] = 2 * (H - 1) - i; cy_ok[2] = (i < H - 1);   // bottom reflection (source >= H)
        int cx[3];
        bool cx_ok[3];
        cx[0] = j;               cx_ok[0] = true;
        cx[1] = -j;              cx_ok[1] = (j > 0);
        cx[2] = 2 * (W - 1) - j; cx_ok[2] = (j < W - 1);

        for (uint dy = 0; dy < k; dy++)
        {
            for (uint dx = 0; dx < k; dx++)
            {
                for (int a = 0; a < 3; a++)
                {
                    if (!cy_ok[a])
                    {
                        continue;
                    }
                    int oy = cy[a] - int(dy) + pad;
                    if (oy < 0 || oy >= H)
                    {
                        continue;
                    }
                    for (int b = 0; b < 3; b++)
                    {
                        if (!cx_ok[b])
                        {
                            continue;
                        }
                        int ox = cx[b] - int(dx) + pad;
                        if (ox < 0 || ox >= W)
                        {
                            continue;
                        }
                        uint go = (((uint(oy) * data->w + uint(ox)) * data->c + c) * k + dy) * k + dx;
                        acc += float(x[go]);
                    }
                }
            }
        }
    }
    else // zero padding
    {
        for (uint dy = 0; dy < k; dy++)
        {
            for (uint dx = 0; dx < k; dx++)
            {
                int oy = i - int(dy) + pad;
                int ox = j - int(dx) + pad;
                if (oy < 0 || oy >= H || ox < 0 || ox >= W)
                {
                    continue;
                }
                uint go = (((uint(oy) * data->w + uint(ox)) * data->c + c) * k + dy) * k + dx;
                acc += float(x[go]);
            }
        }
    }

    y[t] = (half)acc;
}

void _reduce_sum_impl_f32(uint t, TensorReduceData *data)
{
    if (t >= data->n)
    {
        return;
    }
    float* x = (float*)data->x;
    float* y = (float*)data->y;

    uint o = uint(t / data->inner);
    uint j = uint(t % data->inner);

    float acc = 0.f;
    for (uint a = 0; a < data->axis; a++)
    {
        acc += x[(o * data->axis + a) * data->inner + j];
    }
    y[t] = acc;
}

void _reduce_sum_impl_f16(uint t, TensorReduceData *data)
{
    if (t >= data->n)
    {
        return;
    }
    half* x = (half*)data->x;
    half* y = (half*)data->y;

    uint o = uint(t / data->inner);
    uint j = uint(t % data->inner);

    float acc = 0.f;
    for (uint a = 0; a < data->axis; a++)
    {
        acc += x[(o * data->axis + a) * data->inner + j];
    }
    y[t] = (half)acc;
}

void _reduce_max_impl_f32(uint t, TensorReduceData *data)
{
    if (t >= data->n)
    {
        return;
    }
    float* x = (float*)data->x;
    float* y = (float*)data->y;

    uint o = uint(t / data->inner);
    uint j = uint(t % data->inner);

    float m = x[(o * data->axis + 0) * data->inner + j];
    for (uint a = 1; a < data->axis; a++)
    {
        m = max(m, x[(o * data->axis + a) * data->inner + j]);
    }
    y[t] = m;
}

void _reduce_max_impl_f16(uint t, TensorReduceData *data)
{
    if (t >= data->n)
    {
        return;
    }
    half* x = (half*)data->x;
    half* y = (half*)data->y;

    uint o = uint(t / data->inner);
    uint j = uint(t % data->inner);

    float m = x[(o * data->axis + 0) * data->inner + j];
    for (uint a = 1; a < data->axis; a++)
    {
        m = max(m, (float)x[(o * data->axis + a) * data->inner + j]);
    }
    y[t] = (half)m;
}

// backward of max: route grad only to the (first) argmax element of each group.
// x = original input, y = grad_in (output), z = grad_out. n = input element count.
void _reduce_max_backward_impl_f32(uint t, TensorReduceData *data)
{
    if (t >= data->n)
    {
        return;
    }
    float* x = (float*)data->x;
    float* y = (float*)data->y;
    float* g = (float*)data->z;

    uint j = uint(t % data->inner);
    uint a = uint((t / data->inner) % data->axis);
    uint o = uint(t / (data->axis * data->inner));

    float m = x[(o * data->axis + 0) * data->inner + j];
    uint arg = 0;
    for (uint aa = 1; aa < data->axis; aa++)
    {
        float v = x[(o * data->axis + aa) * data->inner + j];
        if (v > m)
        {
            m = v;
            arg = aa;
        }
    }

    y[t] = (a == arg) ? g[o * data->inner + j] : 0.f;
}

void _reduce_max_backward_impl_f16(uint t, TensorReduceData *data)
{
    if (t >= data->n)
    {
        return;
    }
    half* x = (half*)data->x;
    half* y = (half*)data->y;
    half* g = (half*)data->z;

    uint j = uint(t % data->inner);
    uint a = uint((t / data->inner) % data->axis);
    uint o = uint(t / (data->axis * data->inner));

    float m = x[(o * data->axis + 0) * data->inner + j];
    uint arg = 0;
    for (uint aa = 1; aa < data->axis; aa++)
    {
        float v = x[(o * data->axis + aa) * data->inner + j];
        if (v > m)
        {
            m = v;
            arg = aa;
        }
    }

    y[t] = (a == arg) ? g[o * data->inner + j] : (half)0.f;
}

void _expand_impl<T : IArithmetic>(uint t, TensorReduceData *data)
{
    if (t >= data->n)
    {
        return;
    }
    T* x = (T*)data->x;
    T* y = (T*)data->y;

    uint j = uint(t % data->inner);
    uint o = uint(t / (data->axis * data->inner));

    y[t] = x[o * data->inner + j];
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

void _exp_impl_f32(uint t, TensorData* data)
{
    if (t >= data->n)
    {
        return;
    }    
    float* x = (float*)data->x;
    float* z = (float*)data->z;

    z[t] = exp(x[t]);
}

void _exp_impl_f16(uint t, TensorData* data)
{
    if (t >= data->n)
    {
        return;
    }    
    half* x = (half*)data->x;
    half* z = (half*)data->z;

    z[t] = exp(x[t]);
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

void _sin_impl_f32(uint t, TensorData* data)
{
    if (t >= data->n)
    {
        return;
    }
    float* x = (float*)data->x;
    float* z = (float*)data->z;
    z[t] = sin(x[t]);
}

void _sin_impl_f16(uint t, TensorData* data)
{
    if (t >= data->n)
    {
        return;
    }
    half* x = (half*)data->x;
    half* z = (half*)data->z;
    z[t] = sin(x[t]);
}

void _cos_impl_f32(uint t, TensorData* data)
{
    if (t >= data->n)
    {
        return;
    }
    float* x = (float*)data->x;
    float* z = (float*)data->z;
    z[t] = cos(x[t]);
}

void _cos_impl_f16(uint t, TensorData* data)
{
    if (t >= data->n)
    {
        return;
    }
    half* x = (half*)data->x;
    half* z = (half*)data->z;
    z[t] = cos(x[t]);
}

void _tan_impl_f32(uint t, TensorData* data)
{
    if (t >= data->n)
    {
        return;
    }
    float* x = (float*)data->x;
    float* z = (float*)data->z;
    z[t] = tan(x[t]);
}

void _tan_impl_f16(uint t, TensorData* data)
{
    if (t >= data->n)
    {
        return;
    }
    half* x = (half*)data->x;
    half* z = (half*)data->z;
    z[t] = tan(x[t]);
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

[numthreads(64, 1, 1)] void _permute_f32(uint t : SV_DispatchThreadID, TensorPermuteData* data){ _permute_impl<float>(t, data); }
[numthreads(64, 1, 1)] void _permute_f16(uint t : SV_DispatchThreadID, TensorPermuteData* data){ _permute_impl<half>(t, data); }

[numthreads(64, 1, 1)] void _unfold_f32(uint t : SV_DispatchThreadID, TensorUnfoldData* data){ _unfold_impl<float>(t, data); }
[numthreads(64, 1, 1)] void _unfold_f16(uint t : SV_DispatchThreadID, TensorUnfoldData* data){ _unfold_impl<half>(t, data); }

[numthreads(64, 1, 1)] void _unfold_backward_f32(uint t : SV_DispatchThreadID, TensorUnfoldData* data){ _unfold_backward_impl_f32(t, data); }
[numthreads(64, 1, 1)] void _unfold_backward_f16(uint t : SV_DispatchThreadID, TensorUnfoldData* data){ _unfold_backward_impl_f16(t, data); }

[numthreads(64, 1, 1)] void _reduce_sum_f32(uint t : SV_DispatchThreadID, TensorReduceData* data){ _reduce_sum_impl_f32(t, data); }
[numthreads(64, 1, 1)] void _reduce_sum_f16(uint t : SV_DispatchThreadID, TensorReduceData* data){ _reduce_sum_impl_f16(t, data); }
[numthreads(64, 1, 1)] void _reduce_max_f32(uint t : SV_DispatchThreadID, TensorReduceData* data){ _reduce_max_impl_f32(t, data); }
[numthreads(64, 1, 1)] void _reduce_max_f16(uint t : SV_DispatchThreadID, TensorReduceData* data){ _reduce_max_impl_f16(t, data); }
[numthreads(64, 1, 1)] void _reduce_max_backward_f32(uint t : SV_DispatchThreadID, TensorReduceData* data){ _reduce_max_backward_impl_f32(t, data); }
[numthreads(64, 1, 1)] void _reduce_max_backward_f16(uint t : SV_DispatchThreadID, TensorReduceData* data){ _reduce_max_backward_impl_f16(t, data); }

[numthreads(64, 1, 1)] void _expand_f32(uint t : SV_DispatchThreadID, TensorReduceData* data){ _expand_impl<float>(t, data); }
[numthreads(64, 1, 1)] void _expand_f16(uint t : SV_DispatchThreadID, TensorReduceData* data){ _expand_impl<half>(t, data); }

[numthreads(64, 1, 1)] void _mT_f32(uint t : SV_DispatchThreadID, TensorTransposeData* data){ _mT_impl<float>(t, data); }
[numthreads(64, 1, 1)] void _mT_f16(uint t : SV_DispatchThreadID, TensorTransposeData* data){ _mT_impl<half>(t, data); }

[numthreads(64, 1, 1)] void _matmul_f32(uint t : SV_DispatchThreadID, TensorMatMulData* data){ _matmul_impl_f32(t, data); }
[numthreads(64, 1, 1)] void _matmul_f16(uint t : SV_DispatchThreadID, TensorMatMulData* data){ _matmul_impl_f16(t, data); }

[numthreads(1, 1, 1)] void _dot_f32(uint t : SV_DispatchThreadID, TensorData* data){ _dot_impl_f32(t, data); }
[numthreads(1, 1, 1)] void _dot_f16(uint t : SV_DispatchThreadID, TensorData* data){ _dot_impl_f16(t, data); }

[numthreads(64, 1, 1)] void _pow_f32(uint t : SV_DispatchThreadID, TensorData* data){ _pow_impl_f32(t, data); }
[numthreads(64, 1, 1)] void _pow_f16(uint t : SV_DispatchThreadID, TensorData* data){ _pow_impl_f16(t, data); }

[numthreads(64, 1, 1)] void _exp_f32(uint t : SV_DispatchThreadID, TensorData* data){ _exp_impl_f32(t, data); }
[numthreads(64, 1, 1)] void _exp_f16(uint t : SV_DispatchThreadID, TensorData* data){ _exp_impl_f16(t, data); }

[numthreads(64, 1, 1)] void _log_f32(uint t : SV_DispatchThreadID, TensorData* data){ _log_impl_f32(t, data); }
[numthreads(64, 1, 1)] void _log_f16(uint t : SV_DispatchThreadID, TensorData* data){ _log_impl_f16(t, data); }

[numthreads(64, 1, 1)] void _sin_f32(uint t : SV_DispatchThreadID, TensorData* data){ _sin_impl_f32(t, data); }
[numthreads(64, 1, 1)] void _sin_f16(uint t : SV_DispatchThreadID, TensorData* data){ _sin_impl_f16(t, data); }

[numthreads(64, 1, 1)] void _cos_f32(uint t : SV_DispatchThreadID, TensorData* data){ _cos_impl_f32(t, data); }
[numthreads(64, 1, 1)] void _cos_f16(uint t : SV_DispatchThreadID, TensorData* data){ _cos_impl_f16(t, data); }

[numthreads(64, 1, 1)] void _tan_f32(uint t : SV_DispatchThreadID, TensorData* data){ _tan_impl_f32(t, data); }
[numthreads(64, 1, 1)] void _tan_f16(uint t : SV_DispatchThreadID, TensorData* data){ _tan_impl_f16(t, data); }

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