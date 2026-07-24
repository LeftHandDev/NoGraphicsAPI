#include "Common.h"
#include "../common/Random.h"

static const float PI = 3.1415926535897932384626433832795f;

static const uint DISPATCH_X = 65535u * 64u;
uint _lin(uint3 gtid)
{
    return gtid.x + gtid.y * DISPATCH_X;
}


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
    T* y = (T*)data->y; // (OH, OW, C, k, k)

    uint k = data->k;

    // decode output index t over (OH, OW, C, k, k)
    uint dx = t % k;
    uint r1 = t / k;
    uint dy = r1 % k;
    uint r2 = r1 / k;
    uint c = r2 % data->c;
    uint r3 = r2 / data->c;
    uint col = r3 % data->ow;
    uint row = r3 / data->ow;

    int sy = int(row) * int(data->stride) + int(dy) - int(data->pad);
    int sx = int(col) * int(data->stride) + int(dx) - int(data->pad);

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
    float* x = (float*)data->x; // grad_out (OH, OW, C, k, k)
    float* y = (float*)data->y; // grad_in  (H, W, C)

    uint k = data->k;

    // decode input index t over (H, W, C)
    uint c = t % data->c;
    uint r1 = t / data->c;
    uint col = r1 % data->w;
    uint row = r1 / data->w;

    int H = int(data->h);
    int W = int(data->w);
    int OH = int(data->oh);
    int OW = int(data->ow);
    int pad = int(data->pad);
    int stride = int(data->stride);
    int i = int(row);
    int j = int(col);

    // source coords on each axis that map to this input pixel. candidate 0 is the
    // direct (in-bounds) read; 1/2 are mirrored reads that only exist in reflect mode.
    int cy[3];
    bool cy_ok[3];
    cy[0] = i;               cy_ok[0] = true;
    cy[1] = -i;              cy_ok[1] = (data->mode == 1u) && (i > 0);
    cy[2] = 2 * (H - 1) - i; cy_ok[2] = (data->mode == 1u) && (i < H - 1);
    int cx[3];
    bool cx_ok[3];
    cx[0] = j;               cx_ok[0] = true;
    cx[1] = -j;              cx_ok[1] = (data->mode == 1u) && (j > 0);
    cx[2] = 2 * (W - 1) - j; cx_ok[2] = (data->mode == 1u) && (j < W - 1);

    float acc = 0.f;

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
                // forward: source = oy*stride + dy - pad; invert for oy.
                int numy = cy[a] - int(dy) + pad;
                if (numy < 0 || (numy % stride) != 0)
                {
                    continue;
                }
                int oy = numy / stride;
                if (oy < 0 || oy >= OH)
                {
                    continue;
                }
                for (int b = 0; b < 3; b++)
                {
                    if (!cx_ok[b])
                    {
                        continue;
                    }
                    int numx = cx[b] - int(dx) + pad;
                    if (numx < 0 || (numx % stride) != 0)
                    {
                        continue;
                    }
                    int ox = numx / stride;
                    if (ox < 0 || ox >= OW)
                    {
                        continue;
                    }
                    uint go = (((uint(oy) * data->ow + uint(ox)) * data->c + c) * k + dy) * k + dx;
                    acc += x[go];
                }
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
    half* x = (half*)data->x; // grad_out (OH, OW, C, k, k)
    half* y = (half*)data->y; // grad_in  (H, W, C)

    uint k = data->k;

    // decode input index t over (H, W, C)
    uint c = t % data->c;
    uint r1 = t / data->c;
    uint col = r1 % data->w;
    uint row = r1 / data->w;

    int H = int(data->h);
    int W = int(data->w);
    int OH = int(data->oh);
    int OW = int(data->ow);
    int pad = int(data->pad);
    int stride = int(data->stride);
    int i = int(row);
    int j = int(col);

    int cy[3];
    bool cy_ok[3];
    cy[0] = i;               cy_ok[0] = true;
    cy[1] = -i;              cy_ok[1] = (data->mode == 1u) && (i > 0);
    cy[2] = 2 * (H - 1) - i; cy_ok[2] = (data->mode == 1u) && (i < H - 1);
    int cx[3];
    bool cx_ok[3];
    cx[0] = j;               cx_ok[0] = true;
    cx[1] = -j;              cx_ok[1] = (data->mode == 1u) && (j > 0);
    cx[2] = 2 * (W - 1) - j; cx_ok[2] = (data->mode == 1u) && (j < W - 1);

    float acc = 0.f;

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
                int numy = cy[a] - int(dy) + pad;
                if (numy < 0 || (numy % stride) != 0)
                {
                    continue;
                }
                int oy = numy / stride;
                if (oy < 0 || oy >= OH)
                {
                    continue;
                }
                for (int b = 0; b < 3; b++)
                {
                    if (!cx_ok[b])
                    {
                        continue;
                    }
                    int numx = cx[b] - int(dx) + pad;
                    if (numx < 0 || (numx % stride) != 0)
                    {
                        continue;
                    }
                    int ox = numx / stride;
                    if (ox < 0 || ox >= OW)
                    {
                        continue;
                    }
                    uint go = (((uint(oy) * data->ow + uint(ox)) * data->c + c) * k + dy) * k + dx;
                    acc += float(x[go]);
                }
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


groupshared float g_reduce_block[256];

void _reduce_block_impl_f32(uint3 gid, uint3 tid, TensorReduceBlockData* data)
{
    uint t = tid.x;  // lane within the workgroup (0..255)
    uint g = gid.x;  // axis group
    uint s = gid.y;  // segment index = o * inner + j
    uint o = uint(s / data->inner);
    uint j = uint(s % data->inner);

    uint a = g * 256 + t;
    float v = 0.f;
    if (a < data->axis)
    {
        uint64_t idx = (uint64_t(o) * data->axis + a) * data->inner + j;
        v = ((float*)data->x)[idx];
    }
    g_reduce_block[t] = v;
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = 128; stride > 0; stride >>= 1)
    {
        if (t < stride)
        {
            g_reduce_block[t] += g_reduce_block[t + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (t == 0)
    {
        uint64_t outIdx = (uint64_t(o) * data->groups + g) * data->inner + j;
        ((float*)data->y)[outIdx] = g_reduce_block[0];
    }
}

void _reduce_block_impl_f16(uint3 gid, uint3 tid, TensorReduceBlockData* data)
{
    uint t = tid.x;
    uint g = gid.x;
    uint s = gid.y;
    uint o = uint(s / data->inner);
    uint j = uint(s % data->inner);

    uint a = g * 256 + t;
    float v = 0.f;
    if (a < data->axis)
    {
        uint64_t idx = (uint64_t(o) * data->axis + a) * data->inner + j;
        v = float(((half*)data->x)[idx]);
    }
    g_reduce_block[t] = v;
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = 128; stride > 0; stride >>= 1)
    {
        if (t < stride)
        {
            g_reduce_block[t] += g_reduce_block[t + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (t == 0)
    {
        uint64_t outIdx = (uint64_t(o) * data->groups + g) * data->inner + j;
        ((half*)data->y)[outIdx] = half(g_reduce_block[0]);
    }
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

// concat forward: scatter a narrow slab into the wide combined tensor at `offset`.
void _concat_scatter_impl<T : IArithmetic>(uint t, TensorConcatData *data)
{
    if (t >= data->n)
    {
        return;
    }
    T* x = (T*)data->x; // slab   (outer, axis, inner)
    T* y = (T*)data->y; // combined (outer, total, inner)

    uint j = uint(t % data->inner);
    uint k = uint((t / data->inner) % data->axis);
    uint o = uint(t / (data->axis * data->inner));

    uint64_t dst = (uint64_t(o) * data->total + (uint64_t(k) + data->offset)) * data->inner + uint64_t(j);
    y[dst] = x[t];
}

// concat backward: gather this slab's grad back out of the wide grad_out at `offset`.
void _concat_gather_impl<T : IArithmetic>(uint t, TensorConcatData *data)
{
    if (t >= data->n)
    {
        return;
    }
    T* x = (T*)data->x; // grad_out (outer, total, inner)
    T* y = (T*)data->y; // grad_in  (outer, axis,  inner)

    uint j = uint(t % data->inner);
    uint k = uint((t / data->inner) % data->axis);
    uint o = uint(t / (data->axis * data->inner));

    uint64_t src = (uint64_t(o) * data->total + (uint64_t(k) + data->offset)) * data->inner + uint64_t(j);
    y[t] = x[src];
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

[shader("compute")]
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

// Shared-memory tiled GEMM: z(a,c) = x(a,b) * y(b,c), row-major.
// 16x16 output tile per workgroup; accumulate in fp32 for both dtypes.
#define MATMUL_TILE 16
groupshared float g_matmul_tileA[MATMUL_TILE * MATMUL_TILE];
groupshared float g_matmul_tileB[MATMUL_TILE * MATMUL_TILE];

void _matmul_tiled_f32(uint3 gid, uint3 tid, TensorMatMulData* data)
{
    float* x = (float*)data->x;
    float* y = (float*)data->y;
    float* z = (float*)data->z;

    uint A = data->a;
    uint B = data->b;
    uint C = data->c;

    uint col = gid.x * MATMUL_TILE + tid.x; // output col in [0, C)
    uint tiles = (B + MATMUL_TILE - 1) / MATMUL_TILE;

    // Row tiles can exceed the 65535 grid.y limit; stride over them so a capped
    // grid still covers every row (the host caps grid.y at the same value).
    uint rowTiles = (A + MATMUL_TILE - 1) / MATMUL_TILE;
    uint yStride = min(rowTiles, 65535u);
    for (uint ry = gid.y; ry < rowTiles; ry += yStride)
    {
        uint row = ry * MATMUL_TILE + tid.y; // output row in [0, A)

        float acc = 0.f;
        for (uint tt = 0; tt < tiles; tt++)
        {
            uint aCol = tt * MATMUL_TILE + tid.x;
            g_matmul_tileA[tid.y * MATMUL_TILE + tid.x] = (row < A && aCol < B) ? x[row * B + aCol] : 0.f;

            uint bRow = tt * MATMUL_TILE + tid.y;
            g_matmul_tileB[tid.y * MATMUL_TILE + tid.x] = (bRow < B && col < C) ? y[bRow * C + col] : 0.f;

            GroupMemoryBarrierWithGroupSync();

            for (uint kk = 0; kk < MATMUL_TILE; kk++)
            {
                acc += g_matmul_tileA[tid.y * MATMUL_TILE + kk] * g_matmul_tileB[kk * MATMUL_TILE + tid.x];
            }

            GroupMemoryBarrierWithGroupSync();
        }

        if (row < A && col < C)
        {
            z[row * C + col] = acc;
        }
    }
}

void _matmul_tiled_f16(uint3 gid, uint3 tid, TensorMatMulData* data)
{
    half* x = (half*)data->x;
    half* y = (half*)data->y;
    half* z = (half*)data->z;

    uint A = data->a;
    uint B = data->b;
    uint C = data->c;

    uint col = gid.x * MATMUL_TILE + tid.x; // output col in [0, C)
    uint tiles = (B + MATMUL_TILE - 1) / MATMUL_TILE;

    uint rowTiles = (A + MATMUL_TILE - 1) / MATMUL_TILE;
    uint yStride = min(rowTiles, 65535u);
    for (uint ry = gid.y; ry < rowTiles; ry += yStride)
    {
        uint row = ry * MATMUL_TILE + tid.y; // output row in [0, A)

        float acc = 0.f;
        for (uint tt = 0; tt < tiles; tt++)
        {
            uint aCol = tt * MATMUL_TILE + tid.x;
            g_matmul_tileA[tid.y * MATMUL_TILE + tid.x] = (row < A && aCol < B) ? float(x[row * B + aCol]) : 0.f;

            uint bRow = tt * MATMUL_TILE + tid.y;
            g_matmul_tileB[tid.y * MATMUL_TILE + tid.x] = (bRow < B && col < C) ? float(y[bRow * C + col]) : 0.f;

            GroupMemoryBarrierWithGroupSync();

            for (uint kk = 0; kk < MATMUL_TILE; kk++)
            {
                acc += g_matmul_tileA[tid.y * MATMUL_TILE + kk] * g_matmul_tileB[kk * MATMUL_TILE + tid.x];
            }

            GroupMemoryBarrierWithGroupSync();
        }

        if (row < A && col < C)
        {
            z[row * C + col] = half(acc);
        }
    }
}

// Split-K GEMM: partitions the contraction (b) across grid.z workgroups, each
// summing its slice into a partial at z[(split, a, c)]. A separate reduction
// (reduce_block over the split axis) then sums the partials into the final
// (a, c). Used when the output is too small to fill the GPU but the contraction
// is huge (e.g. weight gradients: cols^T @ grad, contraction M = H*W).
void _matmul_splitk_impl_f32(uint3 gid, uint3 tid, TensorMatMulData* data)
{
    float* x = (float*)data->x; // A (a, b)
    float* y = (float*)data->y; // B (b, c)
    float* z = (float*)data->z; // partials (splits, a, c)

    uint A = data->a;
    uint B = data->b;
    uint C = data->c;
    uint S = data->splits;
    uint s = gid.z;

    uint row = gid.y * MATMUL_TILE + tid.y;
    uint col = gid.x * MATMUL_TILE + tid.x;

    uint bTiles = (B + MATMUL_TILE - 1) / MATMUL_TILE;
    uint tilesPerSplit = (bTiles + S - 1) / S;
    uint startTile = s * tilesPerSplit;
    uint endTile = min(startTile + tilesPerSplit, bTiles);

    float acc = 0.f;
    for (uint tt = startTile; tt < endTile; tt++)
    {
        uint aCol = tt * MATMUL_TILE + tid.x;
        // transposeA: x is stored (b, a) (e.g. cols), so A[row, aCol] = x[aCol*a + row].
        float av = 0.f;
        if (data->transposeA != 0u)
        {
            av = (row < A && aCol < B) ? x[aCol * A + row] : 0.f;
        }
        else
        {
            av = (row < A && aCol < B) ? x[row * B + aCol] : 0.f;
        }
        g_matmul_tileA[tid.y * MATMUL_TILE + tid.x] = av;

        uint bRow = tt * MATMUL_TILE + tid.y;
        g_matmul_tileB[tid.y * MATMUL_TILE + tid.x] = (bRow < B && col < C) ? y[bRow * C + col] : 0.f;

        GroupMemoryBarrierWithGroupSync();

        for (uint kk = 0; kk < MATMUL_TILE; kk++)
        {
            acc += g_matmul_tileA[tid.y * MATMUL_TILE + kk] * g_matmul_tileB[kk * MATMUL_TILE + tid.x];
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (row < A && col < C)
    {
        uint64_t outIdx = (uint64_t(s) * A + row) * C + col;
        z[outIdx] = acc;
    }
}

void _matmul_splitk_impl_f16(uint3 gid, uint3 tid, TensorMatMulData* data)
{
    half* x = (half*)data->x;
    half* y = (half*)data->y;
    half* z = (half*)data->z;

    uint A = data->a;
    uint B = data->b;
    uint C = data->c;
    uint S = data->splits;
    uint s = gid.z;

    uint row = gid.y * MATMUL_TILE + tid.y;
    uint col = gid.x * MATMUL_TILE + tid.x;

    uint bTiles = (B + MATMUL_TILE - 1) / MATMUL_TILE;
    uint tilesPerSplit = (bTiles + S - 1) / S;
    uint startTile = s * tilesPerSplit;
    uint endTile = min(startTile + tilesPerSplit, bTiles);

    float acc = 0.f;
    for (uint tt = startTile; tt < endTile; tt++)
    {
        uint aCol = tt * MATMUL_TILE + tid.x;
        float av = 0.f;
        if (data->transposeA != 0u)
        {
            av = (row < A && aCol < B) ? float(x[aCol * A + row]) : 0.f;
        }
        else
        {
            av = (row < A && aCol < B) ? float(x[row * B + aCol]) : 0.f;
        }
        g_matmul_tileA[tid.y * MATMUL_TILE + tid.x] = av;

        uint bRow = tt * MATMUL_TILE + tid.y;
        g_matmul_tileB[tid.y * MATMUL_TILE + tid.x] = (bRow < B && col < C) ? float(y[bRow * C + col]) : 0.f;

        GroupMemoryBarrierWithGroupSync();

        for (uint kk = 0; kk < MATMUL_TILE; kk++)
        {
            acc += g_matmul_tileA[tid.y * MATMUL_TILE + kk] * g_matmul_tileB[kk * MATMUL_TILE + tid.x];
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (row < A && col < C)
    {
        uint64_t outIdx = (uint64_t(s) * A + row) * C + col;
        z[outIdx] = half(acc);
    }
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

[shader("compute")]
[numthreads(64, 1, 1)]
void _fp32_fp16(uint3 _g : SV_DispatchThreadID, TensorData* data)
{
    uint t = _lin(_g);
    if (t >= data->n)
    {
        return;
    }
    half* x = (half*)data->x;
    float* z = (float*)data->z;
    z[t] = x[t];
}

[shader("compute")]
[numthreads(64, 1, 1)]
void _fp16_fp32(uint3 _g : SV_DispatchThreadID, TensorData* data)
{
    uint t = _lin(_g);
    if (t >= data->n)
    {
        return;
    }
    float* x = (float*)data->x;
    half* z = (half*)data->z;
    z[t] = (half)x[t];
}

[shader("compute")]
[numthreads(64, 1, 1)] void _rand_f32(uint3 _g : SV_DispatchThreadID, TensorData* data)
{
    uint t = _lin(_g);
    if (t >= data->n)
    {
        return;
    }
    float* z = (float*)data->z;
    z[t] = _rand_uniform(t, (uint)data->m);
}

[shader("compute")]
[numthreads(64, 1, 1)] void _rand_f16(uint3 _g : SV_DispatchThreadID, TensorData* data)
{
    uint t = _lin(_g);
    if (t >= data->n)
    {
        return;
    }
    half* z = (half*)data->z;
    z[t] = (half)_rand_uniform(t, (uint)data->m);
}

[shader("compute")][numthreads(64, 1, 1)] void _add_f32(uint3 _g : SV_DispatchThreadID, TensorData* data){ _add_impl<float>(_lin(_g), data); }
[shader("compute")][numthreads(64, 1, 1)] void _add_f16(uint3 _g : SV_DispatchThreadID, TensorData* data){ _add_impl<half>(_lin(_g), data); }

[shader("compute")][numthreads(64, 1, 1)] void _sub_f32(uint3 _g : SV_DispatchThreadID, TensorData* data){ _sub_impl<float>(_lin(_g), data); }
[shader("compute")][numthreads(64, 1, 1)] void _sub_f16(uint3 _g : SV_DispatchThreadID, TensorData* data){ _sub_impl<half>(_lin(_g), data); }

[shader("compute")][numthreads(64, 1, 1)] void _mul_f32(uint3 _g : SV_DispatchThreadID, TensorData* data){ _mul_impl<float>(_lin(_g), data); }
[shader("compute")][numthreads(64, 1, 1)] void _mul_f16(uint3 _g : SV_DispatchThreadID, TensorData* data){ _mul_impl<half>(_lin(_g), data); }

[shader("compute")][numthreads(64, 1, 1)] void _div_f32(uint3 _g : SV_DispatchThreadID, TensorData* data){ _div_impl<float>(_lin(_g), data); }
[shader("compute")][numthreads(64, 1, 1)] void _div_f16(uint3 _g : SV_DispatchThreadID, TensorData* data){ _div_impl<half>(_lin(_g), data); }

[shader("compute")][numthreads(64, 1, 1)] void _permute_f32(uint3 _g : SV_DispatchThreadID, TensorPermuteData* data){ _permute_impl<float>(_lin(_g), data); }
[shader("compute")][numthreads(64, 1, 1)] void _permute_f16(uint3 _g : SV_DispatchThreadID, TensorPermuteData* data){ _permute_impl<half>(_lin(_g), data); }

[shader("compute")][numthreads(64, 1, 1)] void _unfold_f32(uint3 _g : SV_DispatchThreadID, TensorUnfoldData* data){ _unfold_impl<float>(_lin(_g), data); }
[shader("compute")][numthreads(64, 1, 1)] void _unfold_f16(uint3 _g : SV_DispatchThreadID, TensorUnfoldData* data){ _unfold_impl<half>(_lin(_g), data); }

[shader("compute")][numthreads(64, 1, 1)] void _unfold_backward_f32(uint3 _g : SV_DispatchThreadID, TensorUnfoldData* data){ _unfold_backward_impl_f32(_lin(_g), data); }
[shader("compute")][numthreads(64, 1, 1)] void _unfold_backward_f16(uint3 _g : SV_DispatchThreadID, TensorUnfoldData* data){ _unfold_backward_impl_f16(_lin(_g), data); }

[shader("compute")][numthreads(64, 1, 1)] void _reduce_sum_f32(uint3 _g : SV_DispatchThreadID, TensorReduceData* data){ _reduce_sum_impl_f32(_lin(_g), data); }
[shader("compute")][numthreads(64, 1, 1)] void _reduce_sum_f16(uint3 _g : SV_DispatchThreadID, TensorReduceData* data){ _reduce_sum_impl_f16(_lin(_g), data); }
[shader("compute")][numthreads(256, 1, 1)] void _reduce_block_f32(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, TensorReduceBlockData* data){ _reduce_block_impl_f32(gid, tid, data); }
[shader("compute")][numthreads(256, 1, 1)] void _reduce_block_f16(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, TensorReduceBlockData* data){ _reduce_block_impl_f16(gid, tid, data); }
[shader("compute")][numthreads(64, 1, 1)] void _reduce_max_f32(uint3 _g : SV_DispatchThreadID, TensorReduceData* data){ _reduce_max_impl_f32(_lin(_g), data); }
[shader("compute")][numthreads(64, 1, 1)] void _reduce_max_f16(uint3 _g : SV_DispatchThreadID, TensorReduceData* data){ _reduce_max_impl_f16(_lin(_g), data); }
[shader("compute")][numthreads(64, 1, 1)] void _reduce_max_backward_f32(uint3 _g : SV_DispatchThreadID, TensorReduceData* data){ _reduce_max_backward_impl_f32(_lin(_g), data); }
[shader("compute")][numthreads(64, 1, 1)] void _reduce_max_backward_f16(uint3 _g : SV_DispatchThreadID, TensorReduceData* data){ _reduce_max_backward_impl_f16(_lin(_g), data); }

[shader("compute")][numthreads(64, 1, 1)] void _expand_f32(uint3 _g : SV_DispatchThreadID, TensorReduceData* data){ _expand_impl<float>(_lin(_g), data); }
[shader("compute")][numthreads(64, 1, 1)] void _expand_f16(uint3 _g : SV_DispatchThreadID, TensorReduceData* data){ _expand_impl<half>(_lin(_g), data); }

[shader("compute")][numthreads(64, 1, 1)] void _concat_scatter_f32(uint3 _g : SV_DispatchThreadID, TensorConcatData* data){ _concat_scatter_impl<float>(_lin(_g), data); }
[shader("compute")][numthreads(64, 1, 1)] void _concat_scatter_f16(uint3 _g : SV_DispatchThreadID, TensorConcatData* data){ _concat_scatter_impl<half>(_lin(_g), data); }
[shader("compute")][numthreads(64, 1, 1)] void _concat_gather_f32(uint3 _g : SV_DispatchThreadID, TensorConcatData* data){ _concat_gather_impl<float>(_lin(_g), data); }
[shader("compute")][numthreads(64, 1, 1)] void _concat_gather_f16(uint3 _g : SV_DispatchThreadID, TensorConcatData* data){ _concat_gather_impl<half>(_lin(_g), data); }

[shader("compute")][numthreads(64, 1, 1)] void _mT_f32(uint3 _g : SV_DispatchThreadID, TensorTransposeData* data){ _mT_impl<float>(_lin(_g), data); }
[shader("compute")][numthreads(64, 1, 1)] void _mT_f16(uint3 _g : SV_DispatchThreadID, TensorTransposeData* data){ _mT_impl<half>(_lin(_g), data); }

[shader("compute")][numthreads(16, 16, 1)] void _matmul_f32(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, TensorMatMulData* data){ _matmul_tiled_f32(gid, tid, data); }
[shader("compute")][numthreads(16, 16, 1)] void _matmul_f16(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, TensorMatMulData* data){ _matmul_tiled_f16(gid, tid, data); }
[shader("compute")][numthreads(16, 16, 1)] void _matmul_splitk_f32(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, TensorMatMulData* data){ _matmul_splitk_impl_f32(gid, tid, data); }
[shader("compute")][numthreads(16, 16, 1)] void _matmul_splitk_f16(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, TensorMatMulData* data){ _matmul_splitk_impl_f16(gid, tid, data); }

[shader("compute")][numthreads(1, 1, 1)] void _dot_f32(uint t : SV_DispatchThreadID, TensorData* data){ _dot_impl_f32(t, data); }
[shader("compute")][numthreads(1, 1, 1)] void _dot_f16(uint t : SV_DispatchThreadID, TensorData* data){ _dot_impl_f16(t, data); }

[shader("compute")][numthreads(64, 1, 1)] void _pow_f32(uint3 _g : SV_DispatchThreadID, TensorData* data){ _pow_impl_f32(_lin(_g), data); }
[shader("compute")][numthreads(64, 1, 1)] void _pow_f16(uint3 _g : SV_DispatchThreadID, TensorData* data){ _pow_impl_f16(_lin(_g), data); }

[shader("compute")][numthreads(64, 1, 1)] void _exp_f32(uint3 _g : SV_DispatchThreadID, TensorData* data){ _exp_impl_f32(_lin(_g), data); }
[shader("compute")][numthreads(64, 1, 1)] void _exp_f16(uint3 _g : SV_DispatchThreadID, TensorData* data){ _exp_impl_f16(_lin(_g), data); }

[shader("compute")][numthreads(64, 1, 1)] void _log_f32(uint3 _g : SV_DispatchThreadID, TensorData* data){ _log_impl_f32(_lin(_g), data); }
[shader("compute")][numthreads(64, 1, 1)] void _log_f16(uint3 _g : SV_DispatchThreadID, TensorData* data){ _log_impl_f16(_lin(_g), data); }

[shader("compute")][numthreads(64, 1, 1)] void _sin_f32(uint3 _g : SV_DispatchThreadID, TensorData* data){ _sin_impl_f32(_lin(_g), data); }
[shader("compute")][numthreads(64, 1, 1)] void _sin_f16(uint3 _g : SV_DispatchThreadID, TensorData* data){ _sin_impl_f16(_lin(_g), data); }

[shader("compute")][numthreads(64, 1, 1)] void _cos_f32(uint3 _g : SV_DispatchThreadID, TensorData* data){ _cos_impl_f32(_lin(_g), data); }
[shader("compute")][numthreads(64, 1, 1)] void _cos_f16(uint3 _g : SV_DispatchThreadID, TensorData* data){ _cos_impl_f16(_lin(_g), data); }

[shader("compute")][numthreads(64, 1, 1)] void _tan_f32(uint3 _g : SV_DispatchThreadID, TensorData* data){ _tan_impl_f32(_lin(_g), data); }
[shader("compute")][numthreads(64, 1, 1)] void _tan_f16(uint3 _g : SV_DispatchThreadID, TensorData* data){ _tan_impl_f16(_lin(_g), data); }

[shader("compute")][numthreads(64, 1, 1)] void _cosh_f32(uint3 _g : SV_DispatchThreadID, TensorData* data){ _cosh_impl_f32(_lin(_g), data); }
[shader("compute")][numthreads(64, 1, 1)] void _cosh_f16(uint3 _g : SV_DispatchThreadID, TensorData* data){ _cosh_impl_f16(_lin(_g), data); }

[shader("compute")][numthreads(64, 1, 1)] void _tanh_f32(uint3 _g : SV_DispatchThreadID, TensorData* data){ _tanh_impl_f32(_lin(_g), data); }
[shader("compute")][numthreads(64, 1, 1)] void _tanh_f16(uint3 _g : SV_DispatchThreadID, TensorData* data){ _tanh_impl_f16(_lin(_g), data); }

[shader("compute")][numthreads(64, 1, 1)] void _relu_f32(uint3 _g : SV_DispatchThreadID, TensorData* data){ _relu_impl_f32(_lin(_g), data); }
[shader("compute")][numthreads(64, 1, 1)] void _relu_f16(uint3 _g : SV_DispatchThreadID, TensorData* data){ _relu_impl_f16(_lin(_g), data); }

[shader("compute")][numthreads(64, 1, 1)] void _relu_backward_f32(uint3 _g : SV_DispatchThreadID, TensorData* data){ _relu_backward_impl_f32(_lin(_g), data); }
[shader("compute")][numthreads(64, 1, 1)] void _relu_backward_f16(uint3 _g : SV_DispatchThreadID, TensorData* data){ _relu_backward_impl_f16(_lin(_g), data); }

[shader("compute")][numthreads(64, 1, 1)] void _gelu_f32(uint3 _g : SV_DispatchThreadID, TensorData* data){ _gelu_impl_f32(_lin(_g), data); }
[shader("compute")][numthreads(64, 1, 1)] void _gelu_f16(uint3 _g : SV_DispatchThreadID, TensorData* data){ _gelu_impl_f16(_lin(_g), data); }

[shader("compute")][numthreads(64, 1, 1)] void _gelu_backward_f32(uint3 _g : SV_DispatchThreadID, TensorData* data){ _gelu_backward_impl_f32(_lin(_g), data); }
[shader("compute")][numthreads(64, 1, 1)] void _gelu_backward_f16(uint3 _g : SV_DispatchThreadID, TensorData* data){ _gelu_backward_impl_f16(_lin(_g), data); }

[shader("compute")][numthreads(64, 1, 1)] void _adam_f32(uint3 _g : SV_DispatchThreadID, TensorAdamData* data){ _adam_impl_f32(_lin(_g), data); }
[shader("compute")][numthreads(64, 1, 1)] void _adam_f16(uint3 _g : SV_DispatchThreadID, TensorAdamData* data){ _adam_impl_f16(_lin(_g), data); }