#include "Common.h"

[numthreads(64, 1, 1)] void _add(uint t : SV_DispatchThreadID, TensorData* data)
{
    if (t >= data->n)
    {
        return;
    }
    data->z[t] = data->x[t] + data->y[t];
}

    [numthreads(64, 1, 1)] void _sub(uint t : SV_DispatchThreadID, TensorData* data)
{
    if (t >= data->n)
    {
        return;
    }
    data->z[t] = data->x[t] - data->y[t];
}

[numthreads(64, 1, 1)] void _mul(uint t : SV_DispatchThreadID, TensorData* data)
{
    if (t >= data->n)
    {
        return;
    }
    data->z[t] = data->x[t] * data->y[t];
}

    [numthreads(64, 1, 1)] void _div(uint t : SV_DispatchThreadID, TensorData* data)
{
    if (t >= data->n)
    {
        return;
    }
    data->z[t] = data->x[t] / data->y[t];
}

[numthreads(1, 1, 1)] void _dot(uint t : SV_DispatchThreadID, TensorData* data)
{
    if (t >= data->n)
    {
        return;
    }

    float result = 0;
    for (int i = 0; i < data->n; i++)
    {
        result += data->x[i] * data->y[i];
    }
    data->z[0] = result;
}

    [numthreads(1, 1, 1)] void _mT(uint t : SV_DispatchThreadID, TensorTransposeData* data)
{
    if (t >= data->n)
    {
        return;
    }

    for (int i = 0; i < data->r; i++)
    {
        for (int j = 0; j < data->c; j++)
        {
            data->y[j * data->r + i] = data->x[i * data->c + j];
        }
    }
}

[numthreads(64, 1, 1)] void _matmul(uint t : SV_DispatchThreadID, TensorMatMulData* data)
{
    if (t >= data->n)
    {
        return;
    }

    uint64_t x_row = t / data->c;
    uint64_t y_col = t % data->c;

    float d = 0;
    for (int i = 0; i < data->b; i++)
    {
        d += data->x[x_row * data->b + i] * data->y[data->c * i + y_col];
    }

    data->z[t] = d;
}

    [numthreads(64, 1, 1)] void _pow(uint t : SV_DispatchThreadID, TensorData* data)
{
    if (t >= data->n)
    {
        return;
    }
    data->z[t] = pow(data->x[t], data->y[t]);
}

[numthreads(64, 1, 1)] void _tanh(uint t : SV_DispatchThreadID, TensorData* data)
{
    if (t >= data->n)
    {
        return;
    }
    data->z[t] = tanh(data->x[t]);
}
