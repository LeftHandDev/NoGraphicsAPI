#include "Common.h"

[numthreads(64, 1, 1)]
void _add(uint t: SV_DispatchThreadID, TensorData *data)
{
    if (t >= data->n)
    {
        return;
    }
    data->z[t] = data->x[t] + data->y[t];
}

[numthreads(64, 1, 1)]
void _sub(uint t: SV_DispatchThreadID, TensorData *data)
{
    if (t >= data->n)
    {
        return;
    }
    data->z[t] = data->x[t] - data->y[t];
}

[numthreads(64, 1, 1)]
void _mul(uint t: SV_DispatchThreadID, TensorData *data)
{
    if (t >= data->n)
    {
        return;
    }
    data->z[t] = data->x[t] * data->y[t];
}

[numthreads(64, 1, 1)]
void _div(uint t: SV_DispatchThreadID, TensorData *data)
{
    if (t >= data->n)
    {
        return;
    }
    data->z[t] = data->x[t] / data->y[t];
}

[numthreads(1, 1, 1)]
void _dot(uint t: SV_DispatchThreadID, TensorData *data)
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

[numthreads(1, 1, 1)]
void _mT(uint t: SV_DispatchThreadID, TensorTransposeData *data)
{
    if (t >= data->n)
    {
        return;
    }

    // a b c
    // d e f

    // a d
    // b e
    // c f

    // 0 1 2
    // 3 4 5

    // 0 3
    // 1 4
    // 2 5

    // 3
    for (int i = 0; i < data->w; i++)
    {
        // 2
        for (int j = 0; j < data->h; j++)
        {
            data->y[j * data->h + i] = data->x[i * data->w + j];
        }
    }
}
