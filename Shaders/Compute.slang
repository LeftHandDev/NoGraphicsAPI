#include "Compute.h"

[numthreads(64, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID, Data* data)
{
    uint value = data->input[threadId.x];
    data->output[threadId.x + data->offset] = uint(float(value) * data->color.r);
}
