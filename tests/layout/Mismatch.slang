#include "Mismatch.h"

[numthreads(1, 1, 1)] void main(uint3 threadId : SV_DispatchThreadID, MismatchData* data)
{
    data->items[threadId.x].position.x = float(data->flags->after + (data->flags->enabled ? 1 : 0)) + data->shifted->item.position.y;
}
