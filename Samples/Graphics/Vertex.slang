#include "Graphics.h"

struct VertexOut
{
    float4 position : SV_Position;
    float2 uv;
};

VertexOut main(uint vertexId: SV_VertexID, uint instanceId: SV_InstanceID, VertexData *data, PixelData *_)
{
    VertexOut outVertex;
    float4 position = data->vertices[vertexId].vertex;
    outVertex.position = mul(data->viewProjection, mul(data->instances[instanceId].model, position));
    outVertex.uv = data->vertices[vertexId].uv;
    return outVertex;
}