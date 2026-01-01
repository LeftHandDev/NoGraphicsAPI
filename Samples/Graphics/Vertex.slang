#include "Graphics.h"

struct VertexOut
{
    float4 position : SV_Position;
    float2 uv;
};

VertexOut main(uint vertexId: SV_VertexID, VertexData *data, PixelData *_)
{
    VertexOut outVertex;
    float4 position = data->vertices[vertexId].position;
    outVertex.position = mul(data->mvp, position);
    outVertex.uv = data->vertices[vertexId].uv;
    return outVertex;
}