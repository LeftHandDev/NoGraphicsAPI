#include "Cube.h"

struct VertexOut
{
    float4 position : SV_Position;
    float4 color : COLOR0;
};

VertexOut main(uint vertexId : SV_VertexID, VertexData* data, PixelData* _data)
{
    VertexOut outVertex;
    float4 position = data->vertices[vertexId].position;
    outVertex.position = mul(data->mvp, position);
    return outVertex;
}