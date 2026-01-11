#include "Graphics.h"

struct VertexOut
{
    float4 position : SV_Position;
    float2 motionVectors;
    float2 uv;
};

VertexOut main(uint vertexId: SV_VertexID, uint instanceId: SV_InstanceID, VertexData *data, PixelData *_)
{
    VertexOut outVertex;
    float4 position = data->vertices[vertexId].vertex;
    outVertex.position = mul(data->viewProjection, mul(data->instances[instanceId].model, position));

    float4 positionNj = mul(data->viewProjectionNj, mul(data->instances[instanceId].model, position));
    float4 prevPositionNj = mul(data->prevViewProjectionNj, mul(data->instances[instanceId].prevModel, position));
    outVertex.motionVectors = (positionNj.xy / positionNj.w) - (prevPositionNj.xy / prevPositionNj.w);

    outVertex.uv = data->vertices[vertexId].uv;
    
    return outVertex;
}