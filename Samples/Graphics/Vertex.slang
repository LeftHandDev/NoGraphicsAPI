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
    float4 position = float4(data->vertices[vertexId], 1.0);
    outVertex.position = mul(data->viewProjection, mul(data->instances[instanceId].model, position));

    float4 positionNj = mul(data->viewProjectionNj, mul(data->instances[instanceId].model, position));
    float4 prevPositionNj = mul(data->prevViewProjectionNj, mul(data->instances[instanceId].prevModel, position));
    outVertex.motionVectors = (positionNj.xy / positionNj.w) - (prevPositionNj.xy / prevPositionNj.w);

    outVertex.uv = data->uvs[vertexId];
    
    return outVertex;
}