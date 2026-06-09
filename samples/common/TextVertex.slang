#include "Text.h"

struct VertexOut
{
    float4 position : SV_Position;
    float2 uv;
};

// vertexId 0 = bottom-left, 1 = bottom-right, 2 = top-right, 3 = top-left
static float2 quadVertices[4] = {
    float2(0.0, 0.0),
    float2(1.0, 0.0),
    float2(1.0, 1.0),
    float2(0.0, 1.0)
};

VertexOut main(uint vertexId: SV_VertexID, uint instanceId: SV_InstanceID, TextVertexData *data, TextPixelData *_)
{
    VertexOut outVertex;
    
    float2 quadPos = quadVertices[vertexId];
    
    // calculate position based on instance id, multiply by text width, add text height for y
    float2 position = float2(instanceId * data->textWidth, 0.0) + quadPos * float2(data->textWidth, data->textHeight);
    outVertex.position = float4(position / float2(data->width, data->height) * 2.0 - 1.0, 0.0, 1.0);

    // use the char code to calculate UVs into the atlas
    uint charCode = data->text[instanceId];
    uint charsPerRow = data->atlasWidth / data->textWidth;
    uint charX = charCode % charsPerRow;
    uint charY = charCode / charsPerRow;
    outVertex.uv = (float2(charX, charY) + quadPos) * float2(data->textWidth, data->textHeight) / float2(data->atlasWidth, data->atlasHeight);
    
    return outVertex;
}