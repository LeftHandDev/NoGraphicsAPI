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
    outVertex.position.y = -outVertex.position.y; // flip y for screen space

    // use the char code form data->text to calculate UVs, assuming a texture atlas with textWidthxtextHeight characters
    uint charCode = data->text[instanceId];
    uint charsPerRow = 256 / data->textWidth; // assuming 256 characters in the atlas
    uint charX = charCode % charsPerRow;
    uint charY = charCode / charsPerRow;
    outVertex.uv = float2(charX, charY) * float2(data->textWidth, data->textHeight) / 256.0 + quadPos * float2(data->textWidth, data->textHeight) / 256.0;
    
    return outVertex;
}