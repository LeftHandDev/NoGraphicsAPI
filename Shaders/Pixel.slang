#include "Cube.h"

struct PixelIn
{
    float4 position : SV_Position;
    float4 color : COLOR0;
};

float4 main(PixelIn inPixel : SV_Position, VertexData _data, PixelData* data) : SV_Target
{
    return inPixel.color * data->color;
}