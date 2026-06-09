#ifndef NO_GRAPHICS_API_SAMPLER_H
#define NO_GRAPHICS_API_SAMPLER_H

// Part of NoGraphicsAPI.h (which includes this at the end) — not a standalone
// header. It relies on the shared types and shader globals defined there.
#ifndef NO_GRAPHICS_API_H
#error "Include NoGraphicsAPI.h instead of Sampler.h"
#endif

// A sampler is plain data plus shader code — not an API object and not a
// descriptor. Following the blog post, declare one inline in GPU code:
//
//     Sampler sampler = {};        // `= {}` applies the defaults below;
//     sampler.minFilter = LINEAR;  // a bare `Sampler sampler;` declaration
//     sampler.magFilter = LINEAR;  // is uninitialized in slang!
//     float4 color = sampler.SampleLevel(textureHeap[t], uv, 0);
//
// or put one in your own data struct and fill it in on the CPU — C++20
// designated initializers work there (slang has none):
//
//     data.cpu->sampler = Sampler{ .minFilter = LINEAR, .addressU = CLAMP };
//
// It is just a struct, so both work. Filtering runs in shader code; only the
// raw texel fetches go through the texture units (Load), so tiling, format
// decode and caching still come from hardware. There is no hardware sampler
// behind this.

// Filter modes
#define NEAREST 0
#define LINEAR 1

// Address modes
#define WRAP 0
#define CLAMP 1
#define MIRROR 2
#define BORDER 3

struct alignas(16) Sampler
{
    uint minFilter = LINEAR;  // applied when lod > 0 (minification)
    uint magFilter = LINEAR;  // applied when lod <= 0 (magnification)
    uint mipFilter = NEAREST; // NEAREST rounds lod to a level, LINEAR blends two
    uint addressU = WRAP;
    uint addressV = WRAP;
    uint padding[3] = { 0, 0, 0 };
    float4 borderColor = { 0, 0, 0, 0 }; // out-of-range texels when address mode is BORDER
};

#ifndef __cplusplus

// Wrapping modes are applied in float UV space, once per sample: integer
// modulo has no hardware instruction on GPUs (it expands to a ~20-op
// emulation), while frac() is a single op.
float ngapiNormalizeUV(float u, uint mode)
{
    if (mode == WRAP)
        return frac(u);
    if (mode == MIRROR)
    {
        float t = frac(u * 0.5) * 2.0; // [0,2): forward half then mirrored half
        return t < 1.0 ? t : 2.0 - t;
    }
    return u; // CLAMP / BORDER resolve out-of-range texels below
}

// Texel-space fixup. After ngapiNormalizeUV a filter footprint leaves
// [0, size-1] by at most one texel on either side, so the wrapping modes need
// only a single conditional step — callers must normalize first. Returns -1
// when the texel is outside the texture and the border color applies.
int ngapiResolveTexelAddress(int coord, int size, uint mode)
{
    switch (mode)
    {
    case WRAP:
        if (coord < 0)
            return coord + size;
        return coord >= size ? coord - size : coord;
    case MIRROR:
        if (coord < 0)
            return -coord - 1;
        return coord >= size ? 2 * size - 1 - coord : coord;
    case CLAMP:
        return clamp(coord, 0, size - 1);
    default: // BORDER
        return (coord < 0 || coord >= size) ? -1 : coord;
    }
}

float4 ngapiFetchTexel(Texture2D<float4> tex, Sampler s, int2 coord, int2 size, int mip)
{
    int x = ngapiResolveTexelAddress(coord.x, size.x, s.addressU);
    int y = ngapiResolveTexelAddress(coord.y, size.y, s.addressV);
    if (x < 0 || y < 0)
        return s.borderColor;
    return tex.Load(int3(x, y, mip));
}

// `uv` must already be normalized by ngapiNormalizeUV; `baseSize` is mip 0.
float4 ngapiSampleMip(Texture2D<float4> tex, Sampler s, float2 uv, int mip, uint filter, int2 baseSize)
{
    int2 size = max(int2(1, 1), baseSize >> mip); // Vulkan mip sizing: max(1, floor(base / 2^mip))

    if (filter == NEAREST)
    {
        int2 coord = int2(floor(uv * float2(size)));
        return ngapiFetchTexel(tex, s, coord, size, mip);
    }

    // LINEAR: fetch the 2x2 quad around the sample position and blend.
    float2 pos = uv * float2(size) - 0.5;
    float2 floorPos = floor(pos);
    int2 base = int2(floorPos);
    float2 weight = pos - floorPos;

    float4 c00 = ngapiFetchTexel(tex, s, base + int2(0, 0), size, mip);
    float4 c10 = ngapiFetchTexel(tex, s, base + int2(1, 0), size, mip);
    float4 c01 = ngapiFetchTexel(tex, s, base + int2(0, 1), size, mip);
    float4 c11 = ngapiFetchTexel(tex, s, base + int2(1, 1), size, mip);

    return lerp(lerp(c00, c10, weight.x), lerp(c01, c11, weight.x), weight.y);
}

extension Sampler
{
    float4 SampleLevel(Texture2D<float4> tex, float2 uv, float lod = 0.0)
    {
        // The only dimension query per sample; mip sizes derive from it.
        uint width, height, mipCount;
        tex.GetDimensions(0, width, height, mipCount);
        int2 baseSize = int2(int(width), int(height));

        lod = clamp(lod, 0.0, float(mipCount - 1));
        uint filter = lod <= 0.0 ? this.magFilter : this.minFilter;

        uv.x = ngapiNormalizeUV(uv.x, this.addressU);
        uv.y = ngapiNormalizeUV(uv.y, this.addressV);

        if (this.mipFilter == LINEAR)
        {
            int mip0 = int(floor(lod));
            int mip1 = min(mip0 + 1, int(mipCount) - 1);
            float mipWeight = lod - float(mip0);

            float4 color0 = ngapiSampleMip(tex, this, uv, mip0, filter, baseSize);
            if (mipWeight <= 0.0 || mip1 == mip0)
                return color0;
            return lerp(color0, ngapiSampleMip(tex, this, uv, mip1, filter, baseSize), mipWeight);
        }

        return ngapiSampleMip(tex, this, uv, int(round(lod)), filter, baseSize);
    }
}

#endif // !__cplusplus

#endif // NO_GRAPHICS_API_SAMPLER_H
