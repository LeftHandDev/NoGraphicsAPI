#ifndef NO_GRAPHICS_API_SAMPLER_H
#define NO_GRAPHICS_API_SAMPLER_H

// Part of NoGraphicsAPI.h (which includes this at the end) — not a standalone
// header. It relies on the shared types and shader globals defined there.
#ifndef NO_GRAPHICS_API_H
#error "Include NoGraphicsAPI.h instead of Sampler.h"
#endif

// Samplers are not API objects and there is no sampler API: following the
// blog post, a sampler is declared next to the shader code that uses it, with
// the complete definition — filters, addressing, anisotropy, border color,
// comparison and lod control — in the declaration. Two forms:
//
//   STATIC_SAMPLER(name, ...)        named, global scope, full state
//   INLINE_SAMPLER(...)              expression, usable inside functions;
//                                    full state except the lod fields
//                                    (bias 0, unclamped — see below)
//
// Every declaration runs on a real hardware sampler. The state travels inside
// the SPIR-V itself: STATIC_SAMPLER packs it into the default value of a
// 64-bit specialization constant, INLINE_SAMPLER into a tagged 32-bit literal.
// At pipeline creation the implementation scans the IR for tagged values,
// creates/dedups matching VkSamplers in its internal heap, and rewires each
// declaration to its allocated slot (by specializing the constant or patching
// the literal), so the shader-visible value folds to the heap slot.
//
// INLINE_SAMPLER omits the lod fields because its literal must stay a 32-bit
// value used directly as the heap index — a 64-bit literal would be constant-
// folded through the index conversion at shader compile time and become
// unpatchable. Specialization constants cannot fold, so STATIC_SAMPLER
// carries the full 48-bit state.
//
// Reservations: uint64 specialization constants whose default has 0x4E47 in
// the top 16 bits, and uint32 literals with 0x4E in the top byte, are claimed
// by this scheme — pick other values for unrelated constants.

// Filter modes (min/mag/mip)
#define NEAREST 0
#define LINEAR 1

// Address modes (per axis)
#define WRAP 0
#define CLAMP 1
#define MIRROR 2
#define BORDER 3
#define MIRROR_CLAMP 4

// Border colors (texels outside the texture when an axis uses BORDER)
#define TRANSPARENT_BLACK 0
#define OPAQUE_BLACK 1
#define OPAQUE_WHITE 2

// Comparison (COMPARE_NONE = ordinary sampling; anything else makes the
// sampler a comparison sampler for SampleCmp*/shadow lookups)
#define COMPARE_NONE 0
#define COMPARE_NEVER 1
#define COMPARE_LESS 2
#define COMPARE_EQUAL 3
#define COMPARE_LESS_EQUAL 4
#define COMPARE_GREATER 5
#define COMPARE_NOT_EQUAL 6
#define COMPARE_GREATER_EQUAL 7
#define COMPARE_ALWAYS 8

// maxLod value meaning "no clamp" (any value >= 16 encodes as unbounded)
#define LOD_UNBOUNDED 1000.0

// Packed state, shared by both forms in the low bits:
//   bit  0     minFilter        bits 12-16  maxAnisotropy (0/1 = off .. 16)
//   bit  1     magFilter        bits 17-18  border color
//   bit  2     mipFilter        bits 19-22  compare (0 = off, else op + 1)
//   bits 3-5   addressU
//   bits 6-8   addressV         STATIC_SAMPLER only (quantized to 1/16):
//   bits 9-11  addressW         bits 23-30  lodBias + 8.0
//                               bits 31-38  minLod
//                               bits 39-46  maxLod (255 = unbounded)
#define STATIC_SAMPLER_MAGIC_MASK 0xffff000000000000ULL
#define STATIC_SAMPLER_MAGIC 0x4E47000000000000ULL // "NG"
#define INLINE_SAMPLER_MAGIC_MASK 0xff000000u
#define INLINE_SAMPLER_MAGIC 0x4E000000u

#ifndef __cplusplus

// Implementation detail: the hardware sampler heap (descriptor set 2) that
// sampler slots index into, aliased once per sampling flavor. The
// implementation owns its contents — user code never references these.
[[vk::binding(0, 2)]]
SamplerState ngapiSamplerHeap[];
[[vk::binding(0, 2)]]
SamplerComparisonState ngapiSamplerCmpHeap[];

#define NGAPI_SAMPLER_BITS(minF, magF, mipF, aU, aV, aW, maxAniso, border, compare) \
    ((minF) | ((magF) << 1) | ((mipF) << 2) | ((aU) << 3) | ((aV) << 6) | ((aW) << 9) | ((maxAniso) << 12) | ((border) << 17) | ((compare) << 19))

// A hardware sampler reached through a heap slot the host picked at pipeline
// creation. Sample()/SampleCmp() use implicit derivatives (pixel shaders
// only, and where anisotropy applies); the *Level variants take an explicit
// lod. The Cmp variants require a COMPARE_* op in the declaration.
struct HwSampler
{
    uint slot;

    float4 Sample(Texture2D<float4> tex, float2 uv)
    {
        return tex.Sample(ngapiSamplerHeap[slot], uv);
    }

    float4 SampleLevel(Texture2D<float4> tex, float2 uv, float lod = 0.0)
    {
        return tex.SampleLevel(ngapiSamplerHeap[slot], uv, lod);
    }

    float SampleCmp(Texture2D<float4> tex, float2 uv, float compareValue)
    {
        return tex.SampleCmp(ngapiSamplerCmpHeap[slot], uv, compareValue);
    }

    float SampleCmpLevelZero(Texture2D<float4> tex, float2 uv, float compareValue)
    {
        return tex.SampleCmpLevelZero(ngapiSamplerCmpHeap[slot], uv, compareValue);
    }
};

// Named sampler at global scope with the full state. The accessor is a
// function because slang does not allow globals to be initialized from
// specialization constants.
#define STATIC_SAMPLER(name, minF, magF, mipF, aU, aV, aW, maxAniso, border, compare, lodBias, minLod, maxLod) \
    [SpecializationConstant] const uint64_t name##_ngapiSlot =                                                 \
        STATIC_SAMPLER_MAGIC |                                                                                 \
        uint64_t(NGAPI_SAMPLER_BITS(minF, magF, mipF, aU, aV, aW, maxAniso, border, compare)) |                \
        (uint64_t(((lodBias) + 8.0) * 16.0 + 0.5) << 23) |                                                     \
        (uint64_t((minLod) * 16.0 + 0.5) << 31) |                                                              \
        (uint64_t(((maxLod) >= 16.0 ? 15.9375 : (maxLod)) * 16.0 + 0.5) << 39);                                \
    HwSampler name()                                                                                           \
    {                                                                                                          \
        return { uint(name##_ngapiSlot) };                                                                     \
    }

// Expression form, usable inside functions:
//   HwSampler s = INLINE_SAMPLER(LINEAR, LINEAR, NEAREST, CLAMP, CLAMP, CLAMP, 1, TRANSPARENT_BLACK, COMPARE_NONE);
#define INLINE_SAMPLER(minF, magF, mipF, aU, aV, aW, maxAniso, border, compare) \
    HwSampler(INLINE_SAMPLER_MAGIC | uint(NGAPI_SAMPLER_BITS(minF, magF, mipF, aU, aV, aW, maxAniso, border, compare)))

#endif // !__cplusplus

#endif // NO_GRAPHICS_API_SAMPLER_H
