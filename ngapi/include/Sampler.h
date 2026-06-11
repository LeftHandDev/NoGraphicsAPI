#ifndef NO_GRAPHICS_API_SAMPLER_H
#define NO_GRAPHICS_API_SAMPLER_H

// Part of NoGraphicsAPI.h (which includes this at the end) — not a standalone
// header. It relies on the shared types and shader globals defined there.
#ifndef NO_GRAPHICS_API_H
#error "Include NoGraphicsAPI.h instead of Sampler.h"
#endif

// Samplers are not API objects and there is no sampler API: following the
// blog post, a sampler is declared next to the shader code that uses it.
//
//     STATIC_SAMPLER(aniso16Wrap, LINEAR, LINEAR, LINEAR, WRAP, WRAP, 16)
//     ...
//     float4 c = aniso16Wrap().Sample(textureHeap[t], uv);          // implicit lod (pixel shaders)
//     float4 d = aniso16Wrap().SampleLevel(textureHeap[t], uv, 0);  // explicit lod
//
// Each declaration runs on a real hardware sampler at full rate. The state is
// packed into the default value of a specialization constant — tagged with
// the magic high bits below so the implementation can find it by scanning the
// SPIR-V at pipeline creation, with no sidecar metadata. The implementation
// creates/dedups a matching VkSampler in its internal sampler heap and
// specializes the constant to the allocated slot, so the shader-visible value
// is the heap slot, folded to a literal at pipeline compile time.
//
// Note: a uint specialization constant whose default value has these high
// bits is claimed by this scheme; pick other defaults for unrelated
// constants.

// Filter modes
#define NEAREST 0
#define LINEAR 1

// Address modes
#define WRAP 0
#define CLAMP 1
#define MIRROR 2
#define BORDER 3

// Packed state: bit 0 minFilter, bit 1 magFilter, bit 2 mipFilter,
// bits 3-4 addressU, bits 5-6 addressV, bits 7-11 max anisotropy
// (0/1 = isotropic, up to 16; needs Sample(), not SampleLevel(), to matter).
#define STATIC_SAMPLER_MAGIC_MASK 0xffff0000
#define STATIC_SAMPLER_MAGIC 0x4e470000 // "NG"
#define STATIC_SAMPLER_STATE(minF, magF, mipF, aU, aV, maxAniso) \
    (STATIC_SAMPLER_MAGIC | (minF) | ((magF) << 1) | ((mipF) << 2) | ((aU) << 3) | ((aV) << 5) | ((maxAniso) << 7))

#ifndef __cplusplus

// Implementation detail: the hardware sampler heap (descriptor set 2) that
// static sampler slots index into. The implementation owns its contents —
// user code never references it.
[[vk::binding(0, 2)]]
SamplerState ngapiSamplerHeap[];

// A hardware sampler reached through a heap slot the host picked at pipeline
// creation. Sample() uses implicit derivatives (pixel shaders only, and the
// path where anisotropy applies); SampleLevel() takes an explicit lod.
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
};

// Declares a static sampler at global scope (see the header comment). The
// accessor is a function because slang does not allow globals to be
// initialized from specialization constants.
#define STATIC_SAMPLER(name, minF, magF, mipF, aU, aV, maxAniso)                                                     \
    [SpecializationConstant] const uint name##_ngapiSlot = STATIC_SAMPLER_STATE(minF, magF, mipF, aU, aV, maxAniso); \
    HwSampler name()                                                                                                 \
    {                                                                                                                \
        return { name##_ngapiSlot };                                                                                 \
    }

#endif // !__cplusplus

#endif // NO_GRAPHICS_API_SAMPLER_H
