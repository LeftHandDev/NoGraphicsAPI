#ifndef NO_GRAPHICS_API_H
#define NO_GRAPHICS_API_H

#include <cstdint>
#include <span>

#define GPU_DEFAULT_ALIGNMENT 16

#define GPU_DEFINE_HANDLE(object) \
    struct object##_T; \
    using object = object##_T*;

// Vector types
using uint = uint32_t;
using uint2 = struct { uint x, y; };
using uint3 = struct { uint x, y, z; };
using uint4 = struct { uint x, y, z, w; };
using float2 = struct { float x, y; };
using float3 = struct { float x, y, z; };
using float4 = struct { float x, y, z, w; };

// Use standard library span
template<typename T>
using Span = std::span<T>;

// Explicit template for ByteSpan
using ByteSpan = Span<uint8_t>;

// Opaque handles
GPU_DEFINE_HANDLE(GpuPipeline)
GPU_DEFINE_HANDLE(GpuTexture)
GPU_DEFINE_HANDLE(GpuDepthStencilState)
GPU_DEFINE_HANDLE(GpuBlendState)
GPU_DEFINE_HANDLE(GpuQueue)
GPU_DEFINE_HANDLE(GpuCommandBuffer)
GPU_DEFINE_HANDLE(GpuSemaphore)

// Enums
enum MEMORY { MEMORY_DEFAULT, MEMORY_GPU, MEMORY_READBACK };
enum CULL { CULL_CCW, CULL_CW, CULL_ALL, CULL_NONE };
enum DEPTH_FLAGS { DEPTH_UNDEFINED = 0x0, DEPTH_READ = 0x1, DEPTH_WRITE = 0x2 };
enum OP { OP_NEVER, OP_LESS, OP_EQUAL, OP_LESS_EQUAL, OP_GREATER, OP_NOT_EQUAL, OP_GREATER_EQUAL, OP_ALWAYS, OP_KEEP }; 
enum BLEND { BLEND_ADD, BLEND_SUBTRACT, BLEND_REV_SUBTRACT, BLEND_MIN, BLEND_MAX };
enum FACTOR { FACTOR_ZERO, FACTOR_ONE, FACTOR_SRC_COLOR, FACTOR_DST_COLOR, FACTOR_SRC_ALPHA /*, ...*/ };
enum TOPOLOGY { TOPOLOGY_TRIANGLE_LIST, TOPOLOGY_TRIANGLE_STRIP, TOPOLOGY_TRIANGLE_FAN };
enum TEXTURE { TEXTURE_1D, TEXTURE_2D, TEXTURE_3D, TEXTURE_CUBE, TEXTURE_2D_ARRAY, TEXTURE_CUBE_ARRAY };
enum FORMAT { FORMAT_NONE, FORMAT_RGBA8_UNORM, FORMAT_D32_FLOAT, FORMAT_RG11B10_FLOAT, FORMAT_RGB10_A2_UNORM /*, ...*/ };
enum USAGE_FLAGS { USAGE_SAMPLED, USAGE_STORAGE, USAGE_COLOR_ATTACHMENT, USAGE_DEPTH_STENCIL_ATTACHMENT /*, ...*/ };
enum STAGE { STAGE_TRANSFER, STAGE_COMPUTE, STAGE_RASTER_COLOR_OUT, STAGE_PIXEL_SHADER, STAGE_VERTEX_SHADER /*, ...*/ };
enum HAZARD_FLAGS { HAZARD_NONE = 0x0, HAZARD_DRAW_ARGUMENTS = 0x1, HAZARD_DESCRIPTORS = 0x2, HAZARD_DEPTH_STENCIL = 0x4 };
enum SIGNAL { SIGNAL_ATOMIC_SET, SIGNAL_ATOMIC_MAX, SIGNAL_ATOMIC_OR /*, ...*/ };

// View descriptor constants
constexpr uint8_t ALL_MIPS = 0xFF;
constexpr uint16_t ALL_LAYERS = 0xFFFF;

// Structs
struct Stencil 
{
    OP test = OP_ALWAYS;
    OP failOp = OP_KEEP;
    OP passOp = OP_KEEP;
    OP depthFailOp = OP_KEEP;
    uint8_t reference = 0;
};

struct GpuDepthStencilDesc 
{
    DEPTH_FLAGS depthMode = DEPTH_UNDEFINED;
    OP depthTest = OP_ALWAYS;
    float depthBias = 0.0f;
    float depthBiasSlopeFactor = 0.0f;
    float depthBiasClamp = 0.0f;
    uint8_t stencilReadMask = 0xff;
    uint8_t stencilWriteMask = 0xff;
    Stencil stencilFront;
    Stencil stencilBack;
};

struct GpuBlendDesc
{
    BLEND colorOp = BLEND_ADD;
    FACTOR srcColorFactor = FACTOR_ONE;
    FACTOR dstColorFactor = FACTOR_ZERO;
    BLEND alphaOp = BLEND_ADD;
    FACTOR srcAlphaFactor = FACTOR_ONE;
    FACTOR dstAlphaFactor = FACTOR_ZERO;
    uint8_t colorWriteMask = 0xf;
};

struct ColorTarget {
    FORMAT format = FORMAT_NONE;
    uint8_t writeMask = 0xf;
};

struct GpuRasterDesc
{
    TOPOLOGY topology = TOPOLOGY_TRIANGLE_LIST;
    CULL cull = CULL_NONE;
    bool alphaToCoverage = false;
    bool supportDualSourceBlending = false;
    uint8_t sampleCount = 1;
    FORMAT depthFormat = FORMAT_NONE;
    FORMAT stencilFormat = FORMAT_NONE;
    Span<ColorTarget> colorTargets = {};
    GpuBlendDesc* blendstate = nullptr; // optional embedded blend state
};

struct GpuTextureDesc
{ 
    TEXTURE type = TEXTURE_2D;
    uint3 dimensions;
    uint32_t mipCount = 1;
    uint32_t layerCount = 1;
    uint32_t sampleCount = 1;
    FORMAT format = FORMAT_NONE; 
    USAGE_FLAGS usage = USAGE_SAMPLED;
};

struct GpuViewDesc 
{
    FORMAT format = FORMAT_NONE;
    uint8_t baseMip = 0;
    uint8_t mipCount = ALL_MIPS;
    uint16_t baseLayer = 0;
    uint16_t layerCount = ALL_LAYERS;
};

struct GpuRenderPassDesc 
{
    Span<GpuTexture> colorTargets = {};
    GpuTexture* depthStencilTarget = nullptr;
};

struct GpuTextureSizeAlign { size_t size; size_t align; };
struct GpuTextureDescriptorSizeAlignOffset { size_t size; size_t align; size_t offset; };
struct GpuTextureDescriptorHeapDesc { GpuTextureDescriptorSizeAlignOffset textureDesc; GpuTextureDescriptorSizeAlignOffset rwTextureDesc; size_t heapSize; };

// Memory
void* gpuMalloc(size_t bytes, MEMORY memory = MEMORY_DEFAULT);
void* gpuMalloc(size_t bytes, size_t align, MEMORY memory = MEMORY_DEFAULT);
void gpuFree(void *ptr);
void* gpuHostToDevicePointer(void *ptr);

// Textures
GpuTextureSizeAlign gpuTextureSizeAlign(GpuTextureDesc desc);
GpuTexture gpuCreateTexture(GpuTextureDesc desc, void* ptrGpu);
void gpuDestroyTexture(GpuTexture texture);
GpuTextureDescriptorHeapDesc gpuTextureDescriptorHeapDesc();
void gpuTextureViewDescriptor(GpuTexture texture, GpuViewDesc desc, void* descriptor);
void gpuRWTextureViewDescriptor(GpuTexture texture, GpuViewDesc desc, void* descriptor);

// Pipelines
GpuPipeline gpuCreateComputePipeline(ByteSpan computeIR);
GpuPipeline gpuCreateGraphicsPipeline(ByteSpan vertexIR, ByteSpan pixelIR, GpuRasterDesc desc);
GpuPipeline gpuCreateGraphicsMeshletPipeline(ByteSpan meshletIR, ByteSpan pixelIR, GpuRasterDesc desc);
void gpuFreePipeline(GpuPipeline pipeline);

// State objects
GpuDepthStencilState gpuCreateDepthStencilState(GpuDepthStencilDesc desc);
GpuBlendState gpuCreateBlendState(GpuBlendDesc desc);
void gpuFreeDepthStencilState(GpuDepthStencilState state);
void gpuFreeBlendState(GpuBlendState state);

// Queue
GpuQueue gpuCreateQueue(/* DEVICE & QUEUE CREATION DETAILS OMITTED */);
GpuCommandBuffer gpuStartCommandRecording(GpuQueue queue);
void gpuSubmit(GpuQueue queue, Span<GpuCommandBuffer> commandBuffers, GpuSemaphore semaphore, uint64_t value);

// Semaphores
GpuSemaphore gpuCreateSemaphore(uint64_t initValue);
void gpuWaitSemaphore(GpuSemaphore sema, uint64_t value, uint64_t timeout = UINT64_MAX);
void gpuDestroySemaphore(GpuSemaphore sema);

// Commands
void gpuMemCpy(GpuCommandBuffer cb, void* destGpu, void* srcGpu, uint64_t size);
void gpuCopyToTexture(GpuCommandBuffer cb, void* destGpu, void* srcGpu, GpuTexture texture);
void gpuCopyFromTexture(GpuCommandBuffer cb, void* destGpu, void* srcGpu, GpuTexture texture);

void gpuSetActiveTextureHeapPtr(GpuCommandBuffer cb, void *ptrGpu);

void gpuBarrier(GpuCommandBuffer cb, STAGE before, STAGE after, HAZARD_FLAGS hazards = HAZARD_NONE);
void gpuSignalAfter(GpuCommandBuffer cb, STAGE before, void *ptrGpu, uint64_t value, SIGNAL signal);
void gpuWaitBefore(GpuCommandBuffer cb, STAGE after, void *ptrGpu, uint64_t value, OP op, HAZARD_FLAGS hazards = HAZARD_NONE, uint64_t mask = ~0);

void gpuSetPipeline(GpuCommandBuffer cb, GpuPipeline pipeline);
void gpuSetDepthStencilState(GpuCommandBuffer cb, GpuDepthStencilState state);
void gpuSetBlendState(GpuCommandBuffer cb, GpuBlendState state); 

void gpuDispatch(GpuCommandBuffer cb, void* dataGpu, uint3 gridDimensions);
void gpuDispatchIndirect(GpuCommandBuffer cb, void* dataGpu, void* gridDimensionsGpu);

void gpuBeginRenderPass(GpuCommandBuffer cb, GpuRenderPassDesc desc);
void gpuEndRenderPass(GpuCommandBuffer cb);

void gpuDrawIndexedInstanced(GpuCommandBuffer cb, void* vertexDataGpu, void* pixelDataGpu, void* indicesGpu, uint32_t indexCount, uint32_t instanceCount);
void gpuDrawIndexedInstancedIndirect(GpuCommandBuffer cb, void* vertexDataGpu, void* pixelDataGpu, void* indicesGpu, void* argsGpu);
void gpuDrawIndexedInstancedIndirectMulti(GpuCommandBuffer cb, void* dataVxGpu, uint32_t vxStride, void* dataPxGpu, uint32_t pxStride, void* argsGpu, void* drawCountGpu);

void gpuDrawMeshlets(GpuCommandBuffer cb, void* meshletDataGpu, void* pixelDataGpu, uint3 dim);
void gpuDrawMeshletsIndirect(GpuCommandBuffer cb, void* meshletDataGpu, void* pixelDataGpu, void *dimGpu);

#endif // NO_GRAPHICS_API_H