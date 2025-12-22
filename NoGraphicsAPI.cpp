#include "NoGraphicsAPI.h"

struct GpuPipeline_T {};
struct GpuTexture_T {};
struct GpuDepthStencilState_T {};
struct GpuBlendState_T {};
struct GpuQueue_T {};
struct GpuCommandBuffer_T {};
struct GpuSemaphore_T {};

void* gpuMalloc(size_t bytes, MEMORY memory)
{
    return nullptr;
}

void* gpuMalloc(size_t bytes, size_t align, MEMORY memory)
{
    return nullptr;
}

void gpuFree(void *ptr)
{
}

void* gpuHostToDevicePointer(void *ptr)
{
    return nullptr;
}

GpuTextureSizeAlign gpuTextureSizeAlign(GpuTextureDesc desc)
{
    return { 0, 0 };
}

GpuTexture gpuCreateTexture(GpuTextureDesc desc, void* ptrGpu)
{
    return nullptr;
}

GpuTextureDescriptor gpuTextureViewDescriptor(GpuTexture texture, GpuViewDesc desc)
{
    return {};
}

GpuTextureDescriptor gpuRWTextureViewDescriptor(GpuTexture texture, GpuViewDesc desc)
{
    return {};
}

GpuPipeline gpuCreateComputePipeline(ByteSpan computeIR)
{
    return nullptr;
}

GpuPipeline gpuCreateGraphicsPipeline(ByteSpan vertexIR, ByteSpan pixelIR, GpuRasterDesc desc)
{
    return nullptr;
}

GpuPipeline gpuCreateGraphicsMeshletPipeline(ByteSpan meshletIR, ByteSpan pixelIR, GpuRasterDesc desc)
{
    return nullptr;
}

void gpuFreePipeline(GpuPipeline pipeline)
{
}

GpuDepthStencilState gpuCreateDepthStencilState(GpuDepthStencilDesc desc)
{
    return nullptr;
}

GpuBlendState gpuCreateBlendState(GpuBlendDesc desc)
{
    return nullptr;
}

void gpuFreeDepthStencilState(GpuDepthStencilState state)
{
}

void gpuFreeBlendState(GpuBlendState state)
{
}

GpuQueue gpuCreateQueue()
{
    return nullptr;
}

GpuCommandBuffer gpuStartCommandRecording(GpuQueue queue)
{
    return nullptr;
}

void gpuSubmit(GpuQueue queue, Span<GpuCommandBuffer> commandBuffers)
{
}

GpuSemaphore gpuCreateSemaphore(uint64_t initValue)
{
    return nullptr;
}

void gpuWaitSemaphore(GpuSemaphore sema, uint64_t value)
{
}

void gpuDestroySemaphore(GpuSemaphore sema)
{
}

void gpuMemCpy(GpuCommandBuffer cb, void* destGpu, void* srcGpu)
{
}

void gpuCopyToTexture(GpuCommandBuffer cb, void* destGpu, void* srcGpu, GpuTexture texture)
{
}

void gpuCopyFromTexture(GpuCommandBuffer cb, void* destGpu, void* srcGpu, GpuTexture texture)
{
}

void gpuSetActiveTextureHeapPtr(GpuCommandBuffer cb, void *ptrGpu)
{
}

void gpuBarrier(GpuCommandBuffer cb, STAGE before, STAGE after, HAZARD_FLAGS hazards)
{
}

void gpuSignalAfter(GpuCommandBuffer cb, STAGE before, void *ptrGpu, uint64_t value, SIGNAL signal)
{
}

void gpuWaitBefore(GpuCommandBuffer cb, STAGE after, void *ptrGpu, uint64_t value, OP op, HAZARD_FLAGS hazards, uint64_t mask)
{
}

void gpuSetPipeline(GpuCommandBuffer cb, GpuPipeline pipeline)
{
}

void gpuSetDepthStencilState(GpuCommandBuffer cb, GpuDepthStencilState state)
{
}

void gpuSetBlendState(GpuCommandBuffer cb, GpuBlendState state)
{
}

void gpuDispatch(GpuCommandBuffer cb, void* dataGpu, uvec3 gridDimensions)
{
}

void gpuDispatchIndirect(GpuCommandBuffer cb, void* dataGpu, void* gridDimensionsGpu)
{
}

void gpuBeginRenderPass(GpuCommandBuffer cb, GpuRenderPassDesc desc)
{
}

void gpuEndRenderPass(GpuCommandBuffer cb)
{
}

void gpuDrawIndexedInstanced(GpuCommandBuffer cb, void* vertexDataGpu, void* pixelDataGpu, void* indicesGpu, uint32_t indexCount, uint32_t instanceCount)
{
}

void gpuDrawIndexedInstancedIndirect(GpuCommandBuffer cb, void* vertexDataGpu, void* pixelDataGpu, void* indicesGpu, void* argsGpu)
{
}

void gpuDrawIndexedInstancedIndirectMulti(GpuCommandBuffer cb, void* dataVxGpu, uint32_t vxStride, void* dataPxGpu, uint32_t pxStride, void* argsGpu, void* drawCountGpu)
{
}

void gpuDrawMeshlets(GpuCommandBuffer cb, void* meshletDataGpu, void* pixelDataGpu, uvec3 dim)
{
}

void gpuDrawMeshletsIndirect(GpuCommandBuffer cb, void* meshletDataGpu, void* pixelDataGpu, void *dimGpu)
{
}
