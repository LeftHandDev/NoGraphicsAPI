#ifndef UTILITIES_H
#define UTILITIES_H

#include <vector>
#include <filesystem>
#include <fstream>
#include <string>
#include <sstream>

#include "Text.h"

template<typename T>
struct Allocation
{
    T* cpu;
    T* gpu;
    GpuDevice device;

    void free()
    {
        gpuFree(device, cpu);
    }
};

template<typename T>
Allocation<T> allocate(GpuDevice device, int count = 1, MEMORY type = MEMORY_DEFAULT)
{
    auto addr = gpuMalloc(device, sizeof(T) * count, type);
    return { 
        .cpu = static_cast<T*>(addr), 
        .gpu = static_cast<T*>(gpuHostToDevicePointer(device, addr)),
        .device = device
    };
}

class LinearAllocator
{
public:

    LinearAllocator(GpuDevice gpuDevice, size_t size, MEMORY memory = MEMORY_DEFAULT)
        : device(gpuDevice)
    {
        basePtr = gpuMalloc(device, size, memory);
        baseGpuPtr = gpuHostToDevicePointer(device, basePtr);
        currentPtr = basePtr;
        totalSize = size;
        usedSize = 0;
    }

    ~LinearAllocator()
    {
        free();
    }

    template<typename T>
    Allocation<T> allocate(size_t size = 1, size_t align = GPU_DEFAULT_ALIGNMENT)
    {
        size_t padding = 0;
        size_t alignedAddress = (reinterpret_cast<size_t>(currentPtr) + (align - 1)) & ~(align - 1);
        padding = alignedAddress - reinterpret_cast<size_t>(currentPtr);

        if (usedSize + padding + size > totalSize)
        {
            return {}; // Out of memory
        }

        currentPtr = reinterpret_cast<void*>(alignedAddress);
        void* allocatedPtr = currentPtr;
        currentPtr = static_cast<uint8_t*>(currentPtr) + size * sizeof(T);
        usedSize += padding + size * sizeof(T);

        return { 
            .cpu = static_cast<T*>(allocatedPtr), 
            .gpu = reinterpret_cast<T*>(static_cast<char*>(baseGpuPtr) + (usedSize - size * sizeof(T)))
        };
    }

    void reset()
    {
        currentPtr = basePtr;
        usedSize = 0;
    }

    void free()
    {
        if (basePtr == nullptr)
        {
            return;
        }

        gpuFree(device, basePtr);
        basePtr = nullptr;
        baseGpuPtr = nullptr;
        currentPtr = nullptr;
        totalSize = 0;
        usedSize = 0;
    }

private:
    GpuDevice device;
    void* basePtr = nullptr;
    void* baseGpuPtr = nullptr;
    void* currentPtr = nullptr;
    size_t totalSize = 0;
    size_t usedSize = 0;
};

class TextRenderer
{
public:
    TextRenderer(GpuDevice device, GpuTextureDesc desc);
    ~TextRenderer();

    void renderText(GpuCommandBuffer cmd, GpuTexture target, const std::string& text, float x, float y, float scale, float3 color);

private:

    GpuPipeline pipeline;
    
    Allocation<GpuTextureDescriptor> textureHeap;
    Allocation<TextVertexData> vertexData;
    Allocation<TextPixelData> pixelData;
    Allocation<uint32_t> indexData;
    Allocation<uint8_t> textData;
    uint offset = 0;
    GpuDevice device;

    const GpuTextureDesc targetDesc;

    GpuTexture atlas;
    void* atlasPtr = nullptr;

    int atlasWidth = 0;
    int atlasHeight = 0;

    const uint maxTextLength = 1024;
};

template<typename T>
T* gpuMalloc(GpuDevice device, int count = 1, MEMORY type = MEMORY_DEFAULT)
{
    return static_cast<T*>(::gpuMalloc(device, sizeof(T) * count, type));
}

std::vector<uint8_t> loadIR(const std::filesystem::path& path);

void getCube(std::vector<float3>& vertices, std::vector<float3>& normals, std::vector<float2>& uvs, std::vector<uint32_t>& indices);

#endif // UTILITIES_H