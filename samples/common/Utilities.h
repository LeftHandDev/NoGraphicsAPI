#ifndef UTILITIES_H
#define UTILITIES_H

#include <vector>
#include <filesystem>
#include <fstream>
#include <string>
#include <sstream>

#include <glm/glm.hpp>

#include "Text.h"

template <typename T>
struct Allocation
{
    T* cpu;
    T* gpu;

    void free(GpuDevice device)
    {
        gpuFree(device, cpu);
    }
};

template <MEMORY memory = MEMORY_DEFAULT>
class LinearAllocator
{
public:
    LinearAllocator(GpuDevice gpuDevice, size_t pageSize = 65536)
        : device(gpuDevice), pageSize(pageSize)
    {
    }

    ~LinearAllocator()
    {
        free();
    }

    void* allocate(size_t size, size_t align = GPU_DEFAULT_ALIGNMENT)
    {
        return allocate<uint8_t>(size, align).cpu;
    }

    template <typename T>
    Allocation<T> allocate(size_t count = 1, size_t align = GPU_DEFAULT_ALIGNMENT)
    {
        size_t totalBytes = count * sizeof(T);

        if (totalBytes > pageSize)
        {
            return allocateLarge<T>(totalBytes, align);
        }

        if (pages.empty())
        {
            allocatePage(pageSize);
        }

        Page& current = pages.back();
        size_t alignedOffset = (current.used + (align - 1)) & ~(align - 1);

        if (alignedOffset + totalBytes > current.size)
        {
            allocatePage(pageSize);
            alignedOffset = 0;
        }

        Page& page = pages.back();
        page.used = alignedOffset + totalBytes;
        page.allocations++;

        if (isGpuOnly())
        {
            return {
                .cpu = nullptr,
                .gpu = reinterpret_cast<T*>(static_cast<uint8_t*>(page.basePtr) + alignedOffset)
            };
        }

        return {
            .cpu = reinterpret_cast<T*>(static_cast<uint8_t*>(page.basePtr) + alignedOffset),
            .gpu = reinterpret_cast<T*>(static_cast<uint8_t*>(page.baseGpuPtr) + alignedOffset)
        };
    }

    void free(const void* ptr)
    {
        auto u = static_cast<const uint8_t*>(ptr);

        auto large = std::find_if(largePages.begin(), largePages.end(),
                                  [&](const Page& p)
                                  {
                                      auto base = static_cast<const uint8_t*>(p.basePtr);
                                      auto gbase = static_cast<const uint8_t*>(p.baseGpuPtr);
                                      return (base && u >= base && u < base + p.size) || (gbase && u >= gbase && u < gbase + p.size);
                                  });
        if (large != largePages.end())
        {
            gpuFree(device, large->basePtr);
            largePages.erase(large);
            return;
        }

        auto small = std::find_if(pages.begin(), pages.end(),
                                  [&](const Page& p)
                                  {
                                      auto base = static_cast<const uint8_t*>(p.basePtr);
                                      auto gbase = static_cast<const uint8_t*>(p.baseGpuPtr);
                                      return (base && u >= base && u < base + p.size) || (gbase && u >= gbase && u < gbase + p.size);
                                  });
        if (small != pages.end() && small->allocations > 0)
        {
            if (--small->allocations == 0)
            {
                gpuFree(device, small->basePtr);
                pages.erase(small);
            }
        }
    }

    void reset()
    {
        for (auto& page : pages)
        {
            gpuFree(device, page.basePtr);
        }
        pages.clear();

        for (auto& page : largePages)
        {
            gpuFree(device, page.basePtr);
        }
        largePages.clear();
    }

private:
    struct Page
    {
        void* basePtr;
        void* baseGpuPtr;
        size_t size;
        size_t used;
        size_t allocations = 0;
    };

    void allocatePage(size_t size)
    {
        void* ptr = gpuMalloc(device, size, memory);
        void* gpuPtr = isGpuOnly() ? nullptr : gpuHostToDevicePointer(device, ptr);
        pages.push_back({ ptr, gpuPtr, size, 0 });
    }

    template <typename T>
    Allocation<T> allocateLarge(size_t totalBytes, size_t align)
    {
        size_t allocSize = totalBytes + align;
        void* ptr = gpuMalloc(device, allocSize, memory);

        size_t alignedOffset = (reinterpret_cast<size_t>(ptr) + (align - 1)) & ~(align - 1);
        alignedOffset -= reinterpret_cast<size_t>(ptr);

        if (isGpuOnly())
        {
            largePages.push_back({ ptr, nullptr, allocSize, alignedOffset + totalBytes });

            return {
                .cpu = nullptr,
                .gpu = reinterpret_cast<T*>(static_cast<uint8_t*>(ptr) + alignedOffset)
            };
        }

        void* gpuBasePtr = gpuHostToDevicePointer(device, ptr);
        largePages.push_back({ ptr, gpuBasePtr, allocSize, alignedOffset + totalBytes });

        return {
            .cpu = reinterpret_cast<T*>(static_cast<uint8_t*>(ptr) + alignedOffset),
            .gpu = reinterpret_cast<T*>(static_cast<uint8_t*>(gpuBasePtr) + alignedOffset)
        };
    }

    bool isGpuOnly() const
    {
        return memory == MEMORY_GPU;
    }

    GpuDevice device;
    size_t pageSize;
    std::vector<Page> pages;
    std::vector<Page> largePages;
};

template <MEMORY memory = MEMORY_DEFAULT>
class RingBuffer
{
public:
    RingBuffer(GpuDevice gpuDevice, size_t size = 65536)
        : device(gpuDevice), totalSize(size)
    {
        if (memory == MEMORY_GPU)
        {
            baseGpuPtr = gpuMalloc(device, size, memory);
        }
        else
        {
            basePtr = gpuMalloc(device, size, memory);
            baseGpuPtr = gpuHostToDevicePointer(device, basePtr);
        }
    }

    ~RingBuffer()
    {
        free();
    }

    template <typename T>
    Allocation<T> allocate(size_t count = 1, size_t align = GPU_DEFAULT_ALIGNMENT)
    {
        size_t totalBytes = count * sizeof(T);
        if (totalBytes > totalSize)
        {
            return {};
        }

        size_t alignedHead = (head + (align - 1)) & ~(align - 1);

        if (alignedHead + totalBytes > totalSize)
        {
            alignedHead = 0;
        }

        head = alignedHead + totalBytes;

        if (memory == MEMORY_GPU)
        {
            return {
                .cpu = nullptr,
                .gpu = reinterpret_cast<T*>(static_cast<uint8_t*>(baseGpuPtr) + alignedHead)
            };
        }

        return {
            .cpu = reinterpret_cast<T*>(static_cast<uint8_t*>(basePtr) + alignedHead),
            .gpu = reinterpret_cast<T*>(static_cast<uint8_t*>(baseGpuPtr) + alignedHead)
        };
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
        head = 0;
    }

    size_t size()
    {
        return totalSize;
    }

private:
    GpuDevice device;
    void* basePtr = nullptr;
    void* baseGpuPtr = nullptr;
    size_t totalSize = 0;
    size_t head = 0;
};

class TextRenderer
{
public:
    TextRenderer(GpuDevice device, GpuTextureDesc desc);
    ~TextRenderer();

    void renderText(GpuCommandBuffer cmd, GpuTexture target, const std::string& text, float x, float y, float scale, float3 color);

private:
    LinearAllocator<MEMORY_DEFAULT>* allocator = nullptr;

    GpuDevice device;
    GpuPipeline pipeline;

    Allocation<GpuTextureDescriptor> textureHeap;
    Allocation<TextVertexData> vertexData;
    Allocation<TextPixelData> pixelData;
    Allocation<uint32_t> indexData;
    Allocation<uint8_t> textData;
    uint offset = 0;

    const GpuTextureDesc targetDesc;

    GpuTexture atlas;
    void* atlasPtr = nullptr;

    int atlasWidth = 0;
    int atlasHeight = 0;

    const uint maxTextLength = 1024;
};

std::vector<uint8_t> loadIR(const std::filesystem::path& path);

void getCube(std::vector<float3>& vertices, std::vector<float3>& normals, std::vector<float2>& uvs, std::vector<uint32_t>& indices);

std::vector<glm::vec2> haltonSequence(uint length = 16);

#endif // UTILITIES_H