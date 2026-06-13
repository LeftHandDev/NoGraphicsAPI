#ifndef UTILITIES_H
#define UTILITIES_H

#include <vector>
#include <filesystem>
#include <fstream>
#include <string>
#include <sstream>
#include <map>

#include <glm/glm.hpp>

#include "Text.h"

template <typename T>
struct Allocation
{
    T* cpu = nullptr;
    T* gpu = nullptr;

    // byte size
    size_t size = 0;

    void free(GpuDevice device)
    {
        gpuFree(device, cpu);
    }
};

inline size_t align(size_t x, size_t a)
{
    return (x + (a - 1)) & ~(a - 1);
}

template <MEMORY memory>
class GpuMallocator
{
public:
    GpuMallocator(GpuDevice dev)
        : device(dev)
    {
    }

    ~GpuMallocator()
    {
        for (auto allocation : allocations)
        {
            gpuFree(device, allocation.gpu);
        }
    }

    template <typename T>
    Allocation<T> allocate(size_t n)
    {
        Allocation<char> alloc;
        n = sizeof(T) * n;
        if (memory == MEMORY_GPU)
        {
            alloc.gpu = reinterpret_cast<char*>(gpuMalloc(device, n, memory));
        }
        else
        {
            alloc.cpu = reinterpret_cast<char*>(gpuMalloc(device, n, memory));
            alloc.gpu = reinterpret_cast<char*>(gpuHostToDevicePointer(device, alloc.cpu));
        }
        alloc.size = n;
        allocations.push_back(alloc);

        Allocation<T> ret;
        ret.cpu = reinterpret_cast<T*>(alloc.cpu);
        ret.gpu = reinterpret_cast<T*>(alloc.gpu);
        ret.size = n;
        return ret;
    }

    template <typename T>
    void free(Allocation<T> alloc)
    {
        char* ptr = reinterpret_cast<char*>(alloc.gpu);
        for (auto iter = allocations.begin(); iter != allocations.end(); iter++)
        {
            if (iter->gpu == ptr)
            {
                gpuFree(device, alloc.gpu);
                allocations.erase(iter);
                return;
            }
        }
    }

    template <typename T>
    bool owns(Allocation<T> alloc)
    {
        char* ptr = reinterpret_cast<char*>(alloc.gpu);
        for (auto iter = allocations.begin(); iter != allocations.end(); iter++)
        {
            if (iter->gpu == ptr)
            {
                return true;
            }
        }
        return false;
    }

    void reset()
    {
        for (auto allocation : allocations)
        {
            gpuFree(device, allocation.gpu);
        }
        allocations.clear();
    }

private:
    std::vector<Allocation<char>> allocations;
    GpuDevice device;
};

template <size_t size, MEMORY memory>
class StackAllocator
{
public:
    StackAllocator(GpuDevice dev)
        : device(dev)
    {
        if (memory == MEMORY_GPU)
        {
            gpu = reinterpret_cast<char*>(gpuMalloc(device, size, memory));
        }
        else
        {
            cpu = reinterpret_cast<char*>(gpuMalloc(device, size, memory));
            gpu = reinterpret_cast<char*>(gpuHostToDevicePointer(device, cpu));
        }
    }

    ~StackAllocator()
    {
        gpuFree(device, gpu);
    }

    template <typename T>
    Allocation<T> allocate(size_t n)
    {
        Allocation<T> alloc = {};
        n = n * sizeof(T);
        if (n + offset > size)
        {
            return alloc;
        }

        if (memory == MEMORY_GPU)
        {
            alloc.gpu = reinterpret_cast<T*>(gpu + offset);
        }
        else
        {
            alloc.cpu = reinterpret_cast<T*>(cpu + offset);
            alloc.gpu = reinterpret_cast<T*>(gpu + offset);
        }

        alloc.size = size;
        offset = align(offset + n, GPU_DEFAULT_ALIGNMENT);
        return alloc;
    }

    template <typename T>
    void free(Allocation<T> alloc)
    {
        char* ptr = reinterpret_cast<char*>(alloc.gpu);
        size_t off = ptr - gpu;
        off = align(offset + alloc.size, GPU_DEFAULT_ALIGNMENT);
        if (off == offset)
        {
            offset = ptr - gpu;
        }
    }

    template <typename T>
    bool owns(Allocation<T> alloc)
    {
        char* ptr = reinterpret_cast<char*>(alloc.gpu);
        return ptr >= gpu && ptr < (gpu + size);
    }

    void reset()
    {
        offset = 0;
    }

private:
    char* cpu = nullptr;
    char* gpu = nullptr;
    size_t offset = 0;
    GpuDevice device;
};

template <size_t size, MEMORY memory>
class FreeListAllocator
{
public:
    FreeListAllocator(GpuDevice dev)
        : device(dev)
    {
        if (memory == MEMORY_GPU)
        {
            gpu = reinterpret_cast<char*>(gpuMalloc(device, size, memory));
        }
        else
        {
            cpu = reinterpret_cast<char*>(gpuMalloc(device, size, memory));
            gpu = reinterpret_cast<char*>(gpuHostToDevicePointer(device, cpu));
        }

        freeList.emplace(0, size);
    }

    ~FreeListAllocator()
    {
        gpuFree(device, cpu ? cpu : gpu);
    }

    template <typename T>
    Allocation<T> allocate(size_t n)
    {
        size_t bytes = align(n * sizeof(T), GPU_DEFAULT_ALIGNMENT);
        if (bytes == 0)
        {
            bytes = GPU_DEFAULT_ALIGNMENT;
        }

        // first-fit: offsets and block sizes stay aligned because bytes is aligned
        for (auto iter = freeList.begin(); iter != freeList.end(); ++iter)
        {
            size_t rangeOffset = iter->first;
            size_t rangeSize = iter->second;
            if (rangeSize >= bytes)
            {
                freeList.erase(iter);
                if (rangeSize > bytes)
                {
                    freeList.emplace(rangeOffset + bytes, rangeSize - bytes);
                }

                Allocation<T> alloc = {};
                alloc.cpu = cpu ? reinterpret_cast<T*>(cpu + rangeOffset) : nullptr;
                alloc.gpu = reinterpret_cast<T*>(gpu + rangeOffset);
                alloc.size = bytes;
                return alloc;
            }
        }

        return {};
    }

    template <typename T>
    void free(Allocation<T> alloc)
    {
        if (!alloc.gpu)
        {
            return;
        }

        size_t offset = reinterpret_cast<char*>(alloc.gpu) - gpu;
        size_t bytes = alloc.size;

        auto next = freeList.lower_bound(offset);

        // coalesce with the previous free block
        if (next != freeList.begin())
        {
            auto prev = std::prev(next);
            if (prev->first + prev->second == offset)
            {
                offset = prev->first;
                bytes += prev->second;
                freeList.erase(prev);
            }
        }

        // coalesce with the next free block
        if (next != freeList.end() && offset + bytes == next->first)
        {
            bytes += next->second;
            freeList.erase(next);
        }

        freeList.emplace(offset, bytes);
    }

    template <typename T>
    bool owns(Allocation<T> alloc)
    {
        char* ptr = reinterpret_cast<char*>(alloc.gpu);
        return ptr >= gpu && ptr < (gpu + size);
    }

    void reset()
    {
        freeList.clear();
        freeList.emplace(0, size);
    }

private:
    char* cpu = nullptr;
    char* gpu = nullptr;
    GpuDevice device;

    // offset -> size, non-overlapping free ranges sorted by offset
    std::map<size_t, size_t> freeList;
};

// CppCon 2015: Andrei Alexandrescu
// std::allocator Is to Allocation what std::vector Is to Vexation
// https://www.youtube.com/watch?v=LIb3L4vKZ7U
template <class Primary, class Fallback>
class FallbackAllocator : private Primary, private Fallback
{
public:
    FallbackAllocator(GpuDevice device)
        : Primary(device), Fallback(device)
    {
    }

    template <typename T>
    Allocation<T> allocate(size_t n)
    {
        Allocation<T> alloc = Primary::template allocate<T>(n);
        if (!alloc.gpu)
        {
            alloc = Fallback::template allocate<T>(n);
        }
        return alloc;
    }

    template <typename T>
    void free(Allocation<T> alloc)
    {
        Primary::template owns<T>(alloc) ? Primary::template free<T>(alloc) : Fallback::template free<T>(alloc);
    }

    template <typename T>
    bool primary_owns(Allocation<T> alloc)
    {
        return Primary::template owns<T>(alloc);
    }

    template <typename T>
    bool fallback_owns(Allocation<T> alloc)
    {
        return Fallback::template owns<T>(alloc);
    }

    template <typename T>
    bool owns(Allocation<T> alloc)
    {
        return Primary::template owns<T>(alloc) || Fallback::template owns<T>(alloc);
    }

    void reset()
    {
        Primary::template reset();
        Fallback::template reset();
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
        reset();
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

        size_t alignedOffset = 0;
        Page* page = nullptr;
        for (size_t i = 0; i < pages.size(); i++)
        {
            alignedOffset = (pages[i].used + (align - 1)) & ~(align - 1);
            if (alignedOffset + totalBytes < pages[i].size)
            {
                page = &pages[i];
                break;
            }
        }

        if (page == nullptr)
        {
            allocatePage(pageSize);
            page = &pages[pages.size() - 1];
            alignedOffset = 0;
        }

        page->used = alignedOffset + totalBytes;
        page->allocations++;

        if (isGpuOnly())
        {
            return {
                .cpu = nullptr,
                .gpu = reinterpret_cast<T*>(static_cast<uint8_t*>(page->basePtr) + alignedOffset)
            };
        }

        return {
            .cpu = reinterpret_cast<T*>(static_cast<uint8_t*>(page->basePtr) + alignedOffset),
            .gpu = reinterpret_cast<T*>(static_cast<uint8_t*>(page->baseGpuPtr) + alignedOffset)
        };
    }

    void free(const void* ptr)
    {
        auto u = static_cast<const uint8_t*>(ptr);

        for (auto iter = largePages.begin(); iter != largePages.end(); iter++)
        {
            auto base = static_cast<const uint8_t*>(iter->basePtr);
            auto gbase = static_cast<const uint8_t*>(iter->baseGpuPtr);
            if (base && u >= base && u < base + iter->size || gbase && u >= gbase && u < gbase + iter->size)
            {
                gpuFree(device, iter->basePtr);
                largePages.erase(iter);
                return;
            }
        }

        for (auto iter = pages.begin(); iter != pages.end(); iter++)
        {
            auto base = static_cast<const uint8_t*>(iter->basePtr);
            auto gbase = static_cast<const uint8_t*>(iter->baseGpuPtr);
            if (base && u >= base && u < base + iter->size || gbase && u >= gbase && u < gbase + iter->size)
            {
                if (--iter->allocations == 0)
                {
                    gpuFree(device, iter->basePtr);
                    pages.erase(iter);
                    return;
                }
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
    bool wrap(size_t count = 1, size_t align = GPU_DEFAULT_ALIGNMENT)
    {
        size_t totalBytes = count * sizeof(T);
        size_t alignedHead = (head + (align - 1)) & ~(align - 1);
        if (alignedHead + totalBytes > totalSize)
        {
            return true;
        }
        return false;
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
    LinearAllocator<MEMORY_DESCRIPTOR>* descriptorAllocator = nullptr;

    GpuDevice device;
    GpuPipeline pipeline;
    GpuDepthStencilState depthStencilState = nullptr;

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