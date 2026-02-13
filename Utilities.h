#ifndef UTILITIES_H
#define UTILITIES_H

#include <vector>
#include <filesystem>
#include <fstream>
#include <string>
#include <sstream>

#include "NoGraphicsAPI.h"

#include "Samples/Graphics/Graphics.h"
#include "Samples/Raytracing/Raytracing.h"

template<typename T>
struct Allocation
{
    T* cpu;
    T* gpu;

    void free()
    {
        gpuFree(cpu);
    }
};

template<typename T>
Allocation<T> allocate(int count = 1)
{
    auto addr = gpuMalloc(sizeof(T) * count);
    return { 
        .cpu = static_cast<T*>(addr), 
        .gpu = static_cast<T*>(gpuHostToDevicePointer(addr)) 
    };
}

class LinearAllocator
{
public:

    LinearAllocator(size_t size, MEMORY memory = MEMORY_DEFAULT)
    {
        basePtr = gpuMalloc(size, memory);
        baseGpuPtr = gpuHostToDevicePointer(basePtr);
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

        gpuFree(basePtr);
        basePtr = nullptr;
        baseGpuPtr = nullptr;
        currentPtr = nullptr;
        totalSize = 0;
        usedSize = 0;
    }

private:
    void* basePtr = nullptr;
    void* baseGpuPtr = nullptr;
    void* currentPtr = nullptr;
    size_t totalSize = 0;
    size_t usedSize = 0;
};

template<typename T>
T* gpuMalloc(int count = 1, MEMORY type = MEMORY_DEFAULT)
{
    return static_cast<T*>(::gpuMalloc(sizeof(T) * count, type));
}

std::vector<uint8_t> loadIR(const std::filesystem::path& path);

void getCube(std::vector<float3>& vertices, std::vector<float3>& normals, std::vector<float2>& uvs, std::vector<uint32_t>& indices);

#endif // UTILITIES_H