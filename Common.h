#ifndef NO_GRAPHICS_API_COMMON_H
#define NO_GRAPHICS_API_COMMON_H

#include <vector>
#include <filesystem>
#include <fstream>

#include "NoGraphicsAPI.h"

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
};

template<typename T>
T* gpuMalloc(int count = 1)
{
    return static_cast<T*>(::gpuMalloc(sizeof(T) * count));
}

class Utilities
{
public:
    static std::vector<uint8_t> loadIR(const std::filesystem::path& path)
    {
        std::ifstream file { path, std::ios::binary | std::ios::ate };
        if (!file.is_open())
        {
            return {};
        }
        auto size = file.tellg();
        std::vector<uint8_t> buffer(size);
        file.seekg(0);
        file.read(reinterpret_cast<char*>(buffer.data()), size);
        return buffer;
    }
};

#endif // NO_GRAPHICS_API_COMMON_H