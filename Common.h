#ifndef NO_GRAPHICS_API_COMMON_H
#define NO_GRAPHICS_API_COMMON_H

#include <vector>
#include <filesystem>
#include <fstream>
#include <string>
#include <sstream>

#include "NoGraphicsAPI.h"

#include "Samples/Graphics/Graphics.h"

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

    static void loadOBJ(const std::filesystem::path& path, std::vector<Vertex>& vertices, std::vector<uint32_t>& indices)
    {
        std::ifstream file { path };
        if (!file.is_open())
        {
            return;
        }

        std::vector<float3> positions;
        std::vector<float2> uvs;

        std::string line;
        while (std::getline(file, line))
        {
            std::istringstream iss(line);
            std::string prefix;
            iss >> prefix;

            if (prefix == "v")
            {
                float3 pos;
                iss >> pos.x >> pos.y >> pos.z;
                positions.push_back(pos);
            }
            else if (prefix == "vt")
            {
                float2 uv;
                iss >> uv.x >> uv.y;
                uvs.push_back(uv);
            }
            else if (prefix == "f")
            {
                for (int i = 0; i < 3; i++)
                {
                    std::string vertexData;
                    iss >> vertexData;
                    
                    int posIdx, uvIdx, normIdx;
                    sscanf_s(vertexData.c_str(), "%d/%d/%d", &posIdx, &uvIdx, &normIdx);
                    
                    Vertex vertex;
                    vertex.vertex = float4{ positions[posIdx - 1].x, positions[posIdx - 1].y, positions[posIdx - 1].z, 1.0f };
                    vertex.uv = uvs[uvIdx - 1];
                    vertices.push_back(vertex);
                    indices.push_back(static_cast<uint32_t>(indices.size()));
                }
            }
        }
    }

};

#endif // NO_GRAPHICS_API_COMMON_H