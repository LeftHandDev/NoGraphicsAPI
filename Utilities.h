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
};

template<typename T>
T* gpuMalloc(int count = 1, MEMORY type = MEMORY_DEFAULT)
{
    return static_cast<T*>(::gpuMalloc(sizeof(T) * count, type));
}

struct Mesh
{
    std::vector<float4> vertices;
    std::vector<float2> uvs;
    std::vector<float3> normals;

    std::vector<uint32_t> indices;
    std::vector<uint32_t> uvIndices;     
    std::vector<uint32_t> normalIndices; 
    
    void load(MeshData* meshData) const
    {
        meshData->indices = gpuMalloc<uint32_t>(static_cast<int>(indices.size()));
        memcpy(meshData->indices, indices.data(), indices.size() * sizeof(uint32_t));
        meshData->indices = static_cast<uint32_t*>(gpuHostToDevicePointer(meshData->indices));

        meshData->vertices = gpuMalloc<float4>(static_cast<int>(vertices.size()));
        memcpy(meshData->vertices, vertices.data(), vertices.size() * sizeof(float4));
        meshData->vertices = static_cast<float4*>(gpuHostToDevicePointer(meshData->vertices));

        meshData->uvs = gpuMalloc<float2>(static_cast<int>(uvs.size()));
        memcpy(meshData->uvs, uvs.data(), uvs.size() * sizeof(float2));
        meshData->uvs = static_cast<float2*>(gpuHostToDevicePointer(meshData->uvs));

        meshData->uvIndices = gpuMalloc<uint32_t>(static_cast<int>(uvIndices.size()));
        memcpy(meshData->uvIndices, uvIndices.data(), uvIndices.size() * sizeof(uint32_t));
        meshData->uvIndices = static_cast<uint32_t*>(gpuHostToDevicePointer(meshData->uvIndices));

        meshData->normals = gpuMalloc<float3>(static_cast<int>(normals.size()));
        memcpy(meshData->normals, normals.data(), normals.size() * sizeof(float3));
        meshData->normals = static_cast<float3*>(gpuHostToDevicePointer(meshData->normals));

        meshData->normalIndices = gpuMalloc<uint32_t>(static_cast<int>(normalIndices.size()));
        memcpy(meshData->normalIndices, normalIndices.data(), normalIndices.size() * sizeof(uint32_t));
        meshData->normalIndices = static_cast<uint32_t*>(gpuHostToDevicePointer(meshData->normalIndices));
    }
};

std::vector<uint8_t> loadIR(const std::filesystem::path& path);

void loadOBJ(const std::filesystem::path& path, std::vector<Vertex>& vertices, std::vector<uint32_t>& indices);

Mesh createMesh(const std::filesystem::path& path);

#endif // UTILITIES_H