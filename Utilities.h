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
T* gpuMalloc(int count = 1)
{
    return static_cast<T*>(::gpuMalloc(sizeof(T) * count));
}

struct Mesh
{
    // Unique attribute data
    std::vector<float4> vertices;
    std::vector<float2> uvs;
    std::vector<float3> normals;

    // Per-triangle-corner indices (3 per triangle)
    // All index arrays are aligned - use primitiveIndex * 3 + corner
    std::vector<uint32_t> indices;       // into vertices (use for acceleration structure)
    std::vector<uint32_t> uvIndices;     // into uvs
    std::vector<uint32_t> normalIndices; // into normals

    void allocate(MeshData* meshData) const
    {
        meshData->indices = gpuMalloc<uint32_t>(static_cast<int>(indices.size()));
        meshData->vertices = gpuMalloc<float4>(static_cast<int>(vertices.size()));
        meshData->uvs = gpuMalloc<float2>(static_cast<int>(uvs.size()));
        meshData->uvIndices = gpuMalloc<uint32_t>(static_cast<int>(uvIndices.size()));
        meshData->normals = gpuMalloc<float3>(static_cast<int>(normals.size()));
        meshData->normalIndices = gpuMalloc<uint32_t>(static_cast<int>(normalIndices.size()));
    }
    
    // Full load for mesh/raytracing shaders
    void load(const MeshData* meshData) const
    {
        memcpy(meshData->indices, indices.data(), indices.size() * sizeof(uint32_t));
        memcpy(meshData->vertices, vertices.data(), vertices.size() * sizeof(float4));
        memcpy(meshData->uvs, uvs.data(), uvs.size() * sizeof(float2));
        memcpy(meshData->uvIndices, uvIndices.data(), uvIndices.size() * sizeof(uint32_t));
        memcpy(meshData->normals, normals.data(), normals.size() * sizeof(float3));
        memcpy(meshData->normalIndices, normalIndices.data(), normalIndices.size() * sizeof(uint32_t));
    }
};

std::vector<uint8_t> loadIR(const std::filesystem::path& path);

void loadOBJ(const std::filesystem::path& path, std::vector<Vertex>& vertices, std::vector<uint32_t>& indices);

Mesh createMesh(const std::filesystem::path& path);

#endif // UTILITIES_H