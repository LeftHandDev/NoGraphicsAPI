#include "Utilities.h"

std::vector<uint8_t> loadIR(const std::filesystem::path &path)
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

void loadOBJ(const std::filesystem::path &path, std::vector<Vertex> &vertices, std::vector<uint32_t> &indices)
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

Mesh createMesh(const std::filesystem::path &path)
{
    Mesh mesh;
    
    std::ifstream file{ path };
    if (!file.is_open())
    {
        return mesh;
    }

    std::string line;
    while (std::getline(file, line))
    {
        if (line.empty() || line[0] == '#')
            continue;

        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if (prefix == "v")
        {
            float3 pos;
            iss >> pos.x >> pos.y >> pos.z;
            mesh.vertices.push_back({ pos.x, pos.y, pos.z, 1.0f });
        }
        else if (prefix == "vt")
        {
            float2 uv;
            iss >> uv.x >> uv.y;
            mesh.uvs.push_back(uv);
        }
        else if (prefix == "vn")
        {
            float3 normal;
            iss >> normal.x >> normal.y >> normal.z;
            mesh.normals.push_back(normal);
        }
        else if (prefix == "f")
        {
            std::vector<int> faceVertexIndices;
            std::vector<int> faceUvIndices;
            std::vector<int> faceNormalIndices;

            std::string vertexData;
            while (iss >> vertexData)
            {
                int posIdx = 0, uvIdx = 0, normIdx = 0;
                
                // Handle different face formats: v, v/vt, v/vt/vn, v//vn
                size_t firstSlash = vertexData.find('/');
                if (firstSlash == std::string::npos)
                {
                    // Format: v
                    posIdx = std::stoi(vertexData);
                }
                else
                {
                    posIdx = std::stoi(vertexData.substr(0, firstSlash));
                    size_t secondSlash = vertexData.find('/', firstSlash + 1);
                    
                    if (secondSlash == std::string::npos)
                    {
                        // Format: v/vt
                        uvIdx = std::stoi(vertexData.substr(firstSlash + 1));
                    }
                    else
                    {
                        // Format: v/vt/vn or v//vn
                        if (secondSlash > firstSlash + 1)
                        {
                            uvIdx = std::stoi(vertexData.substr(firstSlash + 1, secondSlash - firstSlash - 1));
                        }
                        if (secondSlash + 1 < vertexData.size())
                        {
                            normIdx = std::stoi(vertexData.substr(secondSlash + 1));
                        }
                    }
                }

                // OBJ indices are 1-based, convert to 0-based
                faceVertexIndices.push_back(posIdx - 1);
                if (uvIdx > 0) faceUvIndices.push_back(uvIdx - 1);
                if (normIdx > 0) faceNormalIndices.push_back(normIdx - 1);
            }

            // Triangulate faces (fan triangulation for convex polygons)
            for (size_t i = 1; i + 1 < faceVertexIndices.size(); ++i)
            {
                // Vertex indices (primary index buffer for acceleration structure)
                mesh.indices.push_back(faceVertexIndices[0]);
                mesh.indices.push_back(faceVertexIndices[i]);
                mesh.indices.push_back(faceVertexIndices[i + 1]);

                // UV indices (aligned with vertex indices)
                if (!faceUvIndices.empty())
                {
                    mesh.uvIndices.push_back(faceUvIndices[0]);
                    mesh.uvIndices.push_back(faceUvIndices[i]);
                    mesh.uvIndices.push_back(faceUvIndices[i + 1]);
                }

                // Normal indices (aligned with vertex indices)
                if (!faceNormalIndices.empty())
                {
                    mesh.normalIndices.push_back(faceNormalIndices[0]);
                    mesh.normalIndices.push_back(faceNormalIndices[i]);
                    mesh.normalIndices.push_back(faceNormalIndices[i + 1]);
                }
            }
        }
    }

    return mesh;
}
