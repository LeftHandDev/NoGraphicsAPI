#ifndef NO_GRAPHICS_API_COMMON_H
#define NO_GRAPHICS_API_COMMON_H

#include <vector>
#include <filesystem>
#include <fstream>

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