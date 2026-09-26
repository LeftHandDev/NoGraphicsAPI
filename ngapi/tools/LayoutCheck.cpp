// ngapi-layout-check: verifies that structs shared between C++ and shaders
// have the same memory layout on both sides.
//
//   ngapi-layout-check <shader.spv> <shader.d> <out.cpp> <shader-name>
//
// Shaders read their data through pointers (buffer device addresses), and a
// struct read that way carries its GPU layout in the compiled SPIR-V: slang
// names it <Name>_natural and decorates every member with its byte offset,
// and pointers to it with their array stride. From that, this tool writes a
// C++ file that includes the headers declaring those structs (found through
// slangc's depfile) and static_asserts, for each struct:
//
//   - every member's offset and size match the shader's, and
//   - sizeof matches the shader's array stride, if the shader indexes an
//     array of the struct (pointer arithmetic, or an array member).
//
// compile_shader (cmake/CompileShaders.cmake) compiles that file with the C++
// compiler, so a mismatch fails the build. The usual causes are alignas()
// padding, which only C++ applies (NoGraphicsAPI.h defines it away for
// shaders), and bool, which is 1 byte in C++ but 4 on the GPU. Trailing
// padding on a struct the shader never indexes is harmless and not reported.
// Structs declared only in shader code have no C++ counterpart and are
// skipped.

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
    // SPIR-V opcodes, decorations and storage classes used below.
    enum : uint32_t
    {
        OpUndef = 1,
        OpName = 5,
        OpMemberName = 6,
        OpTypeBool = 20,
        OpTypeInt = 21,
        OpTypeFloat = 22,
        OpTypeVector = 23,
        OpTypeMatrix = 24,
        OpTypeArray = 28,
        OpTypeRuntimeArray = 29,
        OpTypeStruct = 30,
        OpTypePointer = 32,
        OpConstant = 43,
        OpFunctionParameter = 55,
        OpFunctionCall = 57,
        OpVariable = 59,
        OpLoad = 61,
        OpAccessChain = 65,
        OpInBoundsAccessChain = 66,
        OpPtrAccessChain = 67,
        OpInBoundsPtrAccessChain = 70,
        OpDecorate = 71,
        OpMemberDecorate = 72,
        OpCompositeExtract = 81,
        OpCopyObject = 83,
        OpConvertUToPtr = 120,
        OpBitcast = 124,
        OpSelect = 169,
        OpPhi = 245,

        DecorationRowMajor = 4,
        DecorationArrayStride = 6,
        DecorationMatrixStride = 7,
        DecorationOffset = 35,

        StorageClassPhysicalStorageBuffer = 5349,
    };

    struct Instruction
    {
        uint32_t opcode;
        std::vector<uint32_t> operands; // words after the opcode word
    };

    struct Module
    {
        std::unordered_map<uint32_t, Instruction> types; // type id -> OpType* instruction
        std::unordered_map<uint32_t, std::string> names;
        std::map<std::pair<uint32_t, uint32_t>, std::string> memberNames;
        std::map<std::pair<uint32_t, uint32_t>, uint32_t> memberOffsets;
        std::map<std::pair<uint32_t, uint32_t>, uint32_t> matrixStrides;
        std::set<std::pair<uint32_t, uint32_t>> rowMajor;
        std::unordered_map<uint32_t, uint32_t> arrayStrides; // array / pointer type -> stride
        std::unordered_map<uint32_t, uint64_t> constants;    // integer constants
        std::unordered_map<uint32_t, uint32_t> valueTypes;   // value id -> type id
        std::map<uint32_t, uint32_t> indexedStructs;         // struct id -> GPU stride
    };

    std::string literalString(const std::vector<uint32_t>& operands, size_t first)
    {
        std::string s;
        for (size_t i = first; i < operands.size(); i++)
        {
            for (int b = 0; b < 4; b++)
            {
                char c = static_cast<char>((operands[i] >> (8 * b)) & 0xff);
                if (c == 0)
                {
                    return s;
                }
                s += c;
            }
        }
        return s;
    }

    bool parse(const std::vector<uint32_t>& words, Module& m, std::string& error)
    {
        if (words.size() < 5 || words[0] != 0x07230203)
        {
            error = "not a little-endian SPIR-V module";
            return false;
        }

        std::vector<Instruction> ptrAccessChains;
        for (size_t at = 5; at < words.size();)
        {
            const uint32_t length = words[at] >> 16;
            const uint32_t opcode = words[at] & 0xffff;
            if (length == 0 || at + length > words.size())
            {
                error = "truncated instruction";
                return false;
            }
            Instruction inst{ opcode, std::vector<uint32_t>(words.begin() + at + 1, words.begin() + at + length) };
            const auto& o = inst.operands;
            at += length;

            switch (opcode)
            {
            case OpName:
                m.names[o[0]] = literalString(o, 1);
                break;
            case OpMemberName:
                m.memberNames[{ o[0], o[1] }] = literalString(o, 2);
                break;
            case OpDecorate:
                if (o.size() >= 3 && o[1] == DecorationArrayStride)
                {
                    m.arrayStrides[o[0]] = o[2];
                }
                break;
            case OpMemberDecorate:
                if (o.size() >= 4 && o[2] == DecorationOffset)
                {
                    m.memberOffsets[{ o[0], o[1] }] = o[3];
                }
                else if (o.size() >= 4 && o[2] == DecorationMatrixStride)
                {
                    m.matrixStrides[{ o[0], o[1] }] = o[3];
                }
                else if (o.size() >= 3 && o[2] == DecorationRowMajor)
                {
                    m.rowMajor.insert({ o[0], o[1] });
                }
                break;
            case OpTypeBool:
            case OpTypeInt:
            case OpTypeFloat:
            case OpTypeVector:
            case OpTypeMatrix:
            case OpTypeArray:
            case OpTypeRuntimeArray:
            case OpTypeStruct:
            case OpTypePointer:
                m.types[o[0]] = inst;
                break;
            case OpConstant:
                if (o.size() >= 3)
                {
                    m.constants[o[1]] = o[2] | (o.size() >= 4 ? static_cast<uint64_t>(o[3]) << 32 : 0);
                }
                break;
            case OpPtrAccessChain:
            case OpInBoundsPtrAccessChain:
                ptrAccessChains.push_back(inst);
                [[fallthrough]];
            case OpUndef:
            case OpFunctionParameter:
            case OpFunctionCall:
            case OpVariable:
            case OpLoad:
            case OpAccessChain:
            case OpInBoundsAccessChain:
            case OpCompositeExtract:
            case OpCopyObject:
            case OpConvertUToPtr:
            case OpBitcast:
            case OpSelect:
            case OpPhi:
                m.valueTypes[o[1]] = o[0];
                break;
            default:
                break;
            }
        }

        // Pointer arithmetic (p[i]) on a pointer to a struct steps by the
        // pointer type's ArrayStride, so C++'s sizeof must match it.
        for (const auto& chain : ptrAccessChains)
        {
            const auto& o = chain.operands; // result type, result, base, element, indexes...
            if (o.size() < 4)
            {
                continue;
            }
            auto element = m.constants.find(o[3]);
            if (element != m.constants.end() && element->second == 0)
            {
                continue; // p[0] does not depend on the stride
            }
            // Without further indexes the result has the base's type.
            uint32_t pointerType = o.size() == 4 ? o[0] : 0;
            if (auto base = m.valueTypes.find(o[2]); base != m.valueTypes.end())
            {
                pointerType = base->second;
            }
            auto pointer = m.types.find(pointerType);
            auto stride = m.arrayStrides.find(pointerType);
            if (pointer == m.types.end() || pointer->second.opcode != OpTypePointer || stride == m.arrayStrides.end())
            {
                continue;
            }
            auto pointee = m.types.find(pointer->second.operands[2]);
            if (pointee != m.types.end() && pointee->second.opcode == OpTypeStruct)
            {
                m.indexedStructs[pointee->first] = stride->second;
            }
        }
        return true;
    }

    struct Layout
    {
        uint64_t size = 0;
        uint64_t align = 1;
    };

    // The type's size and alignment in slang's natural layout (C rules: a
    // vector or matrix aligns to its component, pointers to 8), or nullopt if
    // it has no fixed size (runtime arrays, opaque types).
    std::optional<Layout> layoutOf(const Module& m, uint32_t type, uint32_t parentStruct = 0, uint32_t member = 0)
    {
        auto it = m.types.find(type);
        if (it == m.types.end())
        {
            return std::nullopt;
        }
        const auto& o = it->second.operands;
        switch (it->second.opcode)
        {
        case OpTypeBool:
            return Layout{ 4, 4 };
        case OpTypeInt:
        case OpTypeFloat:
            return Layout{ o[1] / 8u, o[1] / 8u };
        case OpTypeVector:
        {
            auto component = layoutOf(m, o[1]);
            if (!component)
            {
                return std::nullopt;
            }
            return Layout{ component->size * o[2], component->align };
        }
        case OpTypeMatrix:
        {
            auto column = m.types.find(o[1]);
            auto component = column != m.types.end() ? layoutOf(m, column->second.operands[1]) : std::nullopt;
            if (!component)
            {
                return std::nullopt;
            }
            const uint64_t rows = column->second.operands[2];
            const uint64_t columns = o[2];
            auto stride = m.matrixStrides.find({ parentStruct, member });
            const bool isRowMajor = m.rowMajor.count({ parentStruct, member }) != 0;
            const uint64_t vectorStride = stride != m.matrixStrides.end() ? stride->second
                                                                           : component->size * (isRowMajor ? columns : rows);
            return Layout{ vectorStride * (isRowMajor ? rows : columns), component->align };
        }
        case OpTypePointer:
            if (o[1] != StorageClassPhysicalStorageBuffer)
            {
                return std::nullopt;
            }
            return Layout{ 8, 8 };
        case OpTypeArray:
        {
            auto element = layoutOf(m, o[1], parentStruct, member);
            auto length = m.constants.find(o[2]);
            if (!element || length == m.constants.end())
            {
                return std::nullopt;
            }
            auto stride = m.arrayStrides.find(type);
            return Layout{ length->second * (stride != m.arrayStrides.end() ? stride->second : element->size), element->align };
        }
        case OpTypeStruct:
        {
            Layout layout;
            uint64_t end = 0;
            for (uint32_t i = 1; i < o.size(); i++)
            {
                auto offset = m.memberOffsets.find({ type, i - 1 });
                auto memberLayout = layoutOf(m, o[i], type, i - 1);
                if (offset == m.memberOffsets.end() || !memberLayout)
                {
                    return std::nullopt;
                }
                end = std::max<uint64_t>(end, offset->second + memberLayout->size);
                layout.align = std::max(layout.align, memberLayout->align);
            }
            layout.size = (end + layout.align - 1) / layout.align * layout.align;
            return layout;
        }
        default:
            return std::nullopt;
        }
    }

    bool isIdentifier(const std::string& s)
    {
        return !s.empty() && !std::isdigit(static_cast<unsigned char>(s[0])) &&
               std::all_of(s.begin(), s.end(), [](char c)
                           { return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; });
    }

    // Parses a make-style depfile ("target: dep dep ...", '\ ' escapes spaces,
    // '\' + newline continues the line) into its dependency paths.
    std::vector<std::string> readDepfile(const std::string& path)
    {
        std::ifstream in(path, std::ios::binary);
        std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        size_t colon = text.find(": ");
        std::vector<std::string> deps;
        if (colon == std::string::npos)
        {
            return deps;
        }
        std::string current;
        for (size_t i = colon + 2; i < text.size(); i++)
        {
            const char c = text[i];
            if (c == '\\' && i + 1 < text.size() && (text[i + 1] == ' ' || text[i + 1] == '\n' || text[i + 1] == '\r'))
            {
                if (text[i + 1] == ' ')
                {
                    current += ' ';
                }
                i++;
                continue;
            }
            if (c == ' ' || c == '\n' || c == '\r' || c == '\t')
            {
                if (!current.empty())
                {
                    deps.push_back(current);
                    current.clear();
                }
                continue;
            }
            current += c;
        }
        if (!current.empty())
        {
            deps.push_back(current);
        }
        return deps;
    }

    bool isHeader(const std::string& path)
    {
        std::string ext = std::filesystem::path(path).extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c)
                       { return static_cast<char>(std::tolower(c)); });
        return ext == ".h" || ext == ".hpp" || ext == ".hh" || ext == ".hxx" || ext == ".inl";
    }

    std::string escape(const std::string& s)
    {
        std::string out;
        for (char c : s)
        {
            if (c == '"' || c == '\\')
            {
                out += '\\';
            }
            out += c;
        }
        return out;
    }
} // namespace

int main(int argc, char** argv)
{
    if (argc != 5)
    {
        std::fprintf(stderr, "usage: ngapi-layout-check <shader.spv> <shader.d> <out.cpp> <shader-name>\n");
        return 2;
    }
    const std::string spvPath = argv[1], depPath = argv[2], outPath = argv[3], shaderName = argv[4];

    std::ifstream spv(spvPath, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(spv)), std::istreambuf_iterator<char>());
    std::vector<uint32_t> words(bytes.size() / 4);
    std::memcpy(words.data(), bytes.data(), words.size() * 4);

    Module m;
    std::string error;
    if (!parse(words, m, error))
    {
        std::fprintf(stderr, "ngapi-layout-check: %s: %s\n", spvPath.c_str(), error.c_str());
        return 1;
    }

    // Array members of struct type: the shader steps through them by the
    // array's stride, so their element struct is indexed too.
    for (const auto& [id, type] : m.types)
    {
        if (type.opcode != OpTypeStruct)
        {
            continue;
        }
        for (size_t i = 1; i < type.operands.size(); i++)
        {
            auto member = m.types.find(type.operands[i]);
            if (member == m.types.end() || member->second.opcode != OpTypeArray)
            {
                continue;
            }
            auto element = m.types.find(member->second.operands[1]);
            auto stride = m.arrayStrides.find(member->first);
            if (element != m.types.end() && element->second.opcode == OpTypeStruct && stride != m.arrayStrides.end())
            {
                m.indexedStructs[element->first] = stride->second;
            }
        }
    }

    // The structs the shader reads through pointers, by their C++ name.
    const std::string suffix = "_natural";
    std::map<std::string, uint32_t> structs;
    for (const auto& [id, type] : m.types)
    {
        auto name = m.names.find(id);
        if (type.opcode != OpTypeStruct || name == m.names.end() || name->second.size() <= suffix.size() ||
            name->second.compare(name->second.size() - suffix.size(), suffix.size(), suffix) != 0)
        {
            continue;
        }
        std::string cppName = name->second.substr(0, name->second.size() - suffix.size());
        if (isIdentifier(cppName))
        {
            structs.emplace(cppName, id);
        }
    }

    // Only structs that a C++-visible header declares can be checked; the
    // rest exist only in shader code.
    std::vector<std::string> headers;
    std::map<std::string, std::string> declaredIn;
    for (const auto& dep : readDepfile(depPath))
    {
        if (!isHeader(dep))
        {
            continue;
        }
        std::ifstream in(dep, std::ios::binary);
        const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        bool used = false;
        for (const auto& [cppName, id] : structs)
        {
            const std::regex declaration("\\bstruct\\s+(alignas\\s*\\([^)]*\\)\\s*)?" + cppName + "\\s*(:[^{;]*)?\\{");
            if (!declaredIn.count(cppName) && std::regex_search(text, declaration))
            {
                declaredIn[cppName] = dep;
                used = true;
            }
        }
        if (used)
        {
            headers.push_back(dep);
        }
    }

    std::ostringstream out;
    out << "// Generated by ngapi-layout-check for " << shaderName << " -- do not edit.\n"
        << "// Checks that the structs this shader reads through pointers have the same\n"
        << "// layout in C++ as on the GPU (see ngapi/tools/LayoutCheck.cpp).\n";
    std::vector<std::string> skipped;
    for (const auto& [cppName, id] : structs)
    {
        if (!declaredIn.count(cppName))
        {
            skipped.push_back(cppName);
        }
    }
    if (!skipped.empty())
    {
        out << "// Not checked (declared only in shader code):";
        for (const auto& name : skipped)
        {
            out << " " << name;
        }
        out << "\n";
    }
    out << "#include <cstddef>\n";
    for (const auto& header : headers)
    {
        out << "#include \"" << header << "\"\n";
    }

    const std::string where = "GPU layout mismatch (" + escape(shaderName) + "): ";
    for (const auto& [cppName, id] : structs)
    {
        if (!declaredIn.count(cppName))
        {
            continue;
        }
        const auto& members = m.types.at(id).operands;
        out << "\n// " << cppName << "\n";
        for (uint32_t i = 1; i < members.size(); i++)
        {
            auto name = m.memberNames.find({ id, i - 1 });
            auto offset = m.memberOffsets.find({ id, i - 1 });
            if (name == m.memberNames.end() || offset == m.memberOffsets.end() || !isIdentifier(name->second))
            {
                continue;
            }
            const std::string member = cppName + "::" + name->second;
            out << "static_assert(offsetof(" << cppName << ", " << name->second << ") == " << offset->second
                << ", \"" << where << member << " is at byte " << offset->second
                << " in the shader (alignas() padding applies only in C++)\");\n";
            if (auto layout = layoutOf(m, members[i], id, i - 1))
            {
                auto type = m.types.find(members[i]);
                const bool is32BitInt = type != m.types.end() && type->second.opcode == OpTypeInt && type->second.operands[1] == 32;
                out << "static_assert(sizeof(" << member << ") == " << layout->size
                    << ", \"" << where << member << " is " << layout->size << " bytes in the shader"
                    << (is32BitInt ? " (bool is 4 bytes on the GPU; use uint)" : "") << "\");\n";
            }
        }
        if (auto stride = m.indexedStructs.find(id); stride != m.indexedStructs.end())
        {
            out << "static_assert(sizeof(" << cppName << ") == " << stride->second
                << ", \"" << where << "the shader indexes " << cppName << " arrays with a " << stride->second
                << "-byte stride (alignas() padding applies only in C++)\");\n";
        }
    }

    // Only rewrite when the content changes, so an unchanged check does not
    // look newer than it is.
    std::ifstream existing(outPath, std::ios::binary);
    const std::string previous((std::istreambuf_iterator<char>(existing)), std::istreambuf_iterator<char>());
    if (previous != out.str())
    {
        std::ofstream file(outPath, std::ios::binary | std::ios::trunc);
        file << out.str();
        if (!file)
        {
            std::fprintf(stderr, "ngapi-layout-check: cannot write %s\n", outPath.c_str());
            return 1;
        }
    }
    return 0;
}
