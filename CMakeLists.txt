cmake_minimum_required(VERSION 3.22)

project(NoGraphicsAPI
    VERSION 1.0
    DESCRIPTION "A Vulkan implementation of \"No Graphics API\""
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)
find_package(SDL3 CONFIG QUIET)
find_package(glm QUIET)

if(NOT DEFINED VULKAN_SDK AND DEFINED ENV{VULKAN_SDK})
    set(VULKAN_SDK $ENV{VULKAN_SDK})
endif()

if(NOT DEFINED VULKAN_SDK)
    message(FATAL_ERROR "VULKAN_SDK env var is not set")
endif()

add_executable(NoGraphicsAPI
    External/VkBootstrap.cpp
    SDL_gpu.cpp
    NoGraphicsAPI.cpp
    Utilities.cpp
    Main.cpp
    Samples/Compute/Compute.cpp
    Samples/Graphics/Graphics.cpp
    Samples/Raytracing/Raytracing.cpp
)

if(TARGET Vulkan::Vulkan)
    target_link_libraries(NoGraphicsAPI PRIVATE Vulkan::Vulkan)
else()
    target_include_directories(NoGraphicsAPI PRIVATE ${Vulkan_INCLUDE_DIRS})
    target_link_libraries(NoGraphicsAPI PRIVATE ${Vulkan_LIBRARIES})
endif()

# Shader compilation (slang): compile .slang to .spv using the Vulkan SDK's slangc
set(SLANGC "${VULKAN_SDK}/Bin/slangc.exe")
if(NOT EXISTS "${SLANGC}")
    message(WARNING "slangc.exe not found at ${SLANGC}; skipping shader compilation target.")
else()
    set(SHADER_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/Shaders")

    # Compute (Samples/Compute)
    add_custom_command(
        OUTPUT "${CMAKE_SOURCE_DIR}/Samples/Compute/Compute.spv"
        COMMAND "${SLANGC}" "${CMAKE_SOURCE_DIR}/Samples/Compute/Compute.slang"
                -target spirv -stage compute -entry main
                -I "${SHADER_INCLUDE_DIR}"
                -o "${CMAKE_SOURCE_DIR}/Samples/Compute/Compute.spv"
        DEPENDS "${CMAKE_SOURCE_DIR}/Samples/Compute/Compute.slang" "${SHADER_INCLUDE_DIR}/NoGraphicsAPI.h"
        COMMENT "Compiling Compute.slang -> Compute.spv"
        VERBATIM)

    # Graphics (vertex + fragment)
    add_custom_command(
        OUTPUT "${CMAKE_SOURCE_DIR}/Samples/Graphics/Vertex.spv"
        COMMAND "${SLANGC}" "${CMAKE_SOURCE_DIR}/Samples/Graphics/Vertex.slang"
                -target spirv -stage vertex -entry main
                -I "${SHADER_INCLUDE_DIR}"
                -o "${CMAKE_SOURCE_DIR}/Samples/Graphics/Vertex.spv"
        DEPENDS "${CMAKE_SOURCE_DIR}/Samples/Graphics/Vertex.slang" "${SHADER_INCLUDE_DIR}/NoGraphicsAPI.h"
        COMMENT "Compiling Vertex.slang -> Vertex.spv"
        VERBATIM)

    add_custom_command(
        OUTPUT "${CMAKE_SOURCE_DIR}/Samples/Graphics/Pixel.spv"
        COMMAND "${SLANGC}" "${CMAKE_SOURCE_DIR}/Samples/Graphics/Pixel.slang"
                -target spirv -stage fragment -entry main
                -I "${SHADER_INCLUDE_DIR}"
                -o "${CMAKE_SOURCE_DIR}/Samples/Graphics/Pixel.spv"
        DEPENDS "${CMAKE_SOURCE_DIR}/Samples/Graphics/Pixel.slang" "${SHADER_INCLUDE_DIR}/NoGraphicsAPI.h"
        COMMENT "Compiling Pixel.slang -> Pixel.spv"
        VERBATIM)

    # Raytracing sample currently uses a compute shader
    add_custom_command(
        OUTPUT "${CMAKE_SOURCE_DIR}/Samples/Raytracing/Raytracing.spv"
        COMMAND "${SLANGC}" "${CMAKE_SOURCE_DIR}/Samples/Raytracing/Raytracing.slang"
                -target spirv -stage compute -entry main
                -I "${SHADER_INCLUDE_DIR}"
                -o "${CMAKE_SOURCE_DIR}/Samples/Raytracing/Raytracing.spv"
        DEPENDS "${CMAKE_SOURCE_DIR}/Samples/Raytracing/Raytracing.slang" "${SHADER_INCLUDE_DIR}/NoGraphicsAPI.h"
        COMMENT "Compiling Raytracing.slang -> Raytracing.spv"
        VERBATIM)

    add_custom_target(Shaders ALL
        DEPENDS
            "${CMAKE_SOURCE_DIR}/Samples/Compute/Compute.spv"
            "${CMAKE_SOURCE_DIR}/Samples/Graphics/Vertex.spv"
            "${CMAKE_SOURCE_DIR}/Samples/Graphics/Pixel.spv"
            "${CMAKE_SOURCE_DIR}/Samples/Raytracing/Raytracing.spv")

    add_dependencies(NoGraphicsAPI Shaders)
endif()

# Prefer SDL3 CMake package; fall back to the Vulkan SDK copy via the env var
if(TARGET SDL3::SDL3)
    target_link_libraries(NoGraphicsAPI PRIVATE SDL3::SDL3)
else()
    target_include_directories(NoGraphicsAPI PRIVATE "${VULKAN_SDK}/Include/SDL3")
    target_link_libraries(NoGraphicsAPI PRIVATE "${VULKAN_SDK}/Lib/SDL3.lib")
endif()

if(TARGET glm::glm)
    target_link_libraries(NoGraphicsAPI PRIVATE glm::glm)
else()
    target_include_directories(NoGraphicsAPI PRIVATE "${VULKAN_SDK}/Include/glm")
endif()