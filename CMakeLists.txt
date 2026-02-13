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

# Shader compilation (slang): compile .slang to .spv next to the executable
find_program(SLANGC slangc HINTS "${VULKAN_SDK}/Bin" "${VULKAN_SDK}/bin")
if(NOT SLANGC)
    message(WARNING "slangc not found; skipping shader compilation target.")
else()
    set(SHADER_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/Shaders")
    set(SPV_OUTPUT_DIR "$<TARGET_FILE_DIR:NoGraphicsAPI>")

    # Compute (Samples/Compute)
    add_custom_command(
        OUTPUT "${CMAKE_BINARY_DIR}/Shaders/Compute/Compute.spv"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${SPV_OUTPUT_DIR}/Shaders/Compute"
        COMMAND "${SLANGC}" "${CMAKE_SOURCE_DIR}/Samples/Compute/Compute.slang"
                -target spirv -stage compute -entry main
                -I "${SHADER_INCLUDE_DIR}"
                -o "${SPV_OUTPUT_DIR}/Shaders/Compute/Compute.spv"
        COMMAND ${CMAKE_COMMAND} -E copy
                "${SPV_OUTPUT_DIR}/Shaders/Compute/Compute.spv"
                "${CMAKE_BINARY_DIR}/Shaders/Compute/Compute.spv"
        DEPENDS "${CMAKE_SOURCE_DIR}/Samples/Compute/Compute.slang" "${SHADER_INCLUDE_DIR}/NoGraphicsAPI.h"
        COMMENT "Compiling Compute.slang -> Compute.spv"
        VERBATIM)

    # Graphics (vertex + fragment)
    add_custom_command(
        OUTPUT "${CMAKE_BINARY_DIR}/Shaders/Graphics/Vertex.spv"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${SPV_OUTPUT_DIR}/Shaders/Graphics"
        COMMAND "${SLANGC}" "${CMAKE_SOURCE_DIR}/Samples/Graphics/Vertex.slang"
                -target spirv -stage vertex -entry main
                -I "${SHADER_INCLUDE_DIR}"
                -o "${SPV_OUTPUT_DIR}/Shaders/Graphics/Vertex.spv"
        COMMAND ${CMAKE_COMMAND} -E copy
                "${SPV_OUTPUT_DIR}/Shaders/Graphics/Vertex.spv"
                "${CMAKE_BINARY_DIR}/Shaders/Graphics/Vertex.spv"
        DEPENDS "${CMAKE_SOURCE_DIR}/Samples/Graphics/Vertex.slang" "${SHADER_INCLUDE_DIR}/NoGraphicsAPI.h"
        COMMENT "Compiling Vertex.slang -> Vertex.spv"
        VERBATIM)

    add_custom_command(
        OUTPUT "${CMAKE_BINARY_DIR}/Shaders/Graphics/Pixel.spv"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${SPV_OUTPUT_DIR}/Shaders/Graphics"
        COMMAND "${SLANGC}" "${CMAKE_SOURCE_DIR}/Samples/Graphics/Pixel.slang"
                -target spirv -stage fragment -entry main
                -I "${SHADER_INCLUDE_DIR}"
                -o "${SPV_OUTPUT_DIR}/Shaders/Graphics/Pixel.spv"
        COMMAND ${CMAKE_COMMAND} -E copy
                "${SPV_OUTPUT_DIR}/Shaders/Graphics/Pixel.spv"
                "${CMAKE_BINARY_DIR}/Shaders/Graphics/Pixel.spv"
        DEPENDS "${CMAKE_SOURCE_DIR}/Samples/Graphics/Pixel.slang" "${SHADER_INCLUDE_DIR}/NoGraphicsAPI.h"
        COMMENT "Compiling Pixel.slang -> Pixel.spv"
        VERBATIM)

    add_custom_command(
        OUTPUT "${CMAKE_BINARY_DIR}/Shaders/Graphics/TAA.spv"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${SPV_OUTPUT_DIR}/Shaders/Graphics"
        COMMAND "${SLANGC}" "${CMAKE_SOURCE_DIR}/Samples/Graphics/TAA.slang"
                -target spirv -stage compute -entry main
                -I "${SHADER_INCLUDE_DIR}"
                -o "${SPV_OUTPUT_DIR}/Shaders/Graphics/TAA.spv"
        COMMAND ${CMAKE_COMMAND} -E copy
                "${SPV_OUTPUT_DIR}/Shaders/Graphics/TAA.spv"
                "${CMAKE_BINARY_DIR}/Shaders/Graphics/TAA.spv"
        DEPENDS "${CMAKE_SOURCE_DIR}/Samples/Graphics/TAA.slang" "${SHADER_INCLUDE_DIR}/NoGraphicsAPI.h"
        COMMENT "Compiling TAA.slang -> TAA.spv"
        VERBATIM)

    # Raytracing sample currently uses a compute shader
    add_custom_command(
        OUTPUT "${CMAKE_BINARY_DIR}/Shaders/Raytracing/Raytracing.spv"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${SPV_OUTPUT_DIR}/Shaders/Raytracing"
        COMMAND "${SLANGC}" "${CMAKE_SOURCE_DIR}/Samples/Raytracing/Raytracing.slang"
                -target spirv -stage compute -entry main
                -I "${SHADER_INCLUDE_DIR}"
                -o "${SPV_OUTPUT_DIR}/Shaders/Raytracing/Raytracing.spv"
        COMMAND ${CMAKE_COMMAND} -E copy
                "${SPV_OUTPUT_DIR}/Shaders/Raytracing/Raytracing.spv"
                "${CMAKE_BINARY_DIR}/Shaders/Raytracing/Raytracing.spv"
        DEPENDS "${CMAKE_SOURCE_DIR}/Samples/Raytracing/Raytracing.slang" "${SHADER_INCLUDE_DIR}/NoGraphicsAPI.h"
        COMMENT "Compiling Raytracing.slang -> Raytracing.spv"
        VERBATIM)

    # PatchDescriptorsHeap
    add_custom_command(
        OUTPUT "${CMAKE_BINARY_DIR}/Shaders/PatchDescriptors.spv"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${SPV_OUTPUT_DIR}/Shaders"
        COMMAND "${SLANGC}" "${CMAKE_SOURCE_DIR}/Shaders/PatchDescriptors.slang"
                -target spirv -stage compute -entry main
                -I "${SHADER_INCLUDE_DIR}"
                -o "${SPV_OUTPUT_DIR}/Shaders/PatchDescriptors.spv"
        COMMAND ${CMAKE_COMMAND} -E copy
                "${SPV_OUTPUT_DIR}/Shaders/PatchDescriptors.spv"
                "${CMAKE_BINARY_DIR}/Shaders/PatchDescriptors.spv"
        DEPENDS "${CMAKE_SOURCE_DIR}/Shaders/PatchDescriptors.slang" "${SHADER_INCLUDE_DIR}/NoGraphicsAPI.h"
        COMMENT "Compiling PatchDescriptors.slang -> PatchDescriptors.spv"
        VERBATIM)

    add_custom_target(Shaders ALL
        DEPENDS
            "${CMAKE_BINARY_DIR}/Shaders/Compute/Compute.spv"
            "${CMAKE_BINARY_DIR}/Shaders/Graphics/Vertex.spv"
            "${CMAKE_BINARY_DIR}/Shaders/Graphics/Pixel.spv"
            "${CMAKE_BINARY_DIR}/Shaders/Graphics/TAA.spv"
            "${CMAKE_BINARY_DIR}/Shaders/Raytracing/Raytracing.spv"
            "${CMAKE_BINARY_DIR}/Shaders/PatchDescriptors.spv")

    add_dependencies(NoGraphicsAPI Shaders)
endif()

# Copy assets to the build directory next to the executable
add_custom_command(TARGET NoGraphicsAPI POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:NoGraphicsAPI>/Assets"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/Assets"
        "$<TARGET_FILE_DIR:NoGraphicsAPI>/Assets"
    COMMENT "Copying Assets to build output directory"
    VERBATIM)

# Prefer SDL3 CMake package; fall back to the Vulkan SDK copy via the env var
if(TARGET SDL3::SDL3)
    target_link_libraries(NoGraphicsAPI PRIVATE SDL3::SDL3)
else()
    find_path(SDL3_INCLUDE_DIR SDL3/SDL.h HINTS "${VULKAN_SDK}/Include")
    find_library(SDL3_LIBRARY NAMES SDL3 HINTS "${VULKAN_SDK}/Lib" "${VULKAN_SDK}/lib")
    if(SDL3_INCLUDE_DIR AND SDL3_LIBRARY)
        target_include_directories(NoGraphicsAPI PRIVATE "${SDL3_INCLUDE_DIR}")
        target_link_libraries(NoGraphicsAPI PRIVATE "${SDL3_LIBRARY}")
    else()
        message(FATAL_ERROR "SDL3 not found. Install via Vulkan SDK or provide SDL3 CMake package.")
    endif()
endif()

if(TARGET glm::glm)
    target_link_libraries(NoGraphicsAPI PRIVATE glm::glm)
else()
    find_path(GLM_INCLUDE_DIR glm/glm.hpp HINTS "${VULKAN_SDK}/Include")
    if(GLM_INCLUDE_DIR)
        target_include_directories(NoGraphicsAPI PRIVATE "${GLM_INCLUDE_DIR}")
    else()
        message(FATAL_ERROR "GLM not found. Install via Vulkan SDK or provide glm CMake package.")
    endif()
endif()