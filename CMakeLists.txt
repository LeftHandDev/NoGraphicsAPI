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
    NoGraphicsAPI_Impl.cpp
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
    set(ALL_SHADER_OUTPUTS "")

    # compile_shader(SOURCE <path> STAGE <stage> OUTPUT <rel.spv> [EXTRA_DEPENDS ...])
    #   SOURCE       – .slang path relative to CMAKE_SOURCE_DIR
    #   STAGE        – spirv stage (compute, vertex, fragment, …)
    #   OUTPUT       – output .spv path relative to Shaders/
    #   EXTRA_DEPENDS – optional additional file dependencies
    function(compile_shader)
        cmake_parse_arguments(S "" "SOURCE;STAGE;OUTPUT" "EXTRA_DEPENDS" ${ARGN})
        get_filename_component(SPV_DIR  "${S_OUTPUT}" DIRECTORY)
        get_filename_component(SPV_NAME "${S_OUTPUT}" NAME)

        add_custom_command(
            OUTPUT "${CMAKE_BINARY_DIR}/Shaders/${S_OUTPUT}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${SPV_OUTPUT_DIR}/Shaders/${SPV_DIR}"
            COMMAND "${SLANGC}" "${CMAKE_SOURCE_DIR}/${S_SOURCE}"
                    -target spirv -stage ${S_STAGE} -entry main
                    -I "${SHADER_INCLUDE_DIR}"
                    -o "${SPV_OUTPUT_DIR}/Shaders/${S_OUTPUT}"
            COMMAND ${CMAKE_COMMAND} -E copy
                    "${SPV_OUTPUT_DIR}/Shaders/${S_OUTPUT}"
                    "${CMAKE_BINARY_DIR}/Shaders/${S_OUTPUT}"
            DEPENDS "${CMAKE_SOURCE_DIR}/${S_SOURCE}"
                    "${SHADER_INCLUDE_DIR}/NoGraphicsAPI.h"
                    ${S_EXTRA_DEPENDS}
            COMMENT "Compiling ${SPV_NAME}"
            VERBATIM)

        set(ALL_SHADER_OUTPUTS ${ALL_SHADER_OUTPUTS}
            "${CMAKE_BINARY_DIR}/Shaders/${S_OUTPUT}" PARENT_SCOPE)
    endfunction()

    # Register shaders
    compile_shader(SOURCE Samples/Compute/Compute.slang       STAGE compute  OUTPUT Compute/Compute.spv)
    compile_shader(SOURCE Samples/Graphics/Vertex.slang       STAGE vertex   OUTPUT Graphics/Vertex.spv)
    compile_shader(SOURCE Samples/Graphics/Pixel.slang        STAGE fragment OUTPUT Graphics/Pixel.spv)
    compile_shader(SOURCE Samples/Common/TAA.slang            STAGE compute  OUTPUT Common/TAA.spv)
    compile_shader(SOURCE Samples/Raytracing/Raytracing.slang STAGE compute  OUTPUT Raytracing/Raytracing.spv)
    compile_shader(SOURCE Samples/Raytracing/ReSTIR.slang     STAGE compute  OUTPUT Raytracing/ReSTIR.spv)
    compile_shader(SOURCE Shaders/PatchDescriptors.slang      STAGE compute  OUTPUT PatchDescriptors.spv)

    add_custom_target(Shaders ALL DEPENDS ${ALL_SHADER_OUTPUTS})
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