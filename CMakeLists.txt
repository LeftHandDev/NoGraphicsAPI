cmake_minimum_required(VERSION 3.22)

project(NoGraphicsAPI
    VERSION 1.0
    DESCRIPTION "A Vulkan implementation of \"No Graphics API\""
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# All executables (and any shared libs they need, e.g. SDL3) land in build/bin,
# alongside the shaders/ and assets/ they load at runtime.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_BUILD_RPATH "$ORIGIN")

# Windowing backend: GLFW (default) or SDL3. Build with -DNGA_WINDOW_BACKEND=SDL3 to switch.
set(NGA_WINDOW_BACKEND "GLFW" CACHE STRING "Windowing backend: GLFW or SDL3")
set_property(CACHE NGA_WINDOW_BACKEND PROPERTY STRINGS GLFW SDL3)

find_package(Vulkan REQUIRED)

# ---------------------------------------------------------------------------
# Dependencies (git submodules under external/, built from source)
# ---------------------------------------------------------------------------
add_subdirectory(external/glm)
add_subdirectory(external/vk-bootstrap)

# stb: compile the implementations once into a small static lib.
add_library(stb STATIC external/stb/stb.cpp)
target_include_directories(stb PUBLIC external/stb)

if(NGA_WINDOW_BACKEND STREQUAL "GLFW")
    set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL        OFF CACHE BOOL "" FORCE)
    add_subdirectory(external/glfw)
elseif(NGA_WINDOW_BACKEND STREQUAL "SDL3")
    set(SDL_TEST_LIBRARY    OFF CACHE BOOL "" FORCE)
    set(SDL_TESTS           OFF CACHE BOOL "" FORCE)
    set(SDL_EXAMPLES        OFF CACHE BOOL "" FORCE)
    set(SDL_INSTALL         OFF CACHE BOOL "" FORCE)
    # Build the Wayland backend only. SDL's X11 backend needs several libX*-dev
    # packages (Xss, Xtst, ...) that aren't always installed; enable SDL_X11 if
    # you have them. (GLFW, the default backend, builds X11 + Wayland regardless.)
    set(SDL_X11             OFF CACHE BOOL "" FORCE)
    set(SDL_WAYLAND         ON  CACHE BOOL "" FORCE)
    add_subdirectory(external/SDL)
else()
    message(FATAL_ERROR "NGA_WINDOW_BACKEND must be GLFW or SDL3 (got '${NGA_WINDOW_BACKEND}')")
endif()

# ---------------------------------------------------------------------------
# Core "No Graphics API" library
# ---------------------------------------------------------------------------
add_library(nographicsapi STATIC src/NoGraphicsAPI_Impl.cpp)
target_include_directories(nographicsapi
    PUBLIC  src
    PRIVATE shaders)                       # PatchDescriptors.h
target_link_libraries(nographicsapi
    PRIVATE Vulkan::Vulkan vk-bootstrap::vk-bootstrap)

# ---------------------------------------------------------------------------
# Windowing abstraction (one backend compiled in)
# ---------------------------------------------------------------------------
if(NGA_WINDOW_BACKEND STREQUAL "GLFW")
    add_library(nga-platform STATIC platform/window_glfw.cpp)
    target_link_libraries(nga-platform PRIVATE glfw Vulkan::Vulkan)
else()
    add_library(nga-platform STATIC platform/window_sdl3.cpp)
    target_link_libraries(nga-platform PRIVATE SDL3::SDL3 Vulkan::Vulkan)
endif()
target_include_directories(nga-platform PUBLIC platform)
target_link_libraries(nga-platform PUBLIC nographicsapi)

# ---------------------------------------------------------------------------
# Shared sample utilities
# ---------------------------------------------------------------------------
add_library(nga-samples-common STATIC samples/common/Utilities.cpp)
target_include_directories(nga-samples-common PUBLIC samples/common)
target_link_libraries(nga-samples-common
    PUBLIC  nographicsapi glm::glm
    PRIVATE stb)

# ---------------------------------------------------------------------------
# Shader compilation (slang -> SPIR-V)
# ---------------------------------------------------------------------------
if(NOT DEFINED VULKAN_SDK AND DEFINED ENV{VULKAN_SDK})
    set(VULKAN_SDK $ENV{VULKAN_SDK})
endif()
find_program(SLANGC slangc HINTS "${VULKAN_SDK}/Bin" "${VULKAN_SDK}/bin")
if(NOT SLANGC)
    message(FATAL_ERROR "slangc not found; set VULKAN_SDK or put slangc on PATH.")
endif()

set(NGA_SHADER_OUTPUT_DIR  "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders")
set(NGA_SHADER_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/src")
include(CompileShaders)

set(COMMON_SHADER_DEPS
    "${CMAKE_SOURCE_DIR}/src/NoGraphicsAPI.h"
    "${CMAKE_SOURCE_DIR}/src/NoGraphicsAPI_Impl.h"
    "${CMAKE_SOURCE_DIR}/src/Config.h")

set(NGA_SHADER_OUTPUTS "")
compile_shader(SOURCE samples/compute/Compute.slang     STAGE compute  OUTPUT compute/Compute.spv
    EXTRA_DEPENDS ${COMMON_SHADER_DEPS} "${CMAKE_SOURCE_DIR}/samples/compute/Compute.h")
compile_shader(SOURCE samples/graphics/Vertex.slang     STAGE vertex   OUTPUT graphics/Vertex.spv
    EXTRA_DEPENDS ${COMMON_SHADER_DEPS} "${CMAKE_SOURCE_DIR}/samples/graphics/Graphics.h")
compile_shader(SOURCE samples/graphics/Pixel.slang      STAGE fragment OUTPUT graphics/Pixel.spv
    EXTRA_DEPENDS ${COMMON_SHADER_DEPS} "${CMAKE_SOURCE_DIR}/samples/graphics/Graphics.h")
compile_shader(SOURCE samples/common/TAA.slang          STAGE compute  OUTPUT common/TAA.spv
    EXTRA_DEPENDS ${COMMON_SHADER_DEPS} "${CMAKE_SOURCE_DIR}/samples/common/TAA.h")
compile_shader(SOURCE samples/common/TextVertex.slang   STAGE vertex   OUTPUT common/TextVertex.spv
    EXTRA_DEPENDS ${COMMON_SHADER_DEPS} "${CMAKE_SOURCE_DIR}/samples/common/Text.h")
compile_shader(SOURCE samples/common/TextPixel.slang    STAGE fragment OUTPUT common/TextPixel.spv
    EXTRA_DEPENDS ${COMMON_SHADER_DEPS} "${CMAKE_SOURCE_DIR}/samples/common/Text.h")

set(RT_SHADER_DEPS ${COMMON_SHADER_DEPS}
    "${CMAKE_SOURCE_DIR}/samples/raytracing/Raytracing.h"
    "${CMAKE_SOURCE_DIR}/samples/raytracing/Random.h")
compile_shader(SOURCE samples/raytracing/Raytracing.slang STAGE compute OUTPUT raytracing/Raytracing.spv
    EXTRA_DEPENDS ${RT_SHADER_DEPS})
compile_shader(SOURCE samples/raytracing/ReSTIR.slang   STAGE compute  OUTPUT raytracing/RIS.spv   ENTRY RIS
    EXTRA_DEPENDS ${RT_SHADER_DEPS})
compile_shader(SOURCE samples/raytracing/ReSTIR.slang   STAGE compute  OUTPUT raytracing/Reuse.spv ENTRY Reuse
    EXTRA_DEPENDS ${RT_SHADER_DEPS})
compile_shader(SOURCE samples/raytracing/ReSTIR.slang   STAGE compute  OUTPUT raytracing/Shade.spv ENTRY Shade
    EXTRA_DEPENDS ${RT_SHADER_DEPS})
compile_shader(SOURCE shaders/PatchDescriptors.slang    STAGE compute  OUTPUT PatchDescriptors.spv
    EXTRA_DEPENDS ${COMMON_SHADER_DEPS} "${CMAKE_SOURCE_DIR}/shaders/PatchDescriptors.h")

add_custom_target(shaders ALL DEPENDS ${NGA_SHADER_OUTPUTS})

# Copy assets next to the executables
add_custom_target(copy_assets ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/assets" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/assets"
    COMMENT "Copying assets")

# ---------------------------------------------------------------------------
# Sample executables (one per sample, output to build/bin)
# ---------------------------------------------------------------------------
# Windowed samples: depend on the windowing layer, shared utilities, shaders and assets.
function(add_windowed_sample name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE nographicsapi nga-platform nga-samples-common stb glm::glm)
    add_dependencies(${name} shaders copy_assets)
endfunction()

add_windowed_sample(compute    samples/compute/Compute.cpp)
add_windowed_sample(graphics   samples/graphics/Graphics.cpp)
add_windowed_sample(raytracing samples/raytracing/Raytracing.cpp)

# Headless sample: no window, no shaders/assets.
add_executable(multiplegpus samples/multiplegpus/MultipleGPUs.cpp)
target_link_libraries(multiplegpus PRIVATE nographicsapi)
