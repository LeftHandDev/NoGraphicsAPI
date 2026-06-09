cmake_minimum_required(VERSION 3.22)

project(NoGraphicsAPI
    VERSION 1.0
    DESCRIPTION "A Vulkan implementation of \"No Graphics API\""
    LANGUAGES CXX
)

# ---------------------------------------------------------------------------
# The library
# ---------------------------------------------------------------------------
# The reusable library is the self-contained nga/ folder — that folder (or
# this whole repo) is what you embed in your own project; see nga/README.md.
# Everything below it is the development scaffolding around the library:
# samples, windowing backends and the headless rendering tests.
add_subdirectory(nga)

# Consumers embedding this whole repo still get nga's exported variables
# (NGA_INCLUDE_DIR and the module path that makes include(CompileShaders) work).
if(NOT PROJECT_IS_TOP_LEVEL)
    set(NGA_INCLUDE_DIR "${NGA_INCLUDE_DIR}" PARENT_SCOPE)
    set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" PARENT_SCOPE)
endif()

# Samples and tests are for working on NGA itself; when this repo is consumed
# via add_subdirectory()/FetchContent they default to off and only nga builds.
option(NGA_BUILD_SAMPLES "Build the windowed sample executables" ${PROJECT_IS_TOP_LEVEL})
option(NGA_BUILD_TESTS "Build the headless rendering tests" ${PROJECT_IS_TOP_LEVEL})

if(NOT (NGA_BUILD_SAMPLES OR NGA_BUILD_TESTS))
    return()
endif()

# All executables (and any shared libs they need, e.g. SDL3) land in build/bin,
# alongside the shaders/ and assets/ they load at runtime.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_BUILD_RPATH "$ORIGIN")

# ---------------------------------------------------------------------------
# Sample/test dependencies (git submodules under external/, built from source)
# ---------------------------------------------------------------------------
add_subdirectory(external/glm)

# stb: compile the implementations once into a small static lib.
add_library(stb STATIC external/stb/stb.cpp)
target_include_directories(stb PUBLIC external/stb)

# Windowing backend: GLFW (default) or SDL3. Build with -DNGA_WINDOW_BACKEND=SDL3
# to switch. Only the windowed samples need a backend.
set(NGA_WINDOW_BACKEND "GLFW" CACHE STRING "Windowing backend: GLFW or SDL3")
set_property(CACHE NGA_WINDOW_BACKEND PROPERTY STRINGS GLFW SDL3)

if(NGA_BUILD_SAMPLES)
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
endif() # NGA_BUILD_SAMPLES

# ---------------------------------------------------------------------------
# Windowing abstraction (one backend compiled in) -- only with the samples.
# ---------------------------------------------------------------------------
if(NGA_BUILD_SAMPLES)
if(NGA_WINDOW_BACKEND STREQUAL "GLFW")
    add_library(nga-platform STATIC platform/window_glfw.cpp)
    target_link_libraries(nga-platform PRIVATE glfw Vulkan::Vulkan)
else()
    add_library(nga-platform STATIC platform/window_sdl3.cpp)
    target_link_libraries(nga-platform PRIVATE SDL3::SDL3 Vulkan::Vulkan)
endif()
target_include_directories(nga-platform PUBLIC platform)
target_link_libraries(nga-platform PUBLIC nga::nga)
endif()

# ---------------------------------------------------------------------------
# Shared sample utilities
# ---------------------------------------------------------------------------
add_library(nga-samples-common STATIC samples/common/Utilities.cpp)
target_include_directories(nga-samples-common PUBLIC samples/common)
target_link_libraries(nga-samples-common
    PUBLIC  nga::nga glm::glm
    PRIVATE stb)

# ---------------------------------------------------------------------------
# Shader compilation (slang -> SPIR-V) for the samples and tests
# ---------------------------------------------------------------------------
if(NOT DEFINED VULKAN_SDK AND DEFINED ENV{VULKAN_SDK})
    set(VULKAN_SDK $ENV{VULKAN_SDK})
endif()
find_program(SLANGC slangc HINTS "${VULKAN_SDK}/Bin" "${VULKAN_SDK}/bin")
if(NOT SLANGC)
    message(FATAL_ERROR "slangc not found; set VULKAN_SDK or put slangc on PATH. "
        "(Only the samples and tests need it — consuming the nga/ library does not.)")
endif()

# Multi-config generators (e.g. Visual Studio) automatically append $<CONFIG>
# to the executable output directory, so shaders and assets must follow suit
# to land next to the .exe at runtime.
if(CMAKE_CONFIGURATION_TYPES)
    set(NGA_RUNTIME_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>")
else()
    set(NGA_RUNTIME_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
endif()

set(NGA_SHADER_OUTPUT_DIR  "${NGA_RUNTIME_DIR}/shaders")
set(NGA_SHADER_INCLUDE_DIR "${NGA_INCLUDE_DIR}")
include(nga/cmake/CompileShaders.cmake)

set(COMMON_SHADER_DEPS
    "${NGA_INCLUDE_DIR}/NoGraphicsAPI.h"
    "${NGA_INCLUDE_DIR}/NoGraphicsAPI_Impl.h"
    "${NGA_INCLUDE_DIR}/Config.h")

set(NGA_SHADER_OUTPUTS "")
compile_shader(SOURCE samples/compute/Compute.slang     STAGE compute  OUTPUT compute/Compute.spv
    EXTRA_DEPENDS ${COMMON_SHADER_DEPS} "${CMAKE_CURRENT_SOURCE_DIR}/samples/compute/Compute.h")
compile_shader(SOURCE samples/graphics/Vertex.slang     STAGE vertex   OUTPUT graphics/Vertex.spv
    EXTRA_DEPENDS ${COMMON_SHADER_DEPS} "${CMAKE_CURRENT_SOURCE_DIR}/samples/graphics/Graphics.h")
compile_shader(SOURCE samples/graphics/Pixel.slang      STAGE fragment OUTPUT graphics/Pixel.spv
    EXTRA_DEPENDS ${COMMON_SHADER_DEPS} "${CMAKE_CURRENT_SOURCE_DIR}/samples/graphics/Graphics.h")
compile_shader(SOURCE samples/common/TAA.slang          STAGE compute  OUTPUT common/TAA.spv
    EXTRA_DEPENDS ${COMMON_SHADER_DEPS} "${CMAKE_CURRENT_SOURCE_DIR}/samples/common/TAA.h")
compile_shader(SOURCE samples/common/TextVertex.slang   STAGE vertex   OUTPUT common/TextVertex.spv
    EXTRA_DEPENDS ${COMMON_SHADER_DEPS} "${CMAKE_CURRENT_SOURCE_DIR}/samples/common/Text.h")
compile_shader(SOURCE samples/common/TextPixel.slang    STAGE fragment OUTPUT common/TextPixel.spv
    EXTRA_DEPENDS ${COMMON_SHADER_DEPS} "${CMAKE_CURRENT_SOURCE_DIR}/samples/common/Text.h")

set(RT_SHADER_DEPS ${COMMON_SHADER_DEPS}
    "${CMAKE_CURRENT_SOURCE_DIR}/samples/raytracing/Raytracing.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/samples/raytracing/Random.h")
compile_shader(SOURCE samples/raytracing/Raytracing.slang STAGE compute OUTPUT raytracing/Raytracing.spv
    EXTRA_DEPENDS ${RT_SHADER_DEPS})
compile_shader(SOURCE samples/raytracing/ReSTIR.slang   STAGE compute  OUTPUT raytracing/RIS.spv   ENTRY RIS
    EXTRA_DEPENDS ${RT_SHADER_DEPS})
compile_shader(SOURCE samples/raytracing/ReSTIR.slang   STAGE compute  OUTPUT raytracing/Reuse.spv ENTRY Reuse
    EXTRA_DEPENDS ${RT_SHADER_DEPS})
compile_shader(SOURCE samples/raytracing/ReSTIR.slang   STAGE compute  OUTPUT raytracing/Shade.spv ENTRY Shade
    EXTRA_DEPENDS ${RT_SHADER_DEPS})

add_custom_target(shaders ALL DEPENDS ${NGA_SHADER_OUTPUTS})

# The library's internal PatchDescriptors shader is embedded as a checked-in
# header rather than compiled on every build; after editing the .slang,
# regenerate it with this target (see nga/cmake/EmbedPatchDescriptors.cmake).
add_custom_target(nga-embed-patch-descriptors
    COMMAND ${CMAKE_COMMAND} -DSLANGC=${SLANGC}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/nga/cmake/EmbedPatchDescriptors.cmake"
    COMMENT "Re-embedding PatchDescriptors.slang into nga/src/PatchDescriptorsSpv.h")

# Copy assets next to the executables (samples load these from the run dir; the
# tests read assets straight from the source tree, so this is samples-only).
if(NGA_BUILD_SAMPLES)
add_custom_target(copy_assets ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/assets" "${NGA_RUNTIME_DIR}/assets"
    COMMENT "Copying assets")

# ---------------------------------------------------------------------------
# Sample executables (one per sample, output to build/bin)
# ---------------------------------------------------------------------------
# Windowed samples: depend on the windowing layer, shared utilities, shaders and assets.
function(add_windowed_sample name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE nga::nga nga-platform nga-samples-common stb glm::glm)
    add_dependencies(${name} shaders copy_assets)
endfunction()

add_windowed_sample(compute    samples/compute/Compute.cpp)
add_windowed_sample(graphics   samples/graphics/Graphics.cpp)
add_windowed_sample(raytracing samples/raytracing/Raytracing.cpp)

# Headless sample: no window, no shaders/assets.
add_executable(multiplegpus samples/multiplegpus/MultipleGPUs.cpp)
target_link_libraries(multiplegpus PRIVATE nga::nga)
endif() # NGA_BUILD_SAMPLES

# ---------------------------------------------------------------------------
# Headless rendering tests
# ---------------------------------------------------------------------------
if(NGA_BUILD_TESTS)
    add_subdirectory(tests)
endif()
