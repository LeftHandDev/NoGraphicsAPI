cmake_minimum_required(VERSION 3.22)

project(NoGraphicsAPI
    VERSION 1.0
    DESCRIPTION "A Vulkan implementation of \"No Graphics API\""
    LANGUAGES CXX
)

# ---------------------------------------------------------------------------
# The library
# ---------------------------------------------------------------------------
# The reusable library is the self-contained ngapi/ folder — that folder (or
# this whole repo) is what you embed in your own project; see ngapi/README.md.
# Everything below it is the development scaffolding around the library:
# samples, windowing backends and the headless rendering tests.
add_subdirectory(ngapi)

# Consumers embedding this whole repo still get ngapi's exported variables
# (NGAPI_INCLUDE_DIR and the module path that makes include(CompileShaders) work).
if(NOT PROJECT_IS_TOP_LEVEL)
    set(NGAPI_INCLUDE_DIR "${NGAPI_INCLUDE_DIR}" PARENT_SCOPE)
    set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" PARENT_SCOPE)
endif()

# Samples and tests are for working on NGAPI itself; when this repo is consumed
# via add_subdirectory()/FetchContent they default to off and only ngapi builds.
option(NGAPI_BUILD_SAMPLES "Build the windowed sample executables" ${PROJECT_IS_TOP_LEVEL})
option(NGAPI_BUILD_TESTS "Build the headless rendering tests" ${PROJECT_IS_TOP_LEVEL})

if(NOT (NGAPI_BUILD_SAMPLES OR NGAPI_BUILD_TESTS))
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

# Windowing backend: GLFW (default) or SDL3. Build with -DNGAPI_WINDOW_BACKEND=SDL3
# to switch. Only the windowed samples need a backend.
set(NGAPI_WINDOW_BACKEND "GLFW" CACHE STRING "Windowing backend: GLFW or SDL3")
set_property(CACHE NGAPI_WINDOW_BACKEND PROPERTY STRINGS GLFW SDL3)

if(NGAPI_BUILD_SAMPLES)
if(NGAPI_WINDOW_BACKEND STREQUAL "GLFW")
    set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL        OFF CACHE BOOL "" FORCE)
    add_subdirectory(external/glfw)
elseif(NGAPI_WINDOW_BACKEND STREQUAL "SDL3")
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
    message(FATAL_ERROR "NGAPI_WINDOW_BACKEND must be GLFW or SDL3 (got '${NGAPI_WINDOW_BACKEND}')")
endif()
endif() # NGAPI_BUILD_SAMPLES

# ---------------------------------------------------------------------------
# Windowing abstraction (one backend compiled in) -- only with the samples.
# ---------------------------------------------------------------------------
if(NGAPI_BUILD_SAMPLES)
if(NGAPI_WINDOW_BACKEND STREQUAL "GLFW")
    add_library(ngapi-platform STATIC platform/window_glfw.cpp)
    target_link_libraries(ngapi-platform PRIVATE glfw Vulkan::Vulkan)
else()
    add_library(ngapi-platform STATIC platform/window_sdl3.cpp)
    target_link_libraries(ngapi-platform PRIVATE SDL3::SDL3 Vulkan::Vulkan)
endif()
target_include_directories(ngapi-platform PUBLIC platform)
target_link_libraries(ngapi-platform PUBLIC ngapi::ngapi)
endif()

# ---------------------------------------------------------------------------
# Shared sample utilities
# ---------------------------------------------------------------------------
add_library(ngapi-samples-common STATIC samples/common/Utilities.cpp)
target_include_directories(ngapi-samples-common PUBLIC samples/common)
target_link_libraries(ngapi-samples-common
    PUBLIC  ngapi::ngapi glm::glm
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
        "(Only the samples and tests need it — consuming the ngapi/ library does not.)")
endif()

# Multi-config generators (e.g. Visual Studio) automatically append $<CONFIG>
# to the executable output directory, so shaders and assets must follow suit
# to land next to the .exe at runtime.
if(CMAKE_CONFIGURATION_TYPES)
    set(NGAPI_RUNTIME_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>")
else()
    set(NGAPI_RUNTIME_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
endif()

set(NGAPI_SHADER_OUTPUT_DIR  "${NGAPI_RUNTIME_DIR}/shaders")
set(NGAPI_SHADER_INCLUDE_DIR "${NGAPI_INCLUDE_DIR}")
include(ngapi/cmake/CompileShaders.cmake)

set(COMMON_SHADER_DEPS
    "${NGAPI_INCLUDE_DIR}/NoGraphicsAPI.h"
    "${NGAPI_INCLUDE_DIR}/NoGraphicsAPI_Impl.h"
    "${NGAPI_INCLUDE_DIR}/Config.h"
    "${NGAPI_INCLUDE_DIR}/Sampler.h")

set(NGAPI_SHADER_OUTPUTS "")
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

# All bench entry points in one module (one slangc invocation); the harness
# picks entries at pipeline creation.
compile_shader(SOURCE samples/samplerbench/SamplerBench.slang STAGE compute OUTPUT samplerbench/Bench.spv
    ENTRY benchHardware benchStatic benchInline benchStaticNearest benchManualNearest
    EXTRA_DEPENDS ${COMMON_SHADER_DEPS} "${CMAKE_CURRENT_SOURCE_DIR}/samples/samplerbench/SamplerBench.h")

compile_shader(SOURCE samples/multithreading/Multithreading.slang STAGE compute OUTPUT multithreading/Multithreading.spv
    EXTRA_DEPENDS ${COMMON_SHADER_DEPS} "${CMAKE_CURRENT_SOURCE_DIR}/samples/multithreading/Multithreading.h")

# The learning sample's tensor ops are nine compute entry points in one .slang;
# compile them all into a single Tensor.spv (Device_impl builds a pipeline per
# entry point from that one module).
compile_shader(SOURCE samples/learning/Tensor.slang STAGE compute OUTPUT learning/Tensor.spv
    ENTRY _add _sub _mul _div _dot _mT _matmul _pow _tanh
    EXTRA_DEPENDS ${COMMON_SHADER_DEPS} "${CMAKE_CURRENT_SOURCE_DIR}/samples/learning/Common.h")

add_custom_target(shaders ALL DEPENDS ${NGAPI_SHADER_OUTPUTS})

# The library's internal PatchDescriptors shader is embedded as a checked-in
# header rather than compiled on every build; after editing the .slang,
# regenerate it with this target (see ngapi/cmake/EmbedPatchDescriptors.cmake).
add_custom_target(ngapi-embed-patch-descriptors
    COMMAND ${CMAKE_COMMAND} -DSLANGC=${SLANGC}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/ngapi/cmake/EmbedPatchDescriptors.cmake"
    COMMENT "Re-embedding PatchDescriptors.slang into ngapi/src/PatchDescriptorsSpv.h")

# Copy assets next to the executables (samples load these from the run dir; the
# tests read assets straight from the source tree, so this is samples-only).
if(NGAPI_BUILD_SAMPLES)
add_custom_target(copy_assets ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/assets" "${NGAPI_RUNTIME_DIR}/assets"
    COMMENT "Copying assets")

# ---------------------------------------------------------------------------
# Sample executables (one per sample, output to build/bin)
# ---------------------------------------------------------------------------
# Windowed samples: depend on the windowing layer, shared utilities, shaders and assets.
function(add_windowed_sample name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE ngapi::ngapi ngapi-platform ngapi-samples-common stb glm::glm)
    add_dependencies(${name} shaders copy_assets)
endfunction()

add_windowed_sample(compute    samples/compute/Compute.cpp)
add_windowed_sample(graphics   samples/graphics/Graphics.cpp)
add_windowed_sample(raytracing samples/raytracing/Raytracing.cpp)

# Headless sample: no window, no shaders/assets.
add_executable(multiplegpus samples/multiplegpus/MultipleGPUs.cpp)
target_link_libraries(multiplegpus PRIVATE ngapi::ngapi)

# Scratch sampler regression benchmark (run from build/bin).
add_executable(samplerbench samples/samplerbench/SamplerBench.cpp)
target_link_libraries(samplerbench PRIVATE ngapi::ngapi ngapi-samples-common)
add_dependencies(samplerbench shaders)

# Headless multithreading demo/self-test (run from build/bin).
add_executable(multithreading samples/multithreading/Multithreading.cpp)
target_link_libraries(multithreading PRIVATE ngapi::ngapi ngapi-samples-common)
add_dependencies(multithreading shaders)

# Headless sample that runs compute shaders (a tiny tensor/MLP library): no
# window, but it loads the compiled Tensor.spv at runtime.
add_executable(learning samples/learning/Learning.cpp samples/learning/Tensor.cpp)
target_link_libraries(learning PRIVATE ngapi::ngapi ngapi-samples-common glm::glm)
add_dependencies(learning shaders)
endif() # NGA_BUILD_SAMPLES

# ---------------------------------------------------------------------------
# Headless rendering tests
# ---------------------------------------------------------------------------
if(NGAPI_BUILD_TESTS)
    add_subdirectory(tests)
endif()
