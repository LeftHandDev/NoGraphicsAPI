# NoGraphicsAPI ("nga") — self-contained CMake package.
#
# This file is the single entry point for embedding NGA in another project.
# Any of these work, with or without git submodules:
#
#   add_subdirectory(path/to/nga)            # this folder, copied or submoduled
#   add_subdirectory(path/to/NoGraphicsAPI)  # the whole repo
#   include(path/to/nga/nga.cmake)           # no subdirectory at all
#   FetchContent on the repo (or just this folder)
#
# then:
#
#   target_link_libraries(your_app PRIVATE nga::nga)
#
# Requirements: CMake 3.22+, a C++20 compiler, and the Vulkan loader/headers
# (find_package(Vulkan)). Nothing else — the one library dependency,
# vk-bootstrap, is resolved automatically below, and the library's internal
# compute shader ships pre-compiled (src/PatchDescriptorsSpv.h), so consumers
# do not need a shader compiler.
#
# Variables this file defines for the including scope:
#   NGA_INCLUDE_DIR  - the public include directory. NoGraphicsAPI.h is shared
#                      between C++ and slang, so pass this to slangc -I when
#                      compiling your own shaders (cmake/CompileShaders.cmake
#                      is a ready-made helper for that).

if(TARGET nga)
    return()
endif()

if(CMAKE_VERSION VERSION_LESS 3.22)
    message(FATAL_ERROR "nga requires CMake 3.22 or newer (found ${CMAKE_VERSION})")
endif()

set(NGA_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(NGA_INCLUDE_DIR "${NGA_DIR}/include")

# nga's link interface names Vulkan::Vulkan, which consumers resolve from
# their own directories — but find_package() imported targets are scoped to
# the directory that created them, so promote ours to global. (Skipped when
# the consuming project already found Vulkan itself.)
if(NOT TARGET Vulkan::Vulkan)
    find_package(Vulkan REQUIRED)
    set_target_properties(Vulkan::Vulkan PROPERTIES IMPORTED_GLOBAL TRUE)
endif()

# ---------------------------------------------------------------------------
# vk-bootstrap (the only library dependency), resolved in order:
#   1. a vk-bootstrap target the consuming project already created
#   2. NGA_VK_BOOTSTRAP_DIR — an explicit local checkout
#   3. the external/vk-bootstrap submodule (full-repo layout)
#   4. FetchContent, pinned to the same tag as the submodule
# ---------------------------------------------------------------------------
set(NGA_VK_BOOTSTRAP_TAG "v1.4.353")
set(NGA_VK_BOOTSTRAP_DIR "" CACHE PATH
    "Path to a local vk-bootstrap checkout (skips the submodule/download lookup)")

if(NOT TARGET vk-bootstrap::vk-bootstrap)
    if(NGA_VK_BOOTSTRAP_DIR)
        add_subdirectory("${NGA_VK_BOOTSTRAP_DIR}"
                         "${CMAKE_CURRENT_BINARY_DIR}/nga-vk-bootstrap" EXCLUDE_FROM_ALL)
    elseif(EXISTS "${NGA_DIR}/../external/vk-bootstrap/CMakeLists.txt")
        add_subdirectory("${NGA_DIR}/../external/vk-bootstrap"
                         "${CMAKE_CURRENT_BINARY_DIR}/nga-vk-bootstrap" EXCLUDE_FROM_ALL)
    else()
        include(FetchContent)
        FetchContent_Declare(vk-bootstrap
            GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap.git
            GIT_TAG        ${NGA_VK_BOOTSTRAP_TAG}
            GIT_SHALLOW    TRUE)
        FetchContent_MakeAvailable(vk-bootstrap)
    endif()
endif()

# ---------------------------------------------------------------------------
# The library
# ---------------------------------------------------------------------------
add_library(nga STATIC "${NGA_DIR}/src/NoGraphicsAPI_Impl.cpp")
add_library(nga::nga ALIAS nga)

target_compile_features(nga PUBLIC cxx_std_20)
target_include_directories(nga PUBLIC "${NGA_INCLUDE_DIR}")
target_include_directories(nga PRIVATE "${NGA_DIR}/src" "${NGA_DIR}/shaders")
target_link_libraries(nga PRIVATE Vulkan::Vulkan vk-bootstrap::vk-bootstrap)

# Lets consumers `include(CompileShaders)` for slang -> SPIR-V compilation.
list(APPEND CMAKE_MODULE_PATH "${NGA_DIR}/cmake")
