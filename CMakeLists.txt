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

add_executable(NoGraphicsAPI
    External/VkBootstrap.cpp
    SDL_gpu.cpp
    NoGraphicsAPI.cpp
    Main.cpp
    Samples/Compute/Compute.cpp
    Samples/Graphics/Graphics.cpp
)

if(TARGET Vulkan::Vulkan)
    target_link_libraries(NoGraphicsAPI PRIVATE Vulkan::Vulkan)
else()
    target_include_directories(NoGraphicsAPI PRIVATE ${Vulkan_INCLUDE_DIRS})
    target_link_libraries(NoGraphicsAPI PRIVATE ${Vulkan_LIBRARIES})
endif()

# Prefer SDL3 CMake package; fall back to the Vulkan SDK copy via the env var
if(TARGET SDL3::SDL3)
    target_link_libraries(NoGraphicsAPI PRIVATE SDL3::SDL3)
else()
    if(NOT DEFINED VULKAN_SDK AND DEFINED ENV{VULKAN_SDK})
        set(VULKAN_SDK $ENV{VULKAN_SDK})
    endif()

    if(NOT DEFINED VULKAN_SDK)
        message(FATAL_ERROR "VULKAN_SDK env var is not set")
    endif()

    target_include_directories(NoGraphicsAPI PRIVATE "${VULKAN_SDK}/Include/SDL3")
    target_link_libraries(NoGraphicsAPI PRIVATE "${VULKAN_SDK}/Lib/SDL3.lib")
endif()

if(TARGET glm::glm)
    target_link_libraries(NoGraphicsAPI PRIVATE glm::glm)
else()
    if(NOT DEFINED VULKAN_SDK AND DEFINED ENV{VULKAN_SDK})
        set(VULKAN_SDK $ENV{VULKAN_SDK})
    endif()

    if(NOT DEFINED VULKAN_SDK)
        message(FATAL_ERROR "VULKAN_SDK env var is not set")
    endif()

    target_include_directories(NoGraphicsAPI PRIVATE "${VULKAN_SDK}/Include/glm")
endif()