cmake_minimum_required(VERSION 3.22)

project(NoGraphicsAPI
    VERSION 1.0
    DESCRIPTION "A Vulkan implementation of \"No Graphics API\""
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)

add_executable(NoGraphicsAPI
    External/VkBootstrap.cpp
    NoGraphicsAPI.cpp
    Main.cpp
)

if(TARGET Vulkan::Vulkan)
    target_link_libraries(NoGraphicsAPI PRIVATE Vulkan::Vulkan)
else()
    target_include_directories(NoGraphicsAPI PRIVATE ${Vulkan_INCLUDE_DIRS})
    target_link_libraries(NoGraphicsAPI PRIVATE ${Vulkan_LIBRARIES})
endif()