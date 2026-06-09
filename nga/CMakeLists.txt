# Wrapper so this folder works with add_subdirectory() and FetchContent.
# Everything lives in nga.cmake, which can also be include()d directly.
cmake_minimum_required(VERSION 3.22)

project(nga
    VERSION 1.0
    DESCRIPTION "A Vulkan implementation of \"No Graphics API\""
    LANGUAGES CXX
)

include("${CMAKE_CURRENT_LIST_DIR}/nga.cmake")

# Variables from nga.cmake are scoped to this directory; re-export the ones
# consumers need (see nga.cmake's header comment).
if(NOT PROJECT_IS_TOP_LEVEL)
    set(NGA_INCLUDE_DIR "${NGA_INCLUDE_DIR}" PARENT_SCOPE)
endif()
