# Wrapper so this folder works with add_subdirectory() and FetchContent.
# Everything lives in ngapi.cmake, which can also be include()d directly.
cmake_minimum_required(VERSION 3.22)

project(ngapi
    VERSION 1.0
    DESCRIPTION "A Vulkan implementation of \"No Graphics API\""
    LANGUAGES CXX
)

include("${CMAKE_CURRENT_LIST_DIR}/ngapi.cmake")

# Variables from ngapi.cmake are scoped to this directory; re-export the ones
# consumers need (see ngapi.cmake's header comment) so that NGAPI_INCLUDE_DIR and
# include(CompileShaders) also work from the consuming project's CMakeLists.
if(NOT PROJECT_IS_TOP_LEVEL)
    set(NGAPI_INCLUDE_DIR "${NGAPI_INCLUDE_DIR}" PARENT_SCOPE)
    set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" PARENT_SCOPE)
endif()
