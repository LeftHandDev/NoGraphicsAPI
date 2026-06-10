# Headless rendering tests. No window — they reuse the core API, the shared
# sample utilities, and the compiled shaders, render a fixed deterministic
# sequence of frames offscreen, and compare the readback against a golden PNG.

add_library(test-common STATIC test_common.cpp)
target_include_directories(test-common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test-common PUBLIC ngapi::ngapi PRIVATE stb Vulkan::Vulkan)
target_compile_definitions(test-common PUBLIC
    NGAPI_TEST_REFERENCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/reference"
    NGAPI_TEST_SHADER_DIR="${NGAPI_SHADER_OUTPUT_DIR}"
    NGAPI_TEST_ASSET_DIR="${PROJECT_SOURCE_DIR}/assets")

# add_render_test(<target> <source> <sample-include-dir>)
function(add_render_test name source sample_dir)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE test-common ngapi-samples-common stb glm::glm)
    target_include_directories(${name} PRIVATE "${PROJECT_SOURCE_DIR}/${sample_dir}")
    add_dependencies(${name} shaders)
endfunction()

add_render_test(test_compute    test_compute.cpp    samples/compute)
add_render_test(test_graphics   test_graphics.cpp   samples/graphics)
add_render_test(test_raytracing test_raytracing.cpp samples/raytracing)
