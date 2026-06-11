# Slang -> SPIR-V compilation helper. NGAPI shader code includes NoGraphicsAPI.h
# (the header is shared between C++ and slang), so consumers compiling their
# own shaders can use this too — ngapi.cmake puts this directory on
# CMAKE_MODULE_PATH, making it available as include(CompileShaders).
#
# Expects these variables to be set before use:
#   SLANGC                  - path to the slangc compiler
#   NGAPI_SHADER_OUTPUT_DIR   - directory the .spv files are written to (next to the executables)
#   NGAPI_SHADER_INCLUDE_DIR  - include search path passed to slangc (-I);
#                             defaults to ngapi's public include directory
#
# compile_shader(SOURCE <path> STAGE <stage> OUTPUT <rel.spv> [ENTRY <name>...] [EXTRA_DEPENDS ...])
#   SOURCE        - .slang path, absolute or relative to the calling CMakeLists
#   STAGE         - SPIR-V stage (compute, vertex, fragment, ...)
#   OUTPUT        - output .spv path relative to NGAPI_SHADER_OUTPUT_DIR
#   ENTRY         - optional entry point(s) (defaults to "main"); pass several
#                   names to compile multiple entry points into a single .spv
#   EXTRA_DEPENDS - additional files that should trigger a recompile
#
# Appends the produced output path to NGAPI_SHADER_OUTPUTS in the caller's scope.

if(NOT DEFINED NGAPI_SHADER_INCLUDE_DIR)
    get_filename_component(NGAPI_SHADER_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../include" ABSOLUTE)
endif()

if(NOT SLANGC)
    find_program(SLANGC slangc
        HINTS "${VULKAN_SDK}/Bin" "${VULKAN_SDK}/bin" "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
endif()

function(compile_shader)
    cmake_parse_arguments(S "" "SOURCE;STAGE;OUTPUT" "ENTRY;EXTRA_DEPENDS" ${ARGN})

    # An empty COMMAND would be dropped silently, producing a rule that never
    # compiles anything — fail loudly instead.
    if(NOT SLANGC)
        message(FATAL_ERROR "compile_shader: slangc not found; put it on PATH, set VULKAN_SDK, or set SLANGC")
    endif()

    # Pair each entry point with the stage; slangc requires the -stage option to
    # follow the -entry it applies to, so this lets a single .spv hold several
    # entry points (e.g. the tensor ops in the learning sample).
    if(NOT S_ENTRY)
        set(S_ENTRY main)
    endif()
    set(STAGE_ENTRY_ARGS "")
    foreach(entry IN LISTS S_ENTRY)
        list(APPEND STAGE_ENTRY_ARGS -entry ${entry} -stage ${S_STAGE})
    endforeach()

    if(NOT IS_ABSOLUTE "${S_SOURCE}")
        set(S_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${S_SOURCE}")
    endif()

    set(OUT "${NGAPI_SHADER_OUTPUT_DIR}/${S_OUTPUT}")
    get_filename_component(OUT_DIR "${OUT}" DIRECTORY)

    add_custom_command(
        OUTPUT "${OUT}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${OUT_DIR}"
        COMMAND "${SLANGC}" "${S_SOURCE}"
                -target spirv ${STAGE_ENTRY_ARGS}
                -I "${NGAPI_SHADER_INCLUDE_DIR}"
                -warnings-disable 39001 # Sampler.h aliases the sampler heap binding on purpose
                -o "${OUT}"
        DEPENDS "${S_SOURCE}" ${S_EXTRA_DEPENDS}
        COMMENT "Compiling shader ${S_OUTPUT}"
        VERBATIM)

    list(APPEND NGAPI_SHADER_OUTPUTS "${OUT}")
    set(NGAPI_SHADER_OUTPUTS "${NGAPI_SHADER_OUTPUTS}" PARENT_SCOPE)
endfunction()
