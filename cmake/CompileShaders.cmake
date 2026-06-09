# Slang -> SPIR-V compilation helper.
#
# Expects these variables to be set before use:
#   SLANGC                  - path to the slangc compiler
#   NGA_SHADER_OUTPUT_DIR   - directory the .spv files are written to (next to the executables)
#   NGA_SHADER_INCLUDE_DIR  - include search path passed to slangc (-I)
#
# compile_shader(SOURCE <path> STAGE <stage> OUTPUT <rel.spv> [ENTRY <name>] [EXTRA_DEPENDS ...])
#   SOURCE        - .slang path relative to the project root
#   STAGE         - SPIR-V stage (compute, vertex, fragment, ...)
#   OUTPUT        - output .spv path relative to NGA_SHADER_OUTPUT_DIR
#   ENTRY         - optional entry point (defaults to "main")
#   EXTRA_DEPENDS - additional files that should trigger a recompile
#
# Appends the produced output path to NGA_SHADER_OUTPUTS in the caller's scope.
function(compile_shader)
    cmake_parse_arguments(S "" "SOURCE;STAGE;OUTPUT;ENTRY" "EXTRA_DEPENDS" ${ARGN})

    if(S_ENTRY)
        set(ENTRY_ARG -entry ${S_ENTRY})
    else()
        set(ENTRY_ARG -entry main)
    endif()

    set(OUT "${NGA_SHADER_OUTPUT_DIR}/${S_OUTPUT}")
    get_filename_component(OUT_DIR "${OUT}" DIRECTORY)

    add_custom_command(
        OUTPUT "${OUT}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${OUT_DIR}"
        COMMAND "${SLANGC}" "${CMAKE_SOURCE_DIR}/${S_SOURCE}"
                -target spirv -stage ${S_STAGE} ${ENTRY_ARG}
                -I "${NGA_SHADER_INCLUDE_DIR}"
                -o "${OUT}"
        DEPENDS "${CMAKE_SOURCE_DIR}/${S_SOURCE}" ${S_EXTRA_DEPENDS}
        COMMENT "Compiling shader ${S_OUTPUT}"
        VERBATIM)

    list(APPEND NGA_SHADER_OUTPUTS "${OUT}")
    set(NGA_SHADER_OUTPUTS "${NGA_SHADER_OUTPUTS}" PARENT_SCOPE)
endfunction()
