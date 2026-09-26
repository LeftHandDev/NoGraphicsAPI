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
# compile_shader(SOURCE <path> STAGE <stage> OUTPUT <rel.spv> [ENTRY <name>...] [ALL_ENTRIES] [EXTRA_DEPENDS ...])
#   SOURCE        - .slang path, absolute or relative to the calling CMakeLists
#   STAGE         - SPIR-V stage (compute, vertex, fragment, ...)
#   OUTPUT        - output .spv path relative to NGAPI_SHADER_OUTPUT_DIR
#   ENTRY         - optional entry point(s) (defaults to "main"); pass several
#                   names to compile multiple entry points into a single .spv
#   ALL_ENTRIES   - compile every entry point marked with a [shader("...")]
#                   attribute; slangc discovers them, so no ENTRY/STAGE is
#                   needed (used by the learning sample's many tensor ops)
#   EXTRA_DEPENDS - additional files that should trigger a recompile
#
# Appends the produced output path to NGAPI_SHADER_OUTPUTS in the caller's scope.
#
# Layout check (NGAPI_SHADER_LAYOUT_CHECK, on by default): every struct a shader
# reads through a pointer must have the same layout as the C++ struct of the
# same name, or the shader fails to build. compile_shader reads the struct
# layouts out of the compiled SPIR-V (tools/LayoutCheck.cpp), turns them into
# C++ static_asserts over the headers the shader included, and syntax-checks
# those with the C++ compiler. It catches members that alignas() padding (which
# only C++ applies) or bool (1 byte in C++, 4 on the GPU) moves or resizes, and
# array strides that differ. Needs the ngapi-layout-check target from ngapi.cmake.

if(NOT DEFINED NGAPI_SHADER_INCLUDE_DIR)
    get_filename_component(NGAPI_SHADER_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../include" ABSOLUTE)
endif()

if(NOT SLANGC)
    find_program(SLANGC slangc
        HINTS "${VULKAN_SDK}/Bin" "${VULKAN_SDK}/bin" "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
endif()

option(NGAPI_SHADER_LAYOUT_CHECK
    "Fail the shader build when a struct shared between C++ and a shader has a different layout on each side" ON)

# The C++ compiler command that syntax-checks a generated layout check.
function(_ngapi_layout_check_command out_var source)
    if(MSVC OR CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
        set(command "${CMAKE_CXX_COMPILER}" /nologo /Zs /std:c++20 /EHsc "/I${NGAPI_SHADER_INCLUDE_DIR}" "${source}")
    else()
        set(command "${CMAKE_CXX_COMPILER}" -fsyntax-only -std=c++20 "-I${NGAPI_SHADER_INCLUDE_DIR}" "${source}")
    endif()
    set(${out_var} "${command}" PARENT_SCOPE)
endfunction()

function(compile_shader)
    cmake_parse_arguments(S "ALL_ENTRIES" "SOURCE;STAGE;OUTPUT" "ENTRY;EXTRA_DEPENDS" ${ARGN})

    # An empty COMMAND would be dropped silently, producing a rule that never
    # compiles anything — fail loudly instead.
    if(NOT SLANGC)
        message(FATAL_ERROR "compile_shader: slangc not found; put it on PATH, set VULKAN_SDK, or set SLANGC")
    endif()

    # Pair each entry point with the stage; slangc requires the -stage option to
    # follow the -entry it applies to, so this lets a single .spv hold several
    # entry points (e.g. the tensor ops in the learning sample).
    #
    # With ALL_ENTRIES we pass neither: slangc auto-discovers every entry point
    # from its [shader("...")] attribute (and takes the stage from there too),
    # so the .slang file owns the list instead of it being duplicated here.
    set(STAGE_ENTRY_ARGS "")
    if(NOT S_ALL_ENTRIES)
        if(NOT S_ENTRY)
            set(S_ENTRY main)
        endif()
        foreach(entry IN LISTS S_ENTRY)
            list(APPEND STAGE_ENTRY_ARGS -entry ${entry} -stage ${S_STAGE})
        endforeach()
    endif()

    if(NOT IS_ABSOLUTE "${S_SOURCE}")
        set(S_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${S_SOURCE}")
    endif()

    set(OUT "${NGAPI_SHADER_OUTPUT_DIR}/${S_OUTPUT}")
    get_filename_component(OUT_DIR "${OUT}" DIRECTORY)

    set(SLANGC_COMMAND "${SLANGC}" "${S_SOURCE}"
        -target spirv ${STAGE_ENTRY_ARGS}
        -I "${NGAPI_SHADER_INCLUDE_DIR}"
        -warnings-disable 39001) # Sampler.h aliases the sampler heap binding on purpose

    if(NGAPI_SHADER_LAYOUT_CHECK AND TARGET ngapi-layout-check AND CMAKE_CXX_COMPILER AND NOT CMAKE_CROSSCOMPILING)
        # Compile to a scratch .spv, generate the layout check from it and
        # syntax-check that with the C++ compiler. OUT is only written once the
        # check passes, so a mismatch keeps failing the build until it is fixed.
        set(CHECK "${CMAKE_CURRENT_BINARY_DIR}/ngapi-layout-check/${S_OUTPUT}")
        get_filename_component(CHECK_DIR "${CHECK}" DIRECTORY)
        _ngapi_layout_check_command(CHECK_COMMAND "${CHECK}.cpp")
        add_custom_command(
            OUTPUT "${OUT}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${OUT_DIR}" "${CHECK_DIR}"
            COMMAND ${SLANGC_COMMAND} -o "${CHECK}" -depfile "${CHECK}.d"
            COMMAND $<TARGET_FILE:ngapi-layout-check> "${CHECK}" "${CHECK}.d" "${CHECK}.cpp" "${S_OUTPUT}"
            COMMAND ${CHECK_COMMAND}
            COMMAND ${CMAKE_COMMAND} -E copy "${CHECK}" "${OUT}"
            DEPENDS "${S_SOURCE}" ${S_EXTRA_DEPENDS} ngapi-layout-check
            COMMENT "Compiling shader ${S_OUTPUT} (checking C++/GPU struct layouts)"
            VERBATIM)
    else()
        add_custom_command(
            OUTPUT "${OUT}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${OUT_DIR}"
            COMMAND ${SLANGC_COMMAND} -o "${OUT}"
            DEPENDS "${S_SOURCE}" ${S_EXTRA_DEPENDS}
            COMMENT "Compiling shader ${S_OUTPUT}"
            VERBATIM)
    endif()

    list(APPEND NGAPI_SHADER_OUTPUTS "${OUT}")
    set(NGAPI_SHADER_OUTPUTS "${NGAPI_SHADER_OUTPUTS}" PARENT_SCOPE)
endfunction()
