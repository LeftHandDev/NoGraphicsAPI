# nga — the embeddable NoGraphicsAPI library

This folder is the entire reusable library, packaged so you can drop it into
your own CMake project in one step. Everything outside it (samples, windowing
backends, tests) is development scaffolding for NGA itself and is not needed
to use the API.

## Requirements

- CMake 3.22+
- A C++20 compiler
- Vulkan headers + loader (anything `find_package(Vulkan)` can find, e.g. the
  [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) or your distro's
  `vulkan-headers` / `vulkan-icd-loader` packages)

No shader compiler is needed to build or use the library — its one internal
compute shader ships pre-compiled, embedded in `src/PatchDescriptorsSpv.h`.
You only need `slangc` (or any compiler producing SPIR-V) for your own
shaders.

## Using it

Pick whichever fits your project; all of them end the same way:

```cmake
target_link_libraries(your_app PRIVATE nga::nga)
```

**1. Whole repo as a git submodule (or clone):**

```sh
git submodule add <repo-url> external/NoGraphicsAPI
```
```cmake
add_subdirectory(external/NoGraphicsAPI)
```

Samples and tests are skipped automatically when the repo is not the top-level
project, so only the library (and its one dependency) builds.

**2. Just this folder, copied into your tree:**

```cmake
add_subdirectory(path/to/nga)
```

**3. No subdirectory at all — plain include:**

```cmake
include(path/to/nga/nga.cmake)
```

**4. FetchContent:**

```cmake
include(FetchContent)
FetchContent_Declare(NoGraphicsAPI
    GIT_REPOSITORY <repo-url>
    GIT_TAG        main
    GIT_SUBMODULES "")   # submodules are samples-only; vk-bootstrap is fetched on demand
FetchContent_MakeAvailable(NoGraphicsAPI)
```

Then in your code:

```c++
#include "NoGraphicsAPI.h"
```

See the top-level [README](../README.md) for an API walkthrough.

## The vk-bootstrap dependency

NGA uses [vk-bootstrap](https://github.com/charles-lunarg/vk-bootstrap) for
Vulkan instance/device setup, linked privately (it does not appear in NGA's
headers). `nga.cmake` resolves it automatically, in this order:

1. a `vk-bootstrap::vk-bootstrap` target your project already defines
2. `-DNGA_VK_BOOTSTRAP_DIR=<path>` — a local checkout you point it at
3. the repo's `external/vk-bootstrap` submodule, when this folder lives in the
   full NoGraphicsAPI repo and the submodule is initialized
4. FetchContent — downloaded at configure time, pinned to the same tag as the
   submodule

So with the full repo either initialize the submodule or let FetchContent
grab it; with a copied `nga/` folder it just downloads (or use 1/2 for
offline builds).

## Compiling your own shaders

`NoGraphicsAPI.h` is shared between C++ and slang — pass nga's include
directory (exported as `NGA_INCLUDE_DIR`) to `slangc -I` when compiling your
shader code. `cmake/CompileShaders.cmake` is a ready-made helper for that;
`nga.cmake` adds it to `CMAKE_MODULE_PATH`:

```cmake
include(CompileShaders)
set(NGA_SHADER_OUTPUT_DIR "${CMAKE_BINARY_DIR}/shaders")
compile_shader(SOURCE shaders/MyShader.slang STAGE compute OUTPUT MyShader.spv)
add_custom_target(my_shaders ALL DEPENDS ${NGA_SHADER_OUTPUTS})
```

`slangc` is picked up from `PATH` or the `VULKAN_SDK` environment variable;
pass `-DSLANGC=/path/to/slangc` to override.

## Developing NGA itself

`src/PatchDescriptorsSpv.h` is generated from `shaders/PatchDescriptors.slang`
and checked in. After editing that shader, regenerate it (run these from the
repository root, one level above this folder) — preferably with the slangc
pinned in `ci/Dockerfile` so the header stays reproducible:

```sh
docker build -t nga-ci ci   # once
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD":/src -w /src nga-ci \
    cmake -P nga/cmake/EmbedPatchDescriptors.cmake
```

or, with your own slangc on PATH, `cmake -P nga/cmake/EmbedPatchDescriptors.cmake`
(also available as the `nga-embed-patch-descriptors` build target in the full
repo).
