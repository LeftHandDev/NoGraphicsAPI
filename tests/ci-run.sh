#!/usr/bin/env bash
# Configure, build and run the headless rendering tests.
#
# Designed to run inside the pinned ci/Dockerfile image (Mesa lavapipe + a fixed
# slang), but it works on any machine that has cmake, ninja, slangc on PATH and
# a Vulkan loader with a software ICD. The image already exports VK_ICD_FILENAMES
# and LP_NUM_THREADS; we re-assert LP_NUM_THREADS here so a bare local run is
# deterministic too.
#
#   tests/ci-run.sh              configure, build, compare against goldens
#   tests/ci-run.sh --generate   same, but (re)write the golden PNGs instead
#
# Extra args after the mode are forwarded to each test (e.g. --threshold 8).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${NGAPI_BUILD_DIR:-$ROOT/build-ci}"

MODE_ARGS=()
if [[ "${1:-}" == "--generate" ]]; then
    MODE_ARGS+=(--generate)
    shift
fi
EXTRA_ARGS=("$@")

# Deterministic floating point from lavapipe's rasteriser.
export LP_NUM_THREADS="${LP_NUM_THREADS:-1}"

# The samples are windowed and can't run here, but they are compiled so a
# change that breaks one (on this compiler) fails CI instead of going unnoticed.
echo "==> Configure (core + tests + samples)"
cmake -S "$ROOT" -B "$BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DNGAPI_BUILD_SAMPLES=ON \
    -DNGAPI_BUILD_TESTS=ON

echo "==> Build"
cmake --build "$BUILD" -j"$(nproc)"

# Run from the bin dir: failure artifacts (<name>_actual.png / <name>_diff.png)
# are written to the working directory, and CI collects them from build-ci/bin.
cd "$BUILD/bin"

status=0
for t in test_api test_heaps test_compute test_graphics test_raytracing test_msdf; do
    echo "==> $t ${MODE_ARGS[*]} ${EXTRA_ARGS[*]}"
    if ! "./$t" "${MODE_ARGS[@]}" "${EXTRA_ARGS[@]}"; then
        status=1
    fi
done

# compile_shader's C++/GPU layout check must reject mismatched shared structs:
# tests/layout/Mismatch.h has one of each kind the check reports.
echo "==> test_layout_mismatch (must fail to build)"
if cmake --build "$BUILD" --target test_layout_mismatch > layout_mismatch.log 2>&1; then
    echo "FAIL [layout]: shader with mismatched C++/GPU structs compiled"
    status=1
else
    missing=0
    for expected in "indexes Item arrays with a 12-byte stride" \
                    "Flags::enabled is 4 bytes in the shader" \
                    "Shifted::item is at byte 4 in the shader"; do
        if ! grep -q "$expected" layout_mismatch.log; then
            echo "FAIL [layout]: expected '$expected' in the build output"
            missing=1
        fi
    done
    if [[ $missing -eq 0 ]]; then
        echo "PASS [layout]: mismatched structs rejected"
    else
        cat layout_mismatch.log
        status=1
    fi
fi

if [[ ${#MODE_ARGS[@]} -gt 0 ]]; then
    echo "==> Goldens written to $ROOT/tests/reference"
fi
exit $status
