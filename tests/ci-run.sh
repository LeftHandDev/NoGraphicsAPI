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

echo "==> Configure (core + tests only, samples off)"
cmake -S "$ROOT" -B "$BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DNGAPI_BUILD_SAMPLES=OFF \
    -DNGAPI_BUILD_TESTS=ON

echo "==> Build"
cmake --build "$BUILD" -j"$(nproc)"

# Run from the bin dir: failure artifacts (<name>_actual.png / <name>_diff.png)
# are written to the working directory, and CI collects them from build-ci/bin.
cd "$BUILD/bin"

status=0
for t in test_compute test_graphics test_raytracing; do
    echo "==> $t ${MODE_ARGS[*]} ${EXTRA_ARGS[*]}"
    if ! "./$t" "${MODE_ARGS[@]}" "${EXTRA_ARGS[@]}"; then
        status=1
    fi
done

if [[ ${#MODE_ARGS[@]} -gt 0 ]]; then
    echo "==> Goldens written to $ROOT/tests/reference"
fi
exit $status
