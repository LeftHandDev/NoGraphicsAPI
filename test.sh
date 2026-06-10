#!/usr/bin/env bash
#
# Run the headless render tests exactly as CI does: build the pinned Mesa
# lavapipe + slang image (ci/Dockerfile) and run tests/ci-run.sh inside it. Using
# the same image locally and in CI is the whole point -- there is no second code
# path to drift out of sync with the GitHub Actions workflow.
#
#   ./test.sh              build, then compare against the golden PNGs
#   ./test.sh --generate   build, then (re)write the goldens (do this in the
#                          image so committed goldens always match CI)
#
# Any extra args are forwarded to the tests, e.g.  ./test.sh --threshold 8
#
# Requires Docker. For a bare-metal run (your own slangc + a software Vulkan
# ICD, no container) call tests/ci-run.sh directly instead.
set -euo pipefail

cd "$(dirname "$0")"

IMAGE=ngapi-ci

# The tests only need the core library's submodules; glfw / SDL are samples-only
# and are skipped (NGAPI_BUILD_SAMPLES=OFF), so don't pull them in.
echo "==> Fetching core submodules"
git submodule update --init external/glm external/vk-bootstrap

echo "==> Building $IMAGE image (cached after the first run)"
docker build -t "$IMAGE" ci

echo "==> Running tests in $IMAGE"
exec docker run --rm -v "$PWD":/src -w /src "$IMAGE" tests/ci-run.sh "$@"
