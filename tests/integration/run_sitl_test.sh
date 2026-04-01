#!/usr/bin/env bash
# =============================================================================
# run_sitl_test.sh — Build Docker image and run headless SITL integration tests
#
# Usage:
#   ./tests/integration/run_sitl_test.sh              # build + test
#   ./tests/integration/run_sitl_test.sh --no-build   # test only (image exists)
#   ./tests/integration/run_sitl_test.sh --shell       # drop into shell for debug
#
# Requirements:
#   - Docker (with BuildKit recommended)
#   - ~20GB disk for image
#   - No GPU required (headless rendering)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
IMAGE_NAME="reach-avoid-sim"
IMAGE_TAG="test"
CONTAINER_NAME="reach-avoid-sitl-test"

# Parse args
DO_BUILD=true
DROP_SHELL=false
for arg in "$@"; do
    case "$arg" in
        --no-build) DO_BUILD=false ;;
        --shell)    DROP_SHELL=true ;;
        --help|-h)
            echo "Usage: $0 [--no-build] [--shell]"
            echo "  --no-build  Skip Docker build (use existing image)"
            echo "  --shell     Drop into container shell instead of running tests"
            exit 0
            ;;
    esac
done

cd "$REPO_ROOT"

# ---- Build ----
if $DO_BUILD; then
    echo "============================================"
    echo " Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
    echo "============================================"
    docker build \
        -f Dockerfile.sim \
        -t "${IMAGE_NAME}:${IMAGE_TAG}" \
        .
    echo "Build complete."
fi

# Remove any existing test container
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

# ---- Run ----
if $DROP_SHELL; then
    echo "============================================"
    echo " Dropping into container shell"
    echo "============================================"
    docker run -it --rm \
        --name "$CONTAINER_NAME" \
        "${IMAGE_NAME}:${IMAGE_TAG}" \
        bash
    exit $?
fi

echo "============================================"
echo " Running headless SITL integration tests"
echo "============================================"

# Copy test script into container and run it
# We mount the tests directory so latest changes are always picked up
docker run --rm \
    --name "$CONTAINER_NAME" \
    -v "${REPO_ROOT}/tests/integration:/home/simuser/ws/tests/integration:ro" \
    -e "MESA_GL_VERSION_OVERRIDE=3.3" \
    -e "LIBGL_ALWAYS_SOFTWARE=1" \
    "${IMAGE_NAME}:${IMAGE_TAG}" \
    python3 /home/simuser/ws/tests/integration/test_sitl_headless.py

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "=== SITL Integration Tests: PASSED ==="
elif [ $EXIT_CODE -eq 1 ]; then
    echo "=== SITL Integration Tests: SOME FAILURES ==="
elif [ $EXIT_CODE -eq 2 ]; then
    echo "=== SITL Integration Tests: INFRASTRUCTURE FAILURE ==="
else
    echo "=== SITL Integration Tests: EXIT CODE $EXIT_CODE ==="
fi

exit $EXIT_CODE
