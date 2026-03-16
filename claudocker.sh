#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Build the image
docker build -f "$SCRIPT_DIR/Dockerfile.claude" -t aura-claude "$SCRIPT_DIR"

# Run — mounts aura repo as /workspace and ~/.claude for auth persistence
docker run -it --rm \
  -v "${1:-.}:/workspace" \
  -v ~/.claude:/home/clod/.claude \
  aura-claude "${@:2}"
