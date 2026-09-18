#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# Environment/working directory for uv only; research path semantics use --root.
cd "$ROOT"

uv run research train \
    "${1:?Usage: bash scripts/run.sh CONFIG [OPTIONS...]}" \
    "${@:2}" \
    --root "$ROOT"
