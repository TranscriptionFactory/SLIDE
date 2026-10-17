#!/usr/bin/env bash
set -euo pipefail

module load ffmpeg
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAPE="${ROOT}/docs/tui-screenshots.tape"
OUT_DIR="${ROOT}/docs/assets/tui"
PREPARE_ONLY=0
SKIP_PREPARE=0
usage() {
  cat <<'USAGE'
Usage: scripts/capture-tui-screenshots.sh [--prepare-only] [--skip-prepare]

Builds a deterministic TUI smoke fixture and renders release screenshots/GIF
from docs/tui-screenshots.tape using VHS.

Outputs:
  docs/assets/tui/overview.png
  docs/assets/tui/love-diagnostics.png
  docs/assets/tui/love-loading-matrix.png
  docs/assets/tui/factor-1.png
  docs/assets/tui/factor-2.png
  docs/assets/tui/factor-3.png
  docs/assets/tui/slide-selection.png
  docs/assets/tui/performance.png
  docs/assets/tui/tui-demo.gif

Options:
  --prepare-only  Only build/regenerate .tmp/tui-smoke; do not invoke vhs.
  --skip-prepare  Reuse the existing .tmp/tui-smoke fixture and target/debug binary.
  -h, --help      Show this help.

Dependencies:
  - cargo and python3 for fixture generation
  - vhs for capture (https://github.com/charmbracelet/vhs)
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prepare-only)
      PREPARE_ONLY=1
      ;;
    --skip-prepare)
      SKIP_PREPARE=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

cd "${ROOT}"
mkdir -p "${OUT_DIR}"

# CRC/HPC environments provide Go through environment modules; load it so
# `go install github.com/charmbracelet/vhs@latest` works if VHS is missing.
if type module >/dev/null 2>&1; then
  module load go
fi

if [[ "${SKIP_PREPARE}" -eq 0 ]]; then
  echo "==> Preparing deterministic TUI fixture"
  scripts/test-slide-tui.sh --no-tui
fi

if [[ "${PREPARE_ONLY}" -eq 1 ]]; then
  echo "==> Fixture ready; skipping VHS capture"
  exit 0
fi

if ! command -v vhs >/dev/null 2>&1; then
  cat >&2 <<'ERR'
error: vhs is required to capture screenshots.
Install with one of:
  brew install vhs
  go install github.com/charmbracelet/vhs@latest
  docker run --rm -v "$PWD:/vhs" ghcr.io/charmbracelet/vhs docs/tui-screenshots.tape
ERR
  exit 127
fi

echo "==> Capturing TUI screenshots with VHS"
vhs "${TAPE}"

echo "==> Wrote screenshots to ${OUT_DIR}"
