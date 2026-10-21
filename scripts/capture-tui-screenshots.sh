#!/usr/bin/env bash
set -euo pipefail

if ! command -v ffmpeg >/dev/null 2>&1 && type module >/dev/null 2>&1; then
  module load ffmpeg
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SLIDE_TUI_WORKDIR="${SLIDE_TUI_WORKDIR:-${ROOT}/.tmpfiles/$(date +%F)_tui-smoke}"
TAPE="${ROOT}/docs/tui-screenshots.tape"
OUT_DIR="${SLIDE_TUI_SCREENSHOT_DIR:-${ROOT}/docs/assets/tui/$(date +%F)_visual-outputs}"
# Chromium uses Unix sockets whose paths must fit within 108 bytes.
export TMPDIR="${ROOT}/.tmpfiles"
mkdir -p "${TMPDIR}"
export SLIDE_TUI_OUT_PATH="${SLIDE_TUI_OUT_PATH:-${SLIDE_TUI_WORKDIR}/exported-results}"
PREPARE_ONLY=0
SKIP_PREPARE=0
usage() {
  cat <<'USAGE'
Usage: scripts/capture-tui-screenshots.sh [--prepare-only] [--skip-prepare]

Builds a small TUI smoke fixture and renders release screenshots/GIF
from docs/tui-screenshots.tape using VHS.

Outputs:
  docs/assets/tui/YYYY-MM-DD_visual-outputs/
  Overview, LOVE heatmaps, factor tables/text/network, SLIDE rankings/response,
  final-CV summary/folds/predictions/ROC, exported controls, and tui-demo.gif.

Environment:
  SLIDE_TUI_SCREENSHOT_DIR  Override the dated screenshot directory.
  SLIDE_TUI_WORKDIR         Override the fixture directory (use a new path for a new run).
  SLIDE_TUI_OUT_PATH        Export tables/plots here; must not already exist.
  SLIDE_RS_BIN             Use an existing executable instead of building/testing.

Options:
  --prepare-only  Only build/regenerate the dated .tmpfiles/YYYY-MM-DD_tui-smoke fixture; do not invoke vhs.
  --skip-prepare  Reuse the dated .tmpfiles/YYYY-MM-DD_tui-smoke fixture and target/debug binary.
  -h, --help      Show this help.

Dependencies:
  - cargo and python3 for fixture generation
  - vhs, ttyd >= 1.7.2, ffmpeg, and a Chromium browser for capture
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

# CRC/HPC environments provide Go through environment modules; load it so
# `go install github.com/charmbracelet/vhs@latest` works if VHS is missing.
if type module >/dev/null 2>&1; then
  module load go
fi

if [[ "${SKIP_PREPARE}" -eq 0 ]]; then
  echo "==> Preparing TUI fixture"
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

if [[ -e "${OUT_DIR}" ]]; then
  echo "error: screenshot output already exists: ${OUT_DIR}" >&2
  exit 1
fi
mkdir -p "${OUT_DIR}"

echo "==> Capturing TUI screenshots with VHS"
# VHS initializes its emulator size once; use sections of the same tape per size.
for spec in '160x40 1670 824' '80x24 864 517' '60x20 662 440'; do
  read -r size width height <<< "${spec}"
  output="${OUT_DIR}/tui-demo.gif"
  if [[ "${size}" != 160x40 ]]; then
    output="${SLIDE_TUI_WORKDIR}/tui-${size}.gif"
  fi
  awk -v selected="${size}" '
    /^# Capture / { section = $3; next }
    section == "" || section == selected { print }
  ' "${TAPE}" | sed \
    -e "s|docs/assets/tui/|${OUT_DIR}/|g" \
    -e "s/^Set Width .*/Set Width ${width}/" \
    -e "s/^Set Height .*/Set Height ${height}/" \
    -e "s|^Output .*|Output \"${output}\"|" \
    > "${SLIDE_TUI_WORKDIR}/screenshots-${size}.tape"
  vhs "${SLIDE_TUI_WORKDIR}/screenshots-${size}.tape"
done
for size in 160x40:40\ 160 80x24:24\ 80 60x20:20\ 60; do
  name="${size%%:*}"
  expected="${size#*:}"
  test "$(cat "${SLIDE_TUI_WORKDIR}/terminal-${name}.txt")" = "${expected}"
done
cat > "${OUT_DIR}/README.md" <<EOF
# TUI visual outputs ($(date +%F))

Captured from \`$(git rev-parse --short HEAD)\` plus the working-tree capture tape.
Command: \`scripts/capture-tui-screenshots.sh\`
Fixture: \`${SLIDE_TUI_WORKDIR}\`
Exports: \`${SLIDE_TUI_OUT_PATH}\`
Binary: \`${SLIDE_RS_BIN:-./target/debug/slide}\`

The shared smoke workflow uses 40 HIV features, eight knockoff iterations,
one grid cell, binary response, and three-fold final CV. These are demonstration
outputs, not scientific conclusions. The binary-response fixture has
interactions disabled, so the Interaction Ranking view is empty. Screenshots
are static and do not demonstrate interactive inspection.

The tape captures the standard views at 160x40 plus Overview, LOVE loading,
factor, network, and Performance at 80x24 and 60x20. It records and checks the
actual pty sizes with \`stty size\` before each launch, without overriding the
emulator size. Separate VHS launches use sections of the same maintained tape.
LOVE includes all seven subviews.

Other captures include covariance and block heatmaps, signed feature text,
response dots, exported OLS controls, fold scores, prediction bins, and ROC/PR
output. “Permuted CV” shows randomized in-sample OLS controls, not held-out CV.
EOF
for screenshot in "${OUT_DIR}"/*.png "${OUT_DIR}"/tui-demo.gif; do
  test -s "${screenshot}"
  name="$(basename "${screenshot}")"
  printf '\n- [%s](%s)\n' "${name}" "${name}" >> "${OUT_DIR}/README.md"
done

echo "==> Wrote screenshots to ${OUT_DIR}"
