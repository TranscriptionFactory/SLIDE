#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${SLIDE_TUI_WORKDIR:-${ROOT}/.tmpfiles/$(date +%F)_tui-smoke}"
CONFIG="${WORKDIR}/tui-smoke.yaml"
CHECKPOINT="${WORKDIR}/tui-smoke-checkpoint.json"

PROFILE="debug"
OPEN_TUI=1
FRESH=1
BIN="${SLIDE_RS_BIN:-}"

usage() {
  cat <<'USAGE'
Usage: scripts/test-slide-tui.sh [--release] [--no-tui] [--reuse]

Builds the Rust SLIDE CLI, runs focused TUI/core tests, creates a small
final-CV HIV checkpoint, then opens the TUI on that checkpoint.

Options:
  --release  Build and run target/release/slide instead of debug.
  --no-tui   Stop after creating and validating the checkpoint.
  --reuse    Reuse the existing checkpoint/config in the smoke work directory.

Environment:
  SLIDE_TUI_WORKDIR  Override smoke-test output directory.
  SLIDE_RS_BIN       Test this executable instead of building/running cargo tests.
  SLIDE_TUI_OUT_PATH Export result tables and plots to this new directory.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --release)
      PROFILE="release"
      ;;
    --no-tui)
      OPEN_TUI=0
      ;;
    --reuse)
      FRESH=0
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

CARGO_PROFILE_FLAG=()
if [[ "${PROFILE}" == "release" ]]; then
  CARGO_PROFILE_FLAG=(--release)
fi
cd "${ROOT}"

if [[ -z "${BIN}" ]]; then
  BIN="${ROOT}/target/${PROFILE}/slide"
  echo "==> Running focused tests"
  cargo test -p slide-rs
  cargo test -p slide-core --lib

  echo "==> Building slide (${PROFILE})"
  cargo build -p slide-rs "${CARGO_PROFILE_FLAG[@]}"
fi

SMOKE_ARGS=(--binary "${BIN}" --workdir "${WORKDIR}")
if [[ "${FRESH}" -eq 0 ]]; then
  SMOKE_ARGS+=(--reuse)
fi
if [[ -n "${SLIDE_TUI_OUT_PATH:-}" ]]; then
  SMOKE_ARGS+=(--out-path "${SLIDE_TUI_OUT_PATH}")
fi
"${PYTHON:-python3}" "${ROOT}/scripts/slide_artifact_smoke.py" "${SMOKE_ARGS[@]}"

cat <<EOF

==> TUI smoke checkpoint is ready
Config:     ${CONFIG}
Checkpoint: ${CHECKPOINT}

Manual checks in the TUI:
  - Tab order includes Overview, LOVE, Factors, SLIDE, Performance.
  - Overview has a "final CV:" line.
  - Factors has multiple pages; use PageUp/PageDown to browse factors.
  - Performance shows final CV summary, fold scores, and fold model sizes.
  - Press q to quit.
EOF

if [[ "${OPEN_TUI}" -eq 1 ]]; then
  echo
  echo "==> Opening TUI"
  exec "${BIN}" tui --checkpoint "${CHECKPOINT}" --config "${CONFIG}" --no-follow
fi
