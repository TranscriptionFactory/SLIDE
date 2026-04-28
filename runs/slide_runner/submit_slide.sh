#!/bin/bash
#
# Generic YAML-driven SLURM array submission for SLIDE.
# Uses pixi for reproducible environment management — no conda env building.
#
# Usage (from login node or interactive shell):
#   ./submit_slide.sh config.yaml
#
# The script self-submits: it reads the YAML to determine backends and SLURM
# resources, then calls sbatch on itself. Inside SLURM it runs one backend
# per array task.  All tasks share the same pixi environment (cached in
# PROJECT_DIR/.pixi/).

set -euo pipefail

# ── Argument parsing ──────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    cat >&2 <<'USAGE'
Usage: submit_slide.sh <config.yaml>

  config.yaml   SLIDE run configuration (see example_config.yaml)
USAGE
    exit 1
fi

CONFIG_RAW="$1"; shift

# Resolve config to absolute path
CONFIG="$(cd "$(dirname "$CONFIG_RAW")" && pwd)/$(basename "$CONFIG_RAW")"
CONFIG_DIR="$(dirname "$CONFIG")"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config file not found: $CONFIG" >&2
    exit 1
fi

# ── Locate script + project directories ──────────────────────────────────────
# Inside SLURM, BASH_SOURCE[0] points to the spool copy — use the pre-resolved
# SUBMIT_SCRIPT_DIR if available (set during self-submit on the login node).
SCRIPT_DIR="${SUBMIT_SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

_resolve_project_dir() {
    local git_root
    git_root="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
    if [[ -n "$git_root" ]]; then
        echo "$git_root"
        return
    fi
    echo "ERROR: Could not find loveslide repo from SCRIPT_DIR=$SCRIPT_DIR" >&2
    return 1
}

PROJECT_DIR=/ix/djishnu/Aaron/1_general_use/SLIDE_py
#"$(_resolve_project_dir)"

# ── Locate pixi ──────────────────────────────────────────────────────────────
PIXI="${PIXI:-}"
if [[ -z "$PIXI" ]]; then
    for _try in \
        "${HOME}/.pixi/bin/pixi" \
        "$(command -v pixi 2>/dev/null || true)"; do
        if [[ -n "$_try" && -x "$_try" ]]; then
            PIXI="$_try"
            break
        fi
    done
fi
if [[ -z "$PIXI" || ! -x "$PIXI" ]]; then
    echo "ERROR: pixi not found. Install from https://pixi.sh or set PIXI=/path/to/pixi" >&2
    exit 1
fi

# ── YAML helper (pixi provides python + pyyaml) ─────────────────────────────
yaml_query() {
    "$PIXI" run -e dev --manifest-path "${PROJECT_DIR}/pixi.toml" \
        python -c "
import yaml, sys, os
cfg = yaml.safe_load(open('${CONFIG}'))
config_dir = '${CONFIG_DIR}'
expr = sys.argv[1]
val = eval(expr, {'cfg': cfg, 'os': os, 'config_dir': config_dir})
print(val)
" "$1"
}

# Resolve a path from the YAML: absolute paths pass through, relative paths
# are resolved from the config file's directory.
resolve_path() {
    local raw="$1"
    if [[ "$raw" == /* ]]; then
        echo "$raw"
    else
        echo "$(cd "${CONFIG_DIR}" && realpath -m "$raw")"
    fi
}

# ── Self-submit from login node ──────────────────────────────────────────────
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    # Ensure pixi env is built before submitting (avoids race between tasks)
    echo "=== Ensuring pixi environment is ready ==="
    "$PIXI" install -e dev --manifest-path "${PROJECT_DIR}/pixi.toml"

    N_BACKENDS=$(yaml_query "len(cfg['slide']['backends'])")
    JOB_NAME=$(yaml_query   "cfg.get('slurm',{}).get('job_name','slide_run')")
    SLURM_TIME=$(yaml_query "cfg.get('slurm',{}).get('time','2-12:00:00')")
    SLURM_MEM=$(yaml_query  "cfg.get('slurm',{}).get('mem','50G')")
    SLURM_CPUS=$(yaml_query "cfg.get('slurm',{}).get('cpus_per_task',4)")
    SLURM_CLUSTER=$(yaml_query "cfg.get('slurm',{}).get('cluster','htc')")
    OUT_BASE_RAW=$(yaml_query "cfg.get('output',{}).get('base_dir','./output')")
    LOG_DIR="$(resolve_path "$OUT_BASE_RAW")"

    echo "=== Submitting SLURM array job ==="
    echo "  config:   ${CONFIG}"
    echo "  project:  ${PROJECT_DIR}"
    echo "  pixi:     ${PIXI}"
    echo "  backends: ${N_BACKENDS}"
    echo "  array:    0-$((N_BACKENDS - 1))"
    echo "  cluster:  ${SLURM_CLUSTER}"
    echo "  time:     ${SLURM_TIME}  mem: ${SLURM_MEM}  cpus: ${SLURM_CPUS}"
    echo "  logs:     ${LOG_DIR}/"

    mkdir -p "${LOG_DIR}"

    SBATCH_ARGS=(
        --job-name="${JOB_NAME}"
        --time="${SLURM_TIME}"
        --mem="${SLURM_MEM}"
        --cpus-per-task="${SLURM_CPUS}"
        --cluster="${SLURM_CLUSTER}"
        --array="0-$((N_BACKENDS - 1))"
        --output="${LOG_DIR}/${JOB_NAME}_%A_%a.log"
    )

    # Pre-resolve values needed by SLURM tasks so they don't call pixi/yaml_query.
    # Backends as a space-separated list; each task indexes by SLURM_ARRAY_TASK_ID.
    BACKENDS_LIST=$(yaml_query "' '.join(cfg['slide']['backends'])")

    # Pre-compute pixi activation so SLURM tasks never call pixi (avoids lockfile).
    PIXI_ACTIVATE=$("$PIXI" shell-hook -e dev --manifest-path "${PROJECT_DIR}/pixi.toml")

    # Pass resolved paths to SLURM tasks
    export SUBMIT_SCRIPT_DIR="$SCRIPT_DIR"
    export PIXI
    export SLIDE_BACKENDS="$BACKENDS_LIST"
    export SLIDE_OUT_BASE="$LOG_DIR"
    export PIXI_ACTIVATE

    SLIDE_JOB=$(sbatch --parsable --export=ALL "${SBATCH_ARGS[@]}" \
        "${BASH_SOURCE[0]}" "$CONFIG")

    # Extract numeric job ID (sbatch --parsable may return "JOBID;CLUSTER")
    SLIDE_JOB_ID="${SLIDE_JOB%%;*}"
    echo "  Submitted array job: ${SLIDE_JOB_ID}"
    echo ""
    echo "  When complete, compare results with:"
    echo "    python ${SCRIPT_DIR}/compare_outputs.py --dir ${LOG_DIR}/${SLIDE_JOB_ID}"

    exit 0
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Below runs inside SLURM (one array task per backend)
# ═══════════════════════════════════════════════════════════════════════════════

export PYTHONUNBUFFERED=1

# Use pre-resolved values from the login-node phase (no pixi needed here).
read -ra _backends <<< "${SLIDE_BACKENDS}"
BACKEND="${_backends[${SLURM_ARRAY_TASK_ID}]}"
OUTPUT_DIR="${SLIDE_OUT_BASE}/${SLURM_ARRAY_JOB_ID}"

echo "=== Array task ${SLURM_ARRAY_TASK_ID}: backend=${BACKEND} ==="
echo "  config:  ${CONFIG}"
echo "  project: ${PROJECT_DIR}"
echo "  pixi:    ${PIXI}"
echo "  output:  ${OUTPUT_DIR}/${BACKEND}/"

# Activate pixi env using pre-computed activation from login node (no pixi call needed).
eval "$PIXI_ACTIVATE"
echo ""

python "${SCRIPT_DIR}/run_slide.py" \
    --config "${CONFIG}" \
    --backend "${BACKEND}" \
    --out-dir "${OUTPUT_DIR}" \
    2>&1

echo ""
echo "=== ${BACKEND} backend complete ==="
