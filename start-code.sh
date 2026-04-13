#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
IMAGE="docker://node:20-slim"
INSTANCE="coding"
HOME_HOST="/ihome/djishnu/aar126"
HOME_CONTAINER="/home/node"
PROJECT="${1:-$(pwd)}"
NODE_VERSION="v25.7.0"
NODE_BASE="${HOME_HOST}/.nvm/versions/node/${NODE_VERSION}"

# --- Stop existing instance if running ---
if apptainer instance list | grep -q "${INSTANCE}"; then
    echo "Stopping existing instance '${INSTANCE}'..."
    apptainer instance stop "${INSTANCE}"
fi

# --- Start instance ---
apptainer instance start \
    --containall \
    --writable-tmpfs \
    --no-mount tmp \
    --bind "${PROJECT}:/project" \
    --bind "${NODE_BASE}:/usr/local/node" \
    --bind "${HOME_HOST}/.cargo/bin:/usr/local/cargo/bin" \
    --bind "${HOME_HOST}/.pixi/bin/pixi:/usr/local/pixi/bin/pixi" \
    --home "${HOME_HOST}:${HOME_CONTAINER}" \
    --bind "${HOME_HOST}/.nvm/versions/node/${NODE_VERSION}/bin/claude:/usr/local/bin/claude" \
    --bind "${HOME_HOST}/.nvm/versions/node/${NODE_VERSION}/bin/codex:/usr/local/bin/codex" \
    --bind "/usr/lib64/libatomic.so.1:/usr/lib/x86_64-linux-gnu/libatomic.so.1" \
    --env "PATH=/project/.pixi/envs/default/bin:/usr/local/node/bin:/usr/local/cargo/bin:/usr/local/pixi/bin:/usr/local/bin:/usr/local/sbin:/usr/bin:/bin" \
    --env "NODE_PATH=/usr/local/node/lib/node_modules" \
    "${IMAGE}" "${INSTANCE}"

#         --bind /ihome/djishnu/aar126/.nvm/versions/node/v25.7.0/bin/claude:/usr/local/bin/claude \
echo "Instance '${INSTANCE}' started."
echo "  exec:  apptainer exec instance://${INSTANCE} bash"
echo "  stop:  apptainer instance stop ${INSTANCE}"


