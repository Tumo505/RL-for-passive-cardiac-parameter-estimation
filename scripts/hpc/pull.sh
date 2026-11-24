#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
mkdir -p "$PROJECT_DIR/runs/hpc" "$PROJECT_DIR/logs/hpc"
rsync -av --progress -e "$RSYNC_SSH" "$HPC_TARGET:$REMOTE_DIR/runs/" "$PROJECT_DIR/runs/hpc/"
rsync -av --progress -e "$RSYNC_SSH" "$HPC_TARGET:$REMOTE_DIR/logs/" "$PROJECT_DIR/logs/hpc/"
