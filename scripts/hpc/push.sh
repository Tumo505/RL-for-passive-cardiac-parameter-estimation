#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
# No --delete: remote results and unrelated files are preserved.
ssh "${SSH_ARGS[@]}" "$HPC_TARGET" "mkdir -p '$REMOTE_DIR'"
rsync -av --progress -e "$RSYNC_SSH" \
  --exclude=.venv/ --exclude=.git/ --exclude=__pycache__/ --exclude=.pytest_cache/ \
  --exclude='*.egg-info/' --exclude=data/ --exclude=runs/ --exclude=logs/ --exclude=dist/ \
  --exclude=scripts/hpc/site.env --exclude=scripts/hpc/cluster.env \
  "$PROJECT_DIR/" "$HPC_TARGET:$REMOTE_DIR/"
echo "Transferred source to $HPC_TARGET:$REMOTE_DIR through $HPC_GATEWAY"
