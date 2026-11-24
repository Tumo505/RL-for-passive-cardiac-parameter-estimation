#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
# Setup must already have been run on the submission host with the appropriate site modules.
# Explicit -v forwards only experiment settings, not the login environment.
ssh "${SSH_ARGS[@]}" "$HPC_TARGET" \
  "cd '$REMOTE_DIR' && mkdir -p logs && bash scripts/hpc/submit-local.sh -o logs/pbs.log -v SEED_START=0,RESUME=0,TOTAL_STEPS=300000"
