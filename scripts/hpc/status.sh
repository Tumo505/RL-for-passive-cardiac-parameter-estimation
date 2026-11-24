#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
ssh "${SSH_ARGS[@]}" "$HPC_TARGET" qstat -u "${HPC_TARGET%%@*}"
