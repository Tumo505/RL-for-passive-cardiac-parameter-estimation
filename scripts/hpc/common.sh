#!/usr/bin/env bash
set -euo pipefail
HPC_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "$HPC_SCRIPT_DIR/../.." && pwd)
if [[ ! -f "$HPC_SCRIPT_DIR/site.env" ]]; then
  echo "Copy scripts/hpc/site.env.example to scripts/hpc/site.env and set REMOTE_DIR." >&2
  exit 2
fi
source "$HPC_SCRIPT_DIR/site.env"
for hpc_destination in "$HPC_GATEWAY" "$HPC_TARGET"; do
  if [[ ! "$hpc_destination" =~ ^[[:alnum:]_][[:alnum:]_.@-]*$ ]]; then
    echo "Invalid SSH destination" >&2; exit 2
  fi
done
if [[ ! "$REMOTE_DIR" =~ ^/[[:alnum:]_./-]+$ || "$REMOTE_DIR" == / || "$REMOTE_DIR" == *replace* ]]; then
  echo "REMOTE_DIR must be an absolute cluster path without spaces; replace the example." >&2
  exit 2
fi
SSH_ARGS=(-J "$HPC_GATEWAY")
RSYNC_SSH="ssh -J $HPC_GATEWAY"
