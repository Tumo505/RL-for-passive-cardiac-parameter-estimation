#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
if [[ ! -f scripts/hpc/cluster.env ]]; then
  echo "Copy cluster.env.example to cluster.env and set your site queue and resource." >&2
  exit 2
fi
source scripts/hpc/cluster.env
if [[ ! "${HPC_QUEUE:-}" =~ ^[[:alnum:]_][[:alnum:]_.-]*$ || "$HPC_QUEUE" == your_gpu_queue ]]; then
  echo "Set HPC_QUEUE in scripts/hpc/cluster.env" >&2; exit 2
fi
pbs_args=(-q "$HPC_QUEUE")
if [[ -n "${HPC_SOFTWARE_RESOURCE:-}" ]]; then
  if [[ ! "$HPC_SOFTWARE_RESOURCE" =~ ^[[:alnum:]_]+=[[:alnum:]_.-]+$ ]]; then
    echo "HPC_SOFTWARE_RESOURCE must be a single resource=value pair" >&2; exit 2
  fi
  pbs_args+=(-l "$HPC_SOFTWARE_RESOURCE")
fi
mkdir -p logs
qsub "${pbs_args[@]}" "$@" scripts/hpc/train.pbs
