#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
if [[ -f scripts/hpc/cluster.env ]]; then source scripts/hpc/cluster.env; fi
if [[ -n "${HPC_MODULE_INIT:-}" ]]; then source "$HPC_MODULE_INIT"; fi
HPC_PYTHON=${HPC_PYTHON:-python3}
"$HPC_PYTHON" -c 'import sys; assert (3,10) <= sys.version_info[:2] < (3,14), "Use Python 3.10-3.13"'
"$HPC_PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
  .venv/bin/python -m pip install 'torch>=2.4,<3' --index-url "$TORCH_INDEX_URL"
fi
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pip freeze > requirements.cluster.lock.txt
mkdir -p logs runs data
echo "Environment ready. CUDA will be checked inside the allocated GPU job."
