#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
mkdir -p dist
tar --exclude='__pycache__' --exclude='*.pyc' --exclude='site.env' --exclude='cluster.env' --exclude='scripts/hpc/README.md' \
  -czf dist/sacam-cardiac-rl-hpc.tar.gz \
  pyproject.toml README.md requirements.local.lock.txt configs src scripts tests .gitignore
echo "$PWD/dist/sacam-cardiac-rl-hpc.tar.gz"
