#!/usr/bin/env bash
# Run on the submission host. Queue evaluation before signalling training.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
training_job=${1:?Usage: finish-training.sh TRAINING_JOB_ID}
if [[ ! "$training_job" =~ ^[0-9]+([.][[:alnum:]_-]+)*$ ]]; then
  echo "Invalid PBS training job ID" >&2; exit 2
fi
if [[ -f scripts/hpc/cluster.env ]]; then source scripts/hpc/cluster.env; fi
if [[ ! "${HPC_QUEUE:-}" =~ ^[[:alnum:]_][[:alnum:]_.-]*$ || "$HPC_QUEUE" == your_gpu_queue ]]; then
  echo "Set HPC_QUEUE in cluster.env or the environment" >&2; exit 2
fi
for command_name in qsub qstat qsig; do command -v "$command_name" >/dev/null; done
status=$(qstat -f "$training_job")
if ! grep -Eq 'job_state[[:space:]]*=[[:space:]]*R' <<< "$status"; then
  echo "Training is not running. Inspect its status before switching phases." >&2; exit 2
fi
run_root=${RUN_ROOT:-runs/p100_full}
seed_count=${SEEDS:-4}
seed_start=${SEED_START:-0}
if [[ ! "$seed_count" =~ ^[1-9][0-9]*$ || ! "$seed_start" =~ ^[0-9]+$ ]]; then
  echo "SEEDS and SEED_START must be nonnegative integers (SEEDS > 0)" >&2; exit 2
fi
if [[ "$run_root" == *','* || "$run_root" == *$'\n'* ]]; then
  echo "RUN_ROOT cannot contain commas or newlines" >&2; exit 2
fi
marker="$run_root/evaluation_request_${training_job}.json"
if [[ -f "$marker" ]]; then
  echo "An evaluation request already exists: $marker. Inspect it and qstat; no duplicate job submitted." >&2
  exit 2
fi
# Validate every policy and back up the current selected policies before qsig.
.venv/bin/python - "$run_root" "$seed_start" "$seed_count" "$training_job" <<'PY'
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

root, start, count, job = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
for seed in range(start, start+count):
    directory = root / f"seed_{seed}"
    json.loads((directory / "run.json").read_text())
    json.loads((directory / "best.json").read_text())
    with zipfile.ZipFile(directory / "best.zip") as archive:
        if archive.testzip() is not None:
            raise RuntimeError(f"Unreadable policy for seed {seed}; training was not signalled")
backup = root / "policy_backups" / f"before_evaluation_{job}_{time.time_ns()}"
for seed in range(start, start+count):
    destination = backup / f"seed_{seed}"
    destination.mkdir(parents=True)
    for name in ("best.zip", "best.json", "config.json"):
        source = root / f"seed_{seed}" / name
        if source.exists():
            shutil.copy2(source, destination / name)
print(f"Selected policies backed up to {backup}")
PY
pbs_args=(-q "$HPC_QUEUE" -W "depend=afterany:$training_job")
if [[ -n "${HPC_SOFTWARE_RESOURCE:-}" ]]; then
  if [[ ! "$HPC_SOFTWARE_RESOURCE" =~ ^[[:alnum:]_]+=[[:alnum:]_.-]+$ ]]; then
    echo "Invalid software resource" >&2; exit 2
  fi
  pbs_args+=(-l "$HPC_SOFTWARE_RESOURCE")
fi
mkdir -p logs
evaluation_job=$(qsub "${pbs_args[@]}" \
  -v "RUN_ROOT=$run_root,SEEDS=$seed_count,SEED_START=$seed_start" \
  -o "logs/evaluation_$(date +%Y%m%d_%H%M%S).log" scripts/hpc/evaluate.pbs)
if [[ ! "$evaluation_job" =~ ^[0-9]+([.][[:alnum:]_-]+)*$ ]]; then
  echo "Unexpected qsub response: $evaluation_job. Training was not signalled." >&2; exit 2
fi
.venv/bin/python - "$marker" "$training_job" "$evaluation_job" <<'PY'
import json
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"training_job": sys.argv[2], "evaluation_job": sys.argv[3],
    "dependency": "afterany", "signal_requested": "USR1"}, indent=2)+"\n")
PY
echo "Evaluation queued: $evaluation_job (waits for $training_job to exit)"
if ! qsig -s USR1 "$training_job"; then
  echo "Signal failed. Training remains running; evaluation $evaluation_job remains dependent. Inspect qstat." >&2
  exit 1
fi
echo "Graceful stop requested. Workers save checkpoints; evaluation starts after the training job exits."
echo "The old launcher may report interruption/nonzero exit; afterany allows evaluation to proceed."
