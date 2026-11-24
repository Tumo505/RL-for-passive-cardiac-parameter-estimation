import os
import shutil
import subprocess
from pathlib import Path


def test_pbs_isolates_four_gpus_and_runs_evaluation_after_training(tmp_path):
    source = Path("scripts/hpc/train.pbs").resolve()
    (tmp_path / ".venv/bin").mkdir(parents=True)
    fake = tmp_path / ".venv/bin/python"
    fake.write_text("""#!/usr/bin/env bash
set -eu
printf '%s|%s\\n' "${CUDA_VISIBLE_DEVICES:-unset}" "$*" >> "$CAPTURE"
""")
    fake.chmod(0o755)
    capture = tmp_path / "calls.txt"
    environment = {"PATH": os.environ["PATH"], "PBS_O_WORKDIR": str(tmp_path), "PBS_JOBID": "test",
                   "CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b,GPU-c,GPU-d", "CAPTURE": str(capture)}
    result = subprocess.run(["bash", str(source)], env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    calls = capture.read_text().splitlines()
    trains = [line for line in calls if "cli train " in line]
    evaluates = [line for line in calls if "cli evaluate " in line]
    assert len(trains) == len(evaluates) == 4
    assert {line.split("|")[0] for line in trains} == {"GPU-a", "GPU-b", "GPU-c", "GPU-d"}
    assert all("--device cuda:0" in line for line in trains)
    assert max(calls.index(line) for line in trains) < min(calls.index(line) for line in evaluates)
    assert "aggregate" in calls[-1]


def test_transfer_uses_gateway_and_preserves_remote_files(tmp_path):
    root = tmp_path / "project"
    scripts = root / "scripts/hpc"
    scripts.mkdir(parents=True)
    for name in ("common.sh", "push.sh"):
        shutil.copy(Path("scripts/hpc") / name, scripts / name)
    (scripts / "site.env").write_text("HPC_GATEWAY=demo-user@login.example.org\nHPC_TARGET=demo-user@submission-host\nREMOTE_DIR=/scratch/user/SACAM\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("ssh", "rsync"):
        stub = bin_dir / name
        stub.write_text(f'#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$CAPTURE/{name}"\n')
        stub.chmod(0o755)
    environment = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "CAPTURE": str(tmp_path)}
    subprocess.run(["bash", str(scripts / "push.sh")], env=environment, check=True, capture_output=True)
    assert "-J demo-user@login.example.org demo-user@submission-host" in (tmp_path / "ssh").read_text()
    transfer = (tmp_path / "rsync").read_text()
    assert "--delete" not in transfer
    assert "--exclude=.venv/" in transfer
    assert "demo-user@submission-host:/scratch/user/SACAM/" in transfer


def test_pbs_failure_prevents_evaluation(tmp_path):
    (tmp_path / ".venv/bin").mkdir(parents=True)
    fake = tmp_path / ".venv/bin/python"
    fake.write_text("""#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$CAPTURE"
if [[ "$*" == *"cli train "* && "$*" == *"--seed 1 "* ]]; then exit 1; fi
""")
    fake.chmod(0o755)
    capture = tmp_path / "calls.txt"
    environment = {"PATH": os.environ["PATH"], "PBS_O_WORKDIR": str(tmp_path), "PBS_JOBID": "test",
                   "CAPTURE": str(capture), "CUDA_VISIBLE_DEVICES": "0,1,2,3"}
    result = subprocess.run(["bash", str(Path("scripts/hpc/train.pbs").resolve())], env=environment,
                            capture_output=True)
    assert result.returncode != 0
    assert "cli evaluate " not in capture.read_text()


def test_submission_uses_private_queue_configuration(tmp_path):
    scripts = tmp_path / "scripts/hpc"
    scripts.mkdir(parents=True)
    shutil.copy("scripts/hpc/submit-local.sh", scripts / "submit-local.sh")
    (scripts / "cluster.env").write_text("HPC_QUEUE=demo_gpu\nHPC_SOFTWARE_RESOURCE=software=demo_app\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "qsub"
    stub.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURE"\n')
    stub.chmod(0o755)
    capture = tmp_path / "arguments.txt"
    environment = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "CAPTURE": str(capture)}
    subprocess.run(["bash", str(scripts / "submit-local.sh"), "-v", "TOTAL_STEPS=3000"],
                   env=environment, check=True, capture_output=True)
    args = capture.read_text().splitlines()
    assert args[:4] == ["-q", "demo_gpu", "-l", "software=demo_app"]
    assert args.count("scripts/hpc/train.pbs") == 1
    assert "-A" not in args
