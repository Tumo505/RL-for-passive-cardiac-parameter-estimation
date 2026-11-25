import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


@pytest.mark.parametrize("submission_succeeds", [True, False])
def test_evaluation_is_queued_before_graceful_stop(tmp_path, submission_succeeds):
    scripts = tmp_path / "scripts/hpc"
    scripts.mkdir(parents=True)
    shutil.copy("scripts/hpc/finish-training.sh", scripts / "finish-training.sh")
    (tmp_path / ".venv/bin").mkdir(parents=True)
    (tmp_path / ".venv/bin/python").symlink_to(sys.executable)
    for seed in range(4):
        root = tmp_path / f"runs/p100_full/seed_{seed}"
        root.mkdir(parents=True)
        for name in ("best.json", "run.json", "config.json"):
            (root / name).write_text("{}")
        with zipfile.ZipFile(root / "best.zip", "w") as archive:
            archive.writestr("test", "saved policy")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "calls.txt"
    bodies = {
        "qstat": 'echo "job_state = R"',
        "qsub": 'printf "qsub|%s\\n" "$*" >> "$CAPTURE"\n' + ('echo 456.example' if submission_succeeds else 'exit 1'),
        "qsig": 'printf "qsig|%s\\n" "$*" >> "$CAPTURE"',
    }
    for name, body in bodies.items():
        stub = bin_dir / name
        stub.write_text("#!/usr/bin/env bash\n" + body + "\n")
        stub.chmod(0o755)
    environment = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "CAPTURE": str(capture),
                   "HPC_QUEUE": "demo_gpu", "HPC_SOFTWARE_RESOURCE": "software=demo_app"}
    result = subprocess.run(["bash", str(scripts / "finish-training.sh"), "123.example"],
                            env=environment, capture_output=True, text=True)
    calls = capture.read_text().splitlines()
    assert "depend=afterany:123.example" in calls[0]
    backups = list((tmp_path / "runs/p100_full/policy_backups").glob("*/seed_*/best.zip"))
    assert len(backups) == 4
    if submission_succeeds:
        assert result.returncode == 0, result.stderr
        assert calls[1] == "qsig|-s USR1 123.example"
        marker = json.loads((tmp_path / "runs/p100_full/evaluation_request_123.example.json").read_text())
        assert marker["evaluation_job"] == "456.example"
        # A repeated command cannot accidentally submit a second evaluation job.
        repeated = subprocess.run(["bash", str(scripts / "finish-training.sh"), "123.example"],
                                  env=environment, capture_output=True, text=True)
        assert repeated.returncode != 0
        assert capture.read_text().splitlines() == calls
    else:
        assert result.returncode != 0
        assert len(calls) == 1  # Submission failure must leave training untouched.
