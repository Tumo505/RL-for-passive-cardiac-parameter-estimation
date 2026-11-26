"""Run independent seeds concurrently within the actual PBS resource envelope."""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch

from .config import load_config, save_json
from .data import prepare
from .evaluation import aggregate


def duration_seconds(value):
    parts = [int(part) for part in value.strip().split(":")]
    if len(parts) != 3 or min(parts) < 0 or parts[1] >= 60 or parts[2] >= 60:
        raise ValueError(f"Expected PBS hours:minutes:seconds, got {value}")
    return parts[0]*3600 + parts[1]*60 + parts[2]


def parse_pbs_status(text):
    def field(name):
        match = re.search(r"^\s*"+re.escape(name)+r"\s*=\s*(\S+)", text, re.MULTILINE)
        return match.group(1) if match else None
    walltime = field("Resource_List.walltime")
    if walltime is None:
        raise ValueError("qstat did not report Resource_List.walltime")
    used = field("resources_used.walltime") or "00:00:00"
    cpus = field("Resource_List.ncpus")
    if cpus is None and field("Resource_List.select"):
        chunks = field("Resource_List.select").split("+")
        counts = []
        for chunk in chunks:
            match = re.search(r"(?:^|:)ncpus=(\d+)", chunk)
            if match:
                multiplicity = int(chunk.split(":")[0]) if chunk.split(":")[0].isdigit() else 1
                counts.append(multiplicity*int(match.group(1)))
        cpus = str(sum(counts)) if counts else None
    return {"walltime_seconds": duration_seconds(walltime), "used_seconds": duration_seconds(used),
            "allocated_cpus": int(cpus) if cpus else None, "allocated_memory": field("Resource_List.mem")}


def worker_plan(cpu_count, gpu_count, seed_count=4):
    if min(cpu_count, gpu_count, seed_count) < 1:
        raise ValueError("Need allocated CPUs, at least one visible GPU, and at least one seed")
    # Four independent policies can share a P100 for this small model. Device
    # indices are logical indices under the scheduler's CUDA_VISIBLE_DEVICES.
    workers = min(seed_count, cpu_count)
    if workers < seed_count:
        raise ValueError("Need at least one allocated CPU per simultaneous seed")
    if gpu_count > seed_count:
        raise ValueError("More visible GPUs than seeds; increase --seeds to use all GPUs")
    base, remainder = divmod(cpu_count, workers)
    return [{"slot": i, "device": f"cuda:{i % gpu_count}",
             "threads": base+(i < remainder)} for i in range(workers)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir", default="data/p100_full")
    parser.add_argument("--output", default="runs/p100_full")
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--expected-walltime-hours", type=float, default=168)
    parser.add_argument("--evaluation-reserve-hours", type=float, default=6)
    parser.add_argument("--checkpoint-grace-seconds", type=float, default=120)
    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate existing best policies without starting or resuming training")
    parser.add_argument("--workflow", choices=("legacy", "hybrid"), default="legacy")
    parser.add_argument("--evaluation-split", choices=("validation", "test"), default="validation")
    parser.add_argument("--confirm-test", action="store_true")
    parser.add_argument("--initializer-from", help="Root containing compatible seed_N initializers")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--log-prefix", default="")
    parser.add_argument("--skip-completed-training", action="store_true")
    args = parser.parse_args()
    if args.log_prefix and not re.fullmatch(r"[a-zA-Z0-9_-]+", args.log_prefix):
        parser.error("Log prefix must contain only letters, numbers, underscores or hyphens")
    if args.workflow == "hybrid" and args.evaluation_split == "test" and not args.confirm_test:
        parser.error("Freeze hybrid configuration and pass --confirm-test for the fresh final test")
    job_id = os.environ.get("PBS_JOBID")
    if not job_id:
        parser.error("Run inside a PBS allocation; PBS_JOBID is absent")
    if args.expected_walltime_hours < 0 or args.evaluation_reserve_hours <= 0 or args.checkpoint_grace_seconds <= 0:
        parser.error("Expected walltime must be nonnegative; reserve and checkpoint grace must be positive")
    status = subprocess.run(["qstat", "-f", job_id], check=True, text=True, capture_output=True)
    allocation = parse_pbs_status(status.stdout)
    if allocation["walltime_seconds"] < args.expected_walltime_hours*3600:
        parser.error(f"PBS granted only {allocation['walltime_seconds']/3600:g} hours, "
                     f"below the requested {args.expected_walltime_hours:g} hours")
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this allocation")
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    cpu_limits = [affinity]
    if allocation["allocated_cpus"]:
        cpu_limits.append(allocation["allocated_cpus"])
    if os.environ.get("PBS_NP", "").isdigit():
        cpu_limits.append(int(os.environ["PBS_NP"]))
    cpu_count = min(cpu_limits)
    gpu_count = torch.cuda.device_count()
    plan = worker_plan(cpu_count, gpu_count, args.seeds)
    remaining = allocation["walltime_seconds"]-allocation["used_seconds"]
    deadline = time.monotonic()+remaining
    reserve = args.evaluation_reserve_hours*3600
    grace = args.checkpoint_grace_seconds
    if remaining <= (grace if args.evaluate_only else reserve+grace):
        parser.error("Insufficient remaining allocation time for training plus final evaluation")
    root = Path(args.output)
    logs = Path("logs")
    logs.mkdir(exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    if args.evaluate_only:
        missing = [str(root / f"seed_{seed}" / name)
                   for seed in range(args.seed_start, args.seed_start+args.seeds)
                   for name in (("run.json", "best.zip", "best.json", "initializer.pt", "frozen.json")
                                if args.workflow == "hybrid" else ("run.json", "best.zip", "best.json"))
                   if not (root / f"seed_{seed}" / name).exists()]
        if missing:
            parser.error("Evaluation-only mode requires saved policies: " + ", ".join(missing))
    if args.workflow == "hybrid":
        from .hybrid.setup import configuration, prepare_hybrid
        config = configuration(args.config)
        prepare_hybrid(config, args.data_dir)
    else:
        config = load_config(args.config)
        prepare(config, args.data_dir)
    manifest = {**allocation, "job_id": job_id, "visible_gpus": gpu_count,
                "gpu_names": [torch.cuda.get_device_name(i) for i in range(gpu_count)],
                "usable_cpus": cpu_count, "workers": plan, "seed_start": args.seed_start,
                "evaluation_reserve_hours": args.evaluation_reserve_hours,
                "checkpoint_grace_seconds": grace,
                "training_transition_cap": config["training"]["timesteps"] if args.workflow == "hybrid" else None,
                "evaluate_only": args.evaluate_only,
                "workflow": args.workflow, "evaluation_split": args.evaluation_split if args.workflow == "hybrid" else "test",
                "policy": "Concurrent independent seeds, distributed across scheduler-visible GPUs",
                "memory_policy": "Demand-driven, algorithm/replay settings unchanged"}
    save_json(root / ("evaluation_allocation.json" if args.evaluate_only else "allocation.json"), manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    stopped = {"requested": False}
    children = []

    def stop_children(signum, frame):
        stopped["requested"] = True
        for process in children:
            if process.poll() is None:
                process.send_signal(signal.SIGUSR1)

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
        signal.signal(signum, stop_children)

    def run_phase(phase):
        save_json(root / "phase.json", {"phase": phase, "job_id": job_id, "log_prefix": args.log_prefix})
        children.clear()
        handles = []
        try:
            for worker in plan:
                seed = args.seed_start+worker["slot"]
                directory = root / f"seed_{seed}"
                if phase == "train" and args.skip_completed_training and (directory / "run.json").exists():
                    prior = json.loads((directory / "run.json").read_text())
                    from .config import fingerprint
                    from .hybrid.setup import source_hash
                    if (args.workflow == "hybrid" and prior.get("status") in ("completed", "validation_early_stopping")
                            and prior.get("source_hash") == source_hash() and prior.get("config_fingerprint") == fingerprint(config)
                            and prior.get("seed") == seed and (directory / "frozen.json").exists()):
                        print(f"Reusing completed training seed={seed}", flush=True)
                        continue
                module = "cardiac_rl.hybrid_cli" if args.workflow == "hybrid" else "cardiac_rl.cli"
                command = [sys.executable, "-m", module, phase, "--config", args.config,
                           "--data-dir", args.data_dir, "--output", str(directory), "--seed", str(seed),
                           "--threads", str(worker["threads"])]
                if phase == "train":
                    if args.initializer_from:
                        command += ["--initializer-from", str(Path(args.initializer_from) / f"seed_{seed}")]
                    seconds = deadline-time.monotonic()-reserve-grace
                    if seconds <= 0:
                        raise RuntimeError("No training time remains before final evaluation")
                    command += ["--device", worker["device"], "--max-seconds", str(seconds)]
                    if (directory / "run.json").exists():
                        command.append("--resume")
                else:
                    if args.methods:
                        command += ["--methods", *args.methods]
                    command += ["--backend", "reference", "--direct-device", worker["device"]]
                    if args.workflow == "hybrid":
                        command += ["--split", args.evaluation_split]
                        if args.confirm_test:
                            command.append("--confirm-test")
                environment = os.environ.copy()
                for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                    environment[name] = str(worker["threads"])
                environment["MPLCONFIGDIR"] = str((logs / f"mpl_seed_{seed}").resolve())
                prefix = args.log_prefix+"_" if args.log_prefix else ""
                path = logs / f"{prefix}{phase}_{job_id}_seed_{seed}.log"
                handle = path.open("a")
                handles.append(handle)
                print(f"{phase} seed={seed} device={worker['device']} cpu_threads={worker['threads']} log={path}", flush=True)
                children.append(subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=environment))
            while any(process.poll() is None for process in children):
                failures = [process.returncode for process in children if process.poll() not in (None, 0)]
                if failures or stopped["requested"]:
                    stop_children(signal.SIGUSR1, None)
                    raise RuntimeError(f"{phase} interrupted or failed; exit codes={failures}")
                if time.monotonic() >= deadline-grace:
                    stop_children(signal.SIGUSR1, None)
                    raise RuntimeError(f"{phase} reached the allocation checkpoint deadline")
                time.sleep(2)
            if any(process.returncode != 0 for process in children):
                raise RuntimeError(f"{phase} failed; inspect seed logs")
        finally:
            for process in children:
                if process.poll() is None:
                    process.send_signal(signal.SIGUSR1)
            for process in children:
                try:
                    process.wait(timeout=grace)
                except subprocess.TimeoutExpired:
                    process.terminate()
            for handle in handles:
                handle.close()

    if not args.evaluate_only:
        run_phase("train")
    run_phase("evaluate")
    aggregate(root, root / "aggregate.json")
    save_json(root / "phase.json", {"phase": "complete", "job_id": job_id, "log_prefix": args.log_prefix})
    print(f"Completed full reference comparison: {root / 'aggregate.json'}", flush=True)


if __name__ == "__main__":
    main()
