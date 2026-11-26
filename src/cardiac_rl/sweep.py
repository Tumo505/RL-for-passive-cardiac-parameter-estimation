"""Validation-only accuracy sweep, frozen selection, then a separate final test."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import re
import signal
import subprocess
import sys

import numpy as np
import yaml

from .config import fingerprint, save_json
from .data import dataset_fingerprint
from .hybrid.evaluation import ACCURACY_METHODS
from .hybrid.initializer import initializer_fingerprint
from .hybrid.setup import configuration, source_hash

RL_METHODS = ("hybrid_sac", "hybrid_sac_polish", "hybrid_sac_noise_polish")
BASELINES = tuple(m for m in ACCURACY_METHODS if m not in RL_METHODS)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_suite(path):
    suite = yaml.safe_load(Path(path).read_text())
    if not suite.get("profiles") or suite.get("seeds", 0) < 1:
        raise ValueError("A sweep needs named profiles and positive seed count")
    configs = {}
    for name, filename in suite["profiles"].items():
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
            raise ValueError("Unsafe profile name")
        configs[name] = configuration(filename)
        if not configs[name]["hybrid"].get("enable_accuracy_controls"):
            raise ValueError("Sweep profiles must enable accuracy controls")
        if suite["primary_budget"] not in configs[name]["evaluation"]["budgets"]:
            raise ValueError("Primary budget is absent from evaluation budgets")
    first = next(iter(configs.values()))
    for config in configs.values():
        if (dataset_fingerprint(config) != dataset_fingerprint(first) or
                initializer_fingerprint(config) != initializer_fingerprint(first) or
                config["evaluation"] != first["evaluation"]):
            raise ValueError("Profiles must share data, supervised initialization and evaluation protocol")
    return suite, configs


def bindings(suite, configs, root):
    """Bind every candidate and its validation evidence before opening the test."""
    result = {}
    for profile, config in configs.items():
        runs = {}
        for seed in range(suite.get("seed_start", 0), suite.get("seed_start", 0)+suite["seeds"]):
            directory = Path(root)/profile/f"seed_{seed}"
            frozen = json.loads((directory/"frozen.json").read_text())
            if frozen["source_hash"] != source_hash():
                raise ValueError("Implementation changed after training; revalidate in a new sweep")
            for file, field in (("best.zip", "policy_sha256"), ("initializer.pt", "initializer_sha256")):
                if digest(directory/file) != frozen[field]:
                    raise ValueError("Frozen policy or initializer was modified")
            protocol = json.loads((directory/"evaluation_validation_reference/protocol.json").read_text())
            if (protocol["config_fingerprint"] != fingerprint(config) or protocol["seed"] != seed or
                    protocol["split"] != "validation" or protocol["backend"] != "reference" or
                    protocol["source_hash"] != source_hash() or protocol["methods"] != list(ACCURACY_METHODS) or
                    any(protocol[field] != frozen[field] for field in ("policy_sha256", "initializer_sha256"))):
                raise ValueError("Validation protocol does not match the frozen candidate")
            runs[str(seed)] = {"policy_sha256": frozen["policy_sha256"],
                               "initializer_sha256": frozen["initializer_sha256"],
                               "validation_sha256": digest(directory/"evaluation_validation_reference/results.csv")}
        result[profile] = {"config_fingerprint": fingerprint(config), "runs": runs}
    first = next(iter(result.values()))["runs"]
    if any(r["initializer_sha256"] != first[seed]["initializer_sha256"]
           for profile in result.values() for seed, r in profile["runs"].items()):
        raise ValueError("All profiles must share byte-identical initializers per seed")
    return result


def select(suite_path, root):
    root = Path(root)
    if (root/"final_test_started.json").exists():
        raise ValueError("Final test has been opened; selection can no longer change")
    suite, configs = load_suite(suite_path)
    bound = bindings(suite, configs, root)
    scores, canonical_baselines = [], {}
    for profile, config in configs.items():
        pooled = {method: [] for method in ACCURACY_METHODS}
        for seed in range(suite.get("seed_start", 0), suite.get("seed_start", 0)+suite["seeds"]):
            with (root/profile/f"seed_{seed}/evaluation_validation_reference/results.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            expected = set(itertools.product(range(config["data"]["validation_cases"]),
                config["data"]["noise_levels"], config["data"]["observed_fractions"],
                config["evaluation"]["budgets"], ACCURACY_METHODS))
            seen = set()
            for row in rows:
                key = (int(row["case_id"]), float(row["noise"]), float(row["observed_fraction"]), int(row["budget"]), row["method"])
                if (key in seen or key not in expected or int(row["seed"]) != seed or
                        row["split"] != "validation" or row["backend"] != "reference" or
                        not 1 <= int(row["forward_calls"]) <= int(row["budget"])):
                    raise ValueError("Duplicate, unexpected, or invalid validation row")
                seen.add(key)
                numbers = [float(row[field]) for field in ("mean_log_parameter_error", "online_seconds", "a_estimated_kpa", "af_estimated_kpa")]
                if not np.isfinite(numbers).all() or min(numbers) < 0:
                    raise ValueError("Invalid validation metrics")
                if row["method"] in BASELINES:
                    baseline_key = (seed, *key)
                    value = [float(row[f]) for f in ("a_estimated_kpa", "af_estimated_kpa", "mean_log_parameter_error", "forward_calls")]
                    previous = canonical_baselines.setdefault(baseline_key, value)
                    if not np.allclose(previous, value, rtol=1e-7, atol=1e-10):
                        raise ValueError("Shared baseline predictions differ between profiles")
                if int(row["budget"]) == suite["primary_budget"]:
                    pooled[row["method"]].append(row)
            if seen != expected:
                raise ValueError("Incomplete validation condition coverage")
        for method, rows in pooled.items():
            if profile != next(iter(configs)) and method in BASELINES:
                continue
            scores.append({"profile": profile, "method": method, "observations": len(rows),
                "mean_log_parameter_error": float(np.mean([float(r["mean_log_parameter_error"]) for r in rows])),
                "mean_forward_calls": float(np.mean([int(r["forward_calls"]) for r in rows])),
                "success_rate": float(np.mean([r["success"] == "True" for r in rows]))})
    candidates = [s for s in scores if s["method"] in RL_METHODS]
    winner = min(candidates, key=lambda s: (s["mean_log_parameter_error"], s["mean_forward_calls"], s["profile"], s["method"]))
    report = {"suite_sha256": digest(suite_path), "source_hash": source_hash(),
        "selector_sha256": digest(__file__), "dataset_fingerprint": dataset_fingerprint(next(iter(configs.values()))),
        "primary_budget": suite["primary_budget"], "bindings": bound, "selected": winner, "scores": scores,
        "selection_rule": "Lowest pooled validation mean log parameter error; tie: fewer forward calls, profile and method names",
        "baseline_reporting": "All fixed baselines from the first profile; no favorable timing or baseline selection",
        "test_used": False,
        "warning": "Validation selection is exploratory and cannot establish final-test superiority"}
    if (root/"selection.json").exists() and json.loads((root/"selection.json").read_text()) != report:
        raise ValueError("A different selection is already frozen; use a separate sweep root")
    save_json(root/"selection.json", report)
    print(json.dumps(report["selected"], indent=2), flush=True)
    return report


def execute(command):
    print("Launching: "+" ".join(map(str, command)), flush=True)
    child = subprocess.Popen(list(map(str, command)))
    old = {}
    def stop(signum, frame):
        if child.poll() is None:
            child.send_signal(signal.SIGUSR1)
    try:
        for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
            old[sig] = signal.signal(sig, stop)
        if child.wait() != 0:
            raise RuntimeError("Sweep worker failed or stopped; inspect the profile and seed logs")
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)


def run(suite_path, root, data_dir, local=False, device="cpu"):
    suite, configs = load_suite(suite_path)
    root = Path(root)
    if (root/"final_test_started.json").exists():
        raise ValueError("Cannot retrain after opening the final test")
    if (root/"selection.json").exists():
        return select(suite_path, root)
    first = next(iter(configs))
    for profile in configs:
        save_json(root/"progress.json", {"phase": "training_and_validation", "profile": profile, "profiles": list(configs)})
        common = ["--config", suite["profiles"][profile], "--data-dir", str(data_dir)]
        if local:
            for seed in range(suite.get("seed_start", 0), suite.get("seed_start", 0)+suite["seeds"]):
                directory = root/profile/f"seed_{seed}"
                seed_args = common+["--output", str(directory), "--seed", str(seed)]
                train = [sys.executable, "-m", "cardiac_rl.hybrid_cli", "train", *seed_args, "--device", device]
                if (directory/"run.json").exists():
                    train += ["--resume"]
                if profile != first:
                    train += ["--initializer-from", str(root/first/f"seed_{seed}")]
                previous = json.loads((directory/"run.json").read_text()) if (directory/"run.json").exists() else {}
                if not (previous.get("status") in ("completed", "validation_early_stopping") and
                        previous.get("source_hash") == source_hash() and previous.get("config_fingerprint") == fingerprint(configs[profile])):
                    execute(train)
                execute([sys.executable, "-m", "cardiac_rl.hybrid_cli", "evaluate", *seed_args, "--split", "validation"])
        else:
            command = [sys.executable, "-m", "cardiac_rl.allocation", "--workflow", "hybrid", *common,
                "--output", str(root/profile), "--seeds", str(suite["seeds"]),
                "--seed-start", str(suite.get("seed_start", 0)), "--log-prefix", profile, "--skip-completed-training"]
            if profile != first:
                command += ["--initializer-from", str(root/first)]
            execute(command)
    report = select(suite_path, root)
    save_json(root/"progress.json", {"phase": "validation_complete", "selected": report["selected"], "test_opened": False})
    return report


def final_test(suite_path, root, data_dir):
    root = Path(root)
    suite, configs = load_suite(suite_path)
    report = json.loads((root/"selection.json").read_text())
    if (report["suite_sha256"] != digest(suite_path) or report["source_hash"] != source_hash() or
            report["selector_sha256"] != digest(__file__) or report["bindings"] != bindings(suite, configs, root)):
        raise ValueError("Frozen selection inputs changed; final test refused")
    receipt = {"selection_sha256": digest(root/"selection.json"), "data_dir": str(Path(data_dir).resolve())}
    path = root/"final_test_started.json"
    if path.exists() and json.loads(path.read_text()) != receipt:
        raise ValueError("A different final test was already started")
    save_json(path, receipt)
    profile, method = report["selected"]["profile"], report["selected"]["method"]
    execute([sys.executable, "-m", "cardiac_rl.allocation", "--workflow", "hybrid", "--evaluate-only",
        "--config", suite["profiles"][profile], "--data-dir", str(data_dir), "--output", str(root/profile),
        "--seeds", str(suite["seeds"]), "--seed-start", str(suite.get("seed_start", 0)),
        "--evaluation-split", "test", "--confirm-test", "--log-prefix", "final_"+profile,
        "--methods", *BASELINES, method])
    save_json(root/"progress.json", {"phase": "final_test_complete", "selected": report["selected"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "select", "final-test", "status"))
    parser.add_argument("--suite", default="configs/hybrid_sweep.yaml")
    parser.add_argument("--root", default="runs/hybrid_sweep")
    parser.add_argument("--data-dir", default="data/hybrid_sweep")
    parser.add_argument("--local", action="store_true", help="Sequential CPU/GPU validation smoke checks without PBS")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "run":
        run(args.suite, args.root, args.data_dir, args.local, args.device)
    elif args.command == "select":
        select(args.suite, args.root)
    elif args.command == "final-test":
        final_test(args.suite, args.root, args.data_dir)
    else:
        root = Path(args.root)
        print((root/"progress.json").read_text() if (root/"progress.json").exists() else "Sweep has not started")
        suite, _ = load_suite(args.suite)
        for profile in suite["profiles"]:
            for seed in range(suite.get("seed_start", 0), suite.get("seed_start", 0)+suite["seeds"]):
                directory = root/profile/f"seed_{seed}"
                if (directory/"run.json").exists():
                    run_info = json.loads((directory/"run.json").read_text())
                    best = json.loads((directory/"best.json").read_text()) if (directory/"best.json").exists() else {}
                    print(f"{profile} seed={seed}: {run_info.get('status')}, transitions={run_info.get('timesteps', 'in progress')}, best={best.get('mean_log_parameter_error')}")


if __name__ == "__main__":
    main()
