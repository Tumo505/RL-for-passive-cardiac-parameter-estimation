from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC

from ..config import fingerprint, save_json
from ..data import fixed_tasks, load_cases
from ..evaluation import plot_results, summarise
from .core import CalibrationOracle, converged_lm, converged_lsq, mechanics_inverse
from .environment import HybridEnv
from .features import compress_measurements
from .initializer import load_initializer
from .setup import prepare_hybrid, source_hash


METHODS = ("mechanics_direct", "lsq_midpoint", "neural_only", "neural_lsq", "map_lsq_midpoint",
           "neural_map_lsq", "neural_lm", "hybrid_sac")


def solve(method, config, model, cases, initializer, policy, task, budget, backend):
    settings = config["hybrid"]
    if method == "hybrid_sac":
        env = HybridEnv(config, model, cases, initializer, training=False, backend=backend)
        observation, info = env.reset(options={"task": task, "budget": budget})
        while True:
            action, _ = policy.predict(observation, deterministic=True)
            observation, _, done, _, info = env.step(action)
            if done:
                return env.oracle.best_z.copy(), env.oracle, info["stop_reason"]
    measurement = compress_measurements(model, task.observed, task.mask, settings["noise_floor"])
    if method == "mechanics_direct":
        initial = mechanics_inverse(model, measurement)
        oracle = CalibrationOracle(model, task, budget, measurement.noise, initial, [1., 1.], 0., backend)
        oracle.evaluate(initial)
        return initial, oracle, "direct_stress_space_fit"
    learned = method != "lsq_midpoint"
    initial = initializer.predict(measurement.features) if learned else np.array([.5, .5])
    regularization = settings["regularization_weight"] if method in ("map_lsq_midpoint", "neural_map_lsq", "neural_lm") else 0.
    oracle = CalibrationOracle(model, task, budget, measurement.noise, initial,
                               initializer.prior_std, regularization, backend)
    if method in ("lsq_midpoint", "map_lsq_midpoint"):
        start = np.array([.5, .5])
    else:
        start = initial
    if method == "neural_only":
        oracle.evaluate(initial)
        return initial, oracle, "direct_prediction"
    if method == "neural_lm":
        result = converged_lm(oracle, start, settings)
    elif method in ("lsq_midpoint", "neural_lsq", "map_lsq_midpoint", "neural_map_lsq"):
        result = converged_lsq(oracle, start, settings)
    else:
        raise ValueError(f"Unknown hybrid comparator: {method}")
    return result, oracle, "budget_or_convergence"


def evaluate_hybrid(config, data_dir, output, seed=0, split="validation", confirm_test=False,
                    methods=None, backend="reference", limit=None, threads=None, overwrite=False):
    if split not in ("validation", "test"):
        raise ValueError("Evaluate on validation or test")
    if split == "test" and not confirm_test:
        raise ValueError("Final test is sealed. Freeze configuration first, then pass --confirm-test")
    if threads is not None and threads < 1:
        raise ValueError("Thread count must be positive")
    torch.set_num_threads(threads or config["training"]["threads"])
    directory = Path(output)
    run = json.loads((directory / "run.json").read_text())
    if run["config_fingerprint"] != fingerprint(config) or run["seed"] != seed:
        raise ValueError("Run/configuration/seed mismatch")
    frozen = json.loads((directory / "frozen.json").read_text())
    if frozen.get("source_hash") != source_hash():
        raise ValueError("Hybrid implementation changed after selection; resume to revalidate or start a new run before evaluation")
    for name, field in (("best.zip", "policy_sha256"), ("initializer.pt", "initializer_sha256")):
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != frozen[field]:
            raise ValueError("Selected artifacts changed after validation selection; freeze them through training first")
    methods = list(methods or METHODS)
    if not methods or len(set(methods)) != len(methods) or any(method not in METHODS for method in methods):
        raise ValueError(f"Choose unique methods from {METHODS}")
    destination = directory / f"evaluation_{split}_{backend}"
    if (destination / "results.csv").exists() and not overwrite:
        protocol_path = destination / "protocol.json"
        if protocol_path.exists():
            old = json.loads(protocol_path.read_text())
            with (destination / "results.csv").open() as handle:
                count = sum(1 for _ in csv.DictReader(handle))
            requested_cases = min(config["data"][split+"_cases"], limit) if limit is not None else config["data"][split+"_cases"]
            if (old["config_fingerprint"] == fingerprint(config) and old["seed"] == seed and
                    old.get("source_hash") == frozen["source_hash"] and
                    old["policy_sha256"] == frozen["policy_sha256"] and
                    old["initializer_sha256"] == frozen["initializer_sha256"] and
                    old["methods"] == methods and old["unique_parameter_cases"] == requested_cases and
                    count == old["expected_result_rows"]):
                print(f"Reusing complete identical {split} evaluation: {destination}", flush=True)
                return destination
        raise ValueError("Existing evaluation is incomplete or has different inputs; use --overwrite to recover it deliberately")
    destination.mkdir(parents=True, exist_ok=True)
    mechanics = prepare_hybrid(config, data_dir)
    cases = load_cases(config, data_dir, split)
    tasks = list(fixed_tasks(config, data_dir, split, limit))
    initializer = load_initializer(directory / "initializer.pt", config, seed)
    policy = SAC.load(directory / "best.zip", device="cpu") if "hybrid_sac" in methods else None
    rows = []
    started = time.perf_counter()
    with (destination / "traces.jsonl").open("w") as trace_file:
        for index, task in enumerate(tasks):
            truth = mechanics.decode(task.z)
            for budget in config["evaluation"]["budgets"]:
                # Rotate execution order to reduce systematic warm-cache timing bias.
                offset = (index+seed) % len(methods)
                for method in methods[offset:]+methods[:offset]:
                    start = time.perf_counter()
                    z, oracle, reason = solve(method, config, mechanics, cases, initializer,
                                               policy, task, budget, backend)
                    seconds = time.perf_counter()-start
                    prediction = mechanics.decode(z)
                    relative = np.abs(prediction/truth-1)
                    audit_start = time.perf_counter()
                    audit = mechanics.reference(z)
                    trace = [{**step, "seconds": step["seconds"]+max(0., oracle.start-start)} for step in oracle.trace]
                    hits = []
                    for step in trace:
                        candidate = mechanics.decode(step["z"])
                        if np.max(np.abs(candidate/truth-1)) <= config["evaluation"]["parameter_success_threshold"]:
                            hits.append(step)
                    row = {"seed": seed, "case_id": task.case_id, "noise": task.noise,
                           "observed_fraction": task.fraction, "budget": budget, "method": method,
                           "backend": backend, "split": split, "forward_calls": oracle.calls,
                           "online_seconds": seconds, "stop_reason": reason,
                           "objective_kind": ("stress_space_direct_audit" if method == "mechanics_direct" else
                                              ("empirical_map" if oracle.regularization else "mle")),
                           "a_true_kpa": float(truth[0]), "af_true_kpa": float(truth[1]),
                           "a_estimated_kpa": float(prediction[0]), "af_estimated_kpa": float(prediction[1]),
                           "mean_relative_parameter_error": float(relative.mean()),
                           "max_relative_parameter_error": float(relative.max()),
                           "mean_log_parameter_error": float(np.mean(np.abs(np.log(prediction/truth)))),
                           "success": bool(relative.max() <= config["evaluation"]["parameter_success_threshold"]),
                           "first_success_calls_posthoc": hits[0]["calls"] if hits else "",
                           "first_success_seconds_posthoc": hits[0]["seconds"] if hits else "",
                           "reference_clean_relative_displacement_error": float(np.linalg.norm(audit-task.clean)/max(np.linalg.norm(task.clean), 1e-12)),
                           "reference_observed_mse": float(np.mean((audit[task.mask]-task.observed[task.mask])**2)),
                           "objective": oracle.best_cost, "audit_seconds": time.perf_counter()-audit_start}
                    rows.append(row)
                    trace_file.write(json.dumps({"seed": seed, "case_id": task.case_id, "noise": task.noise,
                        "observed_fraction": task.fraction, "budget": budget, "method": method,
                        "trace": trace})+"\n")
            if (index+1) % 10 == 0:
                print(f"hybrid evaluated {index+1}/{len(tasks)} {split} observation conditions", flush=True)
    with (destination / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(destination / "summary.json", summarise(rows))
    save_json(destination / "protocol.json", {"experiment": "hybrid_v2", "split": split,
        "source_hash": frozen["source_hash"],
        "config_fingerprint": fingerprint(config), "seed": seed, "backend": backend, "methods": methods,
        "unique_parameter_cases": len({task.case_id for task in tasks}), "observation_tasks": len(tasks),
        "evaluation_seconds": time.perf_counter()-started, "expected_result_rows": len(tasks)*len(methods)*len(config["evaluation"]["budgets"]),
        "known_noise_supplied": False, "noise_estimation": "Spatial affine residual degrees of freedom",
        "prior": "Shared initializer mean and validation-calibrated diagonal uncertainty for all MAP methods",
        "budget_unit": "One actual complete forward prediction; FD probes counted; exact-repeat cache available to all methods",
        "first_success": "Post-hoc truth-based diagnostic, never used for stopping or selection",
        "selection": "Best deployable MLE or MAP objective, never oracle parameter error",
        "initialization_in_timing": True, "posthoc_reference_audit_calls": len(rows),
        "online_device": "CPU for every method; offline initializer and SAC devices recorded separately",
        "policy_sha256": frozen["policy_sha256"], "initializer_sha256": frozen["initializer_sha256"]})
    plot_results(rows, destination)
    return destination
