from __future__ import annotations

import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC

from .baselines import Objective, load_direct, optimise, task_features, train_direct
from .config import fingerprint, save_json
from .data import fixed_tasks, load_cases, prepare
from .environment import CalibrationEnv, discrepancy


METHODS = ("sac", "least_squares", "differential_evolution", "bayesian", "direct", "random")


def run_sac(policy, config, mechanics, cases, task, budget, backend):
    env = CalibrationEnv(config, mechanics, cases, backend)
    observation, info = env.reset(options={"task": task, "horizon": budget-1})
    trace = [{"calls": 1, "loss": info["best_loss"], "z": info["best_z"].tolist()}]
    for _ in range(budget-1):
        action, _ = policy.predict(observation, deterministic=True)
        observation, _, done, _, info = env.step(action)
        trace.append({"calls": info["forward_calls"], "loss": info["best_loss"], "z": info["best_z"].tolist()})
        if done:
            break
    return info["best_z"], info["forward_calls"], trace


def identifiability(mechanics, task):
    # Truth is used ONLY in this post-hoc diagnostic, never by estimators/rewards.
    h = 1e-4
    columns = []
    for i in range(2):
        lo, hi = task.z.copy(), task.z.copy()
        lo[i], hi[i] = max(0, lo[i]-h), min(1, hi[i]+h)
        columns.append((mechanics.reference(hi)[task.mask]-mechanics.reference(lo)[task.mask])/(hi[i]-lo[i]))
    singular = np.linalg.svd(np.column_stack(columns), compute_uv=False)
    return {"jacobian_min_singular": float(singular[-1]),
            "jacobian_condition": float(singular[0]/max(singular[-1], 1e-15))}


def evaluate(config, data_dir, directory, methods=None, backend="grid", limit=None, seed=0, direct_device="cpu", threads=None):
    directory = Path(directory)
    metadata = json.loads((directory / "run.json").read_text())
    if metadata["config_fingerprint"] != fingerprint(config):
        raise ValueError("Evaluation configuration differs from training run")
    if metadata["seed"] != seed:
        raise ValueError("Evaluation seed must match the training run seed")
    if threads is not None and threads < 1:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(threads or config["training"]["threads"])
    mechanics = prepare(config, data_dir)
    cases = load_cases(config, data_dir, "test")
    methods = list(methods or METHODS)
    if any(method not in METHODS for method in methods):
        raise ValueError(f"Choose methods from {METHODS}")
    budgets = config["evaluation"]["budgets"]
    if max(budgets)-1 > config["environment"]["horizon"]:
        raise ValueError("Evaluation budget exceeds policy training horizon")
    policy = None
    if "sac" in methods:
        policy = SAC.load(directory / "best.zip", device="cpu")
    direct = None
    if "direct" in methods:
        direct = load_direct(train_direct(config, data_dir, directory, seed, direct_device))
    result_dir = directory / ("evaluation_" + backend)
    result_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    tasks = list(fixed_tasks(config, data_dir, "test", limit=limit))
    start_all = time.perf_counter()
    with (result_dir / "traces.jsonl").open("w") as trace_file:
        for index, task in enumerate(tasks):
            diagnostic = identifiability(mechanics, task)
            truth = mechanics.decode(task.z)
            for budget in budgets:
                for method in methods:
                    start = time.perf_counter()
                    if method == "sac":
                        estimate_z, calls, trace = run_sac(policy, config, mechanics, cases, task, budget, backend)
                    elif method == "direct":
                        with torch.no_grad():
                            estimate_z = direct(torch.tensor(task_features(task))[None]).numpy()[0].astype(float)
                        objective = Objective(mechanics, task, budget, backend)
                        objective(estimate_z)
                        calls, trace = objective.calls, objective.trace
                    else:
                        objective = Objective(mechanics, task, budget, backend)
                        estimate_z, calls, trace = optimise(method, objective, seed+task.case_id)
                    online_seconds = time.perf_counter()-start
                    estimated = mechanics.decode(estimate_z)
                    relative = np.abs(estimated/truth-1)
                    # Audits are excluded from optimisation budgets and timed separately.
                    start_audit = time.perf_counter()
                    reference_prediction = mechanics.reference(estimate_z)
                    reference_loss = discrepancy(reference_prediction, task)
                    clean_error = np.linalg.norm(reference_prediction-task.clean)/max(np.linalg.norm(task.clean), 1e-12)
                    row = {"seed": seed, "case_id": task.case_id, "noise": task.noise,
                           "observed_fraction": task.fraction, "budget": budget, "method": method,
                           "backend": backend, "forward_calls": calls, "online_seconds": online_seconds,
                           "a_true_kpa": float(truth[0]), "af_true_kpa": float(truth[1]),
                           "a_estimated_kpa": float(estimated[0]), "af_estimated_kpa": float(estimated[1]),
                           "mean_relative_parameter_error": float(np.mean(relative)),
                           "max_relative_parameter_error": float(np.max(relative)),
                           "mean_log_parameter_error": float(np.mean(np.abs(np.log(estimated/truth)))),
                           "success": bool(np.max(relative) <= config["evaluation"]["parameter_success_threshold"]),
                           "reference_observed_loss": reference_loss,
                           "reference_clean_relative_displacement_error": float(clean_error),
                           "optimisation_loss": trace[-1]["loss"],
                           "audit_seconds": time.perf_counter()-start_audit, **diagnostic}
                    rows.append(row)
                    trace_file.write(json.dumps({"seed": seed, "case_id": task.case_id, "noise": task.noise,
                        "observed_fraction": task.fraction, "budget": budget, "method": method,
                        "trace": trace})+"\n")
            if (index+1) % 10 == 0:
                print(f"evaluated {index+1}/{len(tasks)} observation conditions", flush=True)
    with (result_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarise(rows)
    save_json(result_dir / "summary.json", summary)
    save_json(result_dir / "protocol.json", {"backend": backend, "methods": methods,
        "config_fingerprint": fingerprint(config), "seed": seed, "observation_tasks": len(tasks),
        "unique_parameter_cases": len({t.case_id for t in tasks}), "evaluation_seconds": time.perf_counter()-start_all,
        "budget_unit": "one complete four-load displacement prediction (not one scalar residual)",
        "initial_candidate_included": True, "finite_difference_calls_included": True,
        "posthoc_reference_audit_calls": len(rows), "identifiability_reference_calls": 4*len(tasks),
        "truth_fields_precomputed": True, "success": "both parameters within relative threshold",
        "uncertainty": "bootstrap over parameter case IDs; repeated noise/masks are not independent cases"})
    plot_results(rows, result_dir)
    return result_dir


def summarise(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["budget"], row["noise"], row["observed_fraction"])].append(row)
    result = []
    rng = np.random.default_rng(19273)
    for key, group in groups.items():
        errors = np.array([x["mean_relative_parameter_error"] for x in group])
        # Within each group there is exactly one observation per unique case.
        bootstrap = np.median(rng.choice(errors, size=(500, len(errors)), replace=True), axis=1)
        result.append({"method": key[0], "budget": key[1], "noise": key[2], "observed_fraction": key[3],
                       "cases": len(group), "median_relative_parameter_error": float(np.median(errors)),
                       "median_error_ci95": np.quantile(bootstrap, [.025, .975]).tolist(),
                       "success_rate": float(np.mean([x["success"] for x in group])),
                       "mean_calls": float(np.mean([x["forward_calls"] for x in group])),
                       "median_online_seconds": float(np.median([x["online_seconds"] for x in group]))})
    return result


def plot_results(rows, output):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(output).resolve() / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = sorted({r["method"] for r in rows})
    budgets = sorted({r["budget"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for method in methods:
        values = [np.median([r["mean_relative_parameter_error"] for r in rows
                             if r["method"] == method and r["budget"] == b]) for b in budgets]
        axes[0].plot(budgets, values, "o-", label=method)
    axes[0].set(xlabel="Allowed forward evaluations", ylabel="Median relative parameter error", yscale="log")
    axes[0].legend(fontsize=7)
    maximum_budget = max(budgets)
    positions = np.arange(len(methods))
    axes[1].bar(positions, [np.mean([r["success"] for r in rows
                                   if r["method"] == m and r["budget"] == maximum_budget]) for m in methods])
    axes[1].set_xticks(positions, methods, rotation=35, ha="right", fontsize=8)
    axes[1].set(ylabel="Success rate", ylim=(0, 1), title=f"Budget = {maximum_budget}")
    fig.savefig(Path(output) / "benchmark.png", dpi=180)
    fig.savefig(Path(output) / "benchmark.pdf")
    plt.close(fig)


def aggregate(root, output):
    """Seed-aware aggregate; never treat repeats from one truth as independent."""
    paths = sorted(Path(root).glob("seed_*/evaluation_*/results.csv"))
    if not paths:
        raise ValueError("No seed_*/evaluation_*/results.csv files found")
    groups = defaultdict(list)
    for path in paths:
        with path.open() as handle:
            for row in csv.DictReader(handle):
                key = (row["backend"], row["method"], int(row["budget"]), float(row["noise"]), float(row["observed_fraction"]), row.get("split", "test"))
                groups[key].append(row)
    rng, report = np.random.default_rng(739), []
    for key, group in groups.items():
        seeds = sorted({r["seed"] for r in group})
        cases = sorted({r["case_id"] for r in group})
        matrix = np.full((len(seeds), len(cases)), np.nan)
        for i, seed in enumerate(seeds):
            lookup = {r["case_id"]: float(r["mean_relative_parameter_error"]) for r in group if r["seed"] == seed}
            for j, case in enumerate(cases):
                if case in lookup:
                    matrix[i, j] = lookup[case]
        if np.isnan(matrix).any():
            raise ValueError("Incomplete paired seed/case matrix; finish all evaluations first")
        draws = []
        for _ in range(1000):
            sample = matrix[rng.integers(len(seeds), size=len(seeds))][:, rng.integers(len(cases), size=len(cases))]
            draws.append(float(np.median(sample)))
        report.append({"backend": key[0], "method": key[1], "budget": key[2], "noise": key[3],
                       "observed_fraction": key[4], "split": key[5], "seeds": len(seeds), "parameter_cases": len(cases),
                       "median_relative_parameter_error": float(np.median(matrix)),
                       "crossed_seed_case_bootstrap_ci95": np.quantile(draws, [.025, .975]).tolist()})
    save_json(output, {"source_files": [str(p) for p in paths], "groups": report})
    return report
