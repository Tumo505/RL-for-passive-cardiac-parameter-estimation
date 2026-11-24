from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


def load_config(path):
    with open(path) as handle:
        config = yaml.safe_load(handle)
    bounds = np.asarray(config["mechanics"]["parameter_bounds_kpa"])
    if bounds.shape != (2, 2) or np.any(bounds <= 0) or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("Need two strictly positive, ordered material parameter bounds")
    mechanics = config["mechanics"]
    loads = np.asarray(mechanics["loads_kpa"])
    if loads.ndim != 2 or loads.shape[1] != 2 or np.any(loads <= 0):
        raise ValueError("Biaxial loads must be positive pairs")
    if mechanics["grid_size"] < 3 or min(mechanics["b"], mechanics["bf"], mechanics["length_mm"]) <= 0:
        raise ValueError("Invalid mechanics configuration")
    for fraction in config["data"]["observed_fractions"]:
        if not 0 < fraction <= 1:
            raise ValueError("Observation fractions must be in (0, 1]")
    if any(x < 0 for x in config["data"]["noise_levels"]):
        raise ValueError("Noise levels cannot be negative")
    if min(config["data"][k] for k in ("train_cases", "validation_cases", "test_cases")) < 1:
        raise ValueError("Each split needs at least one case")
    if min(config["evaluation"]["budgets"]) < 4:
        raise ValueError("Evaluation budgets must be at least four forward calls")
    env, train = config["environment"], config["training"]
    if env["horizon"] < 1 or env["action_scale"] <= 0 or env["step_cost"] < 0:
        raise ValueError("Invalid environment configuration")
    if not 0 < env["reward_discount"] <= 1:
        raise ValueError("Invalid discount")
    if min(train[k] for k in ("timesteps", "n_envs", "eval_frequency", "checkpoint_frequency", "threads")) < 1:
        raise ValueError("Training sizes and frequencies must be positive")
    return config


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
