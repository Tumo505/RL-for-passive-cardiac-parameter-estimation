from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..config import load_config
from ..data import load_cases, prepare


def source_hash():
    directory = Path(__file__).parent
    sources = sorted(directory.glob("*.py")) + [directory.parent/name for name in ("mechanics.py", "data.py")]
    return hashlib.sha256(b"".join(path.read_bytes() for path in sources)).hexdigest()


def configuration(path):
    config = load_config(path)
    settings = config.get("hybrid", {})
    if settings.get("version") != 2:
        raise ValueError("Use a version-two hybrid configuration")
    if settings["reward_mode"] not in ("parameter", "objective"):
        raise ValueError("reward_mode must be parameter or objective")
    for name in ("noise_floor", "prior_std_floor", "finite_difference_step", "initializer_epochs",
                 "initializer_samples", "early_stopping_patience", "validation_cases_per_condition"):
        if settings[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if settings["finite_difference_step"] >= 1 or settings["regularization_weight"] < 0:
        raise ValueError("Invalid finite-difference step or regularization weight")
    if settings["call_penalty"] < 0 or settings.get("polish_noise_max", .02) <= 0 or settings.get("discrepancy_factor", 1.) <= 0:
        raise ValueError("Call penalty must be nonnegative; noise/polishing thresholds must be positive")
    if not -1 < settings["stop_threshold"] < 1:
        raise ValueError("stop_threshold must be in (-1, 1)")
    if min(settings["training_budgets"]+config["evaluation"]["budgets"]) < 4:
        raise ValueError("Hybrid budgets need an initial prediction and one three-call correction")
    return config


def prepare_hybrid(config, directory):
    for previous in config["hybrid"].get("exclude_datasets", []):
        if Path(previous).resolve() == Path(directory).resolve():
            raise ValueError("Use a separate hybrid data directory; legacy datasets must remain unchanged")
    model = prepare(config, directory)
    new = set()
    for split in ("train", "validation", "test"):
        z, _ = load_cases(config, directory, split)
        new.update(map(tuple, z))
    for previous in config["hybrid"].get("exclude_datasets", []):
        path = Path(previous) / "cases.npz"
        if not path.exists() or path.resolve() == (Path(directory)/"cases.npz").resolve():
            continue
        with np.load(path, allow_pickle=False) as old:
            for split in ("train", "validation", "test"):
                if new.intersection(map(tuple, old[split+"_z"])):
                    raise ValueError(f"Fresh dataset overlaps an earlier experiment: {path}")
    return model
