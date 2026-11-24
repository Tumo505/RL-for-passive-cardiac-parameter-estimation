from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from .config import fingerprint, save_json
from .mechanics import SlabMechanics


@dataclass
class Task:
    z: np.ndarray
    clean: np.ndarray
    observed: np.ndarray
    mask: np.ndarray
    noise: float
    fraction: float
    case_id: int


def make_task(z, clean, noise, fraction, seed, case_id=0):
    rng = np.random.default_rng(seed)
    # One mask entry is one measured displacement component across all loads/sensors.
    count = max(2, int(round(len(clean) * fraction)))
    mask = np.zeros(len(clean), dtype=bool)
    mask[rng.choice(len(clean), count, replace=False)] = True
    observed = np.where(mask, clean + rng.normal(0, noise, len(clean)), 0.)
    return Task(np.asarray(z).copy(), np.asarray(clean).copy(), observed, mask,
                float(noise), float(fraction), int(case_id))


def dataset_fingerprint(config):
    return fingerprint({"mechanics": config["mechanics"], "data": config["data"], "seed": config["seed"]})


def prepare(config, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model = SlabMechanics(config["mechanics"])
    if (directory / "grid.npz").exists():
        model.load_grid(directory / "grid.npz")
    else:
        model.build_grid(directory / "grid.npz")
    path = directory / "cases.npz"
    if path.exists():
        load_cases(config, directory, "train")
        return model
    # Split parameter cases before noise/masking augmentation; no noisy copies leak.
    counts = [config["data"][key] for key in ("train_cases", "validation_cases", "test_cases")]
    z = qmc.LatinHypercube(2, seed=config["seed"]).random(sum(counts))
    clean = np.asarray([model.reference(row) for row in z])
    payload = {"fingerprint": dataset_fingerprint(config)}
    start = 0
    for split, count in zip(("train", "validation", "test"), counts):
        payload[split + "_z"] = z[start:start+count]
        payload[split + "_clean"] = clean[start:start+count]
        start += count
    temporary = path.with_name("cases.tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)
    save_json(directory / "cases.json", {
        "fingerprint": dataset_fingerprint(config), "counts": dict(zip(("train", "validation", "test"), counts)),
        "reference_parameter_evaluations": sum(counts),
        "equilibrium_solves": sum(counts) * len(model.loads),
        "truth_backend": "reference", "split_unit": "material parameter pair",
        "noise_definition": "Gaussian sigma = noise_level * 0.25 * length_mm",
        "missingness": "Uniform random displacement components; mask fixed per episode",
    })
    return model


def load_cases(config, directory, split):
    with np.load(Path(directory) / "cases.npz", allow_pickle=False) as data:
        if str(data["fingerprint"]) != dataset_fingerprint(config):
            raise ValueError("Dataset/config mismatch; prepare into a different directory")
        return data[split + "_z"].copy(), data[split + "_clean"].copy()


def fixed_tasks(config, directory, split="test", limit=None):
    z, clean = load_cases(config, directory, split)
    for i in range(len(z) if limit is None else min(limit, len(z))):
        for ni, noise in enumerate(config["data"]["noise_levels"]):
            for fi, fraction in enumerate(config["data"]["observed_fractions"]):
                seed = np.random.SeedSequence([config["seed"], 819 if split == "test" else 419, i, ni, fi])
                yield make_task(z[i], clean[i], noise, fraction, seed, i)
