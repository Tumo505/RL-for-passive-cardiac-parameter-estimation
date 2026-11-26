from __future__ import annotations

import time
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..config import fingerprint, save_json
from ..data import fixed_tasks, load_cases, make_task
from .features import compress_measurements


def initializer_fingerprint(config):
    settings = config["hybrid"]
    keys = ("noise_floor", "prior_std_floor", "initializer_samples", "initializer_epochs",
            "initializer_patience", "initializer_batch_size", "initializer_hidden_sizes")
    directory = Path(__file__).parent
    implementation = b"".join(path.read_bytes() for path in
        [directory/"initializer.py", directory/"features.py", directory.parent/"mechanics.py", directory.parent/"data.py"])
    return fingerprint({"seed": config["seed"], "mechanics": config["mechanics"], "data": config["data"],
                        "settings": {key: settings[key] for key in keys},
                        "implementation": hashlib.sha256(implementation).hexdigest()})


def compatible(payload, config, seed):
    signature = payload.get("initializer_fingerprint")
    return payload["seed"] == seed and (
        signature == initializer_fingerprint(config) if signature is not None
        else payload["config_fingerprint"] == fingerprint(config))


def reuse_initializer(source, directory, config, seed):
    """Share identical supervised weights across penalty variants; audit cost once."""
    source, directory = Path(source), Path(directory)
    payload = torch.load(source / "initializer.pt", map_location="cpu", weights_only=True)
    if not compatible(payload, config, seed):
        raise ValueError("Shared initializer has different data, feature encoder, hyperparameters or seed")
    target = directory / "initializer.pt"
    if target.exists():
        if hashlib.sha256(target.read_bytes()).digest() != hashlib.sha256((source/"initializer.pt").read_bytes()).digest():
            raise ValueError("Run already contains a different initializer; it will not be overwritten")
    else:
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "initializer.pt", target.with_suffix(".pending.pt"))
        target.with_suffix(".pending.pt").replace(target)
    info = json.loads((source / "initializer.json").read_text())
    info.update(reused=True, shared_from=str(source), shared_training_seconds=info.get("shared_training_seconds", info["training_seconds"]),
                training_seconds=0., initializer_fingerprint=initializer_fingerprint(config))
    save_json(directory / "initializer.json", info)


class Initializer(nn.Module):
    def __init__(self, size, hidden_sizes):
        super().__init__()
        layers, previous = [], size
        for width in hidden_sizes:
            layers += [nn.Linear(previous, width), nn.ReLU()]
            previous = width
        self.network = nn.Sequential(*layers, nn.Linear(previous, 2), nn.Sigmoid())

    def forward(self, features):
        return self.network(features)


class Predictor:
    def __init__(self, payload, device="cpu"):
        self.device = device
        self.model = Initializer(payload["inputs"], payload["hidden_sizes"]).to(device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.mean = np.asarray(payload["mean"])
        self.scale = np.asarray(payload["scale"])
        self.prior_std = np.asarray(payload["prior_std"])

    def predict(self, features):
        normalised = (np.asarray(features)-self.mean)/self.scale
        with torch.no_grad():
            estimate = self.model(torch.tensor(normalised, dtype=torch.float32, device=self.device)[None])
        return estimate.cpu().numpy()[0].astype(float)


def load_initializer(path, config, seed, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not compatible(payload, config, seed):
        raise ValueError("Initializer/configuration/seed mismatch")
    return Predictor(payload, device)


def train_initializer(config, data_dir, directory, model, seed, device="cpu"):
    directory = Path(directory)
    path = directory / "initializer.pt"
    if path.exists():
        return load_initializer(path, config, seed)
    start = time.perf_counter()
    settings = config["hybrid"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed+7919)
    z, clean = load_cases(config, data_dir, "train")
    features, targets = [], []
    for _ in range(settings["initializer_samples"]):
        i = rng.integers(len(z))
        task = make_task(z[i], clean[i], rng.choice(config["data"]["noise_levels"]),
                         rng.choice(config["data"]["observed_fractions"]), int(rng.integers(2**32)))
        features.append(compress_measurements(model, task.observed, task.mask, settings["noise_floor"]).features)
        targets.append(z[i])
    features = np.asarray(features)
    mean, scale = features.mean(axis=0), np.maximum(features.std(axis=0), .001)
    x = torch.tensor((features-mean)/scale, dtype=torch.float32, device=device)
    y = torch.tensor(np.asarray(targets), dtype=torch.float32, device=device)
    # Every validation condition participates equally; no test split is read.
    tasks = list(fixed_tasks(config, data_dir, "validation"))
    vx = np.array([compress_measurements(model, t.observed, t.mask, settings["noise_floor"]).features for t in tasks])
    vx = torch.tensor((vx-mean)/scale, dtype=torch.float32, device=device)
    vy = torch.tensor(np.asarray([task.z for task in tasks]), dtype=torch.float32, device=device)
    network = Initializer(x.shape[1], settings["initializer_hidden_sizes"]).to(device)
    optimiser = torch.optim.Adam(network.parameters(), lr=1e-3)
    best, patience, best_state = float("inf"), 0, None
    history = []
    for epoch in range(settings["initializer_epochs"]):
        network.train()
        for batch in np.array_split(rng.permutation(len(x)), max(1, int(np.ceil(len(x)/settings["initializer_batch_size"])))):
            indices = torch.as_tensor(batch, device=device)
            loss = nn.functional.mse_loss(network(x[indices]), y[indices])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        network.eval()
        with torch.no_grad():
            score = float(nn.functional.mse_loss(network(vx), vy).item())
        history.append(score)
        if score < best-1e-7:
            best, patience = score, 0
            best_state = {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}
        else:
            patience += 1
        if patience >= settings["initializer_patience"]:
            break
    network.load_state_dict(best_state)
    with torch.no_grad():
        errors = (network(vx)-vy).cpu().numpy()
    prior_std = np.maximum(np.sqrt(np.mean(errors**2, axis=0)), settings["prior_std_floor"])
    payload = {"state_dict": best_state, "inputs": int(x.shape[1]), "hidden_sizes": settings["initializer_hidden_sizes"],
               "mean": mean.tolist(), "scale": scale.tolist(), "prior_std": prior_std.tolist(),
               "seed": seed, "config_fingerprint": fingerprint(config),
               "initializer_fingerprint": initializer_fingerprint(config)}
    torch.save(payload, path.with_suffix(".pending.pt"))
    path.with_suffix(".pending.pt").replace(path)
    save_json(directory / "initializer.json", {"training_seconds": time.perf_counter()-start,
        "initializer_fingerprint": initializer_fingerprint(config), "reused": False,
        "device": device, "training_examples": len(x), "validation_examples": len(tasks),
        "unique_training_material_cases": len(z), "additional_forward_calls": 0,
        "prior_std_log_unit_coordinates": prior_std.tolist(), "validation_mse": history,
        "prior_calibration": "Diagonal validation RMSE; shared unchanged by every MAP method",
        "normalisation": "Mean and standard deviation fit to training features only"})
    return Predictor(payload)
