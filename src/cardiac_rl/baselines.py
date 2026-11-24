from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import differential_evolution, least_squares
from scipy.special import ndtr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from torch import nn

from .config import fingerprint, save_json
from .data import load_cases, make_task
from .environment import discrepancy


class BudgetExhausted(Exception):
    pass


class Objective:
    """Count every candidate evaluation, including finite-difference probes."""
    def __init__(self, model, task, budget, backend="grid"):
        self.model, self.task, self.budget, self.backend = model, task, budget, backend
        self.calls, self.trace = 0, []
        self.best_loss, self.best_z = float("inf"), None
        self.start = time.perf_counter()

    def residual(self, z):
        if self.calls >= self.budget:
            raise BudgetExhausted
        prediction = self.model.forward(z, self.backend)
        self.calls += 1
        loss = discrepancy(prediction, self.task)
        if loss < self.best_loss:
            self.best_loss, self.best_z = loss, np.asarray(z).copy()
        self.trace.append({"calls": self.calls, "loss": self.best_loss,
                           "z": self.best_z.tolist(), "seconds": time.perf_counter()-self.start})
        return (prediction[self.task.mask]-self.task.observed[self.task.mask]) / np.sqrt(self.task.mask.sum())

    def __call__(self, z):
        residual = self.residual(z)
        return float(residual @ residual)


def optimise(method, objective, seed):
    rng = np.random.default_rng(seed)
    objective(np.array([.5, .5]))
    try:
        if method == "least_squares":
            initial = np.array([.5, .5])
            while objective.calls < objective.budget:
                least_squares(objective.residual, initial, bounds=(np.zeros(2), np.ones(2)),
                              jac="2-point", xtol=1e-10, ftol=1e-10, gtol=1e-10)
                initial = rng.uniform(.02, .98, 2)
        elif method == "differential_evolution":
            differential_evolution(objective, [(0, 1), (0, 1)], seed=seed, popsize=5,
                                   polish=False, maxiter=100000, tol=0, atol=0)
        elif method == "random":
            while objective.calls < objective.budget:
                objective(rng.uniform(size=2))
        elif method == "bayesian":
            x, y = [np.array([.5, .5])], [objective.best_loss]
            while objective.calls < objective.budget:
                if len(x) < 6:
                    candidate = rng.uniform(size=2)
                else:
                    # Fixed kernel prevents hidden, expensive hyperparameter searches.
                    kernel = ConstantKernel(1.) * Matern(length_scale=[.3, .3], nu=2.5) + WhiteKernel(1e-6)
                    gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=True)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        gp.fit(np.asarray(x), np.log10(np.asarray(y)+1e-10))
                    candidates = rng.uniform(size=(512, 2))
                    mean, std = gp.predict(candidates, return_std=True)
                    improvement = np.min(np.log10(np.asarray(y)+1e-10))-mean-.01
                    ratio = improvement / np.maximum(std, 1e-12)
                    ei = improvement*ndtr(ratio) + std*np.exp(-.5*ratio**2)/np.sqrt(2*np.pi)
                    candidate = candidates[np.argmax(ei)]
                x.append(candidate)
                y.append(objective(candidate))
        else:
            raise ValueError(f"Unknown optimiser {method}")
    except BudgetExhausted:
        pass
    return objective.best_z, objective.calls, objective.trace


class DirectEstimator(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(inputs, 128), nn.ReLU(), nn.Linear(128, 128),
                                     nn.ReLU(), nn.Linear(128, 2), nn.Sigmoid())

    def forward(self, x):
        return self.network(x)


def task_features(task):
    return np.concatenate([task.observed, task.mask.astype(float)]).astype(np.float32)


def train_direct(config, data_dir, directory, seed, device="cpu"):
    directory = Path(directory)
    path = directory / "direct.pt"
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["config_fingerprint"] != fingerprint(config) or payload["seed"] != seed:
            raise ValueError("Direct estimator does not match run")
        return path
    start = time.perf_counter()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    z, clean = load_cases(config, data_dir, "train")
    val_z, val_clean = load_cases(config, data_dir, "validation")

    def examples(parameters, fields, count):
        x, y = [], []
        for _ in range(count):
            i = rng.integers(len(parameters))
            task = make_task(parameters[i], fields[i], rng.choice(config["data"]["noise_levels"]),
                             rng.choice(config["data"]["observed_fractions"]), int(rng.integers(2**32)))
            x.append(task_features(task))
            y.append(parameters[i])
        return torch.tensor(np.asarray(x), device=device), torch.tensor(np.asarray(y), dtype=torch.float32, device=device)

    x, y = examples(z, clean, config["evaluation"]["direct_train_samples"])
    vx, vy = examples(val_z, val_clean, max(128, len(val_z)*3))
    model = DirectEstimator(x.shape[1]).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, best_state = float("inf"), None
    history = []
    generator = np.random.default_rng(seed+881)
    for epoch in range(config["evaluation"]["direct_epochs"]):
        model.train()
        order = generator.permutation(len(x))
        for start_index in range(0, len(x), 256):
            indices = torch.as_tensor(order[start_index:start_index+256], device=device)
            loss = nn.functional.mse_loss(model(x[indices]), y[indices])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(nn.functional.mse_loss(model(vx), vy))
        history.append(validation_loss)
        if validation_loss < best:
            best = validation_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save({"state_dict": best_state, "inputs": x.shape[1], "seed": seed,
                "config_fingerprint": fingerprint(config)}, path)
    save_json(directory / "direct.json", {"training_seconds": time.perf_counter()-start,
              "device": device,
              "training_examples": len(x), "validation_examples": len(vx),
              "unique_training_parameter_cases": len(z), "additional_forward_calls": 0,
              "validation_mse": history, "selected_validation_mse": best})
    return path


def load_direct(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = DirectEstimator(payload["inputs"])
    model.load_state_dict(payload["state_dict"])
    return model.eval()
