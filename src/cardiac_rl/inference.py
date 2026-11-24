from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from .config import fingerprint, save_json
from .data import Task, fixed_tasks, load_cases, prepare
from .environment import CalibrationEnv


def write_example(config, data_dir, output):
    mechanics = prepare(config, data_dir)
    task = next(fixed_tasks(config, data_dir, "test", limit=1))
    values = task.observed * mechanics.scale_mm
    save_json(output, {
        "description": "Synthetic held-out observation example; no ground-truth parameters included",
        "mechanics_fingerprint": fingerprint(config["mechanics"]),
        "units": "mm", "shape": [len(mechanics.loads), len(mechanics.sensors), 2],
        "loads_kpa": mechanics.loads.tolist(), "sensor_positions_mm": mechanics.sensors.tolist(),
        "displacements_mm": values.reshape(len(mechanics.loads), len(mechanics.sensors), 2).tolist(),
        "mask": task.mask.reshape(len(mechanics.loads), len(mechanics.sensors), 2).tolist(),
    })


def calibrate(config, data_dir, model_dir, observations, output, budget=None, backend="grid"):
    model_dir = Path(model_dir)
    run = json.loads((model_dir / "run.json").read_text())
    if run["config_fingerprint"] != fingerprint(config):
        raise ValueError("Model and requested configuration do not match")
    mechanics = prepare(config, data_dir)
    payload = json.loads(Path(observations).read_text())
    if payload.get("units") != "mm" or payload.get("mechanics_fingerprint") != fingerprint(config["mechanics"]):
        raise ValueError("Observation units/mechanics must match the trained model")
    if not np.allclose(payload["loads_kpa"], mechanics.loads) or not np.allclose(payload["sensor_positions_mm"], mechanics.sensors):
        raise ValueError("Loads and sensor positions must match the trained measurement protocol")
    expected = (len(mechanics.loads), len(mechanics.sensors), 2)
    displacement = np.asarray(payload["displacements_mm"], dtype=float)
    mask = np.asarray(payload["mask"])
    if displacement.shape != expected or mask.shape != expected:
        raise ValueError(f"Displacement and mask shapes must be {expected}")
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError("Mask entries must be boolean or 0/1")
    mask = mask.astype(bool).ravel()
    displacement = displacement.ravel()/mechanics.scale_mm
    if mask.sum() < 2 or not np.all(np.isfinite(displacement[mask])):
        raise ValueError("Need at least two finite observed displacement components")
    # Placeholder truth/clean arrays are never read by the environment or policy.
    task = Task(np.zeros(2), np.zeros(mechanics.output_size), np.where(mask, displacement, 0.),
                mask, 0., float(mask.mean()), 0)
    horizon = config["environment"]["horizon"]
    budget = budget or horizon+1
    if not 2 <= budget <= horizon+1:
        raise ValueError(f"Budget must be in [2, {horizon+1}]")
    policy = SAC.load(model_dir / "best.zip", device="cpu")
    env = CalibrationEnv(config, mechanics, load_cases(config, data_dir, "train"), backend)
    observation, info = env.reset(options={"task": task, "horizon": budget-1})
    trace = [{"calls": 1, "observed_mse": info["best_loss"]}]
    for _ in range(budget-1):
        action, _ = policy.predict(observation, deterministic=True)
        observation, _, done, _, info = env.step(action)
        trace.append({"calls": info["forward_calls"], "observed_mse": info["best_loss"]})
        if done:
            break
    theta = mechanics.decode(info["best_z"])
    prediction = mechanics.reference(info["best_z"])*mechanics.scale_mm
    result = {"a_kpa": float(theta[0]), "af_kpa": float(theta[1]), "budget": budget,
              "forward_calls": info["forward_calls"], "backend": backend,
              "observed_mse_normalised": info["best_loss"],
              "predicted_displacements_mm": prediction.reshape(expected).tolist(), "trace": trace,
              "reference_audit_calls": 1, "uncertainty_intervals": None,
              "interpretation": "Point estimates for the fixed passive slab model; no clinical validation"}
    save_json(output, result)
    return result
