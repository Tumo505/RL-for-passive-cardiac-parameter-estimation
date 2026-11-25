from __future__ import annotations

import importlib.metadata
import json
import platform
import signal
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from .config import fingerprint, save_json
from .data import fixed_tasks, load_cases, prepare
from .environment import CalibrationEnv


def save_checkpoint(model, directory, label):
    """Write complete temporary files before replacing a checkpoint pair."""
    model.save(directory / f"{label}.pending.zip")
    model.save_replay_buffer(directory / f"{label}.pending.pkl")
    (directory / f"{label}.pending.zip").replace(directory / f"{label}.zip")
    (directory / f"{label}.pending.pkl").replace(directory / f"{label}_replay.pkl")
    save_json(directory / f"{label}.state.json", {"timesteps": model.num_timesteps})


def restore_checkpoint(directory, env, device):
    errors = []
    for label in ("last", "checkpoint"):
        path = directory / f"{label}.zip"
        if not path.exists():
            continue
        try:
            model = SAC.load(path, env=env, device=device)
            state = directory / f"{label}.state.json"
            if state.exists() and json.loads(state.read_text())["timesteps"] != model.num_timesteps:
                raise ValueError("Checkpoint pair was not fully committed")
            replay = directory / f"{label}_replay.pkl"
            if replay.exists():
                model.load_replay_buffer(replay)
            print(f"Restored {path} at {model.num_timesteps} transitions", flush=True)
            return model
        except Exception as error:
            errors.append(f"{label}: {error}")
    raise ValueError("No recoverable checkpoint: " + "; ".join(errors))


def evaluate_policy(model, config, mechanics, cases, tasks):
    env = CalibrationEnv(config, mechanics, cases)
    errors, losses, calls = [], [], []
    for task in tasks:
        observation, info = env.reset(options={"task": task, "horizon": env.max_horizon})
        for _ in range(env.horizon):
            action, _ = model.predict(observation, deterministic=True)
            observation, _, done, _, info = env.step(action)
            if done:
                break
        estimate = mechanics.decode(info["best_z"])
        truth = mechanics.decode(task.z)
        errors.append(float(np.mean(np.abs(np.log(estimate/truth)))))
        losses.append(info["best_loss"])
        calls.append(info["forward_calls"])
    return {"mean_log_parameter_error": float(np.mean(errors)),
            "mean_loss": float(np.mean(losses)), "mean_forward_calls": float(np.mean(calls))}


class ValidationCallback(BaseCallback):
    def __init__(self, config, mechanics, cases, tasks, directory, stop, deadline=None):
        super().__init__()
        self.config, self.mechanics, self.cases = config, mechanics, cases
        self.tasks, self.directory, self.stop = tasks, directory, stop
        self.best = float("inf")
        if (directory / "best.json").exists():
            self.best = json.loads((directory / "best.json").read_text())["mean_log_parameter_error"]
        self.last_eval, self.last_checkpoint = 0, 0
        self.validation_seconds = 0.
        self.validation_calls = 0
        self.deadline = deadline

    def validate(self):
        start = time.perf_counter()
        metrics = evaluate_policy(self.model, self.config, self.mechanics, self.cases, self.tasks)
        metrics["timesteps"] = self.model.num_timesteps
        self.validation_seconds += time.perf_counter()-start
        self.validation_calls += int(round(metrics["mean_forward_calls"]*len(self.tasks)))
        with (self.directory / "validation.jsonl").open("a") as handle:
            handle.write(json.dumps(metrics)+"\n")
        print(f"validation: {json.dumps(metrics)}", flush=True)
        if metrics["mean_log_parameter_error"] < self.best:
            self.best = metrics["mean_log_parameter_error"]
            self.model.save(self.directory / "best.zip")
            save_json(self.directory / "best.json", metrics)

    def _on_training_start(self):
        self.last_eval = self.last_checkpoint = self.model.num_timesteps
        self.validate()

    def _on_step(self):
        if self.stop["requested"]:
            return False
        if self.deadline is not None and time.monotonic() >= self.deadline:
            self.stop["reason"] = "time_budget"
            return False
        if self.num_timesteps-self.last_eval >= self.config["training"]["eval_frequency"]:
            self.validate()
            self.last_eval = self.num_timesteps
        if self.num_timesteps-self.last_checkpoint >= self.config["training"]["checkpoint_frequency"]:
            save_checkpoint(self.model, self.directory, "checkpoint")
            self.last_checkpoint = self.num_timesteps
        return not self.stop["requested"]


def train(config, data_dir, output, seed=0, device="auto", steps=None, resume=False,
          max_seconds=None, threads=None):
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("--max-seconds must be positive")
    if steps is not None and steps <= 0:
        raise ValueError("--steps must be positive")
    if threads is not None and threads < 1:
        raise ValueError("--threads must be positive")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    torch.set_num_threads(threads or config["training"]["threads"])
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    signature = fingerprint(config)
    metadata_path = directory / "run.json"
    previous = None
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if not resume:
            raise ValueError("Run already exists. Choose another output directory or use --resume")
        if previous["config_fingerprint"] != signature or previous["seed"] != seed:
            raise ValueError("Resume requires the same configuration and seed")
    elif resume:
        raise ValueError("Cannot resume a directory without run.json")
    mechanics = prepare(config, data_dir)
    cases = load_cases(config, data_dir, "train")
    validation_cases = load_cases(config, data_dir, "validation")
    tasks = list(fixed_tasks(config, data_dir, "validation"))
    # Deterministic selection across the full set, with a separate validation pool.
    selection = np.linspace(0, len(tasks)-1, min(len(tasks), config["training"]["eval_cases"]), dtype=int)
    tasks = [tasks[i] for i in selection]
    environments = [lambda: Monitor(CalibrationEnv(config, mechanics, cases))
                    for _ in range(config["training"]["n_envs"])]
    vec_env = DummyVecEnv(environments)
    vec_env.seed(seed)
    stop = {"requested": False}

    def request_stop(signum, frame):
        stop["requested"] = True
        print(f"Signal {signum}: saving at next training boundary", flush=True)

    old_handlers = {}
    for name in ("SIGTERM", "SIGINT", "SIGUSR1"):
        if hasattr(signal, name):
            sig = getattr(signal, name)
            old_handlers[sig] = signal.signal(sig, request_stop)
    settings = config["training"]
    if resume:
        model = restore_checkpoint(directory, vec_env, device)
    else:
        model = SAC("MlpPolicy", vec_env, seed=seed, device=device, verbose=0,
                    learning_rate=settings["learning_rate"], buffer_size=settings["buffer_size"],
                    learning_starts=settings["learning_starts"], batch_size=settings["batch_size"],
                    gamma=config["environment"]["reward_discount"], tau=.005, ent_coef="auto_0.1",
                    train_freq=settings["train_freq"], gradient_steps=settings["gradient_steps"],
                    policy_kwargs={"net_arch": settings["hidden_sizes"]})
    save_json(directory / "config.json", config)
    metadata = {"config_fingerprint": signature, "seed": seed, "device": str(model.device),
                "requested_device": device, "python": platform.python_version(),
                "platform": platform.platform(), "status": "running", "resumed": resume,
                "cpu_threads": torch.get_num_threads(), "wall_clock_budget_seconds": max_seconds,
                "recovered_start_timesteps": model.num_timesteps,
                "prior_costs_may_be_incomplete": (previous or {}).get("status") == "running",
                "transition_target": steps if steps is not None else (None if max_seconds is not None else settings["timesteps"]),
                "packages": {name: importlib.metadata.version(name)
                             for name in ("numpy", "scipy", "torch", "gymnasium", "stable-baselines3")}}
    if torch.cuda.is_available():
        metadata["gpu"] = torch.cuda.get_device_name(model.device.index or 0) if model.device.type == "cuda" else None
    save_json(metadata_path, metadata)
    deadline = time.monotonic()+max_seconds if max_seconds is not None else None
    callback = ValidationCallback(config, mechanics, validation_cases, tasks, directory, stop, deadline)
    callback.init_callback(model)
    # In timed mode a sentinel horizon lets the callback end training by elapsed
    # time. Constant SAC learning rates do not depend on this horizon.
    target = int(steps if steps is not None else (2**53 if max_seconds is not None else settings["timesteps"]))
    remaining = target-model.num_timesteps
    start = time.perf_counter()
    initial_steps = model.num_timesteps
    failed = False
    try:
        if remaining > 0:
            model.learn(total_timesteps=remaining, callback=callback, reset_num_timesteps=not resume)
        callback.validate()
    except BaseException:
        failed = True
        raise
    finally:
        save_checkpoint(model, directory, "last")
        env_calls = sum(env.unwrapped.forward_calls for env in vec_env.envs)
        metadata.update({"timesteps": model.num_timesteps,
                         "new_timesteps": model.num_timesteps-initial_steps,
                         "training_forward_calls": env_calls+(previous or {}).get("training_forward_calls", 0),
                         "validation_forward_calls": callback.validation_calls+(previous or {}).get("validation_forward_calls", 0),
                         "total_training_seconds": time.perf_counter()-start+(previous or {}).get("total_training_seconds", 0),
                         "status": "failed" if failed else ("interrupted" if stop["requested"] else
                                   ("time_budget_completed" if stop.get("reason") == "time_budget" else "completed"))})
        save_json(metadata_path, metadata)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        vec_env.close()
    return metadata
