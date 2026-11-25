from __future__ import annotations

import hashlib
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

from ..config import fingerprint, save_json
from ..data import fixed_tasks, load_cases
from ..training import restore_checkpoint, save_checkpoint
from .environment import HybridEnv
from .initializer import train_initializer
from .setup import prepare_hybrid, source_hash


def validation_score(policy, config, model, cases, initializer, tasks):
    env = HybridEnv(config, model, cases, initializer, training=False)
    errors, calls = [], []
    budget = min(8, env.max_budget)
    for task in tasks:
        observation, info = env.reset(options={"task": task, "budget": budget})
        while True:
            action, _ = policy.predict(observation, deterministic=True)
            observation, _, done, _, info = env.step(action)
            if done:
                break
        estimate, truth = model.decode(info["best_z"]), model.decode(task.z)
        errors.append(float(np.mean(np.abs(np.log(estimate/truth)))))
        calls.append(info["forward_calls"])
    return {"mean_log_parameter_error": float(np.mean(errors)), "mean_forward_calls": float(np.mean(calls)),
            "validation_budget": budget, "validation_tasks": len(tasks)}


class HybridCallback(BaseCallback):
    def __init__(self, config, model, cases, initializer, tasks, directory, control, deadline):
        super().__init__()
        self.config, self.mechanics, self.cases = config, model, cases
        self.initializer, self.tasks, self.directory, self.control = initializer, tasks, directory, control
        self.deadline = deadline
        self.best, self.anchor, self.stale = float("inf"), float("inf"), 0
        if (directory / "best.json").exists():
            self.best = json.loads((directory / "best.json").read_text())["mean_log_parameter_error"]
        if (directory / "early_stopping.json").exists():
            state = json.loads((directory / "early_stopping.json").read_text())
            self.anchor, self.stale = state["anchor"], state["stale_checks"]
        self.validation_calls = 0
        self.last_eval = self.last_checkpoint = 0

    def validate(self, count_patience=True):
        metrics = validation_score(self.model, self.config, self.mechanics, self.cases,
                                   self.initializer, self.tasks)
        score = metrics["mean_log_parameter_error"]
        metrics["timesteps"] = self.model.num_timesteps
        self.validation_calls += int(round(metrics["mean_forward_calls"]*len(self.tasks)))
        with (self.directory / "validation.jsonl").open("a") as handle:
            handle.write(json.dumps(metrics)+"\n")
        print("hybrid validation: " + json.dumps(metrics), flush=True)
        if score < self.best:
            self.best = score
            self.model.save(self.directory / "best.pending.zip")
            (self.directory / "best.pending.zip").replace(self.directory / "best.zip")
            save_json(self.directory / "best.json", metrics)
        if score < self.anchor-self.config["hybrid"]["minimum_validation_improvement"]:
            self.anchor, self.stale = score, 0
        elif count_patience:
            self.stale += 1
        save_json(self.directory / "early_stopping.json", {"anchor": self.anchor, "stale_checks": self.stale})

    def _on_training_start(self):
        self.last_eval = self.last_checkpoint = self.model.num_timesteps
        self.validate(count_patience=False)

    def _on_step(self):
        if self.control["stop"]:
            return False
        if self.deadline is not None and time.monotonic() >= self.deadline:
            self.control["reason"] = "time_budget_completed"
            return False
        if self.model.num_timesteps-self.last_eval >= self.config["training"]["eval_frequency"]:
            self.validate()
            self.last_eval = self.model.num_timesteps
            if (self.model.num_timesteps >= self.config["hybrid"]["minimum_timesteps"] and
                    self.stale >= self.config["hybrid"]["early_stopping_patience"]):
                self.control["reason"] = "validation_early_stopping"
                return False
        if self.model.num_timesteps-self.last_checkpoint >= self.config["training"]["checkpoint_frequency"]:
            save_checkpoint(self.model, self.directory, "checkpoint")
            self.last_checkpoint = self.model.num_timesteps
        return True


def train_hybrid(config, data_dir, output, seed=0, device="cpu", steps=None, resume=False,
                 max_seconds=None, threads=None):
    started, monotonic_start = time.perf_counter(), time.monotonic()
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device is unavailable")
    if (steps is not None and steps < 1) or (max_seconds is not None and max_seconds <= 0):
        raise ValueError("Training steps and seconds must be positive")
    if threads is not None and threads < 1:
        raise ValueError("Thread count must be positive")
    torch.set_num_threads(threads or config["training"]["threads"])
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    previous = None
    if (directory / "run.json").exists():
        previous = json.loads((directory / "run.json").read_text())
        if not resume:
            raise ValueError("Hybrid run exists; choose a new directory or --resume")
        if previous["config_fingerprint"] != fingerprint(config) or previous["seed"] != seed:
            raise ValueError("Hybrid resume requires matching configuration and seed")
    elif resume:
        raise ValueError("Cannot resume a missing hybrid run")
    mechanics = prepare_hybrid(config, data_dir)
    initializer = train_initializer(config, data_dir, directory, mechanics, seed, device)
    cases = load_cases(config, data_dir, "train")
    val_cases = load_cases(config, data_dir, "validation")
    tasks = list(fixed_tasks(config, data_dir, "validation", config["hybrid"]["validation_cases_per_condition"]))
    vector = DummyVecEnv([lambda: Monitor(HybridEnv(config, mechanics, cases, initializer))
                          for _ in range(config["training"]["n_envs"])])
    vector.seed(seed)
    settings = config["training"]
    if resume:
        agent = restore_checkpoint(directory, vector, device)
    else:
        agent = SAC("MlpPolicy", vector, seed=seed, device=device, verbose=0,
                    learning_rate=settings["learning_rate"], learning_starts=settings["learning_starts"],
                    batch_size=settings["batch_size"], buffer_size=settings["buffer_size"],
                    train_freq=settings["train_freq"], gradient_steps=settings["gradient_steps"],
                    gamma=config["environment"]["reward_discount"],
                    ent_coef=config["hybrid"]["entropy_coefficient"], target_entropy=config["hybrid"]["target_entropy"],
                    policy_kwargs={"net_arch": settings["hidden_sizes"]})
    control = {"stop": False, "reason": "completed"}
    handlers = {}
    def request_stop(signum, frame):
        control.update(stop=True, reason="interrupted")
        print(f"Signal {signum}: hybrid training will save and stop", flush=True)
    for name in ("SIGINT", "SIGTERM", "SIGUSR1"):
        if hasattr(signal, name):
            sig = getattr(signal, name)
            handlers[sig] = signal.signal(sig, request_stop)
    implementation_hash = source_hash()
    initial_steps = agent.num_timesteps
    metadata = {"experiment": "hybrid_v2", "seed": seed, "status": "running",
                "config_fingerprint": fingerprint(config), "source_hash": implementation_hash,
                "source_changed_on_resume": bool(previous and previous.get("source_hash") != implementation_hash),
                "python": platform.python_version(), "torch": torch.__version__, "device": str(agent.device),
                "cpu_threads": torch.get_num_threads(), "recovered_start_timesteps": initial_steps,
                "reward_mode": config["hybrid"]["reward_mode"], "test_split_used_in_training": False,
                "initializer_sha256": hashlib.sha256((directory / "initializer.pt").read_bytes()).hexdigest(),
                "prior_costs_may_be_incomplete": bool((previous or {}).get("status") == "running" or
                                                     (previous or {}).get("prior_costs_may_be_incomplete", False))}
    save_json(directory / "run.json", metadata)
    save_json(directory / "config.json", config)
    callback = HybridCallback(config, mechanics, val_cases, initializer, tasks, directory, control,
                              monotonic_start+max_seconds if max_seconds is not None else None)
    callback.init_callback(agent)
    target = int(steps or settings["timesteps"])
    rl_started = time.perf_counter()
    failed = False
    try:
        if target > agent.num_timesteps:
            agent.learn(target-agent.num_timesteps, callback=callback, reset_num_timesteps=not resume)
        callback.validate(count_patience=False)
    except BaseException:
        failed = True
        raise
    finally:
        save_checkpoint(agent, directory, "last")
        metadata.update(status="failed" if failed else control["reason"], timesteps=agent.num_timesteps,
                        new_timesteps=agent.num_timesteps-initial_steps,
                        total_training_seconds=time.perf_counter()-rl_started+(previous or {}).get("total_training_seconds", 0),
                        current_process_seconds=time.perf_counter()-started,
                        training_forward_calls=sum(env.unwrapped.forward_calls for env in vector.envs)+(previous or {}).get("training_forward_calls", 0),
                        validation_forward_calls=callback.validation_calls+(previous or {}).get("validation_forward_calls", 0),
                        transition_target=target)
        save_json(directory / "run.json", metadata)
        if not failed:
            save_json(directory / "frozen.json", {"config_fingerprint": fingerprint(config), "seed": seed,
                "source_hash": implementation_hash,
                "initializer_sha256": metadata["initializer_sha256"],
                "policy_sha256": hashlib.sha256((directory / "best.zip").read_bytes()).hexdigest(),
                "selected_timesteps": json.loads((directory / "best.json").read_text())["timesteps"],
                "note": "Artifacts fixed after validation selection; final test must be explicitly requested"})
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        vector.close()
    return metadata
