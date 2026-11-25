from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..data import make_task
from .core import CalibrationOracle, LMStepper
from .features import compress_measurements, prediction_innovations


class HybridEnv(gym.Env):
    """SAC controls damping, trust radius, and stopping of a bounded LM step.

    Synthetic parameter truth is a training-only reward label. It never affects
    observations, candidate selection, physical corrections or stopping rules.
    Evaluation passes training=False, so step() does not even read that label.
    """
    metadata = {"render_modes": []}

    def __init__(self, config, model, cases, initializer, training=True, backend="grid"):
        super().__init__()
        self.config, self.model, self.cases, self.initializer = config, model, cases, initializer
        self.training, self.backend = training, backend
        self.settings = config["hybrid"]
        self.max_budget = max(self.settings["training_budgets"]+config["evaluation"]["budgets"])
        # Compact task features plus innovations, states, prior scales, cost/budget
        # and previous correction diagnostics/action.
        self.observation_size = 8*len(model.loads)+17
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.observation_size,), dtype=np.float32)
        self.action_space = spaces.Box(-1., 1., (3,), dtype=np.float32)
        self.done = True
        self.forward_calls = 0

    def _observation(self):
        innovation = prediction_innovations(self.model, self.stepper.prediction, self.measurements)
        extra = np.concatenate([2*self.stepper.z-1, 2*self.oracle.best_z-1, 2*self.initial-1,
                                self.initializer.prior_std,
                                [np.log1p(self.stepper.cost)/10, np.log1p(self.oracle.best_cost)/10,
                                 self.oracle.remaining/self.max_budget, self.stepper.last_step,
                                 float(self.stepper.last_accepted)], self.previous_action])
        return np.concatenate([self.measurements.features, innovation, extra]).astype(np.float32)

    def _utility(self):
        if self.settings["reward_mode"] == "objective":
            return -np.log1p(self.oracle.best_cost)
        truth = self.model.decode(self.task.z)
        prediction = self.model.decode(self.oracle.best_z)
        return -float(np.mean(np.abs(np.log(prediction/truth))))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.task = options.get("task")
        if self.task is None:
            z, clean = self.cases
            index = self.np_random.integers(len(z))
            self.task = make_task(z[index], clean[index], self.np_random.choice(self.config["data"]["noise_levels"]),
                                  self.np_random.choice(self.config["data"]["observed_fractions"]),
                                  int(self.np_random.integers(2**32)), index)
        self.measurements = compress_measurements(self.model, self.task.observed, self.task.mask, self.settings["noise_floor"])
        self.initial = self.initializer.predict(self.measurements.features)
        budget = int(options.get("budget", self.np_random.choice(self.settings["training_budgets"])))
        if not 4 <= budget <= self.max_budget:
            raise ValueError("Hybrid budget is outside the configured training/evaluation range")
        self.oracle = CalibrationOracle(self.model, self.task, budget, self.measurements.noise,
                                        self.initial, self.initializer.prior_std,
                                        self.settings["regularization_weight"], self.backend)
        self.stepper = LMStepper(self.oracle, self.initial, self.settings["finite_difference_step"])
        self.forward_calls += 1
        self.previous_action = np.zeros(3)
        self.done = False
        return self._observation(), self._info("initialised")

    def _info(self, reason):
        return {"best_z": self.oracle.best_z.copy(), "forward_calls": self.oracle.calls,
                "objective": self.oracle.best_cost, "stop_reason": reason}

    def step(self, action):
        if self.done:
            raise RuntimeError("Reset before stepping a terminated hybrid episode")
        action = np.asarray(action, dtype=float)
        if action.shape != (3,) or not np.all(np.isfinite(action)):
            raise ValueError("Hybrid action must be three finite numbers")
        action = np.clip(action, -1, 1)
        previous = self._utility() if self.training else 0.
        calls = self.oracle.calls
        reason = "refining"
        if action[2] > self.settings["stop_threshold"]:
            self.done, reason = True, "policy_stop"
        elif self.oracle.remaining < 3:
            self.done, reason = True, "budget"
        else:
            damping = 10**(-6+4*(action[0]+1))
            radius = 10**(np.log10(.005)+(action[1]+1)/2*np.log10(1/.005))
            metrics = self.stepper.step(damping, radius)
            if self.oracle.calls == calls:
                self.done, reason = True, "cached_stagnation"
            elif metrics["gradient_norm"] <= self.settings["gradient_tolerance"]:
                self.done, reason = True, "gradient_convergence"
            elif metrics["step_norm"] <= self.settings["step_tolerance"]:
                self.done, reason = True, "step_convergence"
            elif self.oracle.remaining < 3:
                self.done, reason = True, "budget"
        used = self.oracle.calls-calls
        self.forward_calls += used
        self.previous_action = action
        reward = self._utility()-previous-self.settings["call_penalty"]*used if self.training else 0.
        return self._observation(), float(reward), self.done, False, self._info(reason)
