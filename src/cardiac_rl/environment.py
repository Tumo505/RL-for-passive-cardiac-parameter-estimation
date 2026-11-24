from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .data import make_task


def discrepancy(prediction, task):
    return float(np.mean((prediction[task.mask] - task.observed[task.mask])**2))


class CalibrationEnv(gym.Env):
    """Observation never contains true parameters or unobserved target values.

    Hidden truth is used only by evaluation. The fixed episode horizon is part of
    the task (remaining budget is observed), so exhaustion is terminal, not an
    external time-limit truncation that SAC should bootstrap beyond.
    """
    metadata = {"render_modes": []}

    def __init__(self, config, model, cases, backend="grid"):
        super().__init__()
        self.config, self.model, self.cases, self.backend = config, model, cases, backend
        self.max_horizon = config["environment"]["horizon"]
        self.horizon = self.max_horizon
        self.action_space = spaces.Box(-1., 1., (2,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (4*model.output_size+9,), dtype=np.float32)
        self.done = True
        self.forward_calls = 0

    def _observation(self):
        mask = self.task.mask.astype(float)
        residual = (self.prediction-self.task.observed) * mask
        return np.concatenate([
            self.task.observed, mask, self.prediction, residual,
            2*self.z-1, 2*self.best_z-1, self.previous_action,
            [(self.horizon-self.steps) / self.max_horizon, np.log10(self.loss+1e-10)/10,
             np.log10(self.best_loss+1e-10)/10],
        ]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.horizon = int(options.get("horizon", self.np_random.integers(min(8, self.max_horizon), self.max_horizon+1)))
        if not 1 <= self.horizon <= self.max_horizon:
            raise ValueError("Requested horizon is outside the trained budget range")
        self.task = options.get("task")
        if self.task is None:
            z, clean = self.cases
            i = self.np_random.integers(len(z))
            self.task = make_task(z[i], clean[i], self.np_random.choice(self.config["data"]["noise_levels"]),
                                  self.np_random.choice(self.config["data"]["observed_fractions"]),
                                  int(self.np_random.integers(2**32)), i)
        self.z = np.asarray(options.get("initial_z", [.5, .5]), dtype=float).copy()
        self.prediction = self.model.forward(self.z, self.backend)
        self.forward_calls += 1
        self.loss = discrepancy(self.prediction, self.task)
        self.best_z, self.best_loss = self.z.copy(), self.loss
        self.steps, self.done = 0, False
        self.previous_action = np.zeros(2)
        return self._observation(), self._info()

    def _info(self):
        return {"loss": self.loss, "best_loss": self.best_loss, "best_z": self.best_z.copy(),
                "forward_calls": self.steps+1}

    def step(self, action):
        if self.done:
            raise RuntimeError("Reset the environment before another step")
        action = np.asarray(action, dtype=float)
        if action.shape != (2,) or not np.all(np.isfinite(action)):
            raise ValueError("Action must be a finite pair")
        action = np.clip(action, -1, 1)
        previous_potential = np.log(self.loss+1e-10)
        self.z = np.clip(self.z + self.config["environment"]["action_scale"]*action, 0, 1)
        self.previous_action = action.copy()
        self.prediction = self.model.forward(self.z, self.backend)
        self.forward_calls += 1
        self.loss = discrepancy(self.prediction, self.task)
        self.steps += 1
        if self.loss < self.best_loss:
            self.best_loss, self.best_z = self.loss, self.z.copy()
        tolerance = self.config["environment"]["success_tolerance"]
        self.done = bool(self.steps >= self.horizon or self.loss <= tolerance)
        # Discount-consistent potential shaping; terminal potential is zero.
        potential = 0. if self.done else -np.log(self.loss+1e-10)
        reward = self.config["environment"]["reward_discount"]*potential + previous_potential
        reward -= self.config["environment"]["step_cost"]
        # Terminal task objective keeps the shaped reward tied to final calibration.
        if self.done:
            reward += -np.log(self.best_loss+1e-10)
        return self._observation(), float(reward), self.done, False, self._info()
