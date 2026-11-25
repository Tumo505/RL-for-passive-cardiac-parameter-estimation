from __future__ import annotations

import time

import numpy as np
from scipy.optimize import least_squares, lsq_linear

from ..baselines import BudgetExhausted


def mechanics_inverse(model, measurements):
    """Model-specific direct inverse: stress is linear in (a, af) at known F.

    With noisy inferred stretches this is a plug-in weighted stress-space fit,
    not the displacement maximum-likelihood solution. No equilibrium prediction
    is needed until the common one-call output audit. Loads missing either
    stretch component are omitted; rank-deficient data fall back to midpoint.
    """
    stretches = measurements.slopes.reshape(len(model.loads), 2)+1
    information = measurements.information.reshape(len(model.loads), 2)
    matrix, target = [], []
    for stretch, load, info in zip(stretches, model.loads, information):
        if np.min(info) <= 0 or np.min(stretch) <= 0 or np.max(stretch) > 1.8:
            continue
        columns = np.column_stack([model.stress(stretch, [1., 0.]), model.stress(stretch, [0., 1.])])
        weight = np.sqrt(np.min(info))
        matrix.extend(columns*weight)
        target.extend(load*weight)
    if len(matrix) < 2 or np.linalg.matrix_rank(matrix) < 2:
        return np.array([.5, .5])
    result = lsq_linear(np.asarray(matrix), np.asarray(target),
                        bounds=(model.bounds[:, 0], model.bounds[:, 1]), tol=1e-12, max_iter=100)
    if not result.success or not np.isfinite(result.x).all():
        return np.array([.5, .5])
    return np.clip((np.log(result.x)-model.log_bounds[:, 0])/np.diff(model.log_bounds, axis=1).ravel(), 0, 1)


class CalibrationOracle:
    """One actual complete forward prediction = one call, including FD probes.

    All methods may reuse already evaluated points. The MAP prior is the same
    learned initialiser and validation-calibrated diagonal scale for every MAP
    method. MLE comparators use regularization=0 and have no learned prior.
    """
    def __init__(self, model, task, budget, noise, prior, prior_std,
                 regularization=1., backend="grid"):
        self.model, self.task, self.budget, self.backend = model, task, int(budget), backend
        self.noise = float(noise)
        self.prior, self.prior_std = np.asarray(prior), np.asarray(prior_std)
        self.regularization = float(regularization)
        if self.budget < 1 or self.noise <= 0 or np.any(self.prior_std <= 0):
            raise ValueError("Invalid oracle budget/noise/prior scale")
        self.calls, self.cache, self.trace = 0, {}, []
        self.best_cost, self.best_z = float("inf"), None
        self.start = time.perf_counter()

    @property
    def remaining(self):
        return self.budget-self.calls

    def residual(self, z, prediction):
        measured = (prediction[self.task.mask]-self.task.observed[self.task.mask])/self.noise
        if self.regularization:
            prior = np.sqrt(self.regularization)*(z-self.prior)/self.prior_std
            return np.concatenate([measured, prior])
        return measured

    def evaluate(self, z):
        z = np.clip(np.asarray(z, dtype=float), 0., 1.)
        key = tuple(z)
        if key in self.cache:
            return self.cache[key]
        if self.remaining < 1:
            raise BudgetExhausted
        prediction = self.model.forward(z, self.backend)
        residual = self.residual(z, prediction)
        cost = float(.5*(residual @ residual))
        self.calls += 1
        self.cache[key] = (prediction, residual, cost)
        if cost < self.best_cost:
            self.best_cost, self.best_z = cost, z.copy()
        self.trace.append({"calls": self.calls, "cost": self.best_cost,
                           "z": self.best_z.tolist(), "seconds": time.perf_counter()-self.start})
        return self.cache[key]


class LMStepper:
    def __init__(self, oracle, initial, finite_difference_step=1e-3):
        self.oracle, self.z = oracle, np.asarray(initial, dtype=float).copy()
        self.fd = finite_difference_step
        self.prediction, self.residual, self.cost = oracle.evaluate(self.z)
        self.last_gradient, self.last_step = np.zeros(2), 0.
        self.last_accepted, self.converged = False, False

    def step(self, damping, radius):
        """Two finite-difference probes plus one trial, within a hard budget."""
        if self.oracle.remaining < 3:
            raise BudgetExhausted
        jacobian = []
        old_cost = self.cost
        for axis in range(2):
            probe = self.z.copy()
            h = self.fd if probe[axis]+self.fd <= 1 else -self.fd
            probe[axis] += h
            _, residual, _ = self.oracle.evaluate(probe)
            jacobian.append((residual-self.residual)/h)
        jacobian = np.column_stack(jacobian)
        gradient = jacobian.T @ self.residual
        hessian = jacobian.T @ jacobian
        system = hessian + max(float(damping), 1e-12)*np.diag(np.maximum(np.diag(hessian), 1.))
        delta = -np.linalg.solve(system, gradient)
        delta *= min(1., float(radius)/max(np.linalg.norm(delta), 1e-15))
        trial = np.clip(self.z+delta, 0, 1)
        prediction, residual, cost = self.oracle.evaluate(trial)
        self.last_gradient = gradient
        self.last_step = float(np.linalg.norm(trial-self.z))
        self.last_accepted = bool(cost < self.cost)
        if self.last_accepted:
            self.z, self.prediction, self.residual, self.cost = trial, prediction, residual, cost
        return {"gradient_norm": float(np.linalg.norm(gradient, ord=np.inf)),
                "step_norm": self.last_step, "relative_improvement": (old_cost-self.cost)/max(1., old_cost),
                "accepted": self.last_accepted}


def converged_lm(oracle, initial, settings):
    stepper = LMStepper(oracle, initial, settings["finite_difference_step"])
    damping = .01
    while oracle.remaining >= 3:
        calls = oracle.calls
        metrics = stepper.step(damping, 1.)
        if (oracle.calls == calls or metrics["gradient_norm"] <= settings["gradient_tolerance"] or
                metrics["step_norm"] <= settings["step_tolerance"] or
                (metrics["accepted"] and metrics["relative_improvement"] <= settings["cost_tolerance"])):
            break
        damping = max(damping/3, 1e-8) if metrics["accepted"] else min(damping*10, 1e8)
    return oracle.best_z.copy()


def converged_lsq(oracle, initial, settings):
    """Single convergence-aware solve; no artificial budget-filling restarts."""
    oracle.evaluate(initial)
    def residual(z):
        return oracle.evaluate(z)[1]
    try:
        least_squares(residual, initial, bounds=([0., 0.], [1., 1.]), jac="2-point",
                      xtol=settings["step_tolerance"], ftol=settings["cost_tolerance"],
                      gtol=settings["gradient_tolerance"])
    except BudgetExhausted:
        pass
    return oracle.best_z.copy()
