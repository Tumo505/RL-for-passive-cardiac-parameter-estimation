"""Homogeneous incompressible biaxial slab; kPa, mm, nominal traction.

Reduced Holzapfel-Ogden-inspired energy with one tension-only fibre family:
W = a/(2b) [exp(b(I1-3))-1] + af/(2bf) [exp(bf<I4-1>_+^2)-1].
This is not the full orthotropic law or a ventricular finite-element model.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares

from .config import fingerprint, save_json


class SlabMechanics:
    def __init__(self, config):
        self.config = config
        self.bounds = np.asarray(config["parameter_bounds_kpa"], dtype=float)
        self.log_bounds = np.log(self.bounds)
        self.loads = np.asarray(config["loads_kpa"], dtype=float)
        self.length = float(config["length_mm"])
        self.scale_mm = 0.25 * self.length
        self.sensors = self.length * np.array([(x, y) for x in (.2, .5, .8) for y in (.2, .5, .8)])
        self.output_size = len(self.loads) * len(self.sensors) * 2
        self.interpolator = None

    def decode(self, z):
        z = np.asarray(z, dtype=float)
        if z.shape[-1] != 2 or not np.all(np.isfinite(z)) or np.any(z < -1e-8) or np.any(z > 1+1e-8):
            raise ValueError("Normalised parameter coordinates must be finite and in [0, 1]")
        return np.exp(self.log_bounds[:, 0] + z * np.diff(self.log_bounds, axis=1).ravel())

    def stress(self, stretches, theta):
        l1, l2 = np.asarray(stretches)
        a, af = theta
        l3 = 1.0 / (l1 * l2)
        q = l1*l1 + l2*l2 + l3*l3 - 3
        fibre = max(l1*l1 - 1, 0.)
        iso = a * np.exp(self.config["b"] * q)
        fibre_stress = 2 * af * fibre * np.exp(self.config["bf"] * fibre*fibre) * l1
        return np.array([iso * (l1-l3*l3/l1) + fibre_stress,
                         iso * (l2-l3*l3/l2)])

    def solve_stretches(self, z):
        theta = self.decode(z)
        solutions = []
        for load in self.loads:
            result = least_squares(lambda l: self.stress(l, theta) - load,
                                   np.array([1.025, 1.06]), bounds=([.6, .6], [1.8, 1.8]),
                                   xtol=1e-11, ftol=1e-11, gtol=1e-11, max_nfev=100)
            if not result.success or np.linalg.norm(result.fun) > 1e-7:
                raise RuntimeError(f"Equilibrium failed: residual={result.fun}, z={z}")
            solutions.append(result.x)
        return np.asarray(solutions)

    def reference(self, z):
        stretches = self.solve_stretches(z)
        displacement = (stretches[:, None, :] - 1) * self.sensors[None, :, :]
        return displacement.ravel() / self.scale_mm

    def forward(self, z, backend="grid"):
        self.decode(z)  # reject extrapolation
        if backend == "reference":
            return self.reference(z)
        if backend != "grid" or self.interpolator is None:
            raise ValueError("Grid backend must be prepared before use")
        return np.asarray(self.interpolator(np.asarray(z).reshape(1, 2))).ravel()

    def load_grid(self, path):
        with np.load(path, allow_pickle=False) as data:
            if str(data["fingerprint"]) != fingerprint(self.config):
                raise ValueError("Grid does not match mechanics configuration; use a new data directory")
            axis, values = data["axis"], data["values"]
        self.interpolator = RegularGridInterpolator((axis, axis), values, bounds_error=True)

    def build_grid(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        axis = np.linspace(0, 1, self.config["grid_size"])
        values = np.empty((len(axis), len(axis), self.output_size))
        for i, first in enumerate(axis):
            for j, second in enumerate(axis):
                values[i, j] = self.reference([first, second])
        temporary = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(temporary, axis=axis, values=values, fingerprint=fingerprint(self.config))
        temporary.replace(path)
        self.load_grid(path)
        rng = np.random.default_rng(71923)
        errors = []
        for z in rng.uniform(size=(64, 2)):
            exact = self.reference(z)
            errors.append(float(np.linalg.norm(self.forward(z)-exact)/np.linalg.norm(exact)))
        report = {"reference_parameter_evaluations": len(axis)**2 + 64,
                  "equilibrium_solves": (len(axis)**2 + 64) * len(self.loads),
                  "build_seconds": time.perf_counter()-start,
                  "validation_cases": 64, "mean_relative_displacement_error": float(np.mean(errors)),
                  "max_relative_displacement_error": float(np.max(errors)),
                  "mechanics_fingerprint": fingerprint(self.config)}
        save_json(path.with_suffix(".json"), report)
        return report
