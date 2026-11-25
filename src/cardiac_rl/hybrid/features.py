from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Measurements:
    features: np.ndarray
    slopes: np.ndarray
    information: np.ndarray
    noise: float


def compress_measurements(model, observed, mask, noise_floor):
    """Affine sufficient statistics from observations, never synthetic truth.

    For each load/axis: estimate slope in u_normalised = slope*x_normalised.
    Gaussian noise is estimated from spatial residual degrees of freedom.
    Missing components contribute neither zeros nor synthetic measurements.
    """
    shape = (len(model.loads), len(model.sensors), 2)
    values = np.asarray(observed, dtype=float).reshape(shape)
    valid = np.asarray(mask, dtype=bool).reshape(shape)
    if not np.isfinite(values[valid]).all():
        raise ValueError("Observed components must be finite")
    values = np.where(valid, values, 0.)
    design = model.sensors[None] / model.scale_mm
    information = np.sum(valid * design**2, axis=1)
    numerator = np.sum(design * values, axis=1)
    slopes = np.divide(numerator, information, out=np.zeros_like(numerator), where=information > 0)
    residual = np.where(valid, values-slopes[:, None, :]*design, 0.)
    dof = int(valid.sum()-np.count_nonzero(information))
    sigma = np.sqrt(np.sum(residual**2)/max(dof, 1))
    noise = float(max(sigma, noise_floor))
    full_information = np.sum(design**2, axis=1)
    features = np.concatenate([slopes.ravel(), (information/full_information).ravel(),
                               (valid.sum(axis=1)/len(model.sensors)).ravel(),
                               [np.log10(noise)/5.]]).astype(np.float32)
    return Measurements(features, slopes.ravel(), information.ravel(), noise)


def prediction_innovations(model, prediction, measurements):
    displacement = np.asarray(prediction).reshape(len(model.loads), len(model.sensors), 2)
    # The simulator is affine; recover its slope from the first nonzero sensor.
    slopes = displacement[:, 0, :] / (model.sensors[0]/model.scale_mm)
    whitened = ((slopes.ravel()-measurements.slopes) * np.sqrt(measurements.information)
                / measurements.noise)
    return np.arcsinh(whitened).astype(np.float32)
