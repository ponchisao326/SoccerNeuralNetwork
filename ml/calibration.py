"""Post-hoc probability calibration for the Poisson-derived 1X2 simplex.

The network outputs probabilities (not logits), so classic logit temperature
scaling is reformulated as power scaling::

    p_T(c) = p(c)^(1/T) / sum_k p(k)^(1/T)

which is exactly logit temperature scaling applied to ``log p``. ``T > 1``
flattens an overconfident distribution, ``T < 1`` sharpens an underconfident
one, and ``T = 1`` is the identity. The transform is monotone per row, so the
argmax prediction (and therefore raw accuracy) is unchanged: temperature only
repairs the *confidence* of the probabilities, which is what log-loss, Brier
and any downstream expected-value consumer (e.g. betting) care about.

The single scalar ``T`` is fit on the validation split by direct 1-D grid
search on log-loss -- the function is smooth and unimodal in ``T``, so a fine
grid is exact enough and avoids an optimiser dependency.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# Fit bounds: T below 0.5 or above 3.0 would imply a pathologically mis-scaled
# model; the grid step of 0.01 changes val log-loss well below seed noise.
_TEMPERATURE_GRID: np.ndarray = np.round(np.arange(0.50, 3.001, 0.01), 2)
_EPS: float = 1e-12


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Power-scale a ``(N, C)`` probability matrix and renormalise.

    Args:
        probs: Row-stochastic probability matrix.
        temperature: Scaling parameter ``T``; 1.0 returns the input unchanged.

    Returns:
        Calibrated row-stochastic matrix of the same shape.
    """
    if temperature == 1.0:
        return probs
    scaled = np.power(np.clip(probs, _EPS, 1.0), 1.0 / temperature)
    return scaled / scaled.sum(axis=1, keepdims=True)


def _log_loss(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean negative log-likelihood of the true class."""
    true_probs = np.clip(probs[np.arange(labels.shape[0]), labels], _EPS, 1.0)
    return float(-np.log(true_probs).mean())


def fit_temperature(
    probs: np.ndarray, labels: np.ndarray, grid: Optional[np.ndarray] = None
) -> float:
    """Pick the temperature minimising log-loss on a held-out split.

    Must be fit on validation only (never test), mirroring every other
    decision-layer parameter in this pipeline.

    Args:
        probs: ``(N, C)`` validation probabilities.
        labels: ``(N,)`` true class indices.
        grid: Candidate temperatures (defaults to ``0.5 .. 3.0`` step 0.01).

    Returns:
        The log-loss-optimal temperature.
    """
    if grid is None:
        grid = _TEMPERATURE_GRID
    best_temperature, best_loss = 1.0, float("inf")
    for temperature in grid:
        loss = _log_loss(apply_temperature(probs, float(temperature)), labels)
        if loss < best_loss:
            best_loss, best_temperature = loss, float(temperature)
    return best_temperature
