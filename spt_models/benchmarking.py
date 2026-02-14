"""
Synthetic Benchmarks and Metrics for SPT Inference

Implements benchmark utilities aligned with the SPT roadmap:
- Diffusion/fraction recovery bias and variance
- Transition and dwell-time recovery errors
- Posterior interval calibration helpers
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class TwoStateSimulationConfig:
    """Synthetic two-state trajectory simulator configuration."""

    n_tracks: int = 120
    steps_per_track: int = 80
    dt_s: float = 0.02
    diffusion_um2_s: Tuple[float, float] = (0.02, 0.9)
    transition_matrix: Tuple[Tuple[float, float], Tuple[float, float]] = (
        (0.94, 0.06),
        (0.08, 0.92),
    )
    localization_error_um: float = 0.025
    random_state: Optional[int] = None


def simulate_two_state_tracks(config: TwoStateSimulationConfig) -> Dict[str, Any]:
    """Simulate Markov-switching two-state trajectories with localization noise."""
    rng = np.random.default_rng(config.random_state)
    d = np.asarray(config.diffusion_um2_s, dtype=float)
    A = np.asarray(config.transition_matrix, dtype=float)
    A = A / A.sum(axis=1, keepdims=True)
    K = 2

    tracks: List[np.ndarray] = []
    states_all: List[np.ndarray] = []

    for _ in range(config.n_tracks):
        xy = np.zeros((config.steps_per_track, 2), dtype=float)
        states = np.zeros(config.steps_per_track - 1, dtype=int)
        s = int(rng.integers(0, K))
        for t in range(1, config.steps_per_track):
            s = int(rng.choice(np.arange(K), p=A[s]))
            states[t - 1] = s
            sigma_step = np.sqrt(max(2.0 * d[s] * config.dt_s, 1e-12))
            xy[t] = xy[t - 1] + rng.normal(0.0, sigma_step, size=2)
        xy += rng.normal(0.0, config.localization_error_um, size=xy.shape)
        frame = np.arange(config.steps_per_track, dtype=float)
        tracks.append(np.column_stack([frame, xy]))
        states_all.append(states)

    return {
        "tracks": tracks,
        "true_diffusion_um2_s": d.tolist(),
        "true_transition_matrix": A.tolist(),
        "true_states": states_all,
    }


def diffusion_recovery_metrics(true_d: Sequence[float], est_d: Sequence[float]) -> Dict[str, Any]:
    """Compute bias/relative-error metrics for diffusion estimates."""
    t = np.sort(np.asarray(true_d, dtype=float))
    e = np.sort(np.asarray(est_d, dtype=float))
    if t.shape != e.shape:
        raise ValueError("true_d and est_d must have the same length.")
    bias = e - t
    rel = bias / np.maximum(t, 1e-12)
    return {
        "bias_um2_s": bias.tolist(),
        "mean_abs_bias_um2_s": float(np.mean(np.abs(bias))),
        "mean_abs_relative_error": float(np.mean(np.abs(rel))),
    }


def fraction_recovery_metrics(true_f: Sequence[float], est_f: Sequence[float]) -> Dict[str, Any]:
    """Compute fraction recovery errors (state order sorted by diffusion)."""
    t = np.asarray(true_f, dtype=float)
    e = np.asarray(est_f, dtype=float)
    t = t / np.sum(t)
    e = e / np.sum(e)
    diff = e - t
    return {
        "fraction_error": diff.tolist(),
        "mean_abs_fraction_error": float(np.mean(np.abs(diff))),
    }


def transition_recovery_metrics(
    true_transition: Sequence[Sequence[float]],
    est_transition: Sequence[Sequence[float]],
) -> Dict[str, Any]:
    """Compute transition matrix recovery errors."""
    t = np.asarray(true_transition, dtype=float)
    e = np.asarray(est_transition, dtype=float)
    if t.shape != e.shape:
        raise ValueError("Transition matrices must have the same shape.")
    abs_err = np.abs(e - t)
    return {
        "transition_abs_error": abs_err.tolist(),
        "mean_abs_transition_error": float(np.mean(abs_err)),
    }


def dwell_time_from_transition(transition_matrix: Sequence[Sequence[float]], dt_s: float) -> np.ndarray:
    """Approximate dwell times from per-frame transition probabilities."""
    A = np.asarray(transition_matrix, dtype=float)
    p_stay = np.clip(np.diag(A), 1e-12, 1.0 - 1e-9)
    dwell_frames = 1.0 / np.maximum(1.0 - p_stay, 1e-12)
    return dwell_frames * float(dt_s)


def posterior_calibration_metrics(
    true_values: Sequence[float],
    ci95: Sequence[Sequence[float]],
) -> Dict[str, Any]:
    """Evaluate whether true values lie inside reported 95% intervals."""
    t = np.asarray(true_values, dtype=float)
    ci = np.asarray(ci95, dtype=float)
    if ci.shape[0] != t.shape[0] or ci.shape[1] != 2:
        raise ValueError("ci95 must have shape (n_values, 2).")
    covered = (t >= ci[:, 0]) & (t <= ci[:, 1])
    return {
        "covered": covered.tolist(),
        "coverage_rate": float(np.mean(covered)),
    }
