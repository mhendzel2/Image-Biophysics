"""
Bias-Aware Diffusion Population Inference (Spot-On style)

This module implements a practical Spot-On-inspired inference path:
- Corrects for localization error and motion blur in displacement emissions
- Applies an axial defocalization survival correction
- Supports tracking-step truncation correction (max jump distance)
- Produces bootstrap uncertainty intervals from trajectory resampling
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from .trajectory_utils import (
    DisplacementDataset,
    extract_displacements,
    normalize_tracks,
    resample_tracks,
)


_EPS = 1e-12


@dataclass
class SpotOnConfig:
    """
    Configuration for Spot-On-like inference.

    Parameters
    ----------
    frame_interval_s : float
        Camera frame interval in seconds.
    axial_detection_range_um : float
        Effective axial detection range in microns.
    localization_error_um : float
        Per-coordinate localization uncertainty (1-sigma) in microns.
    exposure_time_s : float
        Camera exposure time in seconds.
    n_states : int
        Number of diffusion states.
    max_lag_frames : int
        Max lag frame used to build displacements.
    allow_gap_frames : int
        Allowed missed detection frames when building displacements.
    max_jump_um : float | None
        Tracking max-step constraint for truncation correction.
    fit_localization_error : bool
        Whether to infer localization error jointly.
    fit_axial_range : bool
        Whether to infer effective axial detection range jointly.
    min_diffusion_um2_s : float
        Lower positivity floor for diffusion coefficients.
    bootstrap_samples : int
        Number of bootstrap resamples for uncertainty intervals.
    bootstrap_max_iter : int
        Iteration cap per bootstrap fit.
    random_state : int | None
        Random seed.
    """

    frame_interval_s: float
    axial_detection_range_um: float
    localization_error_um: float
    exposure_time_s: float = 0.0
    n_states: int = 2
    max_lag_frames: int = 4
    allow_gap_frames: int = 0
    max_jump_um: Optional[float] = None
    fit_localization_error: bool = False
    fit_axial_range: bool = False
    min_diffusion_um2_s: float = 1e-5
    bootstrap_samples: int = 0
    bootstrap_max_iter: int = 80
    random_state: Optional[int] = None


def _softmax_with_last_zero(logits: np.ndarray) -> np.ndarray:
    """Convert K-1 logits into K fractions with last logit anchored at zero."""
    if logits.size == 0:
        return np.array([1.0], dtype=float)
    augmented = np.concatenate([logits, np.array([0.0])])
    augmented = augmented - np.max(augmented)
    e = np.exp(augmented)
    return e / np.sum(e)


def _effective_dt(frame_interval_s: float, exposure_time_s: float, dt_frames: np.ndarray) -> np.ndarray:
    """
    Motion-blur corrected effective lag time.

    Uses the common approximation:
    dt_eff = dt - t_exp / 3
    """
    dt = np.asarray(dt_frames, dtype=float) * float(frame_interval_s)
    if exposure_time_s <= 0:
        return np.maximum(dt, _EPS)
    dt_eff = dt - float(exposure_time_s) / 3.0
    return np.maximum(dt_eff, _EPS)


def _axial_survival_probability(
    diffusion_um2_s: float,
    dt_s: np.ndarray,
    axial_range_um: float,
    n_terms: int = 20,
) -> np.ndarray:
    """
    Approximate survival probability in an axial slab with absorbing boundaries.

    S(t) ≈ (8/pi^2) * sum_{m=0}^{n_terms-1} [ 1/(2m+1)^2 * exp(-((2m+1)^2 pi^2 D t)/L^2) ]
    """
    L = max(float(axial_range_um), _EPS)
    D = max(float(diffusion_um2_s), _EPS)
    t = np.asarray(dt_s, dtype=float)
    m = np.arange(n_terms, dtype=float)
    odd = 2.0 * m + 1.0
    coeff = (8.0 / (np.pi**2)) / (odd**2)
    exponent = -((odd**2)[:, None] * (np.pi**2) * D * t[None, :]) / (L**2)
    s = np.sum(coeff[:, None] * np.exp(exponent), axis=0)
    return np.clip(s, _EPS, 1.0)


def _rayleigh_logpdf_with_truncation(r: np.ndarray, sigma2: np.ndarray, max_jump_um: Optional[float]) -> np.ndarray:
    """Log Rayleigh PDF with optional upper truncation at tracking max jump."""
    sigma2 = np.maximum(sigma2, _EPS)
    r = np.asarray(r, dtype=float)
    log_pdf = np.log(np.maximum(r, _EPS)) - np.log(sigma2) - (r**2) / (2.0 * sigma2)

    if max_jump_um is None:
        return log_pdf

    rmax = float(max_jump_um)
    if rmax <= 0:
        raise ValueError("max_jump_um must be > 0 if provided.")
    trunc_cdf = 1.0 - np.exp(-(rmax**2) / (2.0 * sigma2))
    trunc_cdf = np.clip(trunc_cdf, _EPS, 1.0)
    return log_pdf - np.log(trunc_cdf)


class SpotOnLikeInference:
    """Spot-On-inspired diffusion population inference for SPT trajectories."""

    def __init__(self, config: SpotOnConfig):
        if config.n_states < 1:
            raise ValueError("n_states must be >= 1.")
        if config.frame_interval_s <= 0:
            raise ValueError("frame_interval_s must be > 0.")
        self.config = config
        self._rng = np.random.default_rng(config.random_state)

    def _initial_guess(self, dataset: DisplacementDataset) -> np.ndarray:
        """Build an unconstrained initial parameter vector."""
        r2_mean = float(np.mean(dataset.displacements_um**2))
        sigma2 = max(self.config.localization_error_um**2, 1e-8)
        dt_eff_mean = float(
            np.mean(
                _effective_dt(
                    self.config.frame_interval_s,
                    self.config.exposure_time_s,
                    dataset.dt_frames,
                )
            )
        )
        d_base = max((r2_mean - 4.0 * sigma2) / (4.0 * dt_eff_mean), self.config.min_diffusion_um2_s)
        spread = np.geomspace(0.3, 3.0, num=self.config.n_states)
        d_init = np.clip(d_base * spread, self.config.min_diffusion_um2_s, None)

        theta: List[float] = list(np.log(d_init))
        if self.config.n_states > 1:
            theta.extend([0.0] * (self.config.n_states - 1))
        if self.config.fit_localization_error:
            theta.append(np.log(max(self.config.localization_error_um, 1e-4)))
        if self.config.fit_axial_range:
            theta.append(np.log(max(self.config.axial_detection_range_um, 0.05)))
        return np.asarray(theta, dtype=float)

    def _unpack_theta(self, theta: np.ndarray) -> Dict[str, Any]:
        """Decode unconstrained parameter vector."""
        n = self.config.n_states
        idx = 0
        d_vals = np.exp(theta[idx : idx + n])
        idx += n

        if n > 1:
            fracs = _softmax_with_last_zero(theta[idx : idx + n - 1])
            idx += n - 1
        else:
            fracs = np.array([1.0], dtype=float)

        sigma_loc = self.config.localization_error_um
        if self.config.fit_localization_error:
            sigma_loc = float(np.exp(theta[idx]))
            idx += 1

        axial = self.config.axial_detection_range_um
        if self.config.fit_axial_range:
            axial = float(np.exp(theta[idx]))
            idx += 1

        return {
            "D_um2_s": np.clip(d_vals, self.config.min_diffusion_um2_s, None),
            "fractions": fracs / np.sum(fracs),
            "localization_error_um": max(sigma_loc, 1e-5),
            "axial_detection_range_um": max(axial, 0.02),
        }

    def _negative_log_likelihood(self, theta: np.ndarray, dataset: DisplacementDataset) -> float:
        params = self._unpack_theta(theta)
        d_vals = params["D_um2_s"]
        fracs = params["fractions"]
        sigma = params["localization_error_um"]
        axial = params["axial_detection_range_um"]

        r = dataset.displacements_um
        dt_frames = dataset.dt_frames
        dt_s = np.asarray(dt_frames, dtype=float) * self.config.frame_interval_s
        dt_eff_s = _effective_dt(self.config.frame_interval_s, self.config.exposure_time_s, dt_frames)

        comp_log = []
        for k, d in enumerate(d_vals):
            # Per-coordinate displacement variance including localization noise.
            sigma2 = 2.0 * d * dt_eff_s + 2.0 * sigma**2
            log_pdf = _rayleigh_logpdf_with_truncation(r, sigma2, self.config.max_jump_um)
            survival = _axial_survival_probability(d, dt_s, axial)
            log_weight = np.log(np.clip(fracs[k], _EPS, 1.0)) + np.log(np.clip(survival, _EPS, 1.0))
            comp_log.append(log_weight + log_pdf)

        log_components = np.vstack(comp_log)
        # Normalize by total in-focus weight at each observation.
        log_norm_weights = logsumexp(
            np.vstack(
                [
                    np.log(np.clip(fracs[k], _EPS, 1.0))
                    + np.log(
                        np.clip(
                            _axial_survival_probability(d_vals[k], dt_s, axial),
                            _EPS,
                            1.0,
                        )
                    )
                    for k in range(len(d_vals))
                ]
            ),
            axis=0,
        )
        log_mix = logsumexp(log_components, axis=0) - log_norm_weights

        if not np.all(np.isfinite(log_mix)):
            return np.inf
        return float(-np.sum(log_mix))

    def fit_from_displacements(
        self,
        displacements_um: np.ndarray,
        dt_frames: np.ndarray,
        max_iter: int = 300,
    ) -> Dict[str, Any]:
        """Fit directly from displacement observations."""
        dataset = DisplacementDataset(
            displacements_um=np.asarray(displacements_um, dtype=float),
            dt_frames=np.asarray(dt_frames, dtype=int),
            sequence_ids=np.zeros(len(displacements_um), dtype=int),
            sequence_start=np.zeros(len(displacements_um), dtype=int),
        )
        if dataset.n_observations == 0:
            raise ValueError("No displacement observations provided.")

        x0 = self._initial_guess(dataset)
        result = minimize(
            fun=lambda th: self._negative_log_likelihood(th, dataset),
            x0=x0,
            method="L-BFGS-B",
            options={"maxiter": int(max_iter)},
        )

        best = self._unpack_theta(result.x)
        # Order states by diffusion coefficient for stable interpretation.
        order = np.argsort(best["D_um2_s"])
        best["D_um2_s"] = best["D_um2_s"][order]
        best["fractions"] = best["fractions"][order]

        log_like = -self._negative_log_likelihood(result.x, dataset)
        n_params = len(result.x)
        n = dataset.n_observations
        aic = 2 * n_params - 2 * log_like
        bic = n_params * np.log(max(n, 1)) - 2 * log_like

        return {
            "method": "spot_on_like_bias_aware_mle",
            "success": bool(result.success),
            "message": str(result.message),
            "log_likelihood": float(log_like),
            "aic": float(aic),
            "bic": float(bic),
            "n_observations": int(n),
            "n_states": int(self.config.n_states),
            "diffusion_coefficients_um2_s": best["D_um2_s"].tolist(),
            "state_fractions": best["fractions"].tolist(),
            "localization_error_um": float(best["localization_error_um"]),
            "axial_detection_range_um": float(best["axial_detection_range_um"]),
            "config": asdict(self.config),
        }

    def _fit_bootstrap(
        self,
        tracks: Sequence[np.ndarray],
        max_iter: int,
    ) -> Dict[str, Any]:
        if self.config.bootstrap_samples <= 0:
            return {}
        if len(tracks) < 2:
            return {"warning": "Need >= 2 tracks for bootstrap uncertainty."}

        d_samples: List[np.ndarray] = []
        f_samples: List[np.ndarray] = []
        sigma_samples: List[float] = []
        axial_samples: List[float] = []

        for b in range(self.config.bootstrap_samples):
            sampled_tracks = resample_tracks(tracks, random_state=self._rng.integers(0, 2**31 - 1))
            ds = extract_displacements(
                sampled_tracks,
                max_lag_frames=self.config.max_lag_frames,
                allow_gap_frames=self.config.allow_gap_frames,
            )
            if ds.n_observations == 0:
                continue
            try:
                fit = self.fit_from_displacements(
                    ds.displacements_um,
                    ds.dt_frames,
                    max_iter=min(max_iter, self.config.bootstrap_max_iter),
                )
            except Exception:
                continue

            d_samples.append(np.asarray(fit["diffusion_coefficients_um2_s"], dtype=float))
            f_samples.append(np.asarray(fit["state_fractions"], dtype=float))
            sigma_samples.append(float(fit["localization_error_um"]))
            axial_samples.append(float(fit["axial_detection_range_um"]))

        if not d_samples:
            return {"warning": "Bootstrap failed to converge on all resamples."}

        d_arr = np.vstack(d_samples)
        f_arr = np.vstack(f_samples)

        return {
            "n_bootstrap_success": int(d_arr.shape[0]),
            "diffusion_ci95_um2_s": np.percentile(d_arr, [2.5, 97.5], axis=0).T.tolist(),
            "fraction_ci95": np.percentile(f_arr, [2.5, 97.5], axis=0).T.tolist(),
            "localization_error_ci95_um": np.percentile(np.asarray(sigma_samples), [2.5, 97.5]).tolist(),
            "axial_detection_range_ci95_um": np.percentile(np.asarray(axial_samples), [2.5, 97.5]).tolist(),
        }

    def fit(
        self,
        trajectories: Any,
        max_iter: int = 300,
    ) -> Dict[str, Any]:
        """
        Fit Spot-On-like diffusion populations from trajectories.

        `trajectories` can be:
        - trackpy DataFrame with frame/x/y/particle
        - ndarray of one trajectory
        - list of trajectory arrays
        """
        tracks = normalize_tracks(trajectories)
        if len(tracks) == 0:
            raise ValueError("No valid trajectories with length >= 2 were provided.")

        ds = extract_displacements(
            tracks,
            max_lag_frames=self.config.max_lag_frames,
            allow_gap_frames=self.config.allow_gap_frames,
        )
        if ds.n_observations == 0:
            raise ValueError("No displacement observations extracted from trajectories.")

        fit_result = self.fit_from_displacements(ds.displacements_um, ds.dt_frames, max_iter=max_iter)
        fit_result["bootstrap_uncertainty"] = self._fit_bootstrap(tracks, max_iter=max_iter)
        return fit_result
