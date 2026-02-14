"""
Bayesian Trajectory Inference Workflows

Provides Bayesian posterior inference for diffusion populations using the
Spot-On-like bias-aware likelihood, with uncertainty summaries and diagnostics.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

import numpy as np

from .spot_on import SpotOnConfig, SpotOnLikeInference
from .trajectory_utils import extract_displacements, normalize_tracks

try:
    import bayes_traj as _bayes_traj  # type: ignore

    BAYES_TRAJ_AVAILABLE = True
except Exception:
    _bayes_traj = None
    BAYES_TRAJ_AVAILABLE = False


_EPS = 1e-12


def _softmax_with_last_zero(logits: np.ndarray) -> np.ndarray:
    if logits.size == 0:
        return np.array([1.0], dtype=float)
    aug = np.concatenate([logits, np.array([0.0])])
    aug -= np.max(aug)
    e = np.exp(aug)
    return e / np.sum(e)


def _effective_sample_size(samples_1d: np.ndarray, max_lag: int = 200) -> float:
    """Simple autocorrelation-based ESS estimate."""
    x = np.asarray(samples_1d, dtype=float)
    n = len(x)
    if n < 3:
        return float(n)
    x = x - np.mean(x)
    var = np.var(x)
    if var <= 0:
        return float(n)

    max_lag = min(max_lag, n - 1)
    rho_sum = 0.0
    for lag in range(1, max_lag + 1):
        v1 = x[:-lag]
        v2 = x[lag:]
        denom = np.sqrt(np.sum(v1**2) * np.sum(v2**2))
        if denom <= 0:
            break
        rho = float(np.sum(v1 * v2) / denom)
        if not np.isfinite(rho) or rho <= 0:
            break
        rho_sum += 2.0 * rho
    return float(n / max(1.0 + rho_sum, _EPS))


class BayesianDiffusionInference:
    """
    Bayesian inference for diffusion populations with posterior uncertainty.

    Uses Metropolis-Hastings sampling over the Spot-On-like likelihood.
    """

    def __init__(
        self,
        config: SpotOnConfig,
        prior_logD_mean: float = np.log(0.3),
        prior_logD_sd: float = 2.0,
        prior_logit_sd: float = 2.0,
    ):
        self.config = config
        self.prior_logD_mean = float(prior_logD_mean)
        self.prior_logD_sd = float(prior_logD_sd)
        self.prior_logit_sd = float(prior_logit_sd)
        self._spot = SpotOnLikeInference(config)
        self._rng = np.random.default_rng(config.random_state)

    @staticmethod
    def external_backend_available() -> bool:
        """Return True if bayes_traj is importable."""
        return BAYES_TRAJ_AVAILABLE

    def _log_prior(self, theta: np.ndarray) -> float:
        n = self.config.n_states
        idx = 0
        log_d = theta[idx : idx + n]
        idx += n
        lp = -0.5 * np.sum(((log_d - self.prior_logD_mean) / self.prior_logD_sd) ** 2)
        lp += -n * np.log(self.prior_logD_sd + _EPS)

        if n > 1:
            logits = theta[idx : idx + n - 1]
            lp += -0.5 * np.sum((logits / self.prior_logit_sd) ** 2)
            lp += -(n - 1) * np.log(self.prior_logit_sd + _EPS)
            idx += n - 1

        if self.config.fit_localization_error:
            # Weak log-normal prior centered at configured value.
            mu = np.log(max(self.config.localization_error_um, 1e-4))
            lp += -0.5 * ((theta[idx] - mu) / 0.8) ** 2 - np.log(0.8 + _EPS)
            idx += 1

        if self.config.fit_axial_range:
            mu = np.log(max(self.config.axial_detection_range_um, 0.05))
            lp += -0.5 * ((theta[idx] - mu) / 0.8) ** 2 - np.log(0.8 + _EPS)
            idx += 1

        return float(lp)

    def _theta_from_mle(self, mle_result: Dict[str, Any]) -> np.ndarray:
        d = np.asarray(mle_result["diffusion_coefficients_um2_s"], dtype=float)
        f = np.asarray(mle_result["state_fractions"], dtype=float)
        logits = []
        if len(f) > 1:
            base = max(f[-1], _EPS)
            logits = np.log(np.clip(f[:-1], _EPS, 1.0)) - np.log(base)

        theta = list(np.log(np.maximum(d, _EPS)))
        theta.extend(logits)
        if self.config.fit_localization_error:
            theta.append(np.log(max(mle_result["localization_error_um"], 1e-5)))
        if self.config.fit_axial_range:
            theta.append(np.log(max(mle_result["axial_detection_range_um"], 0.02)))
        return np.asarray(theta, dtype=float)

    def sample_posterior(
        self,
        trajectories: Any,
        n_samples: int = 2000,
        burn_in: int = 500,
        thin: int = 2,
        proposal_scale: float = 0.08,
        map_max_iter: int = 250,
    ) -> Dict[str, Any]:
        """
        Sample posterior with random-walk Metropolis-Hastings.

        Returns posterior samples and summary intervals for D and state fractions.
        """
        tracks = normalize_tracks(trajectories)
        if len(tracks) == 0:
            raise ValueError("No valid trajectories with length >= 2 were provided.")

        dataset = extract_displacements(
            tracks,
            max_lag_frames=self.config.max_lag_frames,
            allow_gap_frames=self.config.allow_gap_frames,
        )
        if dataset.n_observations == 0:
            raise ValueError("No displacement observations extracted from trajectories.")

        mle = self._spot.fit_from_displacements(
            displacements_um=dataset.displacements_um,
            dt_frames=dataset.dt_frames,
            max_iter=map_max_iter,
        )
        current = self._theta_from_mle(mle)
        dim = current.size

        def log_posterior(theta: np.ndarray) -> float:
            nll = self._spot._negative_log_likelihood(theta, dataset)  # noqa: SLF001
            if not np.isfinite(nll):
                return -np.inf
            return -nll + self._log_prior(theta)

        current_lp = log_posterior(current)
        if not np.isfinite(current_lp):
            raise RuntimeError("Initial posterior is not finite; check priors/metadata.")

        total_steps = int(burn_in + n_samples * max(thin, 1))
        accepted = 0
        chain: List[np.ndarray] = []

        scale_vec = np.full(dim, float(proposal_scale), dtype=float)
        # Slightly smaller steps for logits improve acceptance.
        if self.config.n_states > 1:
            logit_start = self.config.n_states
            logit_end = logit_start + self.config.n_states - 1
            scale_vec[logit_start:logit_end] *= 0.8

        for step in range(total_steps):
            proposal = current + self._rng.normal(0.0, scale_vec, size=dim)
            prop_lp = log_posterior(proposal)

            if np.isfinite(prop_lp):
                accept_ratio = np.exp(np.clip(prop_lp - current_lp, -100, 100))
                if self._rng.uniform() < min(1.0, accept_ratio):
                    current = proposal
                    current_lp = prop_lp
                    accepted += 1

            if step >= burn_in and ((step - burn_in) % max(thin, 1) == 0):
                chain.append(current.copy())

        if not chain:
            raise RuntimeError("No posterior samples collected. Increase n_samples.")
        samples = np.vstack(chain)

        decoded = [self._spot._unpack_theta(s) for s in samples]  # noqa: SLF001
        d_samples = np.vstack([d["D_um2_s"] for d in decoded])
        f_samples = np.vstack([d["fractions"] for d in decoded])
        sigma_samples = np.asarray([d["localization_error_um"] for d in decoded], dtype=float)
        axial_samples = np.asarray([d["axial_detection_range_um"] for d in decoded], dtype=float)

        # Build diagnostics.
        ess_per_dim = np.array([_effective_sample_size(samples[:, i]) for i in range(samples.shape[1])], dtype=float)
        acceptance_rate = accepted / max(total_steps, 1)

        result = {
            "method": "bayesian_spot_on_like_mh",
            "backend": "bayes_traj" if BAYES_TRAJ_AVAILABLE else "internal_mh",
            "config": asdict(self.config),
            "posterior_samples": {
                "theta": samples.tolist(),
                "diffusion_coefficients_um2_s": d_samples.tolist(),
                "state_fractions": f_samples.tolist(),
                "localization_error_um": sigma_samples.tolist(),
                "axial_detection_range_um": axial_samples.tolist(),
            },
            "posterior_summary": {
                "diffusion_median_um2_s": np.median(d_samples, axis=0).tolist(),
                "diffusion_ci95_um2_s": np.percentile(d_samples, [2.5, 97.5], axis=0).T.tolist(),
                "fraction_median": np.median(f_samples, axis=0).tolist(),
                "fraction_ci95": np.percentile(f_samples, [2.5, 97.5], axis=0).T.tolist(),
                "localization_error_median_um": float(np.median(sigma_samples)),
                "localization_error_ci95_um": np.percentile(sigma_samples, [2.5, 97.5]).tolist(),
                "axial_detection_range_median_um": float(np.median(axial_samples)),
                "axial_detection_range_ci95_um": np.percentile(axial_samples, [2.5, 97.5]).tolist(),
            },
            "diagnostics": {
                "acceptance_rate": float(acceptance_rate),
                "effective_sample_size_min": float(np.min(ess_per_dim)),
                "effective_sample_size_median": float(np.median(ess_per_dim)),
                "effective_sample_size_per_dim": ess_per_dim.tolist(),
                "n_posterior_samples": int(samples.shape[0]),
            },
            "map_estimate": mle,
        }
        return result
