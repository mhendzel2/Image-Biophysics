"""
Switching Diffusion HMM with Localization Error and Motion Blur

This module provides a true hidden Markov model (HMM) for SPT state inference.
It supports:
- Baum-Welch (EM) parameter learning for diffusion states and transitions
- Emissions based on displacement distributions with localization noise
- Motion-blur correction via effective lag time
- Viterbi decoding and posterior state probabilities
- Bootstrap uncertainty intervals from trajectory-level resampling
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import logsumexp

from .trajectory_utils import extract_step_sequences, normalize_tracks, resample_tracks


_EPS = 1e-12


@dataclass
class SwitchingHMMConfig:
    """Configuration for switching diffusion HMM inference."""

    frame_interval_s: float
    localization_error_um: float
    exposure_time_s: float = 0.0
    n_states: int = 2
    allow_gap_frames: int = 0
    max_jump_um: Optional[float] = None
    min_diffusion_um2_s: float = 1e-5
    max_iter: int = 100
    tol: float = 1e-5
    transition_prior: float = 1.1
    fit_localization_error: bool = False
    bootstrap_samples: int = 0
    bootstrap_max_iter: int = 50
    random_state: Optional[int] = None


def _effective_dt(frame_interval_s: float, exposure_time_s: float, dt_frames: np.ndarray) -> np.ndarray:
    """Motion-blur corrected lag time."""
    dt = np.asarray(dt_frames, dtype=float) * float(frame_interval_s)
    if exposure_time_s <= 0:
        return np.maximum(dt, _EPS)
    return np.maximum(dt - float(exposure_time_s) / 3.0, _EPS)


def _rayleigh_logpdf_truncated(r: np.ndarray, sigma2: np.ndarray, max_jump_um: Optional[float]) -> np.ndarray:
    """Rayleigh log-PDF with optional right truncation from tracking step limit."""
    sigma2 = np.maximum(np.asarray(sigma2, dtype=float), _EPS)
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


class SwitchingDiffusionHMM:
    """True HMM for switching diffusion inference from SPT trajectories."""

    def __init__(self, config: SwitchingHMMConfig):
        if config.frame_interval_s <= 0:
            raise ValueError("frame_interval_s must be > 0.")
        if config.n_states < 1:
            raise ValueError("n_states must be >= 1.")
        self.config = config
        self._rng = np.random.default_rng(config.random_state)

        self.diffusion_um2_s: Optional[np.ndarray] = None
        self.transition_matrix_frame: Optional[np.ndarray] = None
        self.initial_probabilities: Optional[np.ndarray] = None
        self.localization_error_um: float = float(config.localization_error_um)
        self._fitted = False

    def _initialize_parameters(self, sequences: Sequence[Dict[str, np.ndarray]]) -> None:
        """Initialize model parameters from observed displacements."""
        r_all = np.concatenate([seq["r_um"] for seq in sequences])
        dtf_all = np.concatenate([seq["dt_frames"] for seq in sequences])
        dt_eff = _effective_dt(self.config.frame_interval_s, self.config.exposure_time_s, dtf_all)

        sigma2 = max(self.localization_error_um**2, 1e-8)
        d_base = max((np.mean(r_all**2) - 4.0 * sigma2) / (4.0 * np.mean(dt_eff)), self.config.min_diffusion_um2_s)
        spread = np.geomspace(0.3, 3.0, self.config.n_states)
        self.diffusion_um2_s = np.clip(d_base * spread, self.config.min_diffusion_um2_s, None)

        if self.config.n_states == 1:
            self.transition_matrix_frame = np.array([[1.0]])
            self.initial_probabilities = np.array([1.0])
        else:
            off = 0.08 / (self.config.n_states - 1)
            A = np.full((self.config.n_states, self.config.n_states), off, dtype=float)
            np.fill_diagonal(A, 0.92)
            A /= A.sum(axis=1, keepdims=True)
            self.transition_matrix_frame = A
            self.initial_probabilities = np.full(self.config.n_states, 1.0 / self.config.n_states)

    def _state_log_emissions(self, r: np.ndarray, dt_frames: np.ndarray) -> np.ndarray:
        """Compute log emission probabilities for all states and observations."""
        if self.diffusion_um2_s is None:
            raise RuntimeError("Model parameters are not initialized.")
        dt_eff = _effective_dt(self.config.frame_interval_s, self.config.exposure_time_s, dt_frames)
        sigma = max(self.localization_error_um, 1e-6)

        K = self.config.n_states
        T = len(r)
        log_b = np.zeros((K, T), dtype=float)
        for k in range(K):
            sigma2 = 2.0 * self.diffusion_um2_s[k] * dt_eff + 2.0 * sigma**2
            log_b[k] = _rayleigh_logpdf_truncated(r, sigma2, self.config.max_jump_um)
        return log_b

    def _transition_matrices_for_sequence(self, dt_frames: np.ndarray) -> np.ndarray:
        """
        Build per-step transition matrices.

        For gap dt > 1 frame, transition matrix is approximated as A^dt.
        """
        if self.transition_matrix_frame is None:
            raise RuntimeError("Transition matrix is not initialized.")
        A = np.clip(self.transition_matrix_frame, _EPS, 1.0)
        A /= A.sum(axis=1, keepdims=True)

        T = len(dt_frames)
        if T <= 1:
            return np.empty((0, self.config.n_states, self.config.n_states), dtype=float)

        mats = np.zeros((T - 1, self.config.n_states, self.config.n_states), dtype=float)
        for t in range(1, T):
            m = max(int(dt_frames[t]), 1)
            mats[t - 1] = np.linalg.matrix_power(A, m)
            mats[t - 1] = np.clip(mats[t - 1], _EPS, 1.0)
            mats[t - 1] /= mats[t - 1].sum(axis=1, keepdims=True)
        return mats

    def _forward_backward(
        self,
        log_b: np.ndarray,
        log_pi: np.ndarray,
        log_a_t: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run forward-backward for one sequence.

        Returns
        -------
        gamma : ndarray (K,T)
            Posterior state probabilities.
        xi : ndarray (T-1,K,K)
            Posterior transition probabilities.
        log_likelihood : float
            Sequence log-likelihood.
        """
        K, T = log_b.shape
        alpha = np.full((K, T), -np.inf, dtype=float)
        beta = np.full((K, T), -np.inf, dtype=float)

        alpha[:, 0] = log_pi + log_b[:, 0]
        for t in range(1, T):
            trans = log_a_t[t - 1]
            alpha[:, t] = log_b[:, t] + logsumexp(alpha[:, t - 1][:, None] + trans, axis=0)

        log_like = float(logsumexp(alpha[:, -1]))

        beta[:, -1] = 0.0
        for t in range(T - 2, -1, -1):
            trans = log_a_t[t]
            beta[:, t] = logsumexp(trans + log_b[:, t + 1][None, :] + beta[:, t + 1][None, :], axis=1)

        log_gamma = alpha + beta - log_like
        gamma = np.exp(log_gamma)
        gamma /= np.maximum(gamma.sum(axis=0, keepdims=True), _EPS)

        if T <= 1:
            return gamma, np.empty((0, K, K), dtype=float), log_like

        xi = np.zeros((T - 1, K, K), dtype=float)
        for t in range(T - 1):
            log_xi_t = (
                alpha[:, t][:, None]
                + log_a_t[t]
                + log_b[:, t + 1][None, :]
                + beta[:, t + 1][None, :]
                - log_like
            )
            xi_t = np.exp(log_xi_t)
            xi[t] = xi_t / np.maximum(np.sum(xi_t), _EPS)
        return gamma, xi, log_like

    def _em_step(
        self,
        sequences: Sequence[Dict[str, np.ndarray]],
    ) -> Tuple[float, List[np.ndarray], List[np.ndarray]]:
        """Perform one EM iteration; returns total log-likelihood and posteriors."""
        if self.diffusion_um2_s is None or self.transition_matrix_frame is None or self.initial_probabilities is None:
            raise RuntimeError("Model parameters are not initialized.")

        K = self.config.n_states
        prior = max(self.config.transition_prior, 1.0)

        pi_acc = np.zeros(K, dtype=float)
        xi_acc = np.zeros((K, K), dtype=float)
        gamma_time_acc = np.zeros(K, dtype=float)

        gamma_list: List[np.ndarray] = []
        xi_list: List[np.ndarray] = []

        # Weighted moments for D and sigma updates.
        r2_num = np.zeros(K, dtype=float)
        dt_num = np.zeros(K, dtype=float)

        total_log_like = 0.0
        for seq in sequences:
            r = np.asarray(seq["r_um"], dtype=float)
            dtf = np.asarray(seq["dt_frames"], dtype=int)
            if r.size == 0:
                continue

            log_b = self._state_log_emissions(r, dtf)
            a_t = self._transition_matrices_for_sequence(dtf)
            log_a_t = np.log(np.clip(a_t, _EPS, 1.0))
            log_pi = np.log(np.clip(self.initial_probabilities, _EPS, 1.0))

            gamma, xi, log_like = self._forward_backward(log_b, log_pi, log_a_t)
            total_log_like += log_like

            gamma_list.append(gamma)
            xi_list.append(xi)

            pi_acc += gamma[:, 0]
            if xi.shape[0] > 0:
                # Approximate per-frame counts by dividing each interval by gap length.
                for t in range(xi.shape[0]):
                    m = max(int(dtf[t + 1]), 1)
                    xi_acc += xi[t] / m
                    gamma_time_acc += gamma[:, t] / m

            dt_eff = _effective_dt(self.config.frame_interval_s, self.config.exposure_time_s, dtf)
            for k in range(K):
                w = gamma[k]
                r2_num[k] += np.sum(w * (r**2))
                dt_num[k] += np.sum(w * dt_eff)

        n_seq = max(len(gamma_list), 1)
        self.initial_probabilities = np.clip(pi_acc / n_seq, _EPS, 1.0)
        self.initial_probabilities /= np.sum(self.initial_probabilities)

        if K > 1 and np.sum(xi_acc) > 0:
            A_new = (xi_acc + (prior - 1.0)) / np.maximum(
                gamma_time_acc[:, None] + K * (prior - 1.0),
                _EPS,
            )
            A_new = np.clip(A_new, _EPS, 1.0)
            A_new /= A_new.sum(axis=1, keepdims=True)
            self.transition_matrix_frame = A_new
        else:
            if K == 1:
                self.transition_matrix_frame = np.array([[1.0]])
            else:
                # Keep a weakly sticky default when transitions are uninformative.
                off = 0.02 / (K - 1)
                A_fallback = np.full((K, K), off, dtype=float)
                np.fill_diagonal(A_fallback, 0.98)
                A_fallback /= A_fallback.sum(axis=1, keepdims=True)
                self.transition_matrix_frame = A_fallback

        sigma2 = max(self.localization_error_um**2, 1e-10)
        for k in range(K):
            denom = max(4.0 * dt_num[k], _EPS)
            numer = r2_num[k] - 4.0 * sigma2 * np.sum([g[k].sum() for g in gamma_list])
            d_hat = numer / denom
            self.diffusion_um2_s[k] = max(d_hat, self.config.min_diffusion_um2_s)

        if self.config.fit_localization_error:
            num = 0.0
            den = 0.0
            for seq, gamma in zip(sequences, gamma_list):
                r = np.asarray(seq["r_um"], dtype=float)
                dtf = np.asarray(seq["dt_frames"], dtype=int)
                dt_eff = _effective_dt(self.config.frame_interval_s, self.config.exposure_time_s, dtf)
                for k in range(K):
                    residual = r**2 - 4.0 * self.diffusion_um2_s[k] * dt_eff
                    num += np.sum(gamma[k] * residual)
                    den += np.sum(gamma[k])
            sigma2_new = max(num / max(4.0 * den, _EPS), 1e-8)
            self.localization_error_um = float(np.sqrt(sigma2_new))

        # Sort states by diffusion coefficient to keep interpretation stable.
        order = np.argsort(self.diffusion_um2_s)
        self.diffusion_um2_s = self.diffusion_um2_s[order]
        self.initial_probabilities = self.initial_probabilities[order]
        self.transition_matrix_frame = self.transition_matrix_frame[order][:, order]

        return total_log_like, gamma_list, xi_list

    def _decode_viterbi(
        self,
        r: np.ndarray,
        dt_frames: np.ndarray,
    ) -> np.ndarray:
        """Viterbi decode for one sequence."""
        if self.diffusion_um2_s is None or self.transition_matrix_frame is None or self.initial_probabilities is None:
            raise RuntimeError("Model must be fitted before decoding.")

        log_b = self._state_log_emissions(r, dt_frames)
        A_t = self._transition_matrices_for_sequence(dt_frames)
        log_A_t = np.log(np.clip(A_t, _EPS, 1.0))
        log_pi = np.log(np.clip(self.initial_probabilities, _EPS, 1.0))

        K, T = log_b.shape
        delta = np.full((K, T), -np.inf, dtype=float)
        psi = np.zeros((K, T), dtype=int)

        delta[:, 0] = log_pi + log_b[:, 0]
        for t in range(1, T):
            trans = delta[:, t - 1][:, None] + log_A_t[t - 1]
            psi[:, t] = np.argmax(trans, axis=0)
            delta[:, t] = np.max(trans, axis=0) + log_b[:, t]

        states = np.zeros(T, dtype=int)
        states[-1] = int(np.argmax(delta[:, -1]))
        for t in range(T - 2, -1, -1):
            states[t] = psi[states[t + 1], t + 1]
        return states

    def fit(
        self,
        trajectories: Any,
        max_iter: Optional[int] = None,
        tol: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Fit switching diffusion HMM from trajectories."""
        tracks = normalize_tracks(trajectories)
        if len(tracks) == 0:
            raise ValueError("No valid trajectories with length >= 2 were provided.")
        sequences = extract_step_sequences(tracks, allow_gap_frames=self.config.allow_gap_frames)
        if len(sequences) == 0:
            raise ValueError("No valid displacement sequences extracted from trajectories.")

        self._initialize_parameters(sequences)
        n_iter = int(max_iter or self.config.max_iter)
        conv_tol = float(tol or self.config.tol)

        ll_history: List[float] = []
        posteriors: List[np.ndarray] = []
        for _ in range(n_iter):
            ll, gamma_list, _ = self._em_step(sequences)
            ll_history.append(float(ll))
            posteriors = gamma_list
            if len(ll_history) > 1:
                delta_ll = ll_history[-1] - ll_history[-2]
                if abs(delta_ll) < conv_tol:
                    break

        self._fitted = True

        viterbi_states = [
            self._decode_viterbi(seq["r_um"], seq["dt_frames"]).tolist()
            for seq in sequences
        ]
        steady_state = self._steady_state_distribution()
        dwell_times_s = self._dwell_times_seconds()
        transition_rates = self._transition_rates_per_second()

        result = {
            "method": "switching_diffusion_hmm_em",
            "success": len(ll_history) > 0,
            "n_states": int(self.config.n_states),
            "n_sequences": int(len(sequences)),
            "n_observations": int(sum(len(seq["r_um"]) for seq in sequences)),
            "log_likelihood": float(ll_history[-1]) if ll_history else float("nan"),
            "log_likelihood_history": ll_history,
            "diffusion_coefficients_um2_s": self.diffusion_um2_s.tolist() if self.diffusion_um2_s is not None else [],
            "localization_error_um": float(self.localization_error_um),
            "transition_matrix_frame": self.transition_matrix_frame.tolist()
            if self.transition_matrix_frame is not None
            else [],
            "steady_state_fractions": steady_state.tolist(),
            "dwell_times_s": dwell_times_s.tolist(),
            "transition_rates_per_s": transition_rates.tolist(),
            "viterbi_states": viterbi_states,
            "posterior_state_probabilities": [gamma.T.tolist() for gamma in posteriors],
            "config": asdict(self.config),
        }
        result["bootstrap_uncertainty"] = self._bootstrap_uncertainty(tracks)
        return result

    def _steady_state_distribution(self) -> np.ndarray:
        """Compute steady-state distribution of current transition matrix."""
        if self.transition_matrix_frame is None:
            return np.array([], dtype=float)
        A = np.asarray(self.transition_matrix_frame, dtype=float)
        eigvals, eigvecs = np.linalg.eig(A.T)
        idx = int(np.argmin(np.abs(eigvals - 1.0)))
        v = np.real(eigvecs[:, idx])
        v = np.maximum(v, 0.0)
        if np.sum(v) <= 0:
            return np.full(A.shape[0], 1.0 / A.shape[0])
        return v / np.sum(v)

    def _dwell_times_seconds(self) -> np.ndarray:
        """Approximate dwell time for each state from self-transition probabilities."""
        if self.transition_matrix_frame is None:
            return np.array([], dtype=float)
        A = np.asarray(self.transition_matrix_frame, dtype=float)
        p_stay = np.clip(np.diag(A), _EPS, 1.0 - 1e-9)
        # Mean dwell in frames of geometric dwell distribution.
        mean_frames = 1.0 / np.maximum(1.0 - p_stay, _EPS)
        return mean_frames * self.config.frame_interval_s

    def _transition_rates_per_second(self) -> np.ndarray:
        """Approximate rate matrix from per-frame transition matrix."""
        if self.transition_matrix_frame is None:
            return np.array([[]], dtype=float)
        A = np.asarray(self.transition_matrix_frame, dtype=float)
        K = A.shape[0]
        rates = np.zeros_like(A)
        dt = self.config.frame_interval_s
        for i in range(K):
            for j in range(K):
                if i == j:
                    continue
                rates[i, j] = A[i, j] / max(dt, _EPS)
            rates[i, i] = -np.sum(rates[i, np.arange(K) != i])
        return rates

    def _bootstrap_uncertainty(self, tracks: Sequence[np.ndarray]) -> Dict[str, Any]:
        """Estimate confidence intervals via trajectory-level bootstrap."""
        if self.config.bootstrap_samples <= 0:
            return {}
        if len(tracks) < 2:
            return {"warning": "Need >= 2 tracks for bootstrap uncertainty."}

        d_samples: List[np.ndarray] = []
        dwell_samples: List[np.ndarray] = []
        trans_samples: List[np.ndarray] = []
        sigma_samples: List[float] = []

        for _ in range(self.config.bootstrap_samples):
            sample = resample_tracks(tracks, random_state=self._rng.integers(0, 2**31 - 1))
            cfg = SwitchingHMMConfig(
                **{
                    **asdict(self.config),
                    "bootstrap_samples": 0,
                    "max_iter": min(self.config.bootstrap_max_iter, self.config.max_iter),
                }
            )
            model = SwitchingDiffusionHMM(cfg)
            try:
                res = model.fit(sample, max_iter=cfg.max_iter, tol=self.config.tol)
            except Exception:
                continue

            d_samples.append(np.asarray(res["diffusion_coefficients_um2_s"], dtype=float))
            dwell_samples.append(np.asarray(res["dwell_times_s"], dtype=float))
            trans_samples.append(np.asarray(res["transition_matrix_frame"], dtype=float))
            sigma_samples.append(float(res["localization_error_um"]))

        if not d_samples:
            return {"warning": "Bootstrap failed to converge on all resamples."}

        d_arr = np.stack(d_samples, axis=0)
        dwell_arr = np.stack(dwell_samples, axis=0)
        trans_arr = np.stack(trans_samples, axis=0)

        return {
            "n_bootstrap_success": int(d_arr.shape[0]),
            "diffusion_ci95_um2_s": np.percentile(d_arr, [2.5, 97.5], axis=0).T.tolist(),
            "dwell_time_ci95_s": np.percentile(dwell_arr, [2.5, 97.5], axis=0).T.tolist(),
            "transition_matrix_ci95": np.percentile(trans_arr, [2.5, 97.5], axis=0).tolist(),
            "localization_error_ci95_um": np.percentile(np.asarray(sigma_samples), [2.5, 97.5]).tolist(),
        }
