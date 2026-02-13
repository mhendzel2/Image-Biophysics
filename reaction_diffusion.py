"""
Reaction-diffusion simulation and inference utilities for FRAP-style analyses.

This module provides a finite-difference 2D reaction-diffusion model with:
- Free and bound species fields
- Optional spatially varying diffusion maps
- Bleach profile initialization
- Optional PSF observation blur
- Optional frame-wise motion correction
- Likelihood evaluation and profile-likelihood intervals
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Iterable, Tuple, List
import warnings

import numpy as np
from scipy import ndimage, optimize


@dataclass
class RDParameters:
    """Core reaction-diffusion parameters."""

    D_free_um2_s: float = 3.0
    D_bound_um2_s: float = 0.01
    k_on_s_inv: float = 0.2
    k_off_s_inv: float = 0.1


class ReactionDiffusionModel:
    """2D finite-difference reaction-diffusion model for FRAP image stacks."""

    def __init__(self, pixel_size_um: float = 0.1, time_step_s: float = 0.02):
        self.pixel_size_um = float(pixel_size_um)
        self.time_step_s = float(time_step_s)

    @staticmethod
    def create_bleach_profile(
        shape: Tuple[int, int],
        center_yx: Tuple[float, float],
        radius_px: float,
        depth: float = 0.8,
        edge_sigma_px: float = 1.5,
    ) -> np.ndarray:
        """Create smooth circular bleach profile in [0,1], where lower means stronger bleach."""
        h, w = shape
        yy, xx = np.mgrid[0:h, 0:w]
        cy, cx = center_yx
        rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        hard = (rr <= radius_px).astype(float)
        smooth = ndimage.gaussian_filter(hard, sigma=max(edge_sigma_px, 1e-6))
        smooth = np.clip(smooth / (smooth.max() + 1e-12), 0.0, 1.0)
        profile = 1.0 - float(depth) * smooth
        return np.clip(profile, 0.0, 1.0)

    @staticmethod
    def apply_psf(field: np.ndarray, sigma_px: float) -> np.ndarray:
        """Apply Gaussian PSF blur to a 2D image."""
        return ndimage.gaussian_filter(field, sigma=max(float(sigma_px), 1e-8), mode="nearest")

    @staticmethod
    def _laplacian_reflect(u: np.ndarray) -> np.ndarray:
        up = np.pad(u, ((1, 1), (1, 1)), mode="edge")
        c = up[1:-1, 1:-1]
        return (
            up[0:-2, 1:-1]
            + up[2:, 1:-1]
            + up[1:-1, 0:-2]
            + up[1:-1, 2:]
            - 4.0 * c
        )

    def _stable_substeps(self, Dmax_um2_s: float) -> int:
        if Dmax_um2_s <= 0:
            return 1
        dx2 = self.pixel_size_um ** 2
        dt_max = dx2 / (4.0 * Dmax_um2_s + 1e-12)
        if self.time_step_s <= dt_max:
            return 1
        n = int(np.ceil(self.time_step_s / dt_max))
        warnings.warn(
            f"time_step_s={self.time_step_s:.4g} exceeds explicit stability {dt_max:.4g}; using {n} substeps",
            RuntimeWarning,
        )
        return max(1, n)

    def simulate_fields(
        self,
        initial_free: np.ndarray,
        initial_bound: np.ndarray,
        n_frames: int,
        params: Optional[RDParameters] = None,
        roi_mask: Optional[np.ndarray] = None,
        D_free_map_um2_s: Optional[np.ndarray] = None,
        D_bound_map_um2_s: Optional[np.ndarray] = None,
        bleach_profile: Optional[np.ndarray] = None,
        psf_sigma_px: Optional[float] = None,
        motion_shifts_yx_px: Optional[Iterable[Tuple[float, float]]] = None,
    ) -> Dict[str, Any]:
        """Simulate spatiotemporal free/bound fields and observed fluorescence stack.

        Returns dict with keys: `free`, `bound`, `total`, `observed`.
        """
        p = params or RDParameters()

        f = np.asarray(initial_free, dtype=float).copy()
        b = np.asarray(initial_bound, dtype=float).copy()
        if f.shape != b.shape:
            raise ValueError("initial_free and initial_bound must have identical shape")

        h, w = f.shape
        if roi_mask is None:
            roi_mask = np.ones((h, w), dtype=bool)
        else:
            roi_mask = np.asarray(roi_mask, dtype=bool)
            if roi_mask.shape != (h, w):
                raise ValueError("roi_mask shape must match field shape")

        if bleach_profile is not None:
            bp = np.asarray(bleach_profile, dtype=float)
            if bp.shape != (h, w):
                raise ValueError("bleach_profile shape must match field shape")
            bp = np.clip(bp, 0.0, 1.0)
            f *= bp
            b *= bp

        if D_free_map_um2_s is None:
            Df_map = np.full((h, w), float(p.D_free_um2_s), dtype=float)
        else:
            Df_map = np.asarray(D_free_map_um2_s, dtype=float)
            if Df_map.shape != (h, w):
                raise ValueError("D_free_map_um2_s shape must match field shape")

        if D_bound_map_um2_s is None:
            Db_map = np.full((h, w), float(p.D_bound_um2_s), dtype=float)
        else:
            Db_map = np.asarray(D_bound_map_um2_s, dtype=float)
            if Db_map.shape != (h, w):
                raise ValueError("D_bound_map_um2_s shape must match field shape")

        Dmax = float(max(np.nanmax(Df_map), np.nanmax(Db_map), 0.0))
        n_sub = self._stable_substeps(Dmax)
        dt = self.time_step_s / n_sub
        dx2 = self.pixel_size_um ** 2

        free_stack = np.zeros((n_frames, h, w), dtype=float)
        bound_stack = np.zeros((n_frames, h, w), dtype=float)

        for t in range(n_frames):
            free_stack[t] = f
            bound_stack[t] = b

            for _ in range(n_sub):
                lap_f = self._laplacian_reflect(f)
                lap_b = self._laplacian_reflect(b)

                react_f = -p.k_on_s_inv * f + p.k_off_s_inv * b
                react_b = p.k_on_s_inv * f - p.k_off_s_inv * b

                f = f + dt * ((Df_map / dx2) * lap_f + react_f)
                b = b + dt * ((Db_map / dx2) * lap_b + react_b)

                f[~roi_mask] = 0.0
                b[~roi_mask] = 0.0
                f = np.clip(f, 0.0, None)
                b = np.clip(b, 0.0, None)

        total = free_stack + bound_stack
        observed = total.copy()

        if psf_sigma_px is not None and psf_sigma_px > 0:
            for t in range(n_frames):
                observed[t] = self.apply_psf(observed[t], sigma_px=float(psf_sigma_px))

        if motion_shifts_yx_px is not None:
            shifts = list(motion_shifts_yx_px)
            if len(shifts) != n_frames:
                raise ValueError("motion_shifts_yx_px length must equal n_frames")
            for t, (dy, dx) in enumerate(shifts):
                observed[t] = ndimage.shift(observed[t], shift=(dy, dx), order=1, mode="nearest")

        return {
            "free": free_stack,
            "bound": bound_stack,
            "total": total,
            "observed": observed,
            "n_substeps": n_sub,
            "dt_substep_s": dt,
        }

    @staticmethod
    def negative_log_likelihood(
        observed: np.ndarray,
        predicted: np.ndarray,
        noise_model: str = "gaussian",
        sigma: float = 1.0,
    ) -> float:
        """Evaluate image-stack likelihood (up to additive constants)."""
        y = np.asarray(observed, dtype=float)
        mu = np.asarray(predicted, dtype=float)
        if y.shape != mu.shape:
            raise ValueError("observed and predicted must have identical shape")

        model = str(noise_model).lower()
        if model == "gaussian":
            s2 = float(max(sigma, 1e-12)) ** 2
            return float(0.5 * np.sum((y - mu) ** 2) / s2)
        if model == "poisson":
            mu_safe = np.clip(mu, 1e-12, None)
            return float(np.sum(mu_safe - y * np.log(mu_safe)))
        raise ValueError("noise_model must be 'gaussian' or 'poisson'")

    def fit_to_stack(
        self,
        observed_stack: np.ndarray,
        initial_free: np.ndarray,
        initial_bound: np.ndarray,
        n_frames: int,
        roi_mask: Optional[np.ndarray] = None,
        bleach_profile: Optional[np.ndarray] = None,
        psf_sigma_px: Optional[float] = None,
        noise_model: str = "gaussian",
        sigma: float = 1.0,
        x0: Optional[Dict[str, float]] = None,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Dict[str, Any]:
        """Fit RD parameters by minimizing image-stack negative log likelihood."""
        x0 = x0 or {
            "D_free_um2_s": 2.0,
            "D_bound_um2_s": 0.05,
            "k_on_s_inv": 0.2,
            "k_off_s_inv": 0.1,
        }
        bounds = bounds or {
            "D_free_um2_s": (1e-4, 50.0),
            "D_bound_um2_s": (0.0, 10.0),
            "k_on_s_inv": (0.0, 20.0),
            "k_off_s_inv": (0.0, 20.0),
        }

        keys = ["D_free_um2_s", "D_bound_um2_s", "k_on_s_inv", "k_off_s_inv"]
        x_init = np.array([float(x0[k]) for k in keys], dtype=float)
        bnds = [bounds[k] for k in keys]

        obs = np.asarray(observed_stack, dtype=float)

        def objective(xvec: np.ndarray) -> float:
            p = RDParameters(
                D_free_um2_s=float(xvec[0]),
                D_bound_um2_s=float(xvec[1]),
                k_on_s_inv=float(xvec[2]),
                k_off_s_inv=float(xvec[3]),
            )
            sim = self.simulate_fields(
                initial_free=initial_free,
                initial_bound=initial_bound,
                n_frames=n_frames,
                params=p,
                roi_mask=roi_mask,
                bleach_profile=bleach_profile,
                psf_sigma_px=psf_sigma_px,
            )
            return self.negative_log_likelihood(obs, sim["observed"], noise_model=noise_model, sigma=sigma)

        result = optimize.minimize(objective, x_init, method="L-BFGS-B", bounds=bnds)
        fitted = {k: float(v) for k, v in zip(keys, result.x)}

        return {
            "status": "success" if result.success else "warning",
            "message": result.message,
            "fitted_parameters": fitted,
            "negative_log_likelihood": float(result.fun),
            "optimizer_success": bool(result.success),
            "n_iterations": int(getattr(result, "nit", -1)),
        }

    def profile_likelihood(
        self,
        observed_stack: np.ndarray,
        initial_free: np.ndarray,
        initial_bound: np.ndarray,
        n_frames: int,
        parameter_name: str,
        scan_values: Iterable[float],
        roi_mask: Optional[np.ndarray] = None,
        bleach_profile: Optional[np.ndarray] = None,
        psf_sigma_px: Optional[float] = None,
        noise_model: str = "gaussian",
        sigma: float = 1.0,
        initial_guess: Optional[Dict[str, float]] = None,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Dict[str, Any]:
        """Compute profile likelihood for one parameter by re-optimizing remaining parameters."""
        initial_guess = initial_guess or {
            "D_free_um2_s": 2.0,
            "D_bound_um2_s": 0.05,
            "k_on_s_inv": 0.2,
            "k_off_s_inv": 0.1,
        }
        bounds = bounds or {
            "D_free_um2_s": (1e-4, 50.0),
            "D_bound_um2_s": (0.0, 10.0),
            "k_on_s_inv": (0.0, 20.0),
            "k_off_s_inv": (0.0, 20.0),
        }

        all_keys = ["D_free_um2_s", "D_bound_um2_s", "k_on_s_inv", "k_off_s_inv"]
        if parameter_name not in all_keys:
            raise ValueError(f"parameter_name must be one of {all_keys}")

        free_keys = [k for k in all_keys if k != parameter_name]
        prof_values: List[float] = []
        nll_values: List[float] = []
        fitted_other: List[Dict[str, float]] = []

        obs = np.asarray(observed_stack, dtype=float)

        for v in scan_values:
            x0 = np.array([initial_guess[k] for k in free_keys], dtype=float)
            bnds = [bounds[k] for k in free_keys]

            def objective(xvec: np.ndarray) -> float:
                params = {parameter_name: float(v)}
                for i, k in enumerate(free_keys):
                    params[k] = float(xvec[i])

                p = RDParameters(
                    D_free_um2_s=params["D_free_um2_s"],
                    D_bound_um2_s=params["D_bound_um2_s"],
                    k_on_s_inv=params["k_on_s_inv"],
                    k_off_s_inv=params["k_off_s_inv"],
                )
                sim = self.simulate_fields(
                    initial_free=initial_free,
                    initial_bound=initial_bound,
                    n_frames=n_frames,
                    params=p,
                    roi_mask=roi_mask,
                    bleach_profile=bleach_profile,
                    psf_sigma_px=psf_sigma_px,
                )
                return self.negative_log_likelihood(obs, sim["observed"], noise_model=noise_model, sigma=sigma)

            res = optimize.minimize(objective, x0, method="L-BFGS-B", bounds=bnds)
            prof_values.append(float(v))
            nll_values.append(float(res.fun))
            fitted_other.append({k: float(val) for k, val in zip(free_keys, res.x)})

        nll_arr = np.asarray(nll_values, dtype=float)
        delta = nll_arr - np.nanmin(nll_arr)

        # Approximate 95% CI criterion for one profiled parameter: ΔNLL <= 1.92
        inside = np.asarray(prof_values, dtype=float)[delta <= 1.92]
        ci95 = (
            (float(np.min(inside)), float(np.max(inside)))
            if inside.size > 0
            else (np.nan, np.nan)
        )

        return {
            "parameter_name": parameter_name,
            "scan_values": np.asarray(prof_values, dtype=float),
            "negative_log_likelihood": nll_arr,
            "delta_nll": delta,
            "fitted_other_parameters": fitted_other,
            "ci95_profile": ci95,
        }
