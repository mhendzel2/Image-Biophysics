"""
Segmented Fluorescence Correlation Spectroscopy (FCS)

Temporal segmentation of an intensity trace (e.g., line-scan / imaging FCS extracted
intensity) with per‑segment autocorrelation, model fitting (2D/3D/anomalous), and
quality metrics. Designed to operate even when optional fitting libs are absent.

Models (normalized to mean intensity):
  2D:   G(τ) = G0 * (1 + τ/τD)^(-1) + offset
  3D:   G(τ) = G0 * (1 + τ/τD)^(-1) * (1 + τ/(κ^2 τD))^(-1/2) + offset
  anom: G(τ) = G0 * (1 + (τ/τD)^α)^(-1) + offset
with D = ω^2 / (4 τD)  (ω = lateral beam waist; κ = axial-to-lateral ratio)

Returns a per‑segment summary DataFrame and fitted curves.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple, Optional

# Optional dependencies
try:
    import multipletau as mt  # type: ignore
except Exception as e:  # pragma: no cover - optional
    mt = None
    warnings.warn(f"multipletau not available; using FFT fallback for ACF ({e})")

try:
    import lmfit  # type: ignore
    from lmfit import Model, Parameters  # type: ignore
except Exception as e:  # pragma: no cover - optional
    lmfit = None
    Model = None
    Parameters = None
    warnings.warn(f"lmfit not available; using heuristic τD estimate ({e})")


def _acf_fft(x: np.ndarray) -> np.ndarray:
    """ACF via FFT (biased, mean-subtracted, normalized to mean^2)."""
    x = np.asarray(x, dtype=float)
    mu = x.mean()
    if mu == 0:
        mu = 1e-12
    x = x - mu
    n = len(x)
    f = np.fft.rfft(x, n=2 * n)
    s = np.fft.irfft(f * np.conjugate(f))[:n]
    norm = (np.arange(n, 0, -1) * (mu ** 2))
    return (s / norm).real


def _r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-12
    return 1.0 - ss_res / ss_tot, ss_res


@dataclass
class SegFCSParams:
    dt: float = 0.001  # seconds per sample
    model: str = "3D"  # "2D" | "3D" | "anomalous"
    beam_waist_um: float = 0.25  # ω (µm)
    kappa: float = 5.0  # axial-to-lateral ratio for 3D
    alpha: float = 1.0  # anomalous exponent
    window_s: float = 5.0  # sliding window length (s)
    step_s: float = 2.5  # step between windows (s)
    m_coarsening: int = 16  # multipletau m
    max_lag_s: Optional[float] = None  # limit τmax (s); default ~window/2
    min_points: int = 200  # minimum points per segment
    detrend: bool = True  # subtract local mean per segment
    normalize: bool = True  # G normalized to mean^2


class SegmentedFCSAnalyzer:
    def __init__(self, defaults: Optional[SegFCSParams] = None):
        self.defaults = defaults or SegFCSParams()

    # ---- Models ----
    @staticmethod
    def model_2d(tau, G0, tauD, offset):  # pylint: disable=invalid-name
        return G0 * (1.0 + tau / tauD) ** (-1.0) + offset

    @staticmethod
    def model_3d(tau, G0, tauD, kappa, offset):  # pylint: disable=invalid-name
        return G0 * (1.0 + tau / tauD) ** (-1.0) * (1.0 + tau / (kappa * kappa * tauD)) ** (-0.5) + offset

    @staticmethod
    def model_anom(tau, G0, tauD, alpha, offset):  # pylint: disable=invalid-name
        return G0 * (1.0 + (tau / tauD) ** alpha) ** (-1.0) + offset

    # ---- Segmentation ----
    def _segments(self, n: int, dt: float, window_s: float, step_s: float, min_points: int) -> List[Tuple[int, int]]:
        w = max(int(round(window_s / dt)), min_points)
        s = max(int(round(step_s / dt)), 1)
        segs: List[Tuple[int, int]] = []
        i = 0
        while i + w <= n:
            segs.append((i, i + w))
            i += s
        if not segs and n >= min_points:
            segs = [(0, n)]
        return segs

    # ---- Autocorrelation ----
    def _acf(self, x: np.ndarray, dt: float, m: int, max_lag_s: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
        if mt is not None:  # pragma: no cover - optional dependency path
            tau, G = mt.autocorrelate(x.astype(float), m=m, deltat=dt, normalize=True)  # type: ignore
            tau = np.asarray(tau)
            G = np.asarray(G)
        else:
            G = _acf_fft(x)
            tau = np.arange(len(G)) * dt
        if max_lag_s is not None:
            keep = tau <= max_lag_s
            tau, G = tau[keep], G[keep]
        return tau, G

    # ---- Fitting ----
    def _fit(self, tau: np.ndarray, G: np.ndarray, params: SegFCSParams) -> Dict[str, Any]:
        if params.model == "2D":
            fn = self.model_2d
            pcount = 3
            if lmfit:
                model = Model(self.model_2d, independent_vars=['tau'])  # type: ignore
                p = Parameters()  # type: ignore
                p.add('G0', value=max(G.max(), 1e-4), min=1e-6, max=1.0)
                p.add('tauD', value=max(tau[1], 1e-4), min=1e-6)
                p.add('offset', value=max(G.min(), 0.0), min=-0.1, max=0.2)
        elif params.model == "3D":
            fn = self.model_3d
            pcount = 4
            if lmfit:
                model = Model(self.model_3d, independent_vars=['tau'])  # type: ignore
                p = Parameters()  # type: ignore
                p.add('G0', value=max(G.max(), 1e-4), min=1e-6, max=1.0)
                p.add('tauD', value=max(tau[1], 1e-4), min=1e-6)
                p.add('kappa', value=params.kappa, min=1.0, max=15.0)
                p.add('offset', value=max(G.min(), 0.0), min=-0.1, max=0.2)
        else:  # anomalous
            fn = self.model_anom
            pcount = 4
            if lmfit:
                model = Model(self.model_anom, independent_vars=['tau'])  # type: ignore
                p = Parameters()  # type: ignore
                p.add('G0', value=max(G.max(), 1e-4), min=1e-6, max=1.0)
                p.add('tauD', value=max(tau[1], 1e-4), min=1e-6)
                p.add('alpha', value=params.alpha, min=0.3, max=1.3)
                p.add('offset', value=max(G.min(), 0.0), min=-0.1, max=0.2)

        if lmfit:
            try:  # pragma: no cover - numeric fitting
                fit = model.fit(G, tau=tau, params=p, nan_policy='omit')  # type: ignore
                best = fit.best_values  # type: ignore
                yhat = fit.eval(tau=tau)  # type: ignore
                r2, ss_res = _r2(G, yhat)
                dof = max(len(G) - pcount, 1)
                chi2_red = ss_res / dof
                return {'ok': True, 'params': best, 'yhat': yhat, 'r2': r2, 'chi2_red': chi2_red}
            except Exception as e:  # pragma: no cover
                warnings.warn(f"lmfit failed; using heuristic τD ({e})")

        # Heuristic fallback: τD ~ lag where G drops to half of (G[0]-baseline)
        G0_est = max(G[0], 1e-6)
        offset = np.median(G[-max(10, len(G) // 10):])
        half = offset + 0.5 * (G0_est - offset)
        idx = np.argmin(np.abs(G - half))
        tauD = max(tau[idx], tau[1] if len(tau) > 1 else 1e-3)
        params_out = {'G0': float(G0_est), 'tauD': float(tauD)}
        if params.model == "3D":
            params_out['kappa'] = float(params.kappa)
        if params.model == "anomalous":
            params_out['alpha'] = float(params.alpha)
        params_out['offset'] = float(offset)
        yhat = fn(tau, **params_out)
        r2, ss_res = _r2(G, yhat)
        dof = max(len(G) - pcount, 1)
        chi2_red = ss_res / dof
        return {'ok': True, 'params': params_out, 'yhat': yhat, 'r2': r2, 'chi2_red': chi2_red}

    # ---- Main entry point ----
    def analyze(self, intensity: np.ndarray, **overrides) -> Dict[str, Any]:
        pars = self.defaults
        if overrides:
            pars = SegFCSParams(**{**asdict(pars), **overrides})

        x = np.asarray(intensity, dtype=float).squeeze()
        if x.ndim != 1:
            raise ValueError("SegmentedFCS expects a 1D intensity time series.")
        n = len(x)
        if n < pars.min_points:
            return {'status': 'error', 'message': f'too few points ({n})'}

        segs = self._segments(n, pars.dt, pars.window_s, pars.step_s, pars.min_points)
        results: List[Dict[str, Any]] = []
        curves: List[Dict[str, Any]] = []

        for k, (i0, i1) in enumerate(segs, start=1):
            seg = x[i0:i1]
            if pars.detrend:
                mu = seg.mean()
                if mu != 0 and pars.normalize:
                    seg = seg / mu
                seg = seg - seg.mean()
            tau, G = self._acf(seg, pars.dt, pars.m_coarsening, pars.max_lag_s or (0.5 * pars.window_s))
            fit = self._fit(tau, G, pars)
            if not fit['ok']:
                continue

            p = fit['params']
            tauD = float(p['tauD'])
            D = (pars.beam_waist_um ** 2) / (4.0 * tauD)  # µm^2/s
            G0 = float(p['G0'])
            N = 1.0 / max(G0, 1e-12)  # Particle number approximation

            results.append({
                'segment': k,
                'start_idx': i0,
                'end_idx': i1,
                'start_s': i0 * pars.dt,
                'end_s': i1 * pars.dt,
                'model': pars.model,
                'G0': G0,
                'N_est': N,
                'tauD_s': tauD,
                'D_um2_s': D,
                'kappa': float(p.get('kappa', np.nan)),
                'alpha': float(p.get('alpha', np.nan)),
                'offset': float(p['offset']),
                'R2': float(fit['r2']),
                'chi2_red': float(fit['chi2_red'])
            })
            curves.append({
                'segment': k,
                'tau_s': tau,
                'G': G,
                'G_fit': fit['yhat']
            })

        df = pd.DataFrame(results)
        if df.empty:
            return {'status': 'error', 'message': 'no valid segments'}

        summary = {
            'segments': df,
            'curves': curves,
            'parameters_used': asdict(pars),
            'status': 'success',
            'n_segments': int(len(df)),
            'median_D_um2_s': float(df['D_um2_s'].median()),
            'median_tauD_s': float(df['tauD_s'].median()),
            'median_N_est': float(df['N_est'].median())
        }
        return summary


def get_segmented_fcs_parameters() -> Dict[str, Any]:
    """Convenience default parameters for UI."""
    p = SegFCSParams()
    return asdict(p)
