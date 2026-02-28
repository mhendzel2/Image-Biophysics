"""
Reaction-Diffusion Model Fitting

Fit classical RD FRAP model to experimental data using
likelihood-based optimization.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import Dict, Any, Tuple, Optional, List
import warnings

from frap_models import ReactionDiffusionModel
from .likelihoods import LikelihoodFunction, compute_aic, compute_bic


def _approximate_hessian(
    objective,
    x_opt: np.ndarray,
    bounds: List[Tuple[float, float]],
    step_scale: float = 1e-3
) -> np.ndarray:
    """Approximate Hessian of scalar objective with central finite differences."""
    x0 = np.asarray(x_opt, dtype=float)
    n = len(x0)
    h = np.zeros(n, dtype=float)
    for i in range(n):
        lo, hi = bounds[i]
        span = max(float(hi) - float(lo), 1.0)
        h[i] = max(abs(x0[i]) * step_scale, span * 1e-4, 1e-6)

    f0 = float(objective(x0))
    H = np.full((n, n), np.nan, dtype=float)

    for i in range(n):
        ei = np.zeros(n, dtype=float)
        ei[i] = h[i]
        xp = np.clip(x0 + ei, [b[0] for b in bounds], [b[1] for b in bounds])
        xm = np.clip(x0 - ei, [b[0] for b in bounds], [b[1] for b in bounds])
        fp = float(objective(xp))
        fm = float(objective(xm))
        denom = max((xp[i] - x0[i]) * (x0[i] - xm[i]), 1e-12)
        # For symmetric spacing, this equals h^2.
        H[i, i] = (fp - 2.0 * f0 + fm) / denom

    for i in range(n):
        for j in range(i + 1, n):
            ei = np.zeros(n, dtype=float)
            ej = np.zeros(n, dtype=float)
            ei[i] = h[i]
            ej[j] = h[j]
            xpp = np.clip(x0 + ei + ej, [b[0] for b in bounds], [b[1] for b in bounds])
            xpm = np.clip(x0 + ei - ej, [b[0] for b in bounds], [b[1] for b in bounds])
            xmp = np.clip(x0 - ei + ej, [b[0] for b in bounds], [b[1] for b in bounds])
            xmm = np.clip(x0 - ei - ej, [b[0] for b in bounds], [b[1] for b in bounds])
            fpp = float(objective(xpp))
            fpm = float(objective(xpm))
            fmp = float(objective(xmp))
            fmm = float(objective(xmm))
            denom = max((xpp[i] - xmp[i]) * (xpp[j] - xpm[j]), 1e-12)
            Hij = (fpp - fpm - fmp + fmm) / denom
            H[i, j] = Hij
            H[j, i] = Hij

    return H


def _profile_flatness(
    objective,
    x_opt: np.ndarray,
    bounds: List[Tuple[float, float]],
    param_names: List[str],
    n_profile_points: int = 11
) -> Dict[str, float]:
    """
    Estimate profile flatness per parameter with other parameters fixed.

    Returns the fraction of tested points with delta-NLL <= 1.92
    (approximate 95% likelihood region for one parameter).
    """
    x0 = np.asarray(x_opt, dtype=float)
    flat_fraction: Dict[str, float] = {}
    for i, name in enumerate(param_names):
        lo, hi = bounds[i]
        center = x0[i]
        lower = max(lo, center * 0.5)
        upper = min(hi, center * 1.5 if center > 0 else lo + 0.5 * (hi - lo))
        if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
            flat_fraction[name] = np.nan
            continue
        values = np.linspace(lower, upper, int(max(n_profile_points, 5)))
        nll = []
        for v in values:
            x = x0.copy()
            x[i] = float(v)
            nll.append(float(objective(x)))
        nll = np.asarray(nll, dtype=float)
        if not np.any(np.isfinite(nll)):
            flat_fraction[name] = np.nan
            continue
        min_nll = np.nanmin(nll)
        flat_fraction[name] = float(np.mean((nll - min_nll) <= 1.92))
    return flat_fraction


def _assess_identifiability(
    objective,
    x_opt: np.ndarray,
    bounds: List[Tuple[float, float]],
    param_names: List[str],
    condition_threshold: float = 1e8,
    relative_se_threshold: float = 1.0,
    flat_profile_threshold: float = 0.6,
    n_profile_points: int = 11
) -> Dict[str, Any]:
    """Assess local parameter identifiability around optimum."""
    H = _approximate_hessian(objective, x_opt, bounds)
    cond = np.inf
    if np.all(np.isfinite(H)):
        try:
            cond = float(np.linalg.cond(H))
        except Exception:
            cond = np.inf

    covariance = np.full_like(H, np.nan, dtype=float)
    if np.all(np.isfinite(H)):
        try:
            covariance = np.linalg.pinv(H)
        except Exception:
            covariance = np.full_like(H, np.nan, dtype=float)

    stderr = np.full(len(param_names), np.nan, dtype=float)
    if np.all(np.isfinite(covariance)):
        diag = np.diag(covariance)
        diag = np.where(diag < 0, np.nan, diag)
        stderr = np.sqrt(diag)

    x = np.asarray(x_opt, dtype=float)
    relative_se = stderr / np.maximum(np.abs(x), 1e-12)

    ci95 = {}
    relative_se_dict = {}
    for i, name in enumerate(param_names):
        ci95[name] = (
            float(x[i] - 1.96 * stderr[i]) if np.isfinite(stderr[i]) else np.nan,
            float(x[i] + 1.96 * stderr[i]) if np.isfinite(stderr[i]) else np.nan,
        )
        relative_se_dict[name] = float(relative_se[i]) if np.isfinite(relative_se[i]) else np.nan

    flat_fraction = _profile_flatness(
        objective,
        x_opt,
        bounds,
        param_names,
        n_profile_points=n_profile_points,
    )

    warnings_list: List[str] = []
    if not np.isfinite(cond) or cond > condition_threshold:
        warnings_list.append(
            "Fisher/Hessian matrix is ill-conditioned; FRAP parameters may be non-identifiable."
        )
    for name in param_names:
        rse = relative_se_dict.get(name, np.nan)
        if np.isfinite(rse) and rse > relative_se_threshold:
            warnings_list.append(
                f"Large uncertainty for {name} (relative SE={rse:.2f}); parameter may be poorly identifiable."
            )
        flat = flat_fraction.get(name, np.nan)
        if np.isfinite(flat) and flat > flat_profile_threshold:
            warnings_list.append(
                f"Flat likelihood profile for {name} (flat fraction={flat:.2f}); FRAP-only fit may be degenerate."
            )

    # Deduplicate while preserving order
    deduped = []
    for msg in warnings_list:
        if msg not in deduped:
            deduped.append(msg)

    return {
        'hessian': H,
        'hessian_condition_number': float(cond),
        'stderr': {name: float(stderr[i]) if np.isfinite(stderr[i]) else np.nan for i, name in enumerate(param_names)},
        'relative_se': relative_se_dict,
        'approx_ci95': ci95,
        'profile_flat_fraction': flat_fraction,
        'warnings': deduped,
        'identifiable': len(deduped) == 0
    }


def fit_reaction_diffusion(
    observed_recovery: np.ndarray,
    timepoints: np.ndarray,
    geometry: Dict[str, Any],
    initial_guess: Dict[str, float] = None,
    likelihood_type: str = 'gaussian',
    method: str = 'differential_evolution',
    assess_identifiability: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Fit reaction-diffusion model to FRAP data.
    
    Parameters
    ----------
    observed_recovery : np.ndarray
        Observed normalized recovery curve
    timepoints : np.ndarray
        Time points [s]
    geometry : dict
        Geometry specification for simulation
    initial_guess : dict, optional
        Initial parameter guess. If None, uses reasonable defaults.
    likelihood_type : str
        'gaussian' or 'poisson'
    method : str
        Optimization method: 'differential_evolution' (global) or 'minimize' (local)
    **kwargs
        Additional arguments for optimizer
    
    Returns
    -------
    result : dict
        Fitting results with:
        - 'params': Best-fit parameters
        - 'log_likelihood': Maximum log-likelihood
        - 'aic': Akaike Information Criterion
        - 'bic': Bayesian Information Criterion
        - 'predicted': Model prediction at best fit
        - 'success': Whether optimization succeeded
    """
    # Initialize model
    model = ReactionDiffusionModel()
    
    # Set up likelihood function
    likelihood_func = LikelihoodFunction(
        model, observed_recovery, timepoints, geometry, likelihood_type
    )
    
    # Parameter bounds
    bounds_dict = model.get_parameter_bounds()
    param_names = ['D', 'k_on', 'k_off']
    bounds = [bounds_dict[name] for name in param_names]
    
    # Initial guess
    if initial_guess is None:
        initial_guess = {
            'D': 1.0,
            'k_on': 1.0,
            'k_off': 1.0,
            'bleach_depth': 0.9
        }
    
    # Wrapper for optimizer (takes array, not dict)
    def objective(param_array):
        params = {
            'D': param_array[0],
            'k_on': param_array[1],
            'k_off': param_array[2],
            'bleach_depth': initial_guess.get('bleach_depth', 0.9)
        }
        return likelihood_func(params)
    
    # Optimize
    if method == 'differential_evolution':
        # Global optimization
        result = differential_evolution(
            objective,
            bounds,
            seed=kwargs.get('seed', 42),
            maxiter=kwargs.get('maxiter', 100),
            atol=kwargs.get('atol', 1e-6),
            tol=kwargs.get('tol', 0.01)
        )
    else:
        # Local optimization
        x0 = np.array([initial_guess[name] for name in param_names])
        result = minimize(
            objective,
            x0,
            bounds=bounds,
            method=kwargs.get('optimizer', 'L-BFGS-B')
        )
    
    # Extract results
    best_params = {
        'D': result.x[0],
        'k_on': result.x[1],
        'k_off': result.x[2],
        'bleach_depth': initial_guess.get('bleach_depth', 0.9)
    }
    
    # Compute model prediction
    predicted = model.simulate(best_params, geometry, timepoints)
    
    # Compute information criteria
    neg_log_like = result.fun
    log_like = -neg_log_like
    n_params = len(param_names)
    n_data = len(observed_recovery)
    
    aic = compute_aic(log_like, n_params)
    bic = compute_bic(log_like, n_params, n_data)
    
    identifiability = None
    if assess_identifiability:
        identifiability = _assess_identifiability(
            objective=objective,
            x_opt=result.x,
            bounds=bounds,
            param_names=param_names,
            condition_threshold=float(kwargs.get('identifiability_condition_threshold', 1e8)),
            relative_se_threshold=float(kwargs.get('identifiability_relative_se_threshold', 1.0)),
            flat_profile_threshold=float(kwargs.get('identifiability_flat_profile_threshold', 0.6)),
            n_profile_points=int(kwargs.get('identifiability_profile_points', 11)),
        )
        if identifiability['warnings']:
            warnings.warn(
                "Parameter identifiability warning: "
                + " ".join(identifiability['warnings']),
                UserWarning,
            )

    out = {
        'params': best_params,
        'log_likelihood': log_like,
        'aic': aic,
        'bic': bic,
        'predicted': predicted,
        'success': result.success,
        'message': result.message if hasattr(result, 'message') else 'Optimization complete',
        'n_iterations': result.nfev
    }
    if identifiability is not None:
        out['identifiability'] = identifiability
        out['identifiability_warning'] = bool(identifiability['warnings'])
    return out


def estimate_parameter_uncertainty(
    observed_recovery: np.ndarray,
    timepoints: np.ndarray,
    geometry: Dict[str, Any],
    best_params: Dict[str, float],
    likelihood_type: str = 'gaussian',
    n_bootstrap: int = 100
) -> Dict[str, Tuple[float, float]]:
    """
    Estimate parameter uncertainty via bootstrap.
    
    Parameters
    ----------
    observed_recovery : np.ndarray
        Observed data
    timepoints : np.ndarray
        Time points
    geometry : dict
        Geometry specification
    best_params : dict
        Best-fit parameters
    likelihood_type : str
        Likelihood type
    n_bootstrap : int
        Number of bootstrap samples
    
    Returns
    -------
    uncertainties : dict
        Parameter uncertainties as {param: (lower, upper)} at 95% CI
    """
    model = ReactionDiffusionModel()
    predicted = model.simulate(best_params, geometry, timepoints)
    
    # Estimate noise level
    residuals = observed_recovery - predicted
    sigma = np.std(residuals)
    
    # Bootstrap
    param_samples = {key: [] for key in ['D', 'k_on', 'k_off']}
    
    for i in range(n_bootstrap):
        # Generate synthetic data with noise
        synthetic = predicted + np.random.normal(0, sigma, len(predicted))
        
        # Fit to synthetic data
        try:
            result = fit_reaction_diffusion(
                synthetic, timepoints, geometry,
                initial_guess=best_params,
                likelihood_type=likelihood_type,
                method='minimize',
                maxiter=50
            )
            
            if result['success']:
                for key in param_samples:
                    param_samples[key].append(result['params'][key])
        except:
            continue
    
    # Compute confidence intervals
    uncertainties = {}
    for key, values in param_samples.items():
        if len(values) > 10:
            lower = np.percentile(values, 2.5)
            upper = np.percentile(values, 97.5)
            uncertainties[key] = (lower, upper)
        else:
            uncertainties[key] = (np.nan, np.nan)
    
    return uncertainties
