"""
Reaction-Diffusion Model Fitting

Fit classical RD FRAP model to experimental data using
likelihood-based optimization.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import Dict, Any, Tuple, Optional
import warnings

from frap_models import ReactionDiffusionModel
from .likelihoods import LikelihoodFunction, compute_aic, compute_bic


def fit_reaction_diffusion(
    observed_recovery: np.ndarray,
    timepoints: np.ndarray,
    geometry: Dict[str, Any],
    initial_guess: Dict[str, float] = None,
    likelihood_type: str = 'gaussian',
    method: str = 'differential_evolution',
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
    
    return {
        'params': best_params,
        'log_likelihood': log_like,
        'aic': aic,
        'bic': bic,
        'predicted': predicted,
        'success': result.success,
        'message': result.message if hasattr(result, 'message') else 'Optimization complete',
        'n_iterations': result.nfev
    }


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
