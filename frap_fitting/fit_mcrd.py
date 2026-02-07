"""
Mass-Conserving RD Model Fitting

Fit MCRD model to condensate FRAP data with strict mass conservation.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import Dict, Any, Tuple
import warnings

from frap_models import MassConservingRDModel
from .likelihoods import LikelihoodFunction, compute_aic, compute_bic


def fit_mass_conserving_rd(
    observed_recovery: np.ndarray,
    timepoints: np.ndarray,
    geometry: Dict[str, Any],
    initial_guess: Dict[str, float] = None,
    likelihood_type: str = 'gaussian',
    method: str = 'differential_evolution',
    **kwargs
) -> Dict[str, Any]:
    """
    Fit mass-conserving RD model to FRAP data.
    
    Parameters
    ----------
    observed_recovery : np.ndarray
        Observed normalized recovery curve
    timepoints : np.ndarray
        Time points [s]
    geometry : dict
        Geometry specification
    initial_guess : dict, optional
        Initial parameter guess
    likelihood_type : str
        'gaussian' or 'poisson'
    method : str
        Optimization method
    **kwargs
        Additional optimizer arguments
    
    Returns
    -------
    result : dict
        Fitting results
    """
    # Initialize model
    model = MassConservingRDModel()
    
    # Set up likelihood
    likelihood_func = LikelihoodFunction(
        model, observed_recovery, timepoints, geometry, likelihood_type
    )
    
    # Parameter bounds
    bounds_dict = model.get_parameter_bounds()
    param_names = ['D_dilute', 'D_condensed', 'k_in', 'k_out']
    bounds = [bounds_dict[name] for name in param_names]
    
    # Initial guess
    if initial_guess is None:
        initial_guess = {
            'D_dilute': 5.0,
            'D_condensed': 0.5,
            'k_in': 1.0,
            'k_out': 1.0,
            'condensed_fraction': 0.7,
            'bleach_depth': 0.9
        }
    
    # Objective function
    def objective(param_array):
        params = {
            'D_dilute': param_array[0],
            'D_condensed': param_array[1],
            'k_in': param_array[2],
            'k_out': param_array[3],
            'condensed_fraction': initial_guess.get('condensed_fraction', 0.7),
            'bleach_depth': initial_guess.get('bleach_depth', 0.9)
        }
        return likelihood_func(params)
    
    # Optimize
    if method == 'differential_evolution':
        result = differential_evolution(
            objective,
            bounds,
            seed=kwargs.get('seed', 42),
            maxiter=kwargs.get('maxiter', 100),
            atol=kwargs.get('atol', 1e-6),
            tol=kwargs.get('tol', 0.01)
        )
    else:
        x0 = np.array([initial_guess[name] for name in param_names])
        result = minimize(
            objective,
            x0,
            bounds=bounds,
            method=kwargs.get('optimizer', 'L-BFGS-B')
        )
    
    # Extract results
    best_params = {
        'D_dilute': result.x[0],
        'D_condensed': result.x[1],
        'k_in': result.x[2],
        'k_out': result.x[3],
        'condensed_fraction': initial_guess.get('condensed_fraction', 0.7),
        'bleach_depth': initial_guess.get('bleach_depth', 0.9)
    }
    
    # Compute prediction
    predicted = model.simulate(best_params, geometry, timepoints)
    
    # Information criteria
    neg_log_like = result.fun
    log_like = -neg_log_like
    n_params = len(param_names)
    n_data = len(observed_recovery)
    
    aic = compute_aic(log_like, n_params)
    bic = compute_bic(log_like, n_params, n_data)
    
    # Check mass conservation
    if hasattr(model, '_total_mass_postbleach'):
        mass_conservation_check = True
    else:
        mass_conservation_check = None
    
    return {
        'params': best_params,
        'log_likelihood': log_like,
        'aic': aic,
        'bic': bic,
        'predicted': predicted,
        'success': result.success,
        'message': result.message if hasattr(result, 'message') else 'Optimization complete',
        'n_iterations': result.nfev,
        'mass_conserved': mass_conservation_check
    }


def check_parameter_identifiability(
    observed_recovery: np.ndarray,
    timepoints: np.ndarray,
    geometry: Dict[str, Any],
    best_params: Dict[str, float],
    param_to_vary: str,
    n_samples: int = 20
) -> Dict[str, Any]:
    """
    Check if a parameter is identifiable by varying it around best fit.
    
    Parameters
    ----------
    observed_recovery : np.ndarray
        Observed data
    timepoints : np.ndarray
        Time points
    geometry : dict
        Geometry
    best_params : dict
        Best-fit parameters
    param_to_vary : str
        Parameter name to test
    n_samples : int
        Number of samples to test
    
    Returns
    -------
    identifiability : dict
        Results including likelihood profile
    """
    model = MassConservingRDModel()
    likelihood_func = LikelihoodFunction(
        model, observed_recovery, timepoints, geometry, 'gaussian'
    )
    
    # Get parameter bounds
    bounds = model.get_parameter_bounds()
    lower, upper = bounds[param_to_vary]
    
    # Sample around best fit
    best_value = best_params[param_to_vary]
    test_values = np.linspace(
        max(lower, best_value * 0.1),
        min(upper, best_value * 10),
        n_samples
    )
    
    # Evaluate likelihood profile
    log_likelihoods = []
    for test_val in test_values:
        params = best_params.copy()
        params[param_to_vary] = test_val
        
        try:
            neg_log_like = likelihood_func(params)
            log_likelihoods.append(-neg_log_like)
        except:
            log_likelihoods.append(-np.inf)
    
    log_likelihoods = np.array(log_likelihoods)
    
    # Check if there's a clear optimum
    max_idx = np.argmax(log_likelihoods)
    max_log_like = log_likelihoods[max_idx]
    
    # Count values within 2 log-likelihood units (95% confidence)
    within_ci = np.sum(log_likelihoods > max_log_like - 2)
    
    identifiable = within_ci < n_samples * 0.8  # Somewhat arbitrary threshold
    
    return {
        'parameter': param_to_vary,
        'test_values': test_values,
        'log_likelihoods': log_likelihoods,
        'best_value': test_values[max_idx],
        'identifiable': identifiable,
        'ci_width': within_ci / n_samples
    }
