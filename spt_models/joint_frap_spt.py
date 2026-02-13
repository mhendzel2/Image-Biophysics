"""
Joint FRAP-SPT Modeling

Combine FRAP and SPT data to resolve parameter degeneracies.

Key insight: FRAP alone often cannot uniquely determine D, k_on, k_off.
SPT provides independent constraints through:
- Dwell times -> k_on, k_off
- Trajectory diffusion -> D_free, D_bound
- Bound fraction -> equilibrium
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, Any, Tuple, Optional
import warnings

from frap_models import ReactionDiffusionModel
from .dwell_time import DwellTimeModel


class JointFRAPSPT:
    """
    Joint likelihood for FRAP and SPT data.
    
    Shares parameters:
    - k_on, k_off (appear in both FRAP recovery and SPT dwell times)
    - Bound fraction (FRAP equilibrium and SPT statistics)
    - D (FRAP recovery and SPT trajectory diffusion)
    """
    
    def __init__(
        self,
        frap_model=None,
        spt_model=None
    ):
        """
        Initialize joint model.
        
        Parameters
        ----------
        frap_model : FRAPModel, optional
            FRAP model instance
        spt_model : DwellTimeModel, optional
            SPT dwell time model
        """
        self.frap_model = frap_model or ReactionDiffusionModel()
        self.spt_model = spt_model or DwellTimeModel()
    
    def joint_log_likelihood(
        self,
        params: Dict[str, float],
        frap_data: Dict[str, Any],
        spt_data: Dict[str, Any],
        weights: Dict[str, float] = None
    ) -> float:
        """
        Compute joint log-likelihood.
        
        Parameters
        ----------
        params : dict
            Shared parameters: D, k_on, k_off
        frap_data : dict
            FRAP data with 'recovery', 'timepoints', 'geometry'
        spt_data : dict
            SPT data with 'bound_dwells', 'unbound_dwells'
        weights : dict, optional
            Relative weights for FRAP vs SPT (default: equal)
        
        Returns
        -------
        log_likelihood : float
            Joint log-likelihood
        """
        if weights is None:
            weights = {'frap': 1.0, 'spt': 1.0}
        
        log_like = 0.0
        
        # FRAP likelihood
        if 'recovery' in frap_data:
            frap_pred = self.frap_model.simulate(
                params,
                frap_data['geometry'],
                frap_data['timepoints']
            )
            
            # Gaussian likelihood
            residuals = frap_data['recovery'] - frap_pred
            sigma = np.std(residuals, ddof=1)
            if sigma > 0:
                frap_log_like = -0.5 * np.sum((residuals / sigma)**2)
                log_like += weights['frap'] * frap_log_like
        
        # SPT likelihood (bound dwells)
        if 'bound_dwells' in spt_data and len(spt_data['bound_dwells']) > 0:
            dwells = spt_data['bound_dwells']
            k_off = params['k_off']
            
            # Exponential likelihood
            spt_log_like_bound = np.sum(
                np.log(k_off) - k_off * dwells
            )
            log_like += weights['spt'] * spt_log_like_bound
        
        # SPT likelihood (unbound dwells)
        if 'unbound_dwells' in spt_data and len(spt_data['unbound_dwells']) > 0:
            dwells = spt_data['unbound_dwells']
            k_on = params['k_on']
            concentration = spt_data.get('concentration', 1.0)
            rate = k_on * concentration
            
            spt_log_like_unbound = np.sum(
                np.log(rate) - rate * dwells
            )
            log_like += weights['spt'] * spt_log_like_unbound
        
        return log_like
    
    def demonstrate_degeneracy(
        self,
        frap_data: Dict[str, Any],
        param_to_vary: str = 'k_on'
    ) -> Dict[str, Any]:
        """
        Demonstrate parameter degeneracy in FRAP-only fitting.
        
        Shows that multiple parameter combinations can produce
        similar FRAP curves, resolved by adding SPT data.
        
        Parameters
        ----------
        frap_data : dict
            FRAP data
        param_to_vary : str
            Parameter to vary to show degeneracy
        
        Returns
        -------
        result : dict
            Demonstration of degeneracy
        """
        warnings.warn(
            "FRAP-only fits often have degenerate parameters. "
            "Multiple (D, k_on, k_off) combinations can produce similar curves. "
            "SPT data breaks this degeneracy.",
            UserWarning
        )
        
        # This is a placeholder for full implementation
        return {
            'message': 'FRAP-only parameters are often non-unique',
            'resolution': 'Add SPT dwell time data to constrain kinetics'
        }


def fit_joint_model(
    frap_data: Dict[str, Any],
    spt_data: Dict[str, Any],
    initial_guess: Dict[str, float] = None,
    weights: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Fit joint FRAP-SPT model.
    
    Parameters
    ----------
    frap_data : dict
        FRAP recovery data
    spt_data : dict
        SPT dwell time data
    initial_guess : dict, optional
        Initial parameter guess
    weights : dict, optional
        Relative weights for data types
    
    Returns
    -------
    result : dict
        Joint fit results
    """
    joint_model = JointFRAPSPT()
    
    if initial_guess is None:
        initial_guess = {
            'D': 1.0,
            'k_on': 1.0,
            'k_off': 1.0,
            'bleach_depth': 0.9
        }
    
    # Objective function
    def objective(x):
        params = {
            'D': x[0],
            'k_on': x[1],
            'k_off': x[2],
            'bleach_depth': initial_guess.get('bleach_depth', 0.9)
        }
        
        log_like = joint_model.joint_log_likelihood(
            params, frap_data, spt_data, weights
        )
        
        return -log_like  # Minimize negative log-likelihood
    
    # Bounds
    bounds = [
        (0.01, 100.0),  # D
        (0.01, 100.0),  # k_on
        (0.01, 100.0),  # k_off
    ]
    
    # Initial point
    x0 = np.array([
        initial_guess['D'],
        initial_guess['k_on'],
        initial_guess['k_off']
    ])
    
    # Optimize
    result = minimize(
        objective,
        x0,
        bounds=bounds,
        method='L-BFGS-B'
    )
    
    # Extract results
    best_params = {
        'D': result.x[0],
        'k_on': result.x[1],
        'k_off': result.x[2],
        'bleach_depth': initial_guess.get('bleach_depth', 0.9)
    }
    
    # Compute log-likelihood
    log_like = -result.fun
    
    return {
        'params': best_params,
        'log_likelihood': log_like,
        'success': result.success,
        'message': 'Joint FRAP-SPT fit complete',
        'bound_fraction': best_params['k_on'] / (best_params['k_on'] + best_params['k_off'])
    }


def compare_frap_only_vs_joint(
    frap_data: Dict[str, Any],
    spt_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare FRAP-only vs joint FRAP-SPT fits.
    
    Demonstrates how SPT resolves parameter ambiguities.
    
    Parameters
    ----------
    frap_data : dict
        FRAP data
    spt_data : dict
        SPT data
    
    Returns
    -------
    comparison : dict
        Comparison results
    """
    # This would use fit_rd for FRAP-only and fit_joint_model for joint
    # Placeholder for full implementation
    
    return {
        'message': 'Joint fitting resolves degeneracies',
        'recommendation': 'Use SPT to constrain kinetic parameters'
    }
