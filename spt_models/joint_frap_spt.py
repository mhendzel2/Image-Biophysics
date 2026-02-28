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
        weights: Dict[str, float] = None,
        normalize_by_observation_count: bool = True
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
        details = self._joint_log_likelihood_components(
            params=params,
            frap_data=frap_data,
            spt_data=spt_data,
            weights=weights,
            normalize_by_observation_count=normalize_by_observation_count,
        )
        return float(details['joint_log_likelihood'])

    def _joint_log_likelihood_components(
        self,
        params: Dict[str, float],
        frap_data: Dict[str, Any],
        spt_data: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None,
        normalize_by_observation_count: bool = True
    ) -> Dict[str, Any]:
        """
        Compute modality-wise and joint likelihood contributions.

        By default, each modality is normalized by its observation count
        (mean log-likelihood per observation) before weighting, which prevents
        high-volume modalities from dominating solely due to sample size.
        """
        if weights is None:
            weights = {'frap': 1.0, 'spt': 1.0}
        w_frap = float(weights.get('frap', 1.0))
        w_spt = float(weights.get('spt', 1.0))

        frap_raw = 0.0
        frap_n = 0
        frap_norm = 0.0

        spt_raw = 0.0
        spt_n = 0
        spt_norm = 0.0
        spt_log_like_bound = 0.0
        spt_log_like_unbound = 0.0

        # FRAP likelihood
        if 'recovery' in frap_data:
            frap_pred = self.frap_model.simulate(
                params,
                frap_data['geometry'],
                frap_data['timepoints']
            )

            residuals = np.asarray(frap_data['recovery']) - np.asarray(frap_pred)
            frap_n = int(np.size(residuals))
            sigma = np.std(residuals, ddof=1) if frap_n > 1 else np.std(residuals)
            if sigma > 0 and np.isfinite(sigma):
                frap_raw = float(-0.5 * np.sum((residuals / sigma) ** 2))
            else:
                frap_raw = 0.0

        if normalize_by_observation_count:
            frap_norm = frap_raw / max(frap_n, 1)
        else:
            frap_norm = frap_raw

        # SPT likelihood (bound dwells)
        if 'bound_dwells' in spt_data and len(spt_data['bound_dwells']) > 0:
            dwells = np.asarray(spt_data['bound_dwells'], dtype=float)
            k_off = max(float(params['k_off']), 1e-12)
            spt_log_like_bound = float(np.sum(np.log(k_off) - k_off * dwells))
            spt_raw += spt_log_like_bound
            spt_n += int(dwells.size)

        # SPT likelihood (unbound dwells)
        if 'unbound_dwells' in spt_data and len(spt_data['unbound_dwells']) > 0:
            dwells = np.asarray(spt_data['unbound_dwells'], dtype=float)
            k_on = max(float(params['k_on']), 1e-12)
            concentration = max(float(spt_data.get('concentration', 1.0)), 1e-12)
            rate = max(k_on * concentration, 1e-12)

            spt_log_like_unbound = float(np.sum(np.log(rate) - rate * dwells))
            spt_raw += spt_log_like_unbound
            spt_n += int(dwells.size)

        if normalize_by_observation_count:
            spt_norm = spt_raw / max(spt_n, 1)
        else:
            spt_norm = spt_raw

        joint_log_like = w_frap * frap_norm + w_spt * spt_norm

        return {
            'joint_log_likelihood': float(joint_log_like),
            'weights': {'frap': w_frap, 'spt': w_spt},
            'normalize_by_observation_count': bool(normalize_by_observation_count),
            'frap': {
                'raw_log_likelihood': float(frap_raw),
                'normalized_log_likelihood': float(frap_norm),
                'n_observations': int(frap_n),
            },
            'spt': {
                'raw_log_likelihood': float(spt_raw),
                'normalized_log_likelihood': float(spt_norm),
                'n_observations': int(spt_n),
                'bound_log_likelihood': float(spt_log_like_bound),
                'unbound_log_likelihood': float(spt_log_like_unbound),
            },
        }
    
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
    weights: Dict[str, float] = None,
    normalize_by_observation_count: bool = True
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
            params,
            frap_data,
            spt_data,
            weights=weights,
            normalize_by_observation_count=normalize_by_observation_count
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
    
    # Compute log-likelihood with component diagnostics
    details = joint_model._joint_log_likelihood_components(
        params=best_params,
        frap_data=frap_data,
        spt_data=spt_data,
        weights=weights,
        normalize_by_observation_count=normalize_by_observation_count,
    )
    log_like = float(details['joint_log_likelihood'])
    
    return {
        'params': best_params,
        'log_likelihood': log_like,
        'success': result.success,
        'message': 'Joint FRAP-SPT fit complete',
        'bound_fraction': best_params['k_on'] / (best_params['k_on'] + best_params['k_off']),
        'normalize_by_observation_count': bool(normalize_by_observation_count),
        'likelihood_components': details
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
