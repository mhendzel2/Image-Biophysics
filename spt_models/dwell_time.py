"""
Dwell Time Analysis for SPT

Analyze dwell times of bound and unbound states to constrain
kinetic parameters that appear in FRAP models.
"""

import numpy as np
from scipy.optimize import minimize
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import warnings


class DwellTimeModel:
    """
    Dwell time distribution model for SPT.
    
    Assumes exponential dwell times:
    - Bound state: P(t) = k_off * exp(-k_off * t)
    - Unbound state: P(t) = k_on * C * exp(-k_on * C * t)
    
    where C is concentration of binding sites.
    """
    
    def __init__(self):
        """Initialize dwell time model."""
        self.params = {}
    
    def bound_dwell_pdf(self, times: np.ndarray, k_off: float) -> np.ndarray:
        """
        PDF of bound state dwell times.
        
        Parameters
        ----------
        times : np.ndarray
            Dwell times [s]
        k_off : float
            Unbinding rate [1/s]
        
        Returns
        -------
        pdf : np.ndarray
            Probability density
        """
        return k_off * np.exp(-k_off * times)
    
    def unbound_dwell_pdf(
        self,
        times: np.ndarray,
        k_on: float,
        concentration: float = 1.0
    ) -> np.ndarray:
        """
        PDF of unbound state dwell times.
        
        Parameters
        ----------
        times : np.ndarray
            Dwell times [s]
        k_on : float
            Binding rate [1/s]
        concentration : float
            Effective concentration of binding sites
        
        Returns
        -------
        pdf : np.ndarray
            Probability density
        """
        rate = k_on * concentration
        return rate * np.exp(-rate * times)
    
    def fit_bound_dwells(
        self,
        dwell_times: np.ndarray
    ) -> Dict[str, float]:
        """
        Fit k_off from bound state dwell times.
        
        Parameters
        ----------
        dwell_times : np.ndarray
            Observed bound dwell times [s]
        
        Returns
        -------
        result : dict
            Fit results with k_off and uncertainty
        """
        # Maximum likelihood estimate for exponential
        k_off_mle = 1.0 / np.mean(dwell_times)
        
        # Standard error
        se = k_off_mle / np.sqrt(len(dwell_times))
        
        # 95% confidence interval
        ci_lower = k_off_mle - 1.96 * se
        ci_upper = k_off_mle + 1.96 * se
        
        return {
            'k_off': k_off_mle,
            'k_off_se': se,
            'k_off_ci': (max(0, ci_lower), ci_upper),
            'n_dwells': len(dwell_times)
        }
    
    def fit_unbound_dwells(
        self,
        dwell_times: np.ndarray,
        concentration: float = 1.0
    ) -> Dict[str, float]:
        """
        Fit k_on from unbound state dwell times.
        
        Parameters
        ----------
        dwell_times : np.ndarray
            Observed unbound dwell times [s]
        concentration : float
            Effective concentration
        
        Returns
        -------
        result : dict
            Fit results
        """
        # MLE for exponential
        rate_mle = 1.0 / np.mean(dwell_times)
        k_on_mle = rate_mle / concentration
        
        se = k_on_mle / np.sqrt(len(dwell_times))
        ci_lower = k_on_mle - 1.96 * se
        ci_upper = k_on_mle + 1.96 * se
        
        return {
            'k_on': k_on_mle,
            'k_on_se': se,
            'k_on_ci': (max(0, ci_lower), ci_upper),
            'n_dwells': len(dwell_times)
        }
    
    def compute_bound_fraction(
        self,
        k_on: float,
        k_off: float
    ) -> float:
        """
        Compute equilibrium bound fraction.
        
        f_bound = k_on / (k_on + k_off)
        
        Parameters
        ----------
        k_on : float
            Binding rate
        k_off : float
            Unbinding rate
        
        Returns
        -------
        bound_fraction : float
            Fraction bound at equilibrium
        """
        return k_on / (k_on + k_off)


def fit_dwell_times(
    bound_dwells: np.ndarray,
    unbound_dwells: np.ndarray = None,
    concentration: float = 1.0
) -> Dict[str, Any]:
    """
    Fit kinetic parameters from SPT dwell times.
    
    Parameters
    ----------
    bound_dwells : np.ndarray
        Bound state dwell times [s]
    unbound_dwells : np.ndarray, optional
        Unbound state dwell times [s]
    concentration : float
        Effective binding site concentration
    
    Returns
    -------
    results : dict
        Fitted kinetic parameters
    """
    model = DwellTimeModel()
    
    results = {}
    
    # Fit bound dwells
    if bound_dwells is not None and len(bound_dwells) > 0:
        bound_result = model.fit_bound_dwells(bound_dwells)
        results.update(bound_result)
    
    # Fit unbound dwells
    if unbound_dwells is not None and len(unbound_dwells) > 0:
        unbound_result = model.fit_unbound_dwells(unbound_dwells, concentration)
        results.update(unbound_result)
    
    # Compute bound fraction if both available
    if 'k_on' in results and 'k_off' in results:
        bound_frac = model.compute_bound_fraction(
            results['k_on'],
            results['k_off']
        )
        results['bound_fraction'] = bound_frac
    
    return results


def assess_exponential_assumption(
    dwell_times: np.ndarray,
    state: str = 'bound'
) -> Dict[str, Any]:
    """
    Test if dwell times are exponentially distributed.
    
    Uses Kolmogorov-Smirnov test.
    
    Parameters
    ----------
    dwell_times : np.ndarray
        Observed dwell times
    state : str
        'bound' or 'unbound'
    
    Returns
    -------
    result : dict
        Test results
    """
    # Fit exponential
    rate_mle = 1.0 / np.mean(dwell_times)
    
    # K-S test
    ks_stat, p_value = stats.kstest(
        dwell_times,
        lambda x: stats.expon.cdf(x, scale=1/rate_mle)
    )
    
    # Interpret
    is_exponential = p_value > 0.05
    
    if not is_exponential:
        warnings.warn(
            f"{state} dwell times deviate from exponential (p={p_value:.4f}). "
            "Consider multi-state binding or correlated unbinding events.",
            UserWarning
        )
    
    return {
        'ks_statistic': ks_stat,
        'p_value': p_value,
        'is_exponential': is_exponential,
        'rate_mle': rate_mle
    }
