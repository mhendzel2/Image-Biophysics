"""
Likelihood Functions for FRAP Fitting

Implements proper statistical models for FRAP data:
- Gaussian likelihood (high photon counts)
- Poisson likelihood (photon-limited regime)
- Information criteria (AIC, BIC)
"""

import numpy as np
from typing import Dict, Any, Callable
from scipy import stats


def gaussian_log_likelihood(
    observed: np.ndarray,
    predicted: np.ndarray,
    sigma: float = None
) -> float:
    """
    Gaussian log-likelihood for FRAP data.
    
    Appropriate when:
    - Photon counts are high (>100 per time point)
    - Noise is approximately Gaussian (Central Limit Theorem)
    
    Parameters
    ----------
    observed : np.ndarray
        Observed fluorescence recovery
    predicted : np.ndarray
        Model-predicted recovery
    sigma : float, optional
        Noise standard deviation. If None, estimated from residuals.
    
    Returns
    -------
    log_likelihood : float
        Log-likelihood value
    """
    residuals = observed - predicted
    
    if sigma is None:
        # Estimate from data (maximum likelihood estimate)
        sigma = np.std(residuals, ddof=1)
    
    # Prevent numerical issues
    if sigma <= 0:
        sigma = 1e-10
    
    # Gaussian log-likelihood
    n = len(observed)
    log_like = -0.5 * n * np.log(2 * np.pi * sigma**2)
    log_like -= 0.5 * np.sum((residuals / sigma)**2)
    
    return log_like


def poisson_log_likelihood(
    observed_counts: np.ndarray,
    predicted_counts: np.ndarray
) -> float:
    """
    Poisson log-likelihood for photon-limited FRAP.
    
    Appropriate when:
    - Low photon counts (<100 per time point)
    - Photon shot noise dominates
    
    Parameters
    ----------
    observed_counts : np.ndarray
        Observed photon counts (must be integers or counts)
    predicted_counts : np.ndarray
        Model-predicted counts (continuous, will be used as λ)
    
    Returns
    -------
    log_likelihood : float
        Poisson log-likelihood
    """
    # Ensure positive predictions
    predicted_counts = np.maximum(predicted_counts, 1e-10)
    
    # Poisson log-likelihood: log P(k|λ) = k·log(λ) - λ - log(k!)
    # Sum over all observations
    log_like = np.sum(
        observed_counts * np.log(predicted_counts) -
        predicted_counts -
        stats.gammaln(observed_counts + 1)  # log(k!)
    )
    
    return log_like


def weighted_least_squares(
    observed: np.ndarray,
    predicted: np.ndarray,
    weights: np.ndarray = None
) -> float:
    """
    Weighted least squares objective.
    
    Use only for comparison with published results that use least squares.
    Prefer likelihood-based methods for inference.
    
    Parameters
    ----------
    observed : np.ndarray
        Observed data
    predicted : np.ndarray
        Predicted data
    weights : np.ndarray, optional
        Weights for each point (default: uniform)
    
    Returns
    -------
    chi_squared : float
        Weighted sum of squared residuals
    """
    residuals = observed - predicted
    
    if weights is None:
        weights = np.ones_like(observed)
    
    chi_squared = np.sum(weights * residuals**2)
    
    return chi_squared


def compute_aic(log_likelihood: float, n_params: int) -> float:
    """
    Compute Akaike Information Criterion.
    
    AIC = 2k - 2·ln(L)
    
    Lower AIC indicates better model considering complexity.
    
    Parameters
    ----------
    log_likelihood : float
        Maximum log-likelihood
    n_params : int
        Number of model parameters
    
    Returns
    -------
    aic : float
        Akaike Information Criterion
    """
    return 2 * n_params - 2 * log_likelihood


def compute_aicc(log_likelihood: float, n_params: int, n_data: int) -> float:
    """
    Compute corrected AIC for small sample sizes.
    
    AICc = AIC + 2k(k+1)/(n-k-1)
    
    Use when n_data / n_params < 40.
    """
    aic = compute_aic(log_likelihood, n_params)
    
    if n_data - n_params - 1 > 0:
        correction = 2 * n_params * (n_params + 1) / (n_data - n_params - 1)
    else:
        correction = np.inf
    
    return aic + correction


def compute_bic(log_likelihood: float, n_params: int, n_data: int) -> float:
    """
    Compute Bayesian Information Criterion.
    
    BIC = k·ln(n) - 2·ln(L)
    
    Lower BIC indicates better model. More conservative than AIC.
    
    Parameters
    ----------
    log_likelihood : float
        Maximum log-likelihood
    n_params : int
        Number of parameters
    n_data : int
        Number of data points
    
    Returns
    -------
    bic : float
        Bayesian Information Criterion
    """
    return n_params * np.log(n_data) - 2 * log_likelihood


def compute_evidence_ratio(aic1: float, aic2: float) -> float:
    """
    Compute evidence ratio between two models.
    
    Evidence ratio ≈ exp((AIC_min - AIC_i) / 2)
    
    Interpretation:
    - > 10: Strong evidence for model 1
    - 3-10: Moderate evidence for model 1
    - 1-3: Weak evidence
    
    Parameters
    ----------
    aic1 : float
        AIC of model 1 (should be lower)
    aic2 : float
        AIC of model 2
    
    Returns
    -------
    evidence_ratio : float
        Relative evidence for model 1 vs model 2
    """
    delta_aic = aic2 - aic1
    return np.exp(delta_aic / 2)


class LikelihoodFunction:
    """
    Wrapper for likelihood evaluation with FRAP models.
    """
    
    def __init__(
        self,
        model,
        observed_data: np.ndarray,
        timepoints: np.ndarray,
        geometry: Dict[str, Any],
        likelihood_type: str = 'gaussian'
    ):
        """
        Initialize likelihood function.
        
        Parameters
        ----------
        model : FRAPModel
            FRAP model instance
        observed_data : np.ndarray
            Observed recovery curve
        timepoints : np.ndarray
            Time points
        geometry : dict
            Geometry specification
        likelihood_type : str
            'gaussian' or 'poisson'
        """
        self.model = model
        self.observed = observed_data
        self.timepoints = timepoints
        self.geometry = geometry
        self.likelihood_type = likelihood_type
    
    def __call__(self, params: Dict[str, float]) -> float:
        """
        Evaluate negative log-likelihood (for minimization).
        
        Parameters
        ----------
        params : dict
            Model parameters
        
        Returns
        -------
        neg_log_like : float
            Negative log-likelihood
        """
        try:
            # Simulate model
            predicted = self.model.simulate(params, self.geometry, self.timepoints)
            
            # Compute likelihood
            if self.likelihood_type == 'gaussian':
                log_like = gaussian_log_likelihood(self.observed, predicted)
            elif self.likelihood_type == 'poisson':
                # Convert to counts (assume observed is normalized)
                # This is a simplification - in practice, use actual counts
                scale = 1000  # Typical photon count scale
                obs_counts = self.observed * scale
                pred_counts = predicted * scale
                log_like = poisson_log_likelihood(obs_counts, pred_counts)
            else:
                raise ValueError(f"Unknown likelihood type: {self.likelihood_type}")
            
            return -log_like  # Return negative for minimization
            
        except Exception as e:
            # Return large penalty for invalid parameters
            return 1e10
