"""
FRAP Fitting Module

Likelihood-based fitting for mechanistic FRAP models.

No least-squares shortcuts. Uses proper likelihood functions:
- Gaussian noise (justified for high photon counts)
- Poisson noise (preferred for photon-limited regimes)
"""

from .likelihoods import (
    gaussian_log_likelihood,
    poisson_log_likelihood,
    compute_aic,
    compute_bic
)
from .fit_rd import fit_reaction_diffusion
from .fit_mcrd import fit_mass_conserving_rd
from .model_selection import ModelSelection, compare_models

__all__ = [
    'gaussian_log_likelihood',
    'poisson_log_likelihood',
    'compute_aic',
    'compute_bic',
    'fit_reaction_diffusion',
    'fit_mass_conserving_rd',
    'ModelSelection',
    'compare_models',
]
