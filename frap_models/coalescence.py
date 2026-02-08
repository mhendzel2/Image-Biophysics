"""
Coalescence and Exchange Models

Optional module for condensate FRAP analysis.

WARNING: Coalescence terms are weakly constrained by FRAP alone.
Use these models with caution and corroborate with other techniques (SPT, FCS).

Models:
- Exchange-limited recovery (reaction-dominated)
- Diffusion-limited recovery (geometry-dominated)
- Coalescence/fusion of condensates
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import warnings

from .base import FRAPModel, ParameterError
from .mass_conserving import MassConservingRDModel


class CoalescenceModel(MassConservingRDModel):
    """
    Condensate coalescence model (EXPERIMENTAL).
    
    WARNING: This model includes coalescence terms that are weakly
    constrained by FRAP data alone. Parameters may not be identifiable.
    
    Use only when:
    1. You have complementary data (SPT, particle tracking)
    2. Coalescence is directly observable
    3. You need to test specific hypotheses about fusion kinetics
    
    DO NOT use to "explain away" poor fits.
    """
    
    def __init__(self, simulator=None, enable_coalescence: bool = False):
        """
        Initialize coalescence model.
        
        Parameters
        ----------
        simulator : FRAPSimulator, optional
            Numerical integration engine
        enable_coalescence : bool
            Whether to enable coalescence terms (default: False)
            Must be explicitly enabled to use coalescence
        """
        super().__init__(simulator=simulator)
        self.name = "CoalescenceModel"
        self.enable_coalescence = enable_coalescence
        
        if not enable_coalescence:
            warnings.warn(
                "CoalescenceModel instantiated but coalescence is DISABLED. "
                "This model will behave like MassConservingRDModel. "
                "Set enable_coalescence=True to enable coalescence terms.",
                UserWarning
            )
        else:
            warnings.warn(
                "Coalescence terms are ENABLED. These terms are weakly constrained "
                "by FRAP alone and may lead to non-identifiable parameters. "
                "Ensure you have complementary data to support this choice.",
                UserWarning
            )
    
    def simulate(
        self,
        params: Dict[str, float],
        geometry: Dict[str, Any],
        timepoints: np.ndarray
    ) -> np.ndarray:
        """
        Simulate FRAP with optional coalescence.
        
        Additional parameters (only used if enable_coalescence=True):
        - 'k_fuse': Fusion/coalescence rate [1/s]
        - 'fusion_radius': Spatial range of fusion [μm]
        
        If coalescence is disabled, behaves like MassConservingRDModel.
        """
        if not self.enable_coalescence:
            # Fall back to parent implementation
            return super().simulate(params, geometry, timepoints)
        
        # Validate coalescence-specific parameters
        if 'k_fuse' not in params:
            raise ParameterError("k_fuse required when coalescence is enabled")
        
        # For now, delegate to parent and add warning
        # Full coalescence implementation would require spatial kernel convolution
        warnings.warn(
            "Full coalescence dynamics not yet implemented. "
            "Using mass-conserving RD as approximation.",
            UserWarning
        )
        
        return super().simulate(params, geometry, timepoints)
    
    def validate_parameters(self, params: Dict[str, float]) -> bool:
        """Validate parameters including coalescence terms."""
        # Check parent parameters
        super().validate_parameters(params)
        
        if self.enable_coalescence:
            if 'k_fuse' in params and params['k_fuse'] < 0:
                raise ParameterError("k_fuse must be non-negative")
        
        return True
    
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds including coalescence."""
        bounds = super().get_parameter_bounds()
        
        if self.enable_coalescence:
            bounds.update({
                'k_fuse': (0.0, 10.0),  # 1/s
                'fusion_radius': (0.1, 5.0)  # μm
            })
        
        return bounds


def is_exchange_limited(
    params: Dict[str, float],
    geometry: Dict[str, Any]
) -> bool:
    """
    Determine if recovery is exchange-limited vs diffusion-limited.
    
    Parameters
    ----------
    params : dict
        Model parameters including D, k_on, k_off
    geometry : dict
        Including bleach radius
    
    Returns
    -------
    exchange_limited : bool
        True if recovery is dominated by binding kinetics
    """
    if 'k_on' not in params or 'k_off' not in params:
        return False
    
    if 'D' not in params and 'D_dilute' not in params:
        return False
    
    D = params.get('D', params.get('D_dilute', 1.0))
    k_on = params['k_on']
    k_off = params['k_off']
    
    # Get bleach radius
    bleach_region = geometry.get('bleach_region', {})
    radius = bleach_region.get('radius', 1.0)
    
    # Characteristic times
    tau_diff = radius ** 2 / D  # Diffusion time
    tau_exchange = 1 / (k_on + k_off)  # Exchange time
    
    # If exchange is much slower than diffusion, it's exchange-limited
    return tau_exchange > 3 * tau_diff


def is_diffusion_limited(
    params: Dict[str, float],
    geometry: Dict[str, Any]
) -> bool:
    """
    Determine if recovery is diffusion-limited.
    
    Parameters
    ----------
    params : dict
        Model parameters
    geometry : dict
        Including bleach radius
    
    Returns
    -------
    diffusion_limited : bool
        True if recovery is dominated by diffusion
    """
    return not is_exchange_limited(params, geometry)


def estimate_recovery_regime(
    params: Dict[str, float],
    geometry: Dict[str, Any]
) -> str:
    """
    Estimate the dominant recovery regime.
    
    Returns
    -------
    regime : str
        One of: 'diffusion_limited', 'exchange_limited', 'mixed'
    """
    if 'k_on' not in params or 'D' not in params:
        return 'unknown'
    
    D = params.get('D', params.get('D_dilute', 1.0))
    k_on = params['k_on']
    k_off = params['k_off']
    
    bleach_region = geometry.get('bleach_region', {})
    radius = bleach_region.get('radius', 1.0)
    
    tau_diff = radius ** 2 / D
    tau_exchange = 1 / (k_on + k_off)
    
    ratio = tau_exchange / tau_diff
    
    if ratio > 3:
        return 'exchange_limited'
    elif ratio < 0.3:
        return 'diffusion_limited'
    else:
        return 'mixed'
