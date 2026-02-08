"""
Base FRAP Model Interface

Defines strict abstract interface for all FRAP models to ensure:
- Deterministic simulation given parameters
- Explicit mass conservation tracking
- Standardized bleaching operations
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import warnings


class FRAPModel(ABC):
    """
    Abstract base class for FRAP models.
    
    All FRAP models must implement this interface to ensure:
    1. Deterministic simulation
    2. Explicit initial conditions
    3. Standardized bleaching
    4. Mass conservation tracking
    """
    
    def __init__(self, name: str = "FRAPModel"):
        """
        Initialize FRAP model.
        
        Parameters
        ----------
        name : str
            Model name for identification
        """
        self.name = name
        self._check_conservation()
    
    @abstractmethod
    def simulate(
        self,
        params: Dict[str, float],
        geometry: Dict[str, Any],
        timepoints: np.ndarray
    ) -> np.ndarray:
        """
        Simulate fluorescence recovery curve.
        
        Must be deterministic given params. No stochastic simulation by default.
        
        Parameters
        ----------
        params : dict
            Model parameters (e.g., D, k_on, k_off)
        geometry : dict
            Spatial domain and bleach region specification
            Should contain: 'shape', 'spacing', 'bleach_region'
        timepoints : np.ndarray
            Time points at which to evaluate recovery
        
        Returns
        -------
        recovery : np.ndarray
            Simulated fluorescence recovery curve, normalized to pre-bleach
        """
        pass
    
    @abstractmethod
    def initial_conditions(self, geometry: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Define pre-bleach steady state.
        
        Parameters
        ----------
        geometry : dict
            Spatial domain specification
        
        Returns
        -------
        state : dict
            Initial state fields (e.g., {'free': array, 'bound': array})
        """
        pass
    
    @abstractmethod
    def bleach(
        self,
        state: Dict[str, np.ndarray],
        bleach_geometry: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """
        Apply bleaching operator to state.
        
        Bleaching typically affects both bound and free species unless
        explicitly justified otherwise (e.g., for very fast binding).
        
        Parameters
        ----------
        state : dict
            Pre-bleach state fields
        bleach_geometry : dict
            Bleach region specification (e.g., radius, center, mask)
        
        Returns
        -------
        bleached_state : dict
            Post-bleach state fields
        """
        pass
    
    @abstractmethod
    def conserved_quantities(self) -> List[str]:
        """
        Return list of conserved fields.
        
        If model violates conservation, return empty list and log warning.
        
        Returns
        -------
        conserved : list of str
            Names of conserved quantities (e.g., ['total_mass'])
        """
        pass
    
    def _check_conservation(self):
        """Check and warn if model violates mass conservation."""
        conserved = self.conserved_quantities()
        if not conserved:
            warnings.warn(
                f"{self.name}: This model does not explicitly conserve mass. "
                "This should be treated as an exchange-with-reservoir model. "
                "Ensure this assumption is justified for your experimental system.",
                UserWarning
            )
    
    def validate_parameters(self, params: Dict[str, float]) -> bool:
        """
        Validate parameter values are physically reasonable.
        
        Parameters
        ----------
        params : dict
            Model parameters to validate
        
        Returns
        -------
        valid : bool
            True if parameters are valid
        """
        # Default implementation - override in subclasses
        return True
    
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Get reasonable bounds for model parameters.
        
        Returns
        -------
        bounds : dict
            Parameter bounds as {param_name: (lower, upper)}
        """
        # Default implementation - override in subclasses
        return {}


class GeometryError(Exception):
    """Raised when geometry specification is invalid."""
    pass


class ParameterError(Exception):
    """Raised when parameters are invalid or unphysical."""
    pass
