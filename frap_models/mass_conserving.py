"""
Mass-Conserving Reaction-Diffusion Model

Two-pool MCRD model for condensates:
- Condensed phase (high concentration)
- Dilute phase (low concentration)

Strictly enforces:
∫(c_condensed + c_dilute) dV = constant

This is critical for condensate FRAP analysis where exchange
with reservoirs may not be assumed.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import warnings

from .base import FRAPModel, GeometryError, ParameterError
from .simulators import FRAPSimulator, create_circular_bleach_mask


class MassConservingRDModel(FRAPModel):
    """
    Mass-conserving reaction-diffusion model for condensate FRAP.
    
    Models:
    - Condensed phase: High local concentration, slow effective diffusion
    - Dilute phase: Low concentration, fast diffusion
    - Exchange between phases
    
    Strictly conserves total fluorescence post-bleach.
    """
    
    def __init__(self, simulator: FRAPSimulator = None):
        """
        Initialize mass-conserving RD model.
        
        Parameters
        ----------
        simulator : FRAPSimulator, optional
            Numerical integration engine
        """
        super().__init__(name="MassConservingRDModel")
        self.simulator = simulator or FRAPSimulator(method='implicit')
        self._total_mass_prebleach = None
        self._total_mass_postbleach = None
    
    def simulate(
        self,
        params: Dict[str, float],
        geometry: Dict[str, Any],
        timepoints: np.ndarray
    ) -> np.ndarray:
        """
        Simulate mass-conserving FRAP recovery.
        
        Parameters
        ----------
        params : dict
            Must contain:
            - 'D_dilute': Diffusion in dilute phase [μm²/s]
            - 'D_condensed': Diffusion in condensed phase [μm²/s]
            - 'k_in': Condensation rate [1/s]
            - 'k_out': Dissolution rate [1/s]
            - 'condensed_fraction': Fraction of molecules in condensed phase
        geometry : dict
            Spatial domain and bleach specification
        timepoints : np.ndarray
            Time points for recovery evaluation
        
        Returns
        -------
        recovery : np.ndarray
            Normalized recovery curve
        """
        self._validate_params(params)
        self._validate_geometry(geometry)
        
        # Extract parameters
        D_dilute = params['D_dilute']
        D_condensed = params['D_condensed']
        k_in = params['k_in']
        k_out = params['k_out']
        bleach_depth = params.get('bleach_depth', 1.0)
        
        spacing = geometry['spacing']
        
        # Initial conditions
        state = self.initial_conditions(geometry)
        condensed_pre = state['condensed'].copy()
        dilute_pre = state['dilute'].copy()
        
        # Record pre-bleach mass
        self._total_mass_prebleach = np.sum(condensed_pre + dilute_pre)
        
        # Apply bleaching
        bleach_region = geometry['bleach_region'].copy()
        bleach_region.setdefault('spacing', spacing)
        bleached_state = self.bleach(state, bleach_region)
        condensed = bleached_state['condensed']
        dilute = bleached_state['dilute']
        
        # Record post-bleach mass (should be less due to bleaching)
        self._total_mass_postbleach = np.sum(condensed + dilute)
        
        # Bleach mask for recovery measurement
        bleach_mask = self._create_bleach_mask(geometry)
        F_pre = np.sum((condensed_pre + dilute_pre)[bleach_mask])
        
        # Simulate recovery
        recovery = np.zeros(len(timepoints))
        
        for i, t in enumerate(timepoints):
            if i == 0 and t == 0:
                F_post = np.sum((condensed + dilute)[bleach_mask])
                recovery[i] = F_post / F_pre if F_pre > 0 else 0
            else:
                # Time interval
                if i > 0:
                    dt_interval = timepoints[i] - timepoints[i-1]
                else:
                    dt_interval = t
                
                dt = min(0.01, dt_interval / 10)
                num_steps = int(np.ceil(dt_interval / dt))
                dt = dt_interval / num_steps
                
                # Integrate with phase exchange
                condensed, dilute = self._integrate_two_phase(
                    condensed, dilute,
                    D_condensed, D_dilute,
                    k_in, k_out,
                    spacing, dt, num_steps
                )
                
                # Check mass conservation
                total_mass = np.sum(condensed + dilute)
                if not np.isclose(total_mass, self._total_mass_postbleach, rtol=1e-4):
                    warnings.warn(
                        f"Mass conservation violated at t={t:.2f}s: "
                        f"expected {self._total_mass_postbleach:.6f}, "
                        f"got {total_mass:.6f}",
                        UserWarning
                    )
                
                # Compute recovery
                F_post = np.sum((condensed + dilute)[bleach_mask])
                recovery[i] = F_post / F_pre if F_pre > 0 else 0
        
        return recovery
    
    def initial_conditions(self, geometry: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Define pre-bleach steady state.
        
        Distributes molecules between condensed and dilute phases
        according to specified fraction.
        """
        shape = geometry['shape']
        total_concentration = geometry.get('total_concentration', 1.0)
        condensed_fraction = geometry.get('condensed_fraction', 0.7)
        
        # Define condensate region (simplified - uniform for now)
        # In practice, this could be defined by geometry['condensate_mask']
        condensed = np.ones(shape) * total_concentration * condensed_fraction
        dilute = np.ones(shape) * total_concentration * (1 - condensed_fraction)
        
        return {'condensed': condensed, 'dilute': dilute}
    
    def bleach(
        self,
        state: Dict[str, np.ndarray],
        bleach_geometry: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """Apply bleaching to both phases proportionally."""
        condensed = state['condensed'].copy()
        dilute = state['dilute'].copy()
        
        # Create bleach mask
        if bleach_geometry['type'] == 'circular':
            shape = condensed.shape
            center = bleach_geometry['center']
            radius = bleach_geometry['radius']
            spacing = bleach_geometry.get('spacing', None)
            if spacing is None:
                # Infer grid spacing from state shape and center coordinates.
                # Assumes the center is at the midpoint of the domain, i.e.
                # center[i] = (shape[i] / 2) * spacing  =>  spacing = 2*center[i]/shape[i].
                # This is only valid when center coordinates are in physical units and
                # the centre lies near the grid midpoint.  Pass 'spacing' explicitly
                # in bleach_geometry to avoid relying on this heuristic.
                max_center = max(center)
                if max_center <= 0:
                    raise ValueError(
                        "Cannot infer grid spacing: bleach center must have positive "
                        "coordinates (in physical units). Pass 'spacing' explicitly in "
                        "bleach_geometry."
                    )
                spacing = 2.0 * max_center / max(shape)
            mask = create_circular_bleach_mask(shape, center, radius, spacing)
        elif bleach_geometry['type'] == 'mask':
            mask = bleach_geometry['mask']
        else:
            raise GeometryError(f"Unknown bleach type: {bleach_geometry['type']}")
        
        # Apply bleaching
        bleach_depth = bleach_geometry.get('bleach_depth', 1.0)
        condensed[mask] *= (1 - bleach_depth)
        dilute[mask] *= (1 - bleach_depth)
        
        return {'condensed': condensed, 'dilute': dilute}
    
    def conserved_quantities(self) -> List[str]:
        """
        Return conserved quantities.
        
        Total mass is strictly conserved post-bleach.
        """
        return ['total_mass']
    
    def validate_parameters(self, params: Dict[str, float]) -> bool:
        """Validate parameters."""
        required = ['D_dilute', 'D_condensed', 'k_in', 'k_out']
        
        for key in required:
            if key not in params:
                raise ParameterError(f"Missing required parameter: {key}")
            if params[key] < 0:
                raise ParameterError(f"Parameter {key} must be non-negative")
        
        # Physical constraints
        if params['D_condensed'] > params['D_dilute']:
            warnings.warn(
                "Condensed phase diffusion is faster than dilute phase. "
                "This is physically unusual for condensates.",
                UserWarning
            )
        
        return True
    
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get reasonable parameter bounds."""
        return {
            'D_dilute': (0.1, 50.0),  # μm²/s
            'D_condensed': (0.001, 5.0),  # μm²/s (slower in condensate)
            'k_in': (0.0, 50.0),  # 1/s
            'k_out': (0.0, 50.0),  # 1/s
            'condensed_fraction': (0.1, 0.9)
        }
    
    def _integrate_two_phase(
        self,
        condensed: np.ndarray,
        dilute: np.ndarray,
        D_condensed: float,
        D_dilute: float,
        k_in: float,
        k_out: float,
        spacing: float,
        dt: float,
        num_steps: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate two-phase system with exchange.
        
        Uses operator splitting:
        1. Exchange between phases
        2. Diffusion in each phase
        """
        for _ in range(num_steps):
            # Exchange step (reaction)
            condensed_new = condensed + dt * (k_in * dilute - k_out * condensed)
            dilute_new = dilute + dt * (k_out * condensed - k_in * dilute)
            
            # Ensure non-negative
            condensed_new = np.maximum(condensed_new, 0)
            dilute_new = np.maximum(dilute_new, 0)
            
            # Diffusion step
            condensed_new = self.simulator.integrate_diffusion_2d(
                condensed_new, D_condensed, spacing, dt, 1
            )
            dilute_new = self.simulator.integrate_diffusion_2d(
                dilute_new, D_dilute, spacing, dt, 1
            )
            
            condensed = condensed_new
            dilute = dilute_new
        
        return condensed, dilute
    
    def _validate_params(self, params: Dict[str, float]):
        """Internal parameter validation."""
        self.validate_parameters(params)
    
    def _validate_geometry(self, geometry: Dict[str, Any]):
        """Internal geometry validation."""
        required = ['shape', 'spacing', 'bleach_region']
        for key in required:
            if key not in geometry:
                raise GeometryError(f"Missing required geometry field: {key}")
    
    def _create_bleach_mask(self, geometry: Dict[str, Any]) -> np.ndarray:
        """Create bleach mask from geometry."""
        bleach_geom = geometry['bleach_region'].copy()
        bleach_geom['spacing'] = geometry['spacing']
        
        if bleach_geom['type'] == 'circular':
            return create_circular_bleach_mask(
                geometry['shape'],
                bleach_geom['center'],
                bleach_geom['radius'],
                geometry['spacing']
            )
        else:
            return bleach_geom['mask']
