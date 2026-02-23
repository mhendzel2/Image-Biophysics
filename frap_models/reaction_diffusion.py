"""
Reaction-Diffusion FRAP Model

Classical RD model with binding/unbinding:
∂F/∂t = D ∇²F - k_on·F + k_off·B
∂B/∂t = k_on·F - k_off·B

Key features:
- Spatial discretization using finite differences
- Bleaching applies to both bound and free species
- Supports circular and arbitrary masks
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import warnings

from .base import FRAPModel, GeometryError, ParameterError
from .simulators import FRAPSimulator, create_circular_bleach_mask


class ReactionDiffusionModel(FRAPModel):
    """
    Classical reaction-diffusion FRAP model.
    
    Models fluorescent molecules as:
    - Free species (F): Diffuses with coefficient D
    - Bound species (B): Immobile, bound to structures
    
    Recovery depends on:
    - Diffusion of free molecules
    - Exchange between free and bound pools
    """
    
    def __init__(self, simulator: FRAPSimulator = None):
        """
        Initialize reaction-diffusion model.
        
        Parameters
        ----------
        simulator : FRAPSimulator, optional
            Numerical integration engine
        """
        super().__init__(name="ReactionDiffusionModel")
        self.simulator = simulator or FRAPSimulator(method='implicit')
    
    def simulate(
        self,
        params: Dict[str, float],
        geometry: Dict[str, Any],
        timepoints: np.ndarray
    ) -> np.ndarray:
        """
        Simulate FRAP recovery curve.
        
        Parameters
        ----------
        params : dict
            Must contain:
            - 'D': Diffusion coefficient [μm²/s]
            - 'k_on': Binding rate [1/s]
            - 'k_off': Unbinding rate [1/s]
            - 'bleach_depth': Fraction of fluorescence bleached (0-1)
        geometry : dict
            Must contain:
            - 'shape': Grid shape (ny, nx)
            - 'spacing': Grid spacing [μm]
            - 'bleach_region': dict with 'type', 'center', 'radius'
        timepoints : np.ndarray
            Time points [s] at which to evaluate recovery
        
        Returns
        -------
        recovery : np.ndarray
            Normalized fluorescence recovery (0 = fully bleached, 1 = pre-bleach)
        """
        # Validate inputs
        self._validate_params(params)
        self._validate_geometry(geometry)
        
        # Extract parameters
        D = params['D']
        k_on = params['k_on']
        k_off = params['k_off']
        bleach_depth = params.get('bleach_depth', 1.0)
        
        spacing = geometry['spacing']
        
        # Set up initial conditions
        state = self.initial_conditions(geometry)
        free_pre = state['free'].copy()
        bound_pre = state['bound'].copy()
        
        # Apply bleaching
        bleach_region = geometry['bleach_region'].copy()
        bleach_region.setdefault('spacing', spacing)
        bleached_state = self.bleach(state, bleach_region)
        free = bleached_state['free']
        bound = bleached_state['bound']
        
        # Pre-bleach fluorescence in ROI
        bleach_mask = self._create_bleach_mask(geometry)
        F_pre = np.sum((free_pre + bound_pre)[bleach_mask])
        
        # Simulate recovery at each time point
        recovery = np.zeros(len(timepoints))
        
        for i, t in enumerate(timepoints):
            if i == 0 and t == 0:
                # Immediately after bleach
                F_post = np.sum((free + bound)[bleach_mask])
                recovery[i] = F_post / F_pre if F_pre > 0 else 0
            else:
                # Integrate forward in time
                if i > 0:
                    dt_interval = timepoints[i] - timepoints[i-1]
                else:
                    dt_interval = t
                
                # Choose appropriate time step
                dt = min(0.01, dt_interval / 10)  # Adaptive
                num_steps = int(np.ceil(dt_interval / dt))
                dt = dt_interval / num_steps  # Adjust to exact interval
                
                # Integrate
                free, bound = self.simulator.integrate_reaction_diffusion_2d(
                    free, bound, D, k_on, k_off,
                    spacing, dt, num_steps
                )
                
                # Compute recovery
                F_post = np.sum((free + bound)[bleach_mask])
                recovery[i] = F_post / F_pre if F_pre > 0 else 0
        
        return recovery
    
    def initial_conditions(self, geometry: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Define pre-bleach steady state.
        
        Assumes uniform distribution at equilibrium between free and bound.
        
        Parameters
        ----------
        geometry : dict
            Spatial domain specification
        
        Returns
        -------
        state : dict
            Initial state with 'free' and 'bound' fields
        """
        shape = geometry['shape']
        
        # Equilibrium fractions (can be overridden with geometry parameters)
        total_concentration = geometry.get('total_concentration', 1.0)
        bound_fraction = geometry.get('bound_fraction', 0.5)
        
        # Uniform initial distribution
        free = np.ones(shape) * total_concentration * (1 - bound_fraction)
        bound = np.ones(shape) * total_concentration * bound_fraction
        
        return {'free': free, 'bound': bound}
    
    def bleach(
        self,
        state: Dict[str, np.ndarray],
        bleach_geometry: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """
        Apply bleaching operator.
        
        Bleaching affects both free and bound species unless they are
        physically separated during the bleach pulse.
        
        Parameters
        ----------
        state : dict
            Pre-bleach state
        bleach_geometry : dict
            Bleach region specification
        
        Returns
        -------
        bleached_state : dict
            Post-bleach state
        """
        free = state['free'].copy()
        bound = state['bound'].copy()
        
        # Create bleach mask
        if bleach_geometry['type'] == 'circular':
            shape = free.shape
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
        free[mask] *= (1 - bleach_depth)
        bound[mask] *= (1 - bleach_depth)
        
        return {'free': free, 'bound': bound}
    
    def conserved_quantities(self) -> List[str]:
        """
        Return conserved quantities.
        
        Total mass (free + bound) is conserved in this model.
        """
        return ['total_mass']
    
    def validate_parameters(self, params: Dict[str, float]) -> bool:
        """Validate parameters are physical."""
        required = ['D', 'k_on', 'k_off']
        
        for key in required:
            if key not in params:
                raise ParameterError(f"Missing required parameter: {key}")
            
            if params[key] < 0:
                raise ParameterError(f"Parameter {key} must be non-negative")
        
        # Check reasonable ranges
        if params['D'] > 100:  # μm²/s
            warnings.warn(
                f"Diffusion coefficient {params['D']:.2f} μm²/s is unusually large "
                "for a cellular protein. Typical range: 0.1-50 μm²/s.",
                UserWarning
            )
        
        return True
    
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get reasonable parameter bounds for fitting."""
        return {
            'D': (0.01, 100.0),  # μm²/s
            'k_on': (0.0, 100.0),  # 1/s
            'k_off': (0.0, 100.0),  # 1/s
            'bleach_depth': (0.5, 1.0)
        }
    
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
        elif bleach_geom['type'] == 'mask':
            return bleach_geom['mask']
        else:
            raise GeometryError(f"Unknown bleach type: {bleach_geom['type']}")
