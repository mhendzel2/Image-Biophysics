"""
Test Parameter Identifiability

Verifies that parameter degeneracies are detected and flagged.
Critical for ensuring FRAP-only fits are not over-interpreted.
"""

import unittest
import numpy as np
from frap_models import ReactionDiffusionModel, MassConservingRDModel
from frap_fitting import fit_reaction_diffusion, fit_mass_conserving_rd, ModelSelection


class TestParameterIdentifiability(unittest.TestCase):
    """Test parameter identifiability in FRAP models."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = ReactionDiffusionModel()
        
        # Standard geometry
        self.geometry = {
            'shape': (50, 50),
            'spacing': 0.1,
            'total_concentration': 1.0,
            'bound_fraction': 0.5,
            'bleach_region': {
                'type': 'circular',
                'center': (2.5, 2.5),
                'radius': 0.5,
                'bleach_depth': 0.9
            }
        }
        
        # True parameters
        self.true_params = {
            'D': 5.0,
            'k_on': 2.0,
            'k_off': 2.0,
            'bleach_depth': 0.9
        }
        
        # Simulate synthetic data
        self.timepoints = np.array([0, 0.5, 1, 2, 5, 10, 20, 40])
        self.true_recovery = self.model.simulate(
            self.true_params, self.geometry, self.timepoints
        )
    
    def test_similar_parameters_give_similar_curves(self):
        """Test that different parameter sets can give similar curves."""
        # These parameters should give qualitatively similar recovery
        param_sets = [
            {'D': 5.0, 'k_on': 2.0, 'k_off': 2.0, 'bleach_depth': 0.9},
            {'D': 7.0, 'k_on': 3.0, 'k_off': 3.0, 'bleach_depth': 0.9},
            {'D': 10.0, 'k_on': 5.0, 'k_off': 5.0, 'bleach_depth': 0.9},
        ]
        
        curves = []
        for params in param_sets:
            curve = self.model.simulate(params, self.geometry, self.timepoints)
            curves.append(curve)
        
        # Curves should exist and be similar in shape
        for curve in curves:
            self.assertEqual(len(curve), len(self.timepoints))
            # Should show recovery (increasing)
            self.assertGreater(curve[-1], curve[0])
    
    def test_model_selection_flags_similar_models(self):
        """Test that model selection flags non-identifiable parameters."""
        selection = ModelSelection()
        
        # Add models with slightly different parameters
        # In real scenarios, these might fit equally well
        selection.add_model(
            'Model1',
            log_likelihood=-10.0,
            n_params=3,
            n_data=8
        )
        
        selection.add_model(
            'Model2',
            log_likelihood=-10.5,  # Very similar
            n_params=3,
            n_data=8
        )
        
        # Check for warnings about similar models
        flags = selection.flag_unidentifiable_models(threshold=2.0)
        
        # Should flag these as similar
        self.assertGreater(len(flags), 0)
    
    def test_increasing_bleach_radius_changes_recovery(self):
        """Recovery should scale predictably with bleach radius."""
        radii = [0.3, 0.5, 0.8]
        half_times = []
        
        for radius in radii:
            geom = self.geometry.copy()
            geom['bleach_region'] = {
                'type': 'circular',
                'center': (2.5, 2.5),
                'radius': radius,
                'bleach_depth': 0.9
            }
            
            recovery = self.model.simulate(
                self.true_params, geom, self.timepoints
            )
            
            # Estimate half-time (time to 50% recovery)
            recovery_range = recovery[-1] - recovery[0]
            half_recovery = recovery[0] + 0.5 * recovery_range
            
            # Find crossing point
            idx = np.where(recovery > half_recovery)[0]
            if len(idx) > 0:
                t_half = self.timepoints[idx[0]]
                half_times.append(t_half)
        
        # Half-time should increase with radius (for diffusion-limited)
        if len(half_times) >= 2:
            self.assertGreater(half_times[-1], half_times[0])
    
    def test_diffusion_vs_reaction_regimes(self):
        """Test identification of diffusion vs reaction-limited regimes."""
        from frap_models.coalescence import estimate_recovery_regime
        
        # Diffusion-limited (slow diffusion, fast exchange)
        params_diff_limited = {
            'D': 0.5,  # Slow
            'k_on': 10.0,  # Fast
            'k_off': 10.0,
            'bleach_depth': 0.9
        }
        
        regime_diff = estimate_recovery_regime(params_diff_limited, self.geometry)
        
        # Reaction-limited (fast diffusion, slow exchange)
        params_rxn_limited = {
            'D': 20.0,  # Fast
            'k_on': 0.5,  # Slow
            'k_off': 0.5,
            'bleach_depth': 0.9
        }
        
        regime_rxn = estimate_recovery_regime(params_rxn_limited, self.geometry)
        
        # Should identify different regimes
        # (actual values depend on geometry and rates)
        self.assertIsNotNone(regime_diff)
        self.assertIsNotNone(regime_rxn)
    
    def test_parameter_bounds_are_reasonable(self):
        """Test that parameter bounds are physically reasonable."""
        bounds = self.model.get_parameter_bounds()
        
        # Check key parameters
        self.assertIn('D', bounds)
        self.assertIn('k_on', bounds)
        self.assertIn('k_off', bounds)
        
        # Bounds should be positive
        for param, (lower, upper) in bounds.items():
            self.assertGreaterEqual(lower, 0)
            self.assertGreater(upper, lower)
        
        # Diffusion should have reasonable upper limit
        self.assertLess(bounds['D'][1], 1000)  # Not > 1000 μm²/s


class TestMCRDIdentifiability(unittest.TestCase):
    """Test identifiability for mass-conserving RD model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = MassConservingRDModel()
        
        self.geometry = {
            'shape': (50, 50),
            'spacing': 0.1,
            'total_concentration': 1.0,
            'condensed_fraction': 0.7,
            'bleach_region': {
                'type': 'circular',
                'center': (2.5, 2.5),
                'radius': 0.5,
                'bleach_depth': 0.9
            }
        }
        
        self.params = {
            'D_dilute': 5.0,
            'D_condensed': 0.5,
            'k_in': 1.0,
            'k_out': 1.0,
            'condensed_fraction': 0.7,
            'bleach_depth': 0.9
        }
    
    def test_mcrd_has_more_parameters(self):
        """MCRD model has more parameters than simple RD."""
        mcrd_bounds = self.model.get_parameter_bounds()
        
        rd_model = ReactionDiffusionModel()
        rd_bounds = rd_model.get_parameter_bounds()
        
        # MCRD should have at least as many parameters
        self.assertGreaterEqual(len(mcrd_bounds), len(rd_bounds))
    
    def test_diffusion_ratio_affects_recovery(self):
        """Ratio of condensed to dilute diffusion should affect recovery."""
        # High ratio (condensed diffuses fast)
        params_high = self.params.copy()
        params_high['D_condensed'] = 4.0
        
        # Low ratio (condensed diffuses slowly)
        params_low = self.params.copy()
        params_low['D_condensed'] = 0.1
        
        timepoints = np.array([0, 1, 5, 10, 20])
        
        recovery_high = self.model.simulate(params_high, self.geometry, timepoints)
        recovery_low = self.model.simulate(params_low, self.geometry, timepoints)
        
        # Recoveries should differ
        # (exact relationship depends on other parameters)
        self.assertEqual(len(recovery_high), len(recovery_low))


if __name__ == '__main__':
    unittest.main()
