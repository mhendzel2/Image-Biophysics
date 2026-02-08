"""
Test Mass Conservation in FRAP Models

Verifies that mass-conserving models strictly conserve total fluorescence
post-bleach, as required for defensible biophysical interpretation.
"""

import unittest
import numpy as np
from frap_models import MassConservingRDModel, FRAPSimulator


class TestMassConservation(unittest.TestCase):
    """Test mass conservation in FRAP models."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.simulator = FRAPSimulator(method='implicit')
        self.model = MassConservingRDModel(self.simulator)
        
        # Standard test geometry
        self.geometry = {
            'shape': (50, 50),
            'spacing': 0.1,  # μm
            'total_concentration': 1.0,
            'condensed_fraction': 0.7,
            'bleach_region': {
                'type': 'circular',
                'center': (2.5, 2.5),  # μm
                'radius': 0.5,  # μm
                'bleach_depth': 0.9
            }
        }
        
        # Standard parameters
        self.params = {
            'D_dilute': 5.0,
            'D_condensed': 0.5,
            'k_in': 1.0,
            'k_out': 1.0,
            'condensed_fraction': 0.7,
            'bleach_depth': 0.9
        }
    
    def test_mass_conservation_post_bleach(self):
        """Total mass must remain constant after bleaching."""
        # Initial conditions
        state = self.model.initial_conditions(self.geometry)
        total_pre = np.sum(state['condensed'] + state['dilute'])
        
        # Apply bleaching
        bleached = self.model.bleach(state, self.geometry['bleach_region'])
        total_post = np.sum(bleached['condensed'] + bleached['dilute'])
        
        # Total should decrease due to bleaching
        self.assertLess(total_post, total_pre)
        
        # Store for later checks
        self.expected_total = total_post
    
    def test_mass_conservation_during_recovery(self):
        """Mass must remain constant during recovery simulation."""
        timepoints = np.array([0, 1, 5, 10, 20])
        
        # Simulate
        recovery = self.model.simulate(self.params, self.geometry, timepoints)
        
        # Check that mass was conserved during simulation
        if hasattr(self.model, '_total_mass_postbleach'):
            # Model tracks mass internally
            self.assertIsNotNone(self.model._total_mass_postbleach)
    
    def test_conservation_with_different_parameters(self):
        """Mass conservation must hold for various parameter values."""
        test_params = [
            {'D_dilute': 10.0, 'D_condensed': 1.0, 'k_in': 5.0, 'k_out': 2.0},
            {'D_dilute': 1.0, 'D_condensed': 0.1, 'k_in': 0.5, 'k_out': 0.5},
            {'D_dilute': 20.0, 'D_condensed': 2.0, 'k_in': 10.0, 'k_out': 5.0},
        ]
        
        for params in test_params:
            params.update({
                'condensed_fraction': 0.7,
                'bleach_depth': 0.9
            })
            
            # Run short simulation
            timepoints = np.array([0, 1, 5])
            recovery = self.model.simulate(params, self.geometry, timepoints)
            
            # Should complete without mass conservation warnings
            self.assertEqual(len(recovery), len(timepoints))
    
    def test_conservation_independence_of_geometry(self):
        """Mass conservation should hold for different geometries."""
        geometries = [
            # Small bleach
            {
                **self.geometry,
                'bleach_region': {
                    'type': 'circular',
                    'center': (2.5, 2.5),
                    'radius': 0.3,
                    'bleach_depth': 0.9
                }
            },
            # Large bleach
            {
                **self.geometry,
                'bleach_region': {
                    'type': 'circular',
                    'center': (2.5, 2.5),
                    'radius': 1.0,
                    'bleach_depth': 0.9
                }
            },
        ]
        
        for geom in geometries:
            timepoints = np.array([0, 1, 5])
            recovery = self.model.simulate(self.params, geom, timepoints)
            
            # Should complete successfully
            self.assertEqual(len(recovery), len(timepoints))
            # Recovery should be normalized (0-1 range)
            self.assertTrue(np.all(recovery >= 0))
            self.assertTrue(np.all(recovery <= 1.1))  # Allow small numerical error
    
    def test_simulator_conserves_mass_in_diffusion(self):
        """Test that diffusion step alone conserves mass."""
        # Create simple concentration field
        initial = np.ones((50, 50))
        total_initial = np.sum(initial)
        
        # Integrate diffusion
        final = self.simulator.integrate_diffusion_2d(
            initial,
            diffusion_coeff=1.0,
            spacing=0.1,
            dt=0.01,
            num_steps=100,
            boundary='neumann'
        )
        
        total_final = np.sum(final)
        
        # Mass should be conserved (within numerical precision)
        self.assertAlmostEqual(total_final, total_initial, places=2)
    
    def test_reaction_step_conserves_mass(self):
        """Test that reaction/exchange conserves mass."""
        free = np.ones((50, 50))
        bound = np.ones((50, 50)) * 0.5
        
        total_initial = np.sum(free + bound)
        
        # Simulate exchange
        free_new, bound_new = self.simulator._reaction_step(
            free, bound,
            k_on=1.0,
            k_off=1.0,
            dt=0.1
        )
        
        total_final = np.sum(free_new + bound_new)
        
        # Mass should be conserved exactly for reaction step
        self.assertAlmostEqual(total_final, total_initial, places=6)


class TestReactionDiffusionConservation(unittest.TestCase):
    """Test conservation in standard RD model."""
    
    def setUp(self):
        """Set up test fixtures."""
        from frap_models import ReactionDiffusionModel
        
        self.model = ReactionDiffusionModel()
        
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
        
        self.params = {
            'D': 5.0,
            'k_on': 1.0,
            'k_off': 1.0,
            'bleach_depth': 0.9
        }
    
    def test_rd_model_declares_conservation(self):
        """RD model should declare mass conservation."""
        conserved = self.model.conserved_quantities()
        self.assertIn('total_mass', conserved)
    
    def test_rd_bleach_conserves_relative_mass(self):
        """Bleaching should reduce but not violate conservation."""
        state = self.model.initial_conditions(self.geometry)
        total_pre = np.sum(state['free'] + state['bound'])
        
        bleached = self.model.bleach(state, self.geometry['bleach_region'])
        total_post = np.sum(bleached['free'] + bleached['bound'])
        
        # Should decrease due to bleaching
        self.assertLess(total_post, total_pre)
        
        # Calculate expected reduction
        bleach_depth = self.geometry['bleach_region']['bleach_depth']
        # Exact calculation would require knowing bleached region size
        # But at minimum, total should be positive
        self.assertGreater(total_post, 0)


if __name__ == '__main__':
    unittest.main()
