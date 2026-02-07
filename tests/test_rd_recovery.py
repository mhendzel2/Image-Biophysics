"""
Test Reaction-Diffusion Recovery

Validates FRAP recovery simulations produce expected behavior.
"""

import unittest
import numpy as np
from frap_models import ReactionDiffusionModel, FRAPSimulator
from frap_models.simulators import create_circular_bleach_mask


class TestRDRecovery(unittest.TestCase):
    """Test reaction-diffusion FRAP recovery."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = ReactionDiffusionModel()
        
        self.geometry = {
            'shape': (50, 50),
            'spacing': 0.1,  # μm
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
        
        self.timepoints = np.array([0, 0.5, 1, 2, 5, 10, 20, 40])
    
    def test_recovery_is_monotonic(self):
        """Recovery curve should be monotonically increasing."""
        recovery = self.model.simulate(
            self.params, self.geometry, self.timepoints
        )
        
        # Check monotonicity (allowing small numerical errors)
        diffs = np.diff(recovery)
        self.assertTrue(np.all(diffs >= -1e-6))
    
    def test_recovery_starts_low(self):
        """Recovery should start < 1 after bleaching."""
        recovery = self.model.simulate(
            self.params, self.geometry, self.timepoints
        )
        
        # First point should be less than pre-bleach (normalized to 1)
        self.assertLess(recovery[0], 0.95)
    
    def test_recovery_approaches_prebleach(self):
        """Recovery should approach pre-bleach level at long times."""
        # Longer timepoints
        long_times = np.array([0, 10, 50, 100, 200])
        
        recovery = self.model.simulate(
            self.params, self.geometry, long_times
        )
        
        # Final recovery should be close to 1
        # (may not be exactly 1 due to boundary conditions)
        self.assertGreater(recovery[-1], 0.8)
    
    def test_no_diffusion_means_no_recovery(self):
        """With D=0 and no exchange, recovery should be minimal."""
        params_no_diff = {
            'D': 0.0,  # No diffusion
            'k_on': 0.0,  # No binding
            'k_off': 0.0,  # No unbinding
            'bleach_depth': 0.9
        }
        
        recovery = self.model.simulate(
            params_no_diff, self.geometry, self.timepoints
        )
        
        # Recovery should be very small (only numerical diffusion)
        self.assertLess(recovery[-1] - recovery[0], 0.2)
    
    def test_fast_diffusion_means_fast_recovery(self):
        """High diffusion should lead to faster recovery."""
        params_slow = self.params.copy()
        params_slow['D'] = 1.0
        
        params_fast = self.params.copy()
        params_fast['D'] = 20.0
        
        recovery_slow = self.model.simulate(
            params_slow, self.geometry, self.timepoints
        )
        recovery_fast = self.model.simulate(
            params_fast, self.geometry, self.timepoints
        )
        
        # At intermediate times, fast should recover more
        mid_idx = len(self.timepoints) // 2
        self.assertGreater(recovery_fast[mid_idx], recovery_slow[mid_idx])
    
    def test_bleach_mask_creation(self):
        """Test circular bleach mask generation."""
        mask = create_circular_bleach_mask(
            shape=(50, 50),
            center=(2.5, 2.5),
            radius=0.5,
            spacing=0.1
        )
        
        # Mask should be boolean
        self.assertEqual(mask.dtype, bool)
        
        # Should have correct shape
        self.assertEqual(mask.shape, (50, 50))
        
        # Should have some True values (bleached region)
        self.assertGreater(np.sum(mask), 0)
        
        # Should not be entirely True
        self.assertLess(np.sum(mask), mask.size)
    
    def test_geometry_validation(self):
        """Test that invalid geometry is caught."""
        from frap_models.base import GeometryError
        
        # Missing required field
        bad_geometry = {
            'shape': (50, 50),
            # Missing 'spacing' and 'bleach_region'
        }
        
        with self.assertRaises(GeometryError):
            self.model.simulate(self.params, bad_geometry, self.timepoints)
    
    def test_parameter_validation(self):
        """Test that invalid parameters are caught."""
        from frap_models.base import ParameterError
        
        # Negative diffusion
        bad_params = {
            'D': -1.0,  # Negative!
            'k_on': 1.0,
            'k_off': 1.0,
            'bleach_depth': 0.9
        }
        
        with self.assertRaises(ParameterError):
            self.model.simulate(bad_params, self.geometry, self.timepoints)
    
    def test_initial_conditions_are_uniform(self):
        """Test that initial conditions are spatially uniform."""
        state = self.model.initial_conditions(self.geometry)
        
        self.assertIn('free', state)
        self.assertIn('bound', state)
        
        # Should be uniform (all values equal)
        self.assertTrue(np.allclose(state['free'], state['free'][0, 0]))
        self.assertTrue(np.allclose(state['bound'], state['bound'][0, 0]))
    
    def test_bleaching_reduces_fluorescence(self):
        """Test that bleaching reduces fluorescence in bleached region."""
        state = self.model.initial_conditions(self.geometry)
        
        total_pre = np.sum(state['free'] + state['bound'])
        
        bleached = self.model.bleach(state, self.geometry['bleach_region'])
        
        total_post = np.sum(bleached['free'] + bleached['bound'])
        
        # Total should decrease
        self.assertLess(total_post, total_pre)
    
    def test_simulator_stability_check(self):
        """Test that simulator checks stability conditions."""
        simulator = FRAPSimulator(method='explicit', stability_check=True)
        
        # This should trigger a stability warning
        # (large D, small dx, large dt)
        with self.assertWarns(UserWarning):
            simulator.integrate_diffusion_2d(
                initial_state=np.ones((10, 10)),
                diffusion_coeff=100.0,  # Large D
                spacing=0.1,  # Small dx
                dt=0.1,  # Large dt
                num_steps=1
            )


class TestSimulatorFunctions(unittest.TestCase):
    """Test low-level simulator functions."""
    
    def setUp(self):
        """Set up simulator."""
        self.simulator = FRAPSimulator()
    
    def test_laplacian_computation(self):
        """Test Laplacian computation."""
        # Create field with known Laplacian
        # For f(x,y) = x^2 + y^2, Laplacian = 4
        x = np.arange(10) * 0.1
        y = np.arange(10) * 0.1
        X, Y = np.meshgrid(x, y)
        field = X**2 + Y**2
        
        laplacian = self.simulator._compute_laplacian(field, dx=0.1, boundary='neumann')
        
        # Laplacian should be approximately constant (4.0)
        # (edges will be different due to boundary conditions)
        interior = laplacian[2:-2, 2:-2]
        self.assertTrue(np.allclose(interior, 4.0, atol=0.5))
    
    def test_diffusion_spreads_concentration(self):
        """Test that diffusion spreads a localized concentration."""
        # Point source in center
        initial = np.zeros((50, 50))
        initial[25, 25] = 100.0
        
        # Diffuse
        final = self.simulator.integrate_diffusion_2d(
            initial,
            diffusion_coeff=1.0,
            spacing=0.1,
            dt=0.01,
            num_steps=100
        )
        
        # Should spread (max should decrease, sum should conserve)
        self.assertLess(np.max(final), np.max(initial))
        self.assertGreater(np.sum(final > 1), 1)  # Spread to multiple pixels


if __name__ == '__main__':
    unittest.main()
