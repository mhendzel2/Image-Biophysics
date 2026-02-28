import unittest
import numpy as np

from spt_models import JointFRAPSPT, fit_joint_model


class TestJointLikelihoodNormalization(unittest.TestCase):
    """Validate FRAP/SPT balancing via observation-count normalization."""

    def setUp(self):
        self.model = JointFRAPSPT()
        self.params = {
            'D': 1.0,
            'k_on': 1.8,
            'k_off': 2.2,
            'bleach_depth': 0.9,
        }
        self.spt_small = {
            'bound_dwells': np.array([0.08, 0.11, 0.15, 0.19]),
            'unbound_dwells': np.array([0.06, 0.10, 0.12]),
            'concentration': 1.0,
        }
        self.spt_large = {
            'bound_dwells': np.tile(self.spt_small['bound_dwells'], 80),
            'unbound_dwells': np.tile(self.spt_small['unbound_dwells'], 80),
            'concentration': 1.0,
        }

    def test_normalized_log_like_is_stable_to_dataset_duplication(self):
        d_small = self.model._joint_log_likelihood_components(
            params=self.params,
            frap_data={},
            spt_data=self.spt_small,
            normalize_by_observation_count=True,
        )
        d_large = self.model._joint_log_likelihood_components(
            params=self.params,
            frap_data={},
            spt_data=self.spt_large,
            normalize_by_observation_count=True,
        )

        self.assertAlmostEqual(
            d_small['spt']['normalized_log_likelihood'],
            d_large['spt']['normalized_log_likelihood'],
            places=10,
        )
        self.assertAlmostEqual(
            d_small['joint_log_likelihood'],
            d_large['joint_log_likelihood'],
            places=10,
        )

    def test_raw_log_like_scales_with_dataset_size(self):
        d_small = self.model._joint_log_likelihood_components(
            params=self.params,
            frap_data={},
            spt_data=self.spt_small,
            normalize_by_observation_count=False,
        )
        d_large = self.model._joint_log_likelihood_components(
            params=self.params,
            frap_data={},
            spt_data=self.spt_large,
            normalize_by_observation_count=False,
        )

        scale = float(len(self.spt_large['bound_dwells'])) / float(len(self.spt_small['bound_dwells']))
        self.assertAlmostEqual(
            d_large['spt']['raw_log_likelihood'],
            scale * d_small['spt']['raw_log_likelihood'],
            places=8,
        )
        self.assertAlmostEqual(
            d_large['joint_log_likelihood'],
            scale * d_small['joint_log_likelihood'],
            places=8,
        )

    def test_fit_joint_model_returns_component_diagnostics(self):
        # Tiny FRAP payload for API-level test.
        frap_data = {
            'timepoints': np.array([0.0, 1.0, 2.0, 3.0]),
            'recovery': np.array([0.2, 0.45, 0.63, 0.75]),
            'geometry': {
                'shape': (20, 20),
                'spacing': 0.2,
                'bound_fraction': 0.5,
                'bleach_region': {
                    'type': 'circular',
                    'center': (2.0, 2.0),
                    'radius': 0.4,
                    'bleach_depth': 0.9,
                },
            },
        }

        out = fit_joint_model(
            frap_data=frap_data,
            spt_data=self.spt_small,
            initial_guess=self.params,
            normalize_by_observation_count=True,
        )

        self.assertIn('likelihood_components', out)
        self.assertIn('frap', out['likelihood_components'])
        self.assertIn('spt', out['likelihood_components'])
        self.assertTrue(out['normalize_by_observation_count'])


if __name__ == '__main__':
    unittest.main()
