"""
Tests for advanced SPT inference modules:
- Spot-On-like bias-aware diffusion inference
- Switching diffusion HMM
- Bayesian posterior workflow
- Synthetic trajectory representation utilities
"""

import unittest
import numpy as np

from spt_models import (
    BayesianDiffusionInference,
    SpotOnConfig,
    SpotOnLikeInference,
    SwitchingDiffusionHMM,
    SwitchingHMMConfig,
    SyntheticTrajectoryConfig,
    TrajectoryFeatureEmbedder,
    TwoStateSimulationConfig,
    diffusion_recovery_metrics,
    fraction_recovery_metrics,
    generate_synthetic_dataset,
    simulate_two_state_tracks,
    transition_recovery_metrics,
)


class TestSpotOnLikeInference(unittest.TestCase):
    """Validate bias-aware diffusion/fraction recovery on synthetic data."""

    def setUp(self):
        self.sim_cfg = TwoStateSimulationConfig(
            n_tracks=36,
            steps_per_track=42,
            dt_s=0.02,
            diffusion_um2_s=(0.03, 0.75),
            transition_matrix=((0.94, 0.06), (0.08, 0.92)),
            localization_error_um=0.02,
            random_state=123,
        )
        sim = simulate_two_state_tracks(self.sim_cfg)
        self.tracks = sim["tracks"]
        self.true_d = np.sort(np.asarray(sim["true_diffusion_um2_s"], dtype=float))
        self.true_A = np.asarray(sim["true_transition_matrix"], dtype=float)

    def test_spot_on_recovers_diffusion_and_fractions(self):
        cfg = SpotOnConfig(
            frame_interval_s=self.sim_cfg.dt_s,
            axial_detection_range_um=10.0,  # effectively no defocalization penalty
            localization_error_um=self.sim_cfg.localization_error_um,
            exposure_time_s=self.sim_cfg.dt_s / 2.0,
            n_states=2,
            max_lag_frames=1,
            allow_gap_frames=0,
            bootstrap_samples=0,
            random_state=123,
        )
        model = SpotOnLikeInference(cfg)
        res = model.fit(self.tracks, max_iter=150)

        est_d = np.sort(np.asarray(res["diffusion_coefficients_um2_s"], dtype=float))
        d_metrics = diffusion_recovery_metrics(self.true_d, est_d)
        self.assertLess(d_metrics["mean_abs_relative_error"], 0.8)

        # Stationary fractions from transition matrix.
        a01 = self.true_A[0, 1]
        a10 = self.true_A[1, 0]
        true_f = np.array([a10 / (a01 + a10), a01 / (a01 + a10)], dtype=float)
        est_f = np.asarray(res["state_fractions"], dtype=float)
        f_metrics = fraction_recovery_metrics(true_f, est_f)
        self.assertLess(f_metrics["mean_abs_fraction_error"], 0.25)


class TestSwitchingDiffusionHMM(unittest.TestCase):
    """Validate transition and kinetic recovery."""

    def setUp(self):
        self.sim_cfg = TwoStateSimulationConfig(
            n_tracks=40,
            steps_per_track=45,
            dt_s=0.02,
            diffusion_um2_s=(0.02, 0.85),
            transition_matrix=((0.95, 0.05), (0.07, 0.93)),
            localization_error_um=0.02,
            random_state=321,
        )
        sim = simulate_two_state_tracks(self.sim_cfg)
        self.tracks = sim["tracks"]
        self.true_A = np.asarray(sim["true_transition_matrix"], dtype=float)
        self.true_d = np.sort(np.asarray(sim["true_diffusion_um2_s"], dtype=float))

    def test_hmm_recovers_transition_structure(self):
        cfg = SwitchingHMMConfig(
            frame_interval_s=self.sim_cfg.dt_s,
            localization_error_um=self.sim_cfg.localization_error_um,
            n_states=2,
            max_iter=70,
            tol=1e-5,
            bootstrap_samples=0,
            random_state=99,
        )
        hmm = SwitchingDiffusionHMM(cfg)
        res = hmm.fit(self.tracks)

        est_A = np.asarray(res["transition_matrix_frame"], dtype=float)
        t_metrics = transition_recovery_metrics(self.true_A, est_A)
        self.assertLess(t_metrics["mean_abs_transition_error"], 0.2)

        est_d = np.sort(np.asarray(res["diffusion_coefficients_um2_s"], dtype=float))
        d_metrics = diffusion_recovery_metrics(self.true_d, est_d)
        self.assertLess(d_metrics["mean_abs_relative_error"], 0.9)

        self.assertIn("dwell_times_s", res)
        self.assertEqual(len(res["dwell_times_s"]), 2)


class TestBayesianWorkflow(unittest.TestCase):
    """Validate Bayesian posterior outputs and diagnostics."""

    def test_bayesian_posterior_returns_intervals(self):
        sim_cfg = TwoStateSimulationConfig(
            n_tracks=24,
            steps_per_track=36,
            dt_s=0.02,
            diffusion_um2_s=(0.03, 0.7),
            transition_matrix=((0.93, 0.07), (0.08, 0.92)),
            localization_error_um=0.02,
            random_state=7,
        )
        sim = simulate_two_state_tracks(sim_cfg)
        tracks = sim["tracks"]

        cfg = SpotOnConfig(
            frame_interval_s=sim_cfg.dt_s,
            axial_detection_range_um=6.0,
            localization_error_um=sim_cfg.localization_error_um,
            exposure_time_s=sim_cfg.dt_s / 2.0,
            n_states=2,
            max_lag_frames=1,
            bootstrap_samples=0,
            random_state=7,
        )
        bayes = BayesianDiffusionInference(cfg)
        res = bayes.sample_posterior(
            tracks,
            n_samples=120,
            burn_in=60,
            thin=1,
            proposal_scale=0.08,
            map_max_iter=120,
        )

        self.assertIn("posterior_summary", res)
        self.assertIn("diagnostics", res)
        self.assertEqual(len(res["posterior_summary"]["diffusion_ci95_um2_s"]), 2)
        self.assertGreaterEqual(res["diagnostics"]["acceptance_rate"], 0.0)
        self.assertLessEqual(res["diagnostics"]["acceptance_rate"], 1.0)
        self.assertGreater(res["diagnostics"]["effective_sample_size_min"], 5.0)


class TestTrajectoryRepresentation(unittest.TestCase):
    """Validate synthetic trajectory generation and feature embedding."""

    def test_generate_synthetic_dataset_and_embed(self):
        cfg = SyntheticTrajectoryConfig(n_steps=40, dt_s=0.02, random_state=11)
        dataset = generate_synthetic_dataset(
            n_per_class=4,
            config=cfg,
            classes=["brownian", "confined", "directed"],
        )
        self.assertEqual(len(dataset["trajectories"]), 12)
        self.assertEqual(len(dataset["labels"]), 12)

        embedder = TrajectoryFeatureEmbedder(dt_s=cfg.dt_s)
        feat = embedder.transform(dataset["trajectories"])
        self.assertEqual(feat.shape[0], 12)
        self.assertGreater(feat.shape[1], 4)


if __name__ == "__main__":
    unittest.main()
