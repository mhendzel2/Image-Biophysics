"""
Tests for material_mechanics.py
"""

import unittest
import numpy as np

from material_mechanics import CV2_AVAILABLE, SKIMAGE_AVAILABLE, MaterialMechanics


def make_translating_blob_stack(t_frames: int = 8, height: int = 64, width: int = 64) -> np.ndarray:
    """Create a synthetic stack with a smoothly translating Gaussian blob."""
    y, x = np.mgrid[:height, :width]
    stack = np.zeros((t_frames, height, width), dtype=np.float32)
    for t in range(t_frames):
        cx = width * 0.35 + t * 0.8
        cy = height * 0.45 + t * 0.5
        blob = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * 7.0**2))
        stack[t] = blob + 0.03 * np.sin(0.2 * x + 0.15 * y + t * 0.3)
    stack -= np.min(stack)
    stack /= np.max(stack) + 1e-8
    return stack


def make_fluctuating_boundary_stack(t_frames: int = 14, height: int = 96, width: int = 96):
    """Create synthetic boundary flicker stack and matching mask stack."""
    y, x = np.mgrid[:height, :width]
    cy, cx = height / 2.0, width / 2.0
    theta = np.arctan2(y - cy, x - cx)
    radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    images = np.zeros((t_frames, height, width), dtype=np.float32)
    masks = np.zeros((t_frames, height, width), dtype=bool)

    for t in range(t_frames):
        phase = 2.0 * np.pi * t / max(t_frames, 1)
        boundary = 24.0 + 1.4 * np.sin(2.0 * theta + phase) + 0.6 * np.sin(3.0 * theta - 1.5 * phase)
        mask = radius <= boundary
        image = mask.astype(np.float32) + 0.04 * np.sin(0.15 * x + 0.11 * y + 0.4 * t)
        images[t] = image
        masks[t] = mask

    images -= images.min()
    images /= images.max() + 1e-8
    return images, masks


class TestMaterialMechanics(unittest.TestCase):
    def setUp(self):
        self.stack = make_translating_blob_stack()
        self.analyzer = MaterialMechanics(pixel_size_um=0.2, time_interval_s=0.5)

    def test_internal_stress_output_shapes(self):
        result = self.analyzer.compute_internal_stress(self.stack)
        if not CV2_AVAILABLE:
            self.assertEqual(result.get("status"), "error")
            return
        self.assertEqual(result.get("status"), "success")
        self.assertEqual(result["divergence_map"].shape, self.stack.shape[1:])
        self.assertEqual(result["shear_rate_map"].shape, self.stack.shape[1:])
        self.assertTrue(np.isfinite(result["summary"]["mean_divergence_per_s"]))

    def test_texture_topology_metrics(self):
        result = self.analyzer.analyze_texture_topology(self.stack)
        self.assertEqual(result.get("status"), "success")
        entropy = result["entropy"]
        fractal = result["fractal"]
        if SKIMAGE_AVAILABLE:
            self.assertEqual(entropy["status"], "success")
            self.assertEqual(entropy["entropy_map"].shape, self.stack.shape[1:])
        self.assertTrue("global_fractal_dimension" in fractal)

    def test_boundary_fluctuation_spectrum(self):
        images, masks = make_fluctuating_boundary_stack()
        result = self.analyzer.analyze_envelope_mechanics(
            images,
            nuclear_mask=masks,
            parameters={"use_active_contours": False, "boundary_n_angles": 96},
        )
        self.assertIn(result.get("status"), ("success", "warning"))
        if result.get("status") == "success":
            self.assertTrue(np.isfinite(result.get("mean_radius_um", np.nan)))
            self.assertEqual(len(result["mode_numbers"]), len(result["power_spectrum"]))

    def test_pipeline_returns_component_map(self):
        params = {
            "run_force_distribution": True,
            "run_stiffness_proxy": True,
            "run_texture_topology": True,
            "run_boundary_mechanics": False,
            "run_fusion_kinetics": False,
        }
        result = self.analyzer.analyze(self.stack, parameters=params)
        self.assertEqual(result.get("status"), "success")
        self.assertIn("results", result)
        self.assertIn("texture_topology", result["results"])
        if CV2_AVAILABLE:
            self.assertIn("force_distribution", result["results"])
            self.assertEqual(result["results"]["force_distribution"].get("status"), "success")


if __name__ == "__main__":
    unittest.main()
