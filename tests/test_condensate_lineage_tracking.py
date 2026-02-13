"""
Tests for condensate_lineage_tracking.py
"""

import unittest
import numpy as np

from condensate_lineage_tracking import (
    NETWORKX_AVAILABLE,
    SKIMAGE_AVAILABLE,
    CondensateLineageTracker,
)


def _make_object(label, bbox, total_intensity, major_axis, minor_axis):
    r0, c0, r1, c1 = bbox
    local_mask = np.ones((r1 - r0, c1 - c0), dtype=bool)
    area = float(local_mask.sum())
    return {
        "label": int(label),
        "bbox": tuple(map(int, bbox)),
        "local_mask": local_mask,
        "area": area,
        "total_intensity": float(total_intensity),
        "major_axis": float(major_axis),
        "minor_axis": float(minor_axis),
        "equiv_radius_px": float(np.sqrt(area / np.pi)),
        "centroid_y": float((r0 + r1) / 2.0),
        "centroid_x": float((c0 + c1) / 2.0),
    }


class TestCondensateLineageTracking(unittest.TestCase):
    def setUp(self):
        self.tracker = CondensateLineageTracker(pixel_size_um=0.2, time_interval_s=1.0)

    def test_seeded_watershed_splits_touching_peaks(self):
        if not SKIMAGE_AVAILABLE:
            self.skipTest("scikit-image not available")

        y, x = np.mgrid[:64, :64]
        image = np.exp(-((x - 24) ** 2 + (y - 32) ** 2) / (2 * 5.0**2))
        image += np.exp(-((x - 40) ** 2 + (y - 32) ** 2) / (2 * 5.0**2))
        image = image.astype(np.float32)
        image /= np.max(image) + 1e-8

        frame_result = self.tracker._segment_frame(
            image,
            frame_mask=None,
            parameters={
                "threshold_mode": "percentile",
                "threshold_percentile": 72.0,
                "h_maxima_h": 0.05,
                "min_area_px": 8,
                "segmentation_smoothing_sigma": 1.0,
            },
        )
        self.assertGreaterEqual(len(frame_result["objects"]), 2)

    def test_build_lineage_detects_fusion_and_fits_tau(self):
        if not NETWORKX_AVAILABLE:
            self.skipTest("networkx not available")

        # t0: two droplets
        frame0 = {
            "frame": 0,
            "labels": np.zeros((64, 64), dtype=np.int32),
            "objects": [
                _make_object(1, (20, 12, 30, 22), total_intensity=100.0, major_axis=8.0, minor_axis=6.0),
                _make_object(2, (20, 26, 30, 36), total_intensity=100.0, major_axis=8.0, minor_axis=6.0),
            ],
            "watershed_dam_fraction": 0.1,
        }
        # t1: merged elongated droplet
        frame1 = {
            "frame": 1,
            "labels": np.zeros((64, 64), dtype=np.int32),
            "objects": [
                _make_object(1, (20, 12, 30, 36), total_intensity=196.0, major_axis=14.0, minor_axis=6.0),
            ],
            "watershed_dam_fraction": 0.02,
        }
        # t2/t3: relaxation toward roundness
        frame2 = {
            "frame": 2,
            "labels": np.zeros((64, 64), dtype=np.int32),
            "objects": [
                _make_object(1, (20, 13, 30, 35), total_intensity=194.0, major_axis=12.0, minor_axis=8.2),
            ],
            "watershed_dam_fraction": 0.01,
        }
        frame3 = {
            "frame": 3,
            "labels": np.zeros((64, 64), dtype=np.int32),
            "objects": [
                _make_object(1, (20, 14, 30, 34), total_intensity=193.0, major_axis=10.8, minor_axis=9.4),
            ],
            "watershed_dam_fraction": 0.01,
        }
        frame4 = {
            "frame": 4,
            "labels": np.zeros((64, 64), dtype=np.int32),
            "objects": [
                _make_object(1, (20, 14, 30, 34), total_intensity=192.0, major_axis=10.2, minor_axis=9.8),
            ],
            "watershed_dam_fraction": 0.01,
        }
        segmentation = [frame0, frame1, frame2, frame3, frame4]

        graph = self.tracker.build_lineage(
            segmentation,
            parameters={"iou_threshold": 0.05, "parent_top_k": 2, "child_top_k": 2},
        )
        summary = self.tracker._summarize_events(graph, parameters={"mass_tolerance": 0.25})
        self.assertGreaterEqual(summary["num_fusions"], 1)

        fusion = self.tracker.analyze_fusion_events(
            graph,
            parameters={
                "mass_tolerance": 0.25,
                "fusion_lookahead_frames": 8,
                "fusion_min_trace_points": 3,
                "fusion_min_initial_ar": 1.05,
            },
        )
        self.assertGreaterEqual(len(fusion), 1)
        self.assertTrue(np.isfinite(fusion[0]["relaxation_time_tau_s"]))

    def test_growth_regime_estimates_beta(self):
        if not NETWORKX_AVAILABLE:
            self.skipTest("networkx not available")

        # Build one growing track with R ~ t^0.5.
        segmentation = []
        for t in range(1, 9):
            radius = 3.0 * np.sqrt(t)
            half = int(max(2, round(radius)))
            obj = _make_object(
                1,
                (32 - half, 32 - half, 32 + half, 32 + half),
                total_intensity=200.0 + 12.0 * t,
                major_axis=2.0 * radius,
                minor_axis=2.0 * radius * 0.95,
            )
            segmentation.append(
                {
                    "frame": t - 1,
                    "labels": np.zeros((96, 96), dtype=np.int32),
                    "objects": [obj],
                    "watershed_dam_fraction": 0.0,
                }
            )

        graph = self.tracker.build_lineage(
            segmentation,
            parameters={"iou_threshold": 0.02, "parent_top_k": 1, "child_top_k": 1},
        )
        tracks = self.tracker.analyze_growth_regimes(
            graph,
            parameters={"growth_min_track_length": 5, "plateau_growth_fraction": 0.01},
        )
        self.assertGreaterEqual(len(tracks), 1)
        self.assertTrue(np.isfinite(tracks[0]["beta"]))


if __name__ == "__main__":
    unittest.main()
