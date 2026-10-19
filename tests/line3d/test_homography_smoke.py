"""Synthetic smoke test for the pre-existing estimateHomography entrypoint.

Purpose: the Line3D work touches the shared UniformRandomSampler and the CMake
build, so verify that the older public homography API still runs end to end.
Uses a synthetic 4-point correspondence set only.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_PYTHON = Path(
    os.environ.get("SUPERANSAC_BUILD_PYTHON", REPO_ROOT / "build-line3d" / "python")
)
sys.path.insert(0, str(BUILD_PYTHON))

import pysuperansac as ps  # noqa: E402


class HomographySmokeTest(unittest.TestCase):
    def test_synthetic_rectangle(self):
        src = np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64
        )
        expected = np.array(
            [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]], dtype=np.float64
        )
        correspondences = np.hstack([src, expected])

        config = ps.RANSACSettings()
        config.inlier_threshold = 1e-6
        config.min_iterations = 100
        config.max_iterations = 100
        config.use_sprt = False
        config.scoring = ps.ScoringType.MSAC
        config.sampler = ps.SamplerType.Uniform
        config.local_optimization = ps.LocalOptimizationType.Nothing
        config.final_optimization = ps.LocalOptimizationType.Nothing

        model, inliers, score, iterations = ps.estimateHomography(
            correspondences, np.array([10.0, 10.0, 10.0, 10.0]), None, config
        )

        self.assertEqual(model.shape, (3, 3))
        self.assertTrue(np.all(np.isfinite(model)))
        self.assertEqual(sorted(inliers), [0, 1, 2, 3])
        projected = np.hstack([src, np.ones((4, 1))]) @ model.T
        projected = projected[:, :2] / projected[:, 2:3]
        np.testing.assert_allclose(projected, expected, atol=1e-8)
        self.assertTrue(np.isfinite(score))
        self.assertGreaterEqual(iterations, 1)


if __name__ == "__main__":
    unittest.main()
