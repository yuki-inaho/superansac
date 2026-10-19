"""Synthetic-only tests for the public `pysuperansac.estimateLine3D` API.

The tests never read production data, never touch private paths, and only use
deterministic synthetic XYZ point sets.
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


def make_line_points(seed: int = 3, n_inliers: int = 80, n_outliers: int = 20,
                     noise: float = 0.002):
    rng = np.random.default_rng(seed)
    direction = np.array([0.3, -0.4, 0.8660254037844386])
    direction /= np.linalg.norm(direction)
    base = np.array([0.5, -0.25, 2.0])

    along = rng.uniform(-1.5, 1.5, n_inliers)
    inliers = base + along[:, None] * direction
    inliers += rng.normal(0.0, noise, (n_inliers, 3))

    outliers = rng.uniform(-2.0, 2.0, (n_outliers, 3))
    return np.vstack([inliers, outliers]), direction


def line_residuals(points: np.ndarray, descriptor: np.ndarray) -> np.ndarray:
    closest = descriptor[0]
    direction = descriptor[1]
    offset = points - closest
    perpendicular = offset - np.outer(offset @ direction, direction)
    return np.einsum("ij,ij->i", perpendicular, perpendicular)


def base_config(scoring=None, local=None, final=None, sampler=None,
                threshold: float = 0.01, iterations: int = 600):
    config = ps.RANSACSettings()
    config.inlier_threshold = threshold
    config.min_iterations = iterations
    config.max_iterations = iterations
    config.use_sprt = False
    config.sampler = ps.SamplerType.Uniform if sampler is None else sampler
    if scoring is not None:
        config.scoring = scoring
    config.local_optimization = (
        ps.LocalOptimizationType.Nothing if local is None else local
    )
    config.final_optimization = (
        ps.LocalOptimizationType.Nothing if final is None else final
    )
    return config


class EstimateLine3DTests(unittest.TestCase):
    def assert_descriptor_contract(self, descriptor: np.ndarray):
        self.assertEqual(descriptor.shape, (2, 3))
        self.assertTrue(np.all(np.isfinite(descriptor)))
        closest, direction = descriptor
        self.assertAlmostEqual(float(np.linalg.norm(direction)), 1.0, places=9)
        self.assertAlmostEqual(float(closest @ direction), 0.0, places=9)
        largest = int(np.argmax(np.abs(direction)))
        self.assertGreater(direction[largest], 0.0)

    def test_default_config_works_for_nx3(self):
        points, direction = make_line_points()
        descriptor, inliers, score, iterations = ps.estimateLine3D(points)

        self.assert_descriptor_contract(descriptor)
        self.assertGreaterEqual(len(inliers), 75)
        self.assertTrue(np.isfinite(score))
        self.assertGreaterEqual(iterations, 1)
        self.assertGreater(
            abs(float(direction @ descriptor[1].copy())), 0.999
        )

    def test_exact_two_points(self):
        points = np.array([[1.0, 2.0, 3.0], [4.0, 7.0, 5.0]])
        descriptor, inliers, score, _ = ps.estimateLine3D(points)

        self.assert_descriptor_contract(descriptor)
        self.assertEqual(sorted(inliers), [0, 1])
        residual = line_residuals(points, descriptor)
        self.assertTrue(np.all(residual < 1e-18))
        self.assertTrue(np.isfinite(score))
        # Recompute the closest point independently.
        expected_direction = (points[1] - points[0]) / np.linalg.norm(
            points[1] - points[0]
        )
        if expected_direction[np.argmax(np.abs(expected_direction))] < 0:
            expected_direction = -expected_direction
        expected_closest = (
            points[0] - (points[0] @ expected_direction) * expected_direction
        )
        np.testing.assert_allclose(descriptor[0], expected_closest, atol=1e-12)
        np.testing.assert_allclose(descriptor[1], expected_direction, atol=1e-12)

    def test_noisy_line_with_outliers(self):
        points, direction = make_line_points()
        config = base_config(
            scoring=ps.ScoringType.RANSAC,
            local=ps.LocalOptimizationType.LSQ,
            final=ps.LocalOptimizationType.LSQ,
            threshold=0.01,
        )
        descriptor, inliers, score, _ = ps.estimateLine3D(points, config)

        self.assert_descriptor_contract(descriptor)
        self.assertGreater(abs(float(direction @ descriptor[1].copy())), 0.999)
        self.assertGreaterEqual(len(inliers), 75)
        self.assertLessEqual(len(inliers), 85)
        self.assertEqual(score, float(len(inliers)))

        residual = line_residuals(points, descriptor)
        recomputed = set(np.flatnonzero(residual < 0.01 ** 2).tolist())
        self.assertEqual(recomputed, set(inliers))

    def test_all_supported_scorings_and_optimizers(self):
        points, direction = make_line_points(n_inliers=60, n_outliers=15)
        for scoring in (
            ps.ScoringType.RANSAC,
            ps.ScoringType.MSAC,
            ps.ScoringType.MAGSAC,
        ):
            for local, final in (
                (None, None),
                (ps.LocalOptimizationType.LSQ, ps.LocalOptimizationType.LSQ),
            ):
                with self.subTest(scoring=scoring, local=local):
                    config = base_config(
                        scoring=scoring, local=local, final=final,
                        threshold=0.01, iterations=400,
                    )
                    descriptor, inliers, score, iterations = ps.estimateLine3D(
                        points, config
                    )
                    self.assert_descriptor_contract(descriptor)
                    self.assertGreaterEqual(len(inliers), 55)
                    self.assertGreater(
                        abs(float(direction @ descriptor[1].copy())), 0.995
                    )
                    self.assertTrue(np.isfinite(score))
                    self.assertGreaterEqual(iterations, 1)
                    residual = line_residuals(points, descriptor)
                    self.assertTrue(np.all(residual[inliers] < 1e-12 + 0.01 ** 2))

    def test_prosac_is_rejected_pending_audit(self):
        points, _ = make_line_points(n_inliers=60, n_outliers=10)
        config = base_config(
            scoring=ps.ScoringType.MSAC,
            sampler=ps.SamplerType.PROSAC,
            threshold=0.01,
            iterations=400,
        )
        with self.assertRaises(ValueError):
            ps.estimateLine3D(points, config)

    def test_rotation_translation_equivariance(self):
        points, _ = make_line_points(n_inliers=60, n_outliers=10)
        config = base_config(scoring=ps.ScoringType.MSAC, threshold=0.01,
                             iterations=500)
        base_descriptor, base_inliers, _, _ = ps.estimateLine3D(points, config)

        angle = 0.7
        axis = np.array([1.0, -2.0, 0.5])
        axis /= np.linalg.norm(axis)
        rotation = _rotation_matrix(angle, axis)
        translation = np.array([10.0, -4.0, 3.0])
        moved = points @ rotation.T + translation

        moved_descriptor, moved_inliers, _, _ = ps.estimateLine3D(moved, config)

        self.assert_descriptor_contract(moved_descriptor)
        self.assertAlmostEqual(
            abs(float((rotation @ base_descriptor[1]) @ moved_descriptor[1])),
            1.0,
            places=6,
        )
        self.assertEqual(set(base_inliers), set(moved_inliers))

        # The line position must transform with the same rigid transform.
        expected_closest = rotation @ base_descriptor[0] + translation
        expected_direction = rotation @ base_descriptor[1]
        expected_closest = (
            expected_closest
            - (expected_closest @ expected_direction) * expected_direction
        )
        np.testing.assert_allclose(
            moved_descriptor[0], expected_closest, atol=1e-7
        )

    def test_metre_millimetre_scale_equivariance(self):
        points, _ = make_line_points(n_inliers=60, n_outliers=10)
        config_m = base_config(scoring=ps.ScoringType.MAGSAC, threshold=0.01,
                               iterations=500)
        config_mm = base_config(scoring=ps.ScoringType.MAGSAC, threshold=10.0,
                                iterations=500)

        descriptor_m, inliers_m, score_m, _ = ps.estimateLine3D(points, config_m)
        descriptor_mm, inliers_mm, score_mm, _ = ps.estimateLine3D(
            points * 1000.0, config_mm
        )

        self.assert_descriptor_contract(descriptor_m)
        self.assert_descriptor_contract(descriptor_mm)
        self.assertEqual(set(inliers_m), set(inliers_mm))
        np.testing.assert_allclose(
            descriptor_mm[0], descriptor_m[0] * 1000.0, rtol=1e-9, atol=1e-7
        )
        self.assertAlmostEqual(
            abs(float(descriptor_m[1] @ descriptor_mm[1])), 1.0, places=9
        )
        self.assertTrue(np.isfinite(score_m) and np.isfinite(score_mm))

    def test_invalid_inputs_raise(self):
        good = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        with self.assertRaises(ValueError):
            ps.estimateLine3D(np.array([[0.0, 0.0, 0.0]]))
        with self.assertRaises(ValueError):
            ps.estimateLine3D(np.zeros((2, 2)))
        with self.assertRaises((ValueError, RuntimeError)):
            ps.estimateLine3D(np.zeros(3))
        bad = good.copy()
        bad[0, 1] = np.nan
        with self.assertRaises(ValueError):
            ps.estimateLine3D(bad)
        bad[0, 1] = np.inf
        with self.assertRaises(ValueError):
            ps.estimateLine3D(bad)

        for threshold in (0.0, -1.0, np.nan, np.inf):
            config = base_config(threshold=threshold)
            with self.assertRaises(ValueError):
                ps.estimateLine3D(good, config)

        for confidence in (0.0, 1.0, np.nan):
            config = base_config()
            config.confidence = confidence
            with self.assertRaises(ValueError):
                ps.estimateLine3D(good, config)

        config = base_config()
        config.use_sprt = True
        with self.assertRaises(ValueError):
            ps.estimateLine3D(good, config)

        for scoring in (ps.ScoringType.ACRANSAC, ps.ScoringType.Grid):
            config = base_config(scoring=scoring)
            with self.assertRaises(ValueError):
                ps.estimateLine3D(good, config)

        config = base_config(sampler=ps.SamplerType.NAPSAC)
        with self.assertRaises(ValueError):
            ps.estimateLine3D(good, config)

        config = base_config(local=ps.LocalOptimizationType.NestedRANSAC)
        with self.assertRaises(ValueError):
            ps.estimateLine3D(good, config)

        config = base_config(final=ps.LocalOptimizationType.IteratedLSQ)
        with self.assertRaises(ValueError):
            ps.estimateLine3D(good, config)

    def test_coincident_points_no_crash(self):
        points = np.ones((5, 3))
        config = base_config(scoring=ps.ScoringType.RANSAC,
                             threshold=0.01, iterations=100)
        descriptor, inliers, score, iterations = ps.estimateLine3D(points, config)
        self.assertEqual(descriptor.shape, (2, 3))
        self.assertTrue(np.all(descriptor == 0.0))
        self.assertEqual(list(inliers), [])
        self.assertEqual(score, 0.0)
        self.assertGreaterEqual(iterations, 1)

    def test_descriptor_is_independent_of_input_permutation(self):
        points, _ = make_line_points(n_inliers=60, n_outliers=0)
        config = base_config(scoring=ps.ScoringType.RANSAC, threshold=0.02,
                             iterations=500)
        first_descriptor, first_inliers, _, _ = ps.estimateLine3D(points, config)

        rng = np.random.default_rng(99)
        order = rng.permutation(len(points))
        second_descriptor, second_inliers, _, _ = ps.estimateLine3D(
            points[order], config
        )

        self.assertGreater(
            abs(float(first_descriptor[1] @ second_descriptor[1])), 0.999
        )
        self.assertEqual(set(order[second_inliers]), set(first_inliers))


def _rotation_matrix(angle: float, axis: np.ndarray) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    one_c = 1.0 - c
    return np.array([
        [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
        [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
        [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
    ])


if __name__ == "__main__":
    unittest.main()
