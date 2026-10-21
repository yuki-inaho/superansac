"""Run with the built Python module on PYTHONPATH; no external data required."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import pysuperansac as sac

spec = importlib.util.spec_from_file_location(
    "line3d_example", Path(__file__).resolve().parents[2] / "examples/example_line3d_fitting.py")
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


@pytest.mark.parametrize("direction", [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, -2, 3]])
def test_exact_lines(direction):
    d = np.array(direction, dtype=float)
    d /= np.linalg.norm(d)
    p = np.array([1.5, -2, 4])
    points = p + np.linspace(-3, 3, 31)[:, None] * d
    for model in [sac.fitLine3D(points), sac.estimateLine3D(points, threshold=1e-5, seed=8)[0]]:
        assert model.shape == (2, 3)
        assert abs(np.dot(model[1], d)) == pytest.approx(1)
        assert np.linalg.norm(np.cross(p - model[0], model[1])) < 1e-10
        assert model[1, np.argmax(abs(model[1]))] > 0


@pytest.mark.parametrize("ratio", [0, 0.3, 0.6, 0.8])
@pytest.mark.parametrize("seed", [0, 7, 42])
def test_noisy_outliers(ratio, seed):
    points, truth, p, d = example.make_cloud(seed, ratio)
    model, indices, score, iterations = sac.estimateLine3D(points, threshold=0.04, seed=seed)
    stats = example.metrics(model, indices, points, truth, p, d)
    assert stats["direction_error_deg"] < 0.25
    assert stats["position_error"] < 0.005
    assert stats["precision"] > 0.98
    assert stats["recall"] > 0.98
    residuals = np.linalg.norm(np.cross(points - model[0], model[1]), axis=1)
    np.testing.assert_array_equal(indices, np.flatnonzero(residuals < 0.04))
    assert score == pytest.approx(-np.minimum(residuals**2, 0.04**2).sum())
    assert 100 <= iterations <= 5000


def test_reproducible_data_and_ransac():
    points = example.make_cloud(15)[0]
    np.testing.assert_array_equal(points, example.make_cloud(15)[0])
    first = sac.estimateLine3D(points, seed=123)
    second = sac.estimateLine3D(points, seed=123)
    np.testing.assert_array_equal(first[0], second[0])
    assert first[1:] == second[1:]
    # Different seeds need not give different final lines after convergence.


def test_pca_refinement_and_baseline():
    points, truth, p, d = example.make_cloud(7)
    result = sac.estimateLine3D(points, threshold=0.04, seed=7)
    raw = sac.estimateLine3D(points, threshold=0.04, seed=7, refine=False)
    baseline = sac.fitLine3D(points)
    robust = example.metrics(result[0], result[1], points, truth, p, d)
    ordinary = example.metrics(baseline, range(len(points)), points, truth, p, d)
    assert robust["direction_error_deg"] < ordinary["direction_error_deg"]
    assert robust["true_inlier_rmse"] < ordinary["true_inlier_rmse"]
    assert result[2] >= raw[2]
    # Verify returned refined direction equals PCA on its stable consensus.
    fit = sac.fitLine3D(points[result[1]])
    assert abs(np.dot(fit[1], result[0][1])) > 1 - 1e-10


def test_weights_and_layouts():
    points = np.array([[0, 2, 3], [2, 2, 3], [4, 2, 3], [50, 50, 50]], dtype=float)
    weights = [1, 2, 1, 0]
    fit = sac.fitLine3D(points, weights)
    np.testing.assert_allclose(fit, [[2, 2, 3], [1, 0, 0]], atol=1e-12)
    for arr in [np.asfortranarray(points), points.astype(np.float32), points.tolist(),
                np.repeat(points, 2, axis=0)[::2]]:
        np.testing.assert_allclose(sac.fitLine3D(arr, weights), fit)
        assert len(sac.estimateLine3D(arr, threshold=0.01)[1]) == 3
    np.testing.assert_allclose(sac.fitLine3D(points, np.array(weights) * 1e100), fit)


def test_translation_scale_and_rotation():
    points, truth, p, d = example.make_cloud(21, 0.3)
    result = sac.estimateLine3D(points, threshold=0.04, seed=21)
    q, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(3, 3)))
    for scale in [1e-6, 10, 1e6]:
        offset = scale * np.array([100, -200, 300])
        transformed = scale * (points @ q.T) + offset
        new = sac.estimateLine3D(transformed, threshold=scale * 0.04, seed=21)
        np.testing.assert_array_equal(new[1], result[1])
        assert abs(np.dot(new[0][1], q @ result[0][1])) > 1 - 1e-10
        assert np.linalg.norm(np.cross(new[0][0] - (scale * (q @ result[0][0]) + offset), new[0][1])) < scale * 1e-9


def test_two_points_and_duplicates():
    points = np.array([[1, 2, 3], [4, 6, 8]], dtype=float)
    result = sac.estimateLine3D(points, threshold=1e-6, min_iterations=1, max_iterations=5)
    assert result[1] == [0, 1]
    cloud = np.repeat(points, 100, axis=0)
    assert len(sac.estimateLine3D(cloud, threshold=1e-6)[1]) == 200


@pytest.mark.parametrize("points", [np.empty((0, 3)), np.ones((1, 3)), np.ones((5, 2)),
                                     np.ones((5, 4)), np.ones((20, 3)),
                                     [[1, 2, 3], [1 + 1e-15, 2, 3]],
                                     [[0, 0, 0], [np.nan, 1, 2]],
                                     [[0, 0, 0], [np.inf, 1, 2]],
                                     [[0, 0, 0], [-np.inf, 1, 2]]])
def test_invalid_points(points):
    for fn in [sac.fitLine3D, sac.estimateLine3D]:
        with pytest.raises(ValueError):
            fn(points)


@pytest.mark.parametrize("weights", [[1], [-1, 1, 1], [0, 0, 0], [1, 0, 0],
                                      [1, np.nan, 1], [1, np.inf, 1]])
def test_invalid_weights(weights):
    with pytest.raises(ValueError):
        sac.fitLine3D([[0, 0, 0], [1, 1, 1], [2, 2, 2]], weights)


@pytest.mark.parametrize("kwargs", [{"threshold": x} for x in [0, -1, np.nan, np.inf, 1e200, 1e-200]] +
                         [{"confidence": x} for x in [0, 1, -0.1, np.nan, np.inf]] +
                         [{"max_iterations": 0}, {"min_iterations": 10, "max_iterations": 5}])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        sac.estimateLine3D([[0, 0, 0], [1, 1, 1]], **kwargs)


def test_isotropic_pca_is_not_unique():
    points = np.vstack([np.eye(3), -np.eye(3)])
    with pytest.raises(ValueError):
        sac.fitLine3D(points)


def test_existing_homography_smoke():
    rng = np.random.default_rng(8)
    xy = rng.uniform(0, 100, (30, 2))
    uv = xy + [10, 20]
    config = sac.RANSACSettings()
    config.sampler = sac.SamplerType.Uniform
    config.scoring = sac.ScoringType.MSAC
    config.local_optimization = sac.LocalOptimizationType.LSQ
    config.final_optimization = sac.LocalOptimizationType.LSQ
    config.use_sprt = False
    config.min_iterations = 30
    config.max_iterations = 100
    h, inliers, _, _ = sac.estimateHomography(np.hstack([xy, uv]), np.array([200.] * 4), config=config)
    np.testing.assert_allclose(h / h[2, 2], [[1, 0, 10], [0, 1, 20], [0, 0, 1]], atol=1e-5)
    assert len(inliers) == len(xy)
