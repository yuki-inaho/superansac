"""Seeded single-line point-cloud validation, with no downloaded datasets.

Run: python examples/example_line3d_fitting.py --seed 7 --outlier-ratio 0.6
Optional: --plot /tmp/line3d.png (requires matplotlib).
"""
import argparse
import json
import numpy as np
import pysuperansac


def make_cloud(seed=7, outlier_ratio=0.6, n=1000, noise=0.01):
    if not 0 <= outlier_ratio < 1 or n < 2 or noise < 0:
        raise ValueError("Require 0 <= outlier_ratio < 1, n >= 2, and noise >= 0")
    rng = np.random.default_rng(seed)
    anchor = np.array([1.0, -2.0, 0.5])
    direction = np.array([1.0, 2.0, -3.0])
    direction /= np.linalg.norm(direction)
    n_inliers = int(round(n * (1 - outlier_ratio)))
    if n_inliers < 2:
        raise ValueError("The cloud must contain at least two true inliers")
    t = rng.uniform(-3, 3, n_inliers)
    # Isotropic 3D measurement noise: the perpendicular residual has 2 DOF.
    inliers = anchor + t[:, None] * direction + rng.normal(0, noise, (n_inliers, 3))
    outliers = anchor + rng.uniform(-4, 4, (n - n_inliers, 3))
    points = np.vstack([inliers, outliers])
    truth = np.arange(n) < n_inliers
    permutation = rng.permutation(n)
    return points[permutation], truth[permutation], anchor, direction


def metrics(model, indices, points, truth, true_anchor, true_direction):
    anchor, direction = model
    cosine = np.clip(abs(np.dot(direction, true_direction)), 0, 1)
    angle = float(np.degrees(np.arccos(cosine)))
    # Transverse displacement at the reference anchor, not distance between
    # two infinite skew lines (which can conceal a large local offset).
    position = float(np.linalg.norm(np.cross(true_anchor - anchor, direction)))
    predicted = np.zeros(len(points), dtype=bool)
    predicted[indices] = True
    tp = int(np.sum(predicted & truth))
    residuals = np.linalg.norm(np.cross(points[truth] - anchor, direction), axis=1)
    return {"direction_error_deg": angle, "position_error": position,
            "precision": tp / max(1, int(predicted.sum())),
            "recall": tp / max(1, int(truth.sum())),
            "true_inlier_rmse": float(np.sqrt(np.mean(residuals**2)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--outlier-ratio", type=float, default=0.6)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--threshold", type=float, default=0.04)
    parser.add_argument("--points", type=int, default=1000)
    parser.add_argument("--plot", help="Save a 3D comparison plot; needs matplotlib")
    args = parser.parse_args()
    points, truth, anchor, direction = make_cloud(args.seed, args.outlier_ratio, args.points, args.noise)
    model, inliers, score, iterations = pysuperansac.estimateLine3D(
        points, threshold=args.threshold, seed=args.seed)
    baseline = pysuperansac.fitLine3D(points)
    report = {"seed": args.seed, "point_count": len(points), "outlier_ratio": args.outlier_ratio,
              "noise_sigma": args.noise, "threshold": args.threshold,
              "model": model.tolist(), "inlier_count": len(inliers), "score": score,
              "iterations": iterations,
              "ransac": metrics(model, inliers, points, truth, anchor, direction),
              "all_point_pca": metrics(baseline, np.arange(len(points)), points, truth, anchor, direction)}
    print(json.dumps(report, indent=2))
    if args.plot:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(*points[~truth].T, s=5, alpha=0.2, label="outliers")
        ax.scatter(*points[truth].T, s=6, label="true inliers")
        for line, label in [(np.vstack([anchor, direction]), "truth"),
                            (model, "RANSAC + PCA"), (baseline, "all-point PCA")]:
            segment = line[0] + np.array([-3, 3])[:, None] * line[1]
            ax.plot(*segment.T, label=label)
        ax.set(xlabel="X", ylabel="Y", zlabel="Z", title="3D line fitting")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)


if __name__ == "__main__":
    main()
