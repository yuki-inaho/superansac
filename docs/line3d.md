# 3D line estimation

`pysuperansac.estimateLine3D` fits a single 3D line to an `Nx3` array of XYZ
points using the existing SupeRANSAC engine. It is a generic geometric
primitive: no dataset-specific defaults, thresholds, camera assumptions or
identity labels are embedded.

## API

```python
import numpy as np
import pysuperansac as ps

points = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.1, 0.0],
    [2.0, -0.1, 0.1],
    [3.0, 0.2, 0.0],
], dtype=np.float64)

descriptor, inliers, score, iterations = ps.estimateLine3D(points)
```

The optional second argument is a `pysuperansac.RANSACSettings` instance. The
default configuration works with a plain `Nx3` array.

### Return value

`descriptor` is a `2x3` float64 matrix:

| row | meaning |
| --- | --- |
| 0 | point on the line closest to the origin |
| 1 | unit direction |

The direction sign is canonicalized so that its component with the largest
absolute value is positive. This is a deterministic representation choice, not
an oriented physical axis: under an arbitrary rotation the same geometric line
can be emitted as `d` or `-d`. All residuals and inlier decisions are
sign-invariant.

`inliers` is a list of row indices, `score` is the robust score from the
selected scoring rule, and `iterations` is the number of RANSAC iterations
actually performed.

## Model and residual

- Minimal sample: 2 distinct finite points.
- Non-minimal fit: weighted PCA over at least 3 points.
- Squared residual: orthogonal point-to-line distance squared,

  ```
  q  = p - a
  r2 = || q - (q . d) d ||^2
  ```

  where `a` is descriptor row 0 and `d` is descriptor row 1.
- An all-coincident sample, a coincident minimal pair, or non-finite input is
  rejected before a model is produced.

The non-minimal solver uses **positional** weights: `weights[i]` belongs to
`sample[i]`, matching how the scoring and local-optimization callers build
their weight arrays. It does not index weights by the global point index.

## Supported modes

| Component | Supported values |
| --- | --- |
| Sampler | `Uniform` (PROSAC rejected pending index-bound audit) |
| Scoring | `RANSAC`, `MSAC`, `MAGSAC` |
| Local optimization | `Nothing`, `LSQ` |
| Final optimization | `Nothing`, `LSQ` |
| Termination | `RANSAC` |

MAGSAC uses `getDegreesOfFreedom() == 2` because an orthogonal point-to-line
distance in 3D has three point coordinates minus one degree of freedom along
the line. This selects the two-DOF gamma model in the existing MAGSAC tables.

The default configuration is `Uniform` + non-SPRT `MAGSAC` + `LSQ` local and
final optimization. The global `RANSACSettings` defaults target 2D
correspondence pipelines and are intentionally not changed.

## Explicitly rejected modes

The entrypoint raises `ValueError` (C++ `std::invalid_argument`) rather than
silently changing a requested mode for:

- `use_sprt=True` — SPRT is a 2D-pipeline optimization and is not part of the
  reviewed 3D-line path.
- `ACRANSAC`, `GRID`, `MINPRAN`, `GAU`, `ML` scoring — 2D-specific or not
  implemented for this estimator.
- `NAPSAC`, `ProgressiveNAPSAC`, `ImportanceSampler`, `ARSampler` samplers —
  they require 2D image bounds or point probabilities.
- local/final optimizers other than `Nothing` and `LSQ`.
- inlier selectors, non-RANSAC termination, non-finite or non-positive
  thresholds, invalid confidence, zero iterations, and `min_iterations >
  max_iterations`.
- input arrays that are not `Nx3`, contain fewer than 2 rows, or contain
  `NaN`/`Inf`.

Malformed input is rejected at the Python boundary before the engine runs, so
the C++ core never sees invalid points or thresholds.

## Units and scale

The threshold is expressed in the same units as the input points. Scaling
points and threshold together (for example metres to millimetres, `x1000`
points and `x1000` threshold) preserves the inlier set and the geometry; the
descriptor scales with the data.

## Building and testing

The extension must be built with finite-number validation enabled. Fast-math
is OFF by default because `-ffast-math` implies `-ffinite-math-only`, which can
let the compiler optimize away `std::isfinite` checks:

```bash
cmake -S . -B build-line3d -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSUPERANSAC_ENABLE_FAST_MATH=OFF \
  -DSUPERANSAC_BUILD_TESTS=ON
cmake --build build-line3d -j2
./build-line3d/tests/test_line3d_core
```

Python API tests (synthetic XYZ only):

```bash
python -m unittest discover -s tests/line3d -v
```

The Python tests locate the extension in `build-line3d/python` by default; set
`SUPERANSAC_BUILD_PYTHON` to another directory when needed.

## Choosing the distance threshold

Configure the supported modes explicitly when changing the threshold:

```python
config = ps.RANSACSettings()
config.sampler = ps.SamplerType.Uniform
config.scoring = ps.ScoringType.MAGSAC
config.local_optimization = ps.LocalOptimizationType.LSQ
config.final_optimization = ps.LocalOptimizationType.LSQ
config.use_sprt = False
config.inlier_threshold = 0.01  # same units as XYZ; choose for your measurements
descriptor, inliers, score, iterations = ps.estimateLine3D(points, config)
```
