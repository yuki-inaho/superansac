# SupeRANSAC: One RANSAC to Rule them All ([Arxiv Paper](https://arxiv.org/abs/2506.04803))

## About

SupeRANSAC is a Python library that provides bindings for an advanced RANSAC C++ implementation using pybind11. 
It supports a wide variety of sampling, scoring, local optimization, and inlier selection techniques for robust model estimation tasks.
It provides estimators for homography, essential, fundamental matrix, rigid and absolute pose estimation.

## Installation and reproducible development

### Recommended: one locked Pixi environment

Use [Pixi](https://pixi.prefix.dev/latest/installation/) **0.82.1**, the version
used in CI. This repository has substantial native dependencies, so Pixi is the
primary workflow rather than a Python-only uv environment: `pixi.lock` records
Python, the C/C++ compiler and sysroot, CMake, Ninja, Eigen, Boost, OpenCV,
pybind11, NumPy, pytest, matplotlib, and their transitive package builds/hashes.
There is deliberately no second uv lockfile or separate just task layer.

The supported locked reference is **Linux x86_64, Python 3.12** (exact patch and
package builds are in `pixi.lock`), tested on Ubuntu 24.04 CI. A Linux host with
kernel 4.18 and glibc 2.28 or newer is required. Other platforms can still use the ordinary
CMake/pip route below, but are not validated by this lockfile. No `sudo`, system
CMake, development headers, or separate Python installation is needed for Pixi.

```sh
git clone https://github.com/yuki-inaho/superansac.git
cd superansac
pixi install --locked
pixi run --locked test
pixi run --locked demo
```

`test` builds and runs the native CTest test, installs the actual Python
extension (and its sibling native core library), then runs the Python tests.
`demo` saves `build/line3d.png` using a headless matplotlib backend and prints
accuracy metrics. Tasks always run the build/install dependencies rather than
assuming a previous build is current. The Python tests import the installed
module, not a manually supplied `PYTHONPATH`.

Useful individual commands:

```sh
pixi run --locked build        # Native CMake build in build/pixi
pixi run --locked test-native
pixi run --locked install      # Build and install into the Pixi Python environment
pixi run --locked test-python
pixi run --locked package      # sdist, then wheel rebuilt from that sdist in dist/
pixi run --locked python -I -c "import pysuperansac; print(pysuperansac.__file__)"
```

For custom demo arguments, install once, then run Python inside the environment:

```sh
pixi run --locked install
pixi run --locked python examples/example_line3d_fitting.py \
  --points 10000 --seed 7 --outlier-ratio 0.6 --plot build/line3d-large.png
```

Pixi tasks use the locked build dependencies with `--no-build-isolation` and
`--no-deps` and `--no-index`; pip does not independently resolve or download dependencies. The native and
Python builds use Release mode and disable host-specific `-march=native` tuning.
The existing fast-math default remains enabled. For the strict variant:

```sh
pixi run --locked configure -- -DSUPERANSAC_ENABLE_FAST_MATH=OFF
SKBUILD_CMAKE_ARGS=-DSUPERANSAC_ENABLE_FAST_MATH=OFF pixi run --locked test
```

The native choice remains in its CMake cache; use the same two commands with
`ON` to switch back. `CMAKE_BUILD_PARALLEL_LEVEL` is set to 2 to avoid excessive
compiler memory use. For a fresh environment, use a fresh clone and run the
quick start above. CI does exactly that with environment caching disabled,
tests fast math ON/OFF, and validates an installed wheel built from the sdist.

Commit `pixi.toml` and `pixi.lock` together. Normal use should keep `--locked`,
which rejects a manifest/lock mismatch instead of silently re-resolving. To
intentionally upgrade, edit constraints or run `pixi update`, review the lock
diff, and rerun the full tests and demo before committing.

The lock reproduces dependency packages, **not the OS kernel, CPU, or bitwise
floating-point results across machines**. It covers the self-contained line
validation, not the optional dataset/feature-matching notebook stacks. Wheels
contain the project core library but still require compatible external native
libraries; they are local environment artifacts, not portable manylinux wheels.
Do not copy a wheel built against one Pixi prefix into an unrelated environment.

### Existing CMake / pip workflows

The native CMake API and Python import/API remain unchanged. Package metadata
allows Python 3.8+, but only the locked Python 3.12 environment is validated here.
A standalone
`python -m pip install .` uses the declared scikit-build-core backend and bundles
both native project libraries correctly; it does not use `pixi.lock`. For that
route, install a C++17 compiler, Python development headers, Eigen3, Boost system,
and OpenCV development files first. For example, on Ubuntu:

```sh
sudo apt-get update
sudo apt-get install build-essential python3-dev libeigen3-dev libboost-system-dev libopencv-dev
python -m pip install .
```

These system packages are not locked by pip. Direct CMake builds are also still
available; supply `pybind11_DIR` if pybind11 is not in CMake's search path. The
legacy `setup.py` is retained, but the PEP 517 `pip install .` command above is
the supported Python packaging entry point.

# Results

Below, the AUC@10&deg; scores are shown averaged over the LaMAR, PhotoTourism, ETH3D, Kitti, 7Scenes, and ScanNet datasets.
We show results both with SuperPoint + LightGlue (top row) and RoMA (bottom) features for essential matrix (left), fundamental matrix (middle) and homography (right) estimation.  

<table align="center">
  <tr>
    <td align="center"><img src="assets/essential_matrix_splg.png" width="250"/></td>
    <td align="center"><img src="assets/fundamental_matrix_splg.png" width="250"/></td>
    <td align="center"><img src="assets/homography_matrix_splg.png" width="250"/></td>
  </tr>
  <tr>
    <td align="center"><img src="assets/essential_matrix_roma.png" width="250"/></td>
    <td align="center"><img src="assets/fundamental_matrix_roma.png" width="250"/></td>
    <td align="center"><img src="assets/homography_matrix_roma.png" width="250"/></td>
  </tr>
  <tr>
    <td align="center"><b>Essential matrix</b></td>
    <td align="center"><b>Fundamental matrix</b></td>
    <td align="center"><b>Homography</b></td>
  </tr>
</table>

Below, absolute pose (from 2D-3D correspondences) and rigid transformation (from 3D-3D matches) estimation results are shown. 
For absolute pose estimation, we integrated SupeRANSAC into the hloc library.
For rigid transformation estimation, we employ GeoTransformer features. 

<table align="center">
  <tr>
    <td align="center"><img src="assets/absolute_pose_experiments.png" width="250"/></td>
    <td align="center"><img src="assets/rigid_transform_experiments.png" width="250"/></td>
  </tr>
  <tr>
    <td align="center"><b>Absolute Pose with hloc</b></td>
    <td align="center"><b>Rigid transform with GeoTransformer</b></td>
  </tr>
</table>



# Jupyter Notebook Examples

- **Homography Fitting with SuperPoint + LightGlue Matches**  
  Example available at: [notebook](examples/example_homography_fitting_splg.ipynb)

- **Homography Fitting with RoMA Matches**  
  Example available at: [notebook](examples/example_homography_fitting_roma.ipynb)

- **Fundamental Matrix Fitting with SuperPoint + LightGlue Matches**  
  Example available at: [notebook](examples/example_fundamental_matrix_fitting_splg.ipynb)

- **Fundamental Matrix Fitting with RoMA Matches**  
  Example available at: [notebook](examples/example_fundamental_matrix_fitting_roma.ipynb)

- **Essential Matrix Fitting with SuperPoint + LightGlue Matches**  
  Example available at: [notebook](examples/example_essential_matrix_fitting_splg.ipynb)

- **Essential Matrix Fitting with RoMA Matches**  
  Example available at: [notebook](examples/example_essential_matrix_fitting_roma.ipynb)

- **Absolute Pose Estimation**  
  Example available at: [notebook](examples/example_absolute_pose_fitting.ipynb)

- **6D Rigid Transformation Estimation**  
  Example available at: [notebook](examples/example_rigid_transform_fitting.ipynb)


## Usage

The library provides several model estimation functions and settings that allow customization of the RANSAC pipeline. Below are the primary features and their usage.

### Importing the Library

```python
import pysuperansac
from pysuperansac import ScoringType, SamplerType, LocalOptimizationType, InlierSelectorType, NeighborhoodType, CameraType, RANSACSettings
```

### Example: Estimate a Homography Matrix

```python
# Example correspondences and image sizes
correspondences = [
    ([x1, y1], [x2, y2]),
    ([x3, y3], [x4, y4]),
    # Add more correspondences...
]
image_sizes = [width1, height1, width2, height2]

# Configure RANSAC settings
config = RANSACSettings()
config.min_iterations = 100
config.max_iterations = 1000
config.inlier_threshold = 3.0
config.confidence = 0.99
config.scoring = ScoringType.MAGSAC
config.sampler = SamplerType.PROSAC
config.local_optimization = LocalOptimizationType.GCRANSAC
config.final_optimization = LocalOptimizationType.LSQ

# Estimate homography
result = pysuperansac.estimateHomography(correspondences, image_sizes, config=config)
print("Estimated Homography Matrix:", result) 
```

### Supported Estimation Functions

#### 1. Homography Estimation
- **Function**: `pysuperansac.estimateHomography`
- **Description**: Estimates a homography matrix from 2D-2D point correspondences.
- **Parameters**:
  - `correspondences`: A list of paired 2D points.
  - `image_sizes`: A tuple of source and target image sizes.
  - `config`: An instance of `RANSACSettings`.
  - `probabilities` *(optional)*: Correspondence probabilities.

#### 2. Fundamental Matrix Estimation
- **Function**: `pysuperansac.estimateFundamentalMatrix`
- **Description**: Estimates a fundamental matrix from 2D-2D point correspondences.
- **Parameters**:
  - `correspondences`: A list of paired 2D points.
  - `image_sizes`: A tuple of source and target image sizes.
  - `config`: An instance of `RANSACSettings`.
  - `probabilities` *(optional)*: Correspondence probabilities.

#### 3. Essential Matrix Estimation
- **Function**: `pysuperansac.estimateEssentialMatrix`
- **Description**: Estimates an essential matrix using 2D-2D point correspondences and intrinsic matrices.
- **Parameters**:
  - `correspondences`: A list of paired 2D points.
  - `image_sizes`: A tuple of source and target image sizes.
  - `config`: An instance of `RANSACSettings`.
  - `intrinsics_src`: Source camera intrinsic matrix.
  - `intrinsics_dst`: Destination camera intrinsic matrix.
  - `probabilities` *(optional)*: Correspondence probabilities.

#### 4. Rigid Transformation Estimation
- **Function**: `pysuperansac.estimateRigidTransform`
- **Description**: Estimates a 6D rigid transformation matrix from 3D-3D point correspondences.
- **Parameters**:
  - `correspondences`: A list of paired 3D points.
  - `bounding_box_sizes`: Size of the bounding box.
  - `config`: An instance of `RANSACSettings`.
  - `probabilities` *(optional)*: Correspondence probabilities.

#### 5. Absolute Pose Estimation
- **Function**: `pysuperansac.estimateAbsolutePose`
- **Description**: Estimates the absolute pose of a camera using 2D-3D correspondences.
- **Parameters**:
  - `correspondences`: A list of paired 2D-3D points.
  - `camera_type`: Type of the camera (e.g., `CameraType.SimpleRadial`).
  - `camera_params`: Camera parameters.
  - `config`: An instance of `RANSACSettings`.
  - `probabilities` *(optional)*: Correspondence probabilities.


### Advanced Configuration: RANSACSettings

The ```RANSACSettings``` class provides fine-grained control over the RANSAC pipeline. Customize it as needed:

```
config = RANSACSettings()
config.min_iterations = 100
config.max_iterations = 1000
config.inlier_threshold = 3.0
config.confidence = 0.99
config.scoring = ScoringType.MAGSAC
config.sampler = SamplerType.PROSAC
config.local_optimization = LocalOptimizationType.LSQ
config.neighborhood = NeighborhoodType.Grid
```

### Enumerations and Their Values

The library supports several enumeration types to customize sampling, scoring, and other pipeline components:

**Scoring Types**
- ScoringType.RANSAC
- ScoringType.MSAC
- ScoringType.MAGSAC
- ScoringType.ACRANSAC

**Sampler Types**
- SamplerType.Uniform
- SamplerType.PROSAC
- SamplerType.NAPSAC
- SamplerType.ProgressiveNAPSAC
- SamplerType.ImportanceSampler
- SamplerType.ARSampler

**Local Optimization Types**
- LocalOptimizationType.Nothing
- LocalOptimizationType.LSQ
- LocalOptimizationType.IteratedLSQ
- LocalOptimizationType.NestedRANSAC
- LocalOptimizationType.GCRANSAC

**Neighborhood Types**
- NeighborhoodType.Grid
- NeighborhoodType.BruteForce

**Camera Types**
- CameraType.SimpleRadial
- CameraType.SimplePinhole

## Evaluation - Python

The testing scripts for SupeRANSAC and the baselines are located in folder `tests/`.
The results for essential, fundamental, and homography matrix estimation are obtained by running
```
python tests/X/tester-X-superansac.py
```
where X is essential/fundamental/homography matrix. Note that this process may take a while. 
However, before running, you need to install the feature detectors and set up the datasets as described below. 

### Install feature detectors

To use SuperPoint + LightGlue features, install LightGlue as follows:
```
git clone https://github.com/cvg/LightGlue.git && cd LightGlue
python -m pip install -e .
```

To use RoMA features, install it as follows:
```
git clone https://github.com/Parskatt/RoMa && cd RoMa
pip install -e .
```

### Evaluation on the PhotoTourism dataset
Download the data from the CVPR tutorial "RANSAC in 2020":
```
wget http://cmp.felk.cvut.cz/~mishkdmy/CVPR-RANSAC-Tutorial-2020/RANSAC-Tutorial-Data-EF.tar
tar -xf  RANSAC-Tutorial-Data-EF.tar
```

Then run the notebook `examples/relative_pose_evaluation_phototourism.ipynb`.

### Evaluation on the ScanNet dataset
Download the data from the test set for relative pose estimation used in SuperGlue (~250Mb for 1500 image pairs only):
```
wget https://www.polybox.ethz.ch/index.php/s/lAZyxm62WUh27Zl/download
unzip ScanNet_test.zip -d <path to extract the ScanNet test set>
```

Then run the notebook `examples/relative_pose_evaluation_scannet.ipynb`.


### Evaluation on the 7Scenes dataset
Download the [7Scenes dataset](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) and put it where it suits you. You can also only download one scene and later specify this scene in the dataloader constructor.

Then run the script `runners/run_7scenes.py`.


### Evaluation on the ETH3D dataset
Download the [ETH3D dataset](https://www.eth3d.net/datasets) (training split of the high-res multi-view, undistorted images + GT extrinsics & intrinsics should be enough) and put it where it suits you. The input argument 'downsize_factor' can be used to downscale the images, because they can be quite large otherwise.

Then run the script `runners/run_eth3d.py`.


### Evaluation on the LaMAR dataset
Download the [CAB scene of the LaMAR dataset](https://cvg-data.inf.ethz.ch/lamar/CAB.zip), and unzip it to your favourite location. Note that we only use the images in `CAB/sessions/query_val_hololens`.


### Evaluation on the KITTI dataset
Download the [KITTI odometry dataset](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) (grayscale images and poses), and unzip them to your favourite location.

## 3D point-cloud line fitting

`fitLine3D` and `estimateLine3D` fit a **single infinite 3D line** to an `N x 3`
XYZ array. They do not project onto an image plane or extract multiple lines.
Both return a `2 x 3` descriptor: row 0 is a point on the line, row 1 is a unit
3D direction, so the line is `anchor + t * direction`. The largest-magnitude
direction component is made positive for deterministic output, but `d` and `-d`
represent the same geometric line. The anchor is the fitting subset's centroid
(weighted for weighted least squares), not a unique closest-to-origin point.

```python
import numpy as np
import pysuperansac

# Replace with your point cloud. Lists, float32, strided and Fortran-order
# arrays are also accepted and converted to a row-major double matrix.
points = np.array([[0, 1, 2], [1, 3, 5], [2, 5, 8], [9, -7, 2]], dtype=float)

# Ordinary orthogonal least squares: sensitive to outliers.
line_lsq = pysuperansac.fitLine3D(points)
# Optional nonnegative weights, one per input point:
line_weighted = pysuperansac.fitLine3D(points, weights=[1, 1, 1, 0])

# Robust fit using the existing native SupeRansac engine.
line, inliers, score, iterations = pysuperansac.estimateLine3D(
    points,
    threshold=0.03,       # perpendicular Euclidean distance, in XYZ units
    confidence=0.99,      # adaptive two-point RANSAC stopping estimate
    min_iterations=100,
    max_iterations=5000,
    seed=7,
    refine=True,
)
anchor, direction = line
```

The implementation supplies a native `Line3DEstimator` and `Line3DSolver` under
`include/estimators`. A two-point hypothesis rejects coincident/numerically
indistinguishable points. Non-minimal fitting uses the principal eigenvector of
the centered, scaled 3D covariance (orthogonal PCA/total least squares), with
optional weights. Residuals are `||(point - anchor) x direction||`. The solver
uses the existing fixed-capacity `ModelMatrix` without changing existing APIs.
C++ applications can include `line3d.h` and call `superansac::fitLine3D` or
`superansac::estimateLine3D` with the same defaults and return ordering as Python.

The robust convenience API deliberately supports **uniform sampling + MSAC
scoring + optional score-gated inlier PCA refinement**. It does not silently
apply image-specific samplers or null models to XYZ data. MSAC returns the
negative sum of squared residuals truncated at `threshold**2` (larger is better).
Inlier indices refer to the original input order and use a strict distance
`< threshold` test. Refinement recollects inliers for up to two additional
score-gated rounds; a refinement that worsens the score is not accepted. The
low-level estimator reports two residual degrees of freedom. AC-RANSAC's
image-area null model is explicitly unsupported.

Reusing a seed on the same input, settings, and build reproduces the same run.
Bit-for-bit agreement across compilers/platforms is not promised. Confidence is
the standard RANSAC stopping estimate, not an accuracy guarantee, especially
with repeated points, highly contaminated data, or a small iteration budget.
The same patch corrects the shared uniform sampler's inclusive upper-bound bug:
it now samples indices in `[0, N)` for the current pool, including after a pool
size change. Other estimators retain their APIs, but corrected samples can
change their numerical results.

Invalid shapes, non-finite coordinates, invalid weights/settings, all-coincident
clouds, or a non-unique PCA direction raise `ValueError` in Python
(`std::invalid_argument` in C++). Failure to find a valid hypothesis within the
budget raises `RuntimeError`. At least two positive-weight, numerically distinct
points are required for weighted fitting. Duplicate points are allowed in an
otherwise valid cloud. Low-level solver weights follow **sample order** when
an index sample is supplied, matching the scoring weight-vector contract.
Finite checks remain active with the project's default fast-math flags.
Extremely large offsets with tiny local variation are limited by double
precision; center/rescale such data first. This is unbounded line fitting,
not segment endpoint estimation or multi-line extraction.

### Reproducible synthetic validation

Use the [locked Pixi environment](#recommended-one-locked-pixi-environment)
from the repository root:

```sh
pixi install --locked
pixi run --locked test
pixi run --locked demo
```

The example generates seeded XYZ line points with isotropic Gaussian noise and
uniform 3D outliers, shuffles them, and compares robust fitting with all-point
PCA. It reports sign-invariant angular error, transverse position error measured
at the ground-truth anchor, true-inlier residual RMSE, and inlier precision and
recall. Position error is not the shortest distance between two infinite skew
lines, which can hide a large local displacement.

Tests cover multiple line orientations, 0/30/60/80% outliers across three seeds,
PCA refinement, threshold/returned-mask consistency, weighted and indexed fits,
rotated/scaled/translated clouds, seed replay, duplicates, exact two-point input,
isotropic degeneracy, NaN/Inf, invalid parameters, array memory layouts, sampler
bounds, and a pre-existing homography-estimation smoke test. GitHub Actions runs
the suite with fast math both enabled and disabled; no external datasets are
needed. Existing dataset-dependent benchmarks are not part of this test suite.
