#pragma once

#include "utils/types.h"
#include <cstdint>
#include <tuple>
#include <vector>

namespace superansac {
using Line3D = Eigen::Matrix<double, 2, 3, Eigen::RowMajor>;
using Line3DResult = std::tuple<Line3D, std::vector<size_t>, double, size_t>;

// Orthogonal (optionally weighted) least squares. Weights are nonnegative,
// one per input row. Throws std::invalid_argument for invalid/degenerate data.
Line3D fitLine3D(const DataMatrix& points, const std::vector<double>& weights = {});

// Uniform two-point RANSAC, existing MSAC score and score-gated inlier PCA
// refinement. Threshold is perpendicular distance in the input coordinate units.
// Returns (2x3 [anchor; unit direction], inlier indices, MSAC score, iterations).
// Direction's largest-magnitude component is positive. Opposite signs describe
// the same line; the anchor is not necessarily the closest point to the origin.
// Throws std::invalid_argument for invalid inputs and std::runtime_error when
// no non-degenerate hypothesis is found within the iteration budget.
Line3DResult estimateLine3D(const DataMatrix& points, double threshold = 0.03,
                          double confidence = 0.99, size_t minIterations = 100,
                          size_t maxIterations = 5000, uint64_t seed = 0,
                          bool refine = true);
}
