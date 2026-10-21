#include "estimators/estimator_line3d.h"
#include "samplers/uniform_random_sampler.h"
#include "line3d.h"
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
int main() {
    using namespace superansac;
    estimator::Line3DEstimator estimator;
    DataMatrix points(5, 3);
    points << 50, 20, 10, 1, 2, 3, 3, 2, 3, 5, 2, 3, 1, 2, 3;
    const size_t sample[] = {1, 2};
    std::vector<models::Model> models;
    require(estimator.estimateModel(points, sample, &models), "minimal solver failed");
    require(estimator.isValidModel(models[0]), "invalid descriptor");
    const double query[] = {40, 5, 7};
    require(std::abs(estimator.squaredResidual(query, models[0]) - 25) < 1e-10, "wrong orthogonal residual");
    const size_t duplicate[] = {1, 4}, outside[] = {1, 5};
    require(!estimator.isValidSample(points, duplicate), "duplicate accepted");
    require(!estimator.isValidSample(points, outside), "out-of-range sample accepted");
    models.clear();
    require(!estimator.estimateModel(points, duplicate, &models) && models.empty(), "degenerate model emitted");
    const size_t subset[] = {1, 3, 0};
    const double weights[] = {1, 3, 0};
    const size_t count = 3;
    require(estimator.estimateModelNonminimal(points, subset, count, &models, weights), "weighted PCA failed");
    require((models.back().getData().row(0) - Eigen::RowVector3d(4, 2, 3)).norm() < 1e-12,
            "subset weights must be in sample order");
    double batch[3];
    estimator.squaredResiduals(points, models.back(), 1, 3, batch);
    for (size_t i = 0; i < 3; ++i)
        require(batch[i] == estimator.squaredResidual(points.row(i + 1).data(), models.back()), "batch residual mismatch");
    const auto nan = std::numeric_limits<double>::quiet_NaN();
    points(1, 0) = nan;
    require(!estimator.isValidSample(points, sample), "NaN accepted by minimal validation");
    models.clear();
    require(!estimator.estimateModelNonminimal(points, subset, count, &models, weights), "NaN accepted by PCA");

    // A regression for the old inclusive-N upper bound, including two-point
    // clouds, shrinking pools, the final row, and seed replay.
    samplers::UniformRandomSampler a, b;
    a.setSeed(42); b.setSeed(42);
    a.initialize(size_t(100)); b.initialize(size_t(99));
    for (size_t n : {size_t(2), size_t(3), size_t(30), size_t(4)}) {
        bool seenLast = false;
        for (size_t i = 0; i < 1000; ++i) {
            size_t x[2], y[2];
            require(a.sample(n, 2, x) && b.sample(n, 2, y), "sampler failed");
            require(x[0] < n && x[1] < n && x[0] != x[1], "out-of-range or repeated index");
            require(x[0] == y[0] && x[1] == y[1], "seed sequence differs");
            seenLast |= x[0] == n - 1 || x[1] == n - 1;
        }
        require(seenLast, "last row excluded from sampling");
    }
    size_t x[2];
    require(!a.sample(size_t(1), 2, x), "oversize sample accepted");
    require(!a.sample(size_t(2), -1, x), "negative sample accepted");
    require(a.sample(size_t(0), 0, x), "empty sample failed");
    DataMatrix exact(2, 3); exact << 1, 2, 3, 4, 6, 8;
    const auto result = estimateLine3D(exact, 1e-6, 0.99, 1, 10, 2);
    require(std::get<1>(result).size() == 2, "two-point RANSAC failed");
    std::cout << "Line3D solver, residual, indexed weighting, sampler, and engine tests passed\n";
}
