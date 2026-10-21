#include "line3d.h"
#include "superansac.h"
#include "estimators/estimator_line3d.h"
#include "samplers/uniform_random_sampler.h"
#include "scoring/msac_scoring.h"
#include "local_optimization/least_squares_optimizer.h"
#include "termination/ransac_criterion.h"
#include <limits>
#include <stdexcept>

namespace superansac {
namespace {
void validatePoints(const DataMatrix& points) {
    if (points.cols() != 3 || points.rows() < 2)
        throw std::invalid_argument("points must have shape (N, 3), with N >= 2.");
    if (!estimator::solver::line3d_detail::finite(points))
        throw std::invalid_argument("points must contain only finite coordinates.");
}
}
Line3D fitLine3D(const DataMatrix& points, const std::vector<double>& weights) {
    validatePoints(points);
    if (!weights.empty() && weights.size() != size_t(points.rows()))
        throw std::invalid_argument("weights must have one entry per point.");
    estimator::solver::Line3DSolver solver;
    std::vector<models::Model> models;
    if (!solver.estimateModel(points, nullptr, points.rows(), models, weights.empty() ? nullptr : weights.data()))
        throw std::invalid_argument("Cannot fit a unique line: degenerate points or invalid weights.");
    return models.front().getData();
}
Line3DResult estimateLine3D(const DataMatrix& points, double threshold, double confidence,
                          size_t minIterations, size_t maxIterations, uint64_t seed, bool refine) {
    validatePoints(points);
    using estimator::solver::line3d_detail::finite;
    if (!finite(threshold) || !(threshold > 0.0) || !finite(threshold * threshold) ||
        !(threshold * threshold > 0.0))
        throw std::invalid_argument("threshold must be positive and have a finite, nonzero square.");
    if (!finite(confidence) || !(confidence > 0.0 && confidence < 1.0))
        throw std::invalid_argument("confidence must be strictly between 0 and 1.");
    if (maxIterations == 0 || minIterations > maxIterations)
        throw std::invalid_argument("Require 0 <= min_iterations <= max_iterations and max_iterations > 0.");
    // Avoid thousands of futile draws for a fully coincident cloud.
    bool hasDistinctPoints = false;
    const Eigen::Vector3d first = points.row(0).transpose();
    for (Eigen::Index i = 1; i < points.rows(); ++i) {
        if (estimator::solver::line3d_detail::distinct(first, points.row(i).transpose())) {
            hasDistinctPoints = true;
            break;
        }
    }
    if (!hasDistinctPoints)
        throw std::invalid_argument("At least two numerically distinct points are required.");

    estimator::Line3DEstimator estimator;
    samplers::UniformRandomSampler sampler;
    sampler.setSeed(seed);
    sampler.initialize(points);
    scoring::MSACScoring score;
    score.setThreshold(threshold);
    termination::RANSACCriterion termination;
    termination.setConfidence(confidence);
    local_optimization::LeastSquaresOptimizer optimizer;
    optimizer.setUseInliers(true);
    optimizer.setExtraRefinementRounds(2);
    RANSACSettings settings;
    settings.inlierThreshold = threshold;
    settings.confidence = confidence;
    settings.minIterations = minIterations;
    settings.maxIterations = maxIterations;
    settings.scoring = scoring::ScoringType::MSAC;
    settings.sampler = samplers::SamplerType::Uniform;
    settings.useSprt = false;
    settings.inlierSelector = inlier_selector::InlierSelectorType::None;
    settings.localOptimizationInsideTheLoop = true; // Enables adaptive termination.
    settings.localOptimization = refine ? local_optimization::LocalOptimizationType::LSQ : local_optimization::LocalOptimizationType::None;
    settings.finalOptimization = settings.localOptimization;

    SupeRansac ransac;
    ransac.setSettings(settings);
    ransac.setEstimator(&estimator);
    ransac.setSampler(&sampler);
    ransac.setScoring(&score);
    ransac.setTerminationCriterion(&termination);
    ransac.setInlierSelector(nullptr);
    ransac.setLocalOptimizer(refine ? &optimizer : nullptr);
    ransac.setFinalOptimizer(refine ? &optimizer : nullptr);
    ransac.run(points);
    if (ransac.getInliers().size() < 2 || !estimator.isValidModel(ransac.getBestModel()))
        throw std::runtime_error("No non-degenerate 3D line found within the RANSAC budget.");
    return {ransac.getBestModel().getData(), ransac.getInliers(),
            ransac.getBestScore().getValue(), ransac.getIterationNumber()};
}
}
