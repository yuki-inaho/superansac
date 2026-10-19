// Synthetic-only unit tests for the generic 3D-line solvers.
// No external data, private paths or production artifacts are referenced.

#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include "estimators/line3d_utils.h"
#include "estimators/solver_line3d_minimal.h"
#include "estimators/solver_line3d_pca.h"
#include "samplers/uniform_random_sampler.h"
#include "models/model.h"
#include "utils/types.h"

namespace
{
	int g_failures = 0;

	void expect(const bool kCondition_, const std::string &kMessage_)
	{
		if (!kCondition_)
		{
			std::cerr << "FAIL: " << kMessage_ << std::endl;
			++g_failures;
		}
	}

	void expectNear(const double kActual_, const double kExpected_,
		const double kTolerance_, const std::string &kMessage_)
	{
		if (!std::isfinite(kActual_) ||
			std::abs(kActual_ - kExpected_) > kTolerance_)
		{
			std::cerr << "FAIL: " << kMessage_ << " (actual=" << kActual_
				<< ", expected=" << kExpected_ << ")" << std::endl;
			++g_failures;
		}
	}

	DataMatrix makeData(const std::vector<Eigen::Vector3d> &kPoints_)
	{
		DataMatrix data(static_cast<Eigen::Index>(kPoints_.size()), 3);
		for (size_t i = 0; i < kPoints_.size(); ++i)
			data.row(static_cast<Eigen::Index>(i)) = kPoints_[i].transpose();
		return data;
	}

	double pointToLineDistance(
		const Eigen::Vector3d &kPoint_,
		const ModelMatrix &kDescriptor_)
	{
		return std::sqrt(superansac::estimator::line3d::squaredResidual(
			kPoint_.data(), kDescriptor_));
	}

	void testMinimalExactLine()
	{
		const Eigen::Vector3d p0(1.0, 2.0, 3.0);
		const Eigen::Vector3d p1(4.0, 7.0, 5.0);
		const DataMatrix data = makeData({p0, p1});
		const size_t sample[2] = {0, 1};

		superansac::estimator::solver::Line3DMinimalSolver solver;
		std::vector<superansac::models::Model> models;
		expect(solver.sampleSize() == 2, "minimal sample size is 2");
		expect(solver.estimateModel(data, sample, 2, models, nullptr),
			"minimal solver accepts two distinct points");
		expect(models.size() == 1, "minimal solver returns one model");

		const ModelMatrix &descriptor = models[0].getData();
		expect(superansac::estimator::line3d::isValidDescriptor(descriptor),
			"minimal descriptor satisfies the contract");
		expectNear(pointToLineDistance(p0, descriptor), 0.0, 1e-12,
			"first point lies on the exact line");
		expectNear(pointToLineDistance(p1, descriptor), 0.0, 1e-12,
			"second point lies on the exact line");

		const Eigen::Vector3d direction = descriptor.row(1).transpose();
		const Eigen::Vector3d expected = (p1 - p0).normalized();
		expectNear(std::abs(direction.dot(expected)), 1.0, 1e-12,
			"minimal direction matches the two-point direction");
		const Eigen::Vector3d closest = descriptor.row(0).transpose();
		expectNear(closest.dot(direction), 0.0, 1e-12,
			"descriptor point is perpendicular to the direction");
		expectNear(
			(closest - superansac::estimator::line3d::closestPointToOrigin(
				p0, direction)).norm(),
			0.0, 1e-12, "descriptor point is the closest point to the origin");

		// Sign canonicalization: the largest-|component| component is positive.
		Eigen::Index largest = 0;
		direction.cwiseAbs().maxCoeff(&largest);
		expect(direction(largest) > 0.0, "direction sign is canonical");
	}

	void testMinimalDegeneracy()
	{
		const Eigen::Vector3d p0(1.0, 2.0, 3.0);
		const DataMatrix data = makeData({p0, p0});
		const size_t sample[2] = {0, 1};
		superansac::estimator::solver::Line3DMinimalSolver solver;
		std::vector<superansac::models::Model> models;
		expect(!solver.estimateModel(data, sample, 2, models, nullptr),
			"coincident minimal sample is rejected");

		DataMatrix nonFinite = data;
		nonFinite(1, 2) = std::numeric_limits<double>::quiet_NaN();
		expect(!solver.estimateModel(nonFinite, sample, 2, models, nullptr),
			"non-finite minimal sample is rejected");
	}

	void testWeightedPcaAndWeightIndexing()
	{
		const std::vector<Eigen::Vector3d> points = {
			{0.0, 0.0, 0.0},
			{1.0, 0.2, 0.0},
			{2.0, -0.1, 0.1},
			{3.0, 0.3, 0.2},
			{4.0, -0.2, -0.1},
			{5.0, 0.1, 0.3}};
		const DataMatrix data = makeData(points);

		// Deliberately non-sorted sample order: positional weights must follow
		// the sample order, not the global point index.
		const size_t sample[4] = {3, 0, 4, 2};
		const double weights[4] = {0.5, 2.0, 1.0, 4.0};

		superansac::estimator::solver::Line3DPCASolver solver;
		std::vector<superansac::models::Model> models;
		expect(solver.sampleSize() == 3, "PCA sample size is 3");
		expect(solver.estimateModel(data, sample, 4, models, weights),
			"weighted PCA accepts four finite non-negative weights");
		expect(models.size() == 1, "PCA solver returns one model");

		auto weightedFit = [&](const bool kPositional_)
		{
			double weightSum = 0.0;
			Eigen::Vector3d mean = Eigen::Vector3d::Zero();
			for (size_t i = 0; i < 4; ++i)
			{
				const double w = kPositional_ ? weights[i] : weights[sample[i]];
				mean += w * points[sample[i]];
				weightSum += w;
			}
			mean /= weightSum;

			Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
			for (size_t i = 0; i < 4; ++i)
			{
				const double w = kPositional_ ? weights[i] : weights[sample[i]];
				const Eigen::Vector3d offset = points[sample[i]] - mean;
				covariance += w * offset * offset.transpose();
			}
			covariance /= weightSum;

			Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(covariance);
			Eigen::Vector3d direction = eig.eigenvectors().col(2);
			Eigen::Vector3d closest =
				mean - mean.dot(direction) * direction;
			return std::make_pair(closest, direction);
		};

		const auto expected = weightedFit(true);
		const auto wrong = weightedFit(false);
		const ModelMatrix &descriptor = models[0].getData();
		const Eigen::Vector3d closest = descriptor.row(0).transpose();
		const Eigen::Vector3d direction = descriptor.row(1).transpose();

		expectNear(std::abs(direction.dot(expected.second)), 1.0, 1e-12,
			"weighted PCA direction matches analytic positional-weight PCA");
		expectNear((closest - expected.first).norm(), 0.0, 1e-12,
			"weighted PCA closest point matches analytic positional weights");
		expect((expected.first - wrong.first).norm() > 1e-6 ||
			std::abs(expected.second.dot(wrong.second)) < 1.0 - 1e-9,
			"the test data distinguishes positional from global-index weights");
		expect((closest - wrong.first).norm() > 1e-9 ||
			std::abs(direction.dot(wrong.second)) < 1.0 - 1e-9,
			"solver must not use kWeights_[sample[i]]");
	}

	void testPcaDegeneracy()
	{
		const Eigen::Vector3d p(1.0, -2.0, 0.5);
		const DataMatrix coincident = makeData({p, p, p, p});
		const size_t sample[4] = {0, 1, 2, 3};
		superansac::estimator::solver::Line3DPCASolver solver;
		std::vector<superansac::models::Model> models;
		expect(!solver.estimateModel(coincident, sample, 4, models, nullptr),
			"all-coincident PCA sample is rejected");

		const std::vector<Eigen::Vector3d> distinct = {
			{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {3.0, 0.0, 0.0}};
		const DataMatrix data = makeData(distinct);
		const double zeroWeights[4] = {0.0, 0.0, 0.0, 0.0};
		expect(!solver.estimateModel(data, sample, 4, models, zeroWeights),
			"zero total weight is rejected");

		const double negativeWeights[4] = {1.0, -1.0, 1.0, 1.0};
		expect(!solver.estimateModel(data, sample, 4, models, negativeWeights),
			"negative weight is rejected");

		const double nanWeights[4] = {1.0, std::numeric_limits<double>::quiet_NaN(),
			1.0, 1.0};
		expect(!solver.estimateModel(data, sample, 4, models, nanWeights),
			"non-finite weight is rejected");

		expect(!solver.estimateModel(data, sample, 2, models, nullptr),
			"PCA rejects fewer than 3 points");
	}

	void testRigidAndScaleEquivariance()
	{
		const std::vector<Eigen::Vector3d> points = {
			{-2.0, 0.1, 0.2}, {-1.0, -0.1, 0.0}, {0.0, 0.2, -0.1},
			{1.0, 0.0, 0.1}, {2.0, -0.2, 0.0}};
		const DataMatrix data = makeData(points);
		const size_t sample[5] = {0, 1, 2, 3, 4};

		superansac::estimator::solver::Line3DPCASolver solver;
		std::vector<superansac::models::Model> models;
		expect(solver.estimateModel(data, sample, 5, models, nullptr),
			"PCA fits the synthetic equivariance line");
		const ModelMatrix &base = models[0].getData();
		const Eigen::Vector3d basePoint = base.row(0).transpose();
		const Eigen::Vector3d baseDirection = base.row(1).transpose();

		// Fixed non-trivial rotation (axis-angle).
		const Eigen::AngleAxisd rotation(
			0.7, Eigen::Vector3d(1.0, -2.0, 0.5).normalized());
		const Eigen::Vector3d translation(10.0, -4.0, 3.0);
		const double scale = 1000.0; // metre -> millimetre style scaling

		std::vector<Eigen::Vector3d> transformed;
		transformed.reserve(points.size());
		for (const auto &point : points)
			transformed.push_back(scale * (rotation * point) + translation);

		const DataMatrix transformedData = makeData(transformed);
		models.clear();
		expect(solver.estimateModel(transformedData, sample, 5, models, nullptr),
			"PCA fits the transformed line");
		const ModelMatrix &moved = models[0].getData();
		const Eigen::Vector3d movedPoint = moved.row(0).transpose();
		const Eigen::Vector3d movedDirection = moved.row(1).transpose();

		expectNear(std::abs(movedDirection.dot(rotation * baseDirection)), 1.0,
			1e-10, "line direction is rotation-equivariant up to canonical sign");

		const Eigen::Vector3d expectedDirection = rotation * baseDirection;
		const Eigen::Vector3d anyPoint =
			scale * (rotation * basePoint) + translation;
		const Eigen::Vector3d expectedClosest =
			anyPoint - anyPoint.dot(expectedDirection) * expectedDirection;
		expectNear((movedPoint - expectedClosest).norm(), 0.0,
			1e-8 * std::max(1.0, expectedClosest.norm()),
			"line position is rigid/scale equivariant");
	}

	void expectDistinctInRange(const size_t *kSample_, const size_t kCount_,
		const size_t kPool_, const std::string &kMessage_)
	{
		for (size_t i = 0; i < kCount_; ++i)
			expect(kSample_[i] < kPool_, kMessage_ + ": index in range");
		for (size_t i = 0; i < kCount_; ++i)
			for (size_t j = i + 1; j < kCount_; ++j)
				expect(kSample_[i] != kSample_[j],
					kMessage_ + ": indices distinct");
	}

	void testUniformSamplerBounds()
	{
		// Regression for the inclusive-upper-bound bug: the sampler used to
		// reset the generator to [0, N] instead of [0, N-1].  With N = 2 this
		// produced index 2 for roughly half of the draws.
		const DataMatrix two = makeData(
			{Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(1.0, 0.0, 0.0)});
		superansac::samplers::UniformRandomSampler sampler;
		sampler.initialize(two);

		size_t sample[10] = {0};
		for (int draw = 0; draw < 2000; ++draw)
		{
			expect(sampler.sample(two, 2, sample),
				"uniform sampler draws a pair for N=2");
			expectDistinctInRange(sample, 2, 2, "N=2");
		}

		// initialize(10) followed by sample(2) and sample(10) must both use
		// the pool size passed to sample.
		const DataMatrix ten = makeData({
			{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0},
			{3.0, 0.0, 0.0}, {4.0, 0.0, 0.0}, {5.0, 0.0, 0.0},
			{6.0, 0.0, 0.0}, {7.0, 0.0, 0.0}, {8.0, 0.0, 0.0},
			{9.0, 0.0, 0.0}});
		sampler.initialize(ten);
		for (int draw = 0; draw < 200; ++draw)
		{
			expect(sampler.sample(ten, 2, sample),
				"initialize(10) then sample(2) succeeds");
			expectDistinctInRange(sample, 2, 10, "initialize(10) sample(2)");

			expect(sampler.sample(ten, 10, sample),
				"initialize(10) then sample(10) succeeds");
			expectDistinctInRange(sample, 10, 10,
				"initialize(10) sample(10)");
		}

		// Grow/shrink 2<->10 between calls without re-initializing.
		sampler.initialize(two);
		for (int cycle = 0; cycle < 200; ++cycle)
		{
			expect(sampler.sample(ten, 2, sample),
				"sample(10,2) after initialize(2) succeeds");
			expectDistinctInRange(sample, 2, 10, "grow 2->10");

			expect(sampler.sample(two, 2, sample),
				"sample(2,2) after sample(10,2) succeeds");
			expectDistinctInRange(sample, 2, 2, "shrink 10->2");
		}

		const DataMatrix one = makeData({Eigen::Vector3d(0.0, 0.0, 0.0)});
		sampler.initialize(one);
		expect(sampler.sample(one, 1, sample),
			"uniform sampler draws the only index for N=1");
		expect(sample[0] == 0, "uniform sampler index is 0 for N=1");
		expect(!sampler.sample(one, 2, sample),
			"uniform sampler rejects k > N (N=1)");

		const DataMatrix empty(0, 3);
		sampler.initialize(empty);
		expect(!sampler.sample(empty, 1, sample),
			"uniform sampler rejects any draw for N=0");
		expect(!sampler.sample(static_cast<size_t>(0), 1, sample),
			"uniform sampler rejects N=0 with the size_t overload");
		expect(!sampler.sample(ten, 0, sample),
			"uniform sampler rejects k=0");
		expect(!sampler.sample(ten, -1, sample),
			"uniform sampler rejects negative k");
		expect(!sampler.sample(ten, 2, nullptr),
			"uniform sampler rejects a null output array");
		expect(!sampler.sample(static_cast<size_t>(3), 4, sample),
			"uniform sampler rejects k > current sample() pool size");

		// An uninitialized sampler must reject rather than dereference.
		superansac::samplers::UniformRandomSampler uninitialized;
		expect(!uninitialized.sample(ten, 2, sample),
			"uniform sampler rejects sample before initialize");

		// Both initialize overloads and reset must retain the N-1 bound.
		sampler.initialize(ten);
		sampler.reset(10);
		for (int draw = 0; draw < 200; ++draw)
		{
			expect(sampler.sample(ten, 10, sample),
				"reset(10) then sample(10) succeeds");
			expectDistinctInRange(sample, 10, 10, "reset(10)");
		}
		sampler.initialize(static_cast<size_t>(10));
		sampler.reset(2);
		for (int draw = 0; draw < 200; ++draw)
		{
			expect(sampler.sample(two, 2, sample),
				"size_t initialize(10) + reset(2) then sample(2) succeeds");
			expectDistinctInRange(sample, 2, 2, "size_t init + reset(2)");
		}
	}

	void testDescriptorValidation()
	{
		ModelMatrix descriptor;
		descriptor.setZero(2, 3);
		descriptor.row(1) << 1.0, 0.0, 0.0;
		expect(superansac::estimator::line3d::isValidDescriptor(descriptor),
			"axis-aligned canonical descriptor is valid");

		descriptor.row(1) << -1.0, 0.0, 0.0;
		expect(!superansac::estimator::line3d::isValidDescriptor(descriptor),
			"non-canonical direction sign is rejected");

		descriptor.row(1) << 2.0, 0.0, 0.0;
		expect(!superansac::estimator::line3d::isValidDescriptor(descriptor),
			"non-unit direction is rejected");

		descriptor.row(1) << 1.0, 0.0, 0.0;
		descriptor.row(0) << 5.0, 1.0, 0.0;
		expect(!superansac::estimator::line3d::isValidDescriptor(descriptor),
			"non-closest descriptor point is rejected");

		descriptor.row(0) << 0.0, 0.0, 0.0;
		descriptor(0, 0) = std::numeric_limits<double>::quiet_NaN();
		expect(!superansac::estimator::line3d::isValidDescriptor(descriptor),
			"non-finite descriptor is rejected");
	}
}

int main()
{
	testMinimalExactLine();
	testMinimalDegeneracy();
	testWeightedPcaAndWeightIndexing();
	testPcaDegeneracy();
	testRigidAndScaleEquivariance();
	testUniformSamplerBounds();
	testDescriptorValidation();

	if (g_failures != 0)
	{
		std::cerr << g_failures << " test(s) failed." << std::endl;
		return 1;
	}
	std::cout << "All line3d core tests passed." << std::endl;
	return 0;
}
