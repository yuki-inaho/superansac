// Copyright (C) 2024 ETH Zurich.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of ETH Zurich nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Daniel Barath (majti89@gmail.com)
#pragma once

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <vector>

#include "abstract_solver.h"
#include "line3d_utils.h"
#include "../models/model.h"
#include "../utils/types.h"

namespace superansac
{
	namespace estimator
	{
		namespace solver
		{
			// Non-minimal weighted 3D-line solver: weighted mean followed by the
			// principal eigenvector of the weighted covariance (weighted PCA).
			//
			// Weight indexing is positional: kWeights_[i] belongs to
			// kSample_[i], matching the contract of the scoring/local-optimization
			// callers (IRLS and LSQ fill weights in sample order).  Indexing by
			// kSample_[i] would silently read an unrelated point's weight.
			class Line3DPCASolver : public AbstractSolver
			{
			public:
				Line3DPCASolver() {}
				~Line3DPCASolver() {}

				bool returnMultipleModels() const override
				{
					return maximumSolutions() > 1;
				}

				size_t maximumSolutions() const override
				{
					return 1;
				}

				size_t sampleSize() const override
				{
					return 3;
				}

				FORCE_INLINE bool estimateModel(
					const DataMatrix &kData_,
					const size_t *kSample_,
					const size_t kSampleNumber_,
					std::vector<models::Model> &models_,
					const double *kWeights_ = nullptr) const override
				{
					if (kSample_ == nullptr || kSampleNumber_ < 3)
						return false;
					if (kData_.cols() < 3)
						return false;

					double weightSum = 0.0;
					Eigen::Vector3d weightedMean = Eigen::Vector3d::Zero();

					for (size_t i = 0; i < kSampleNumber_; ++i)
					{
						const size_t kIndex = kSample_[i];
						if (kIndex >= static_cast<size_t>(kData_.rows()))
							return false;

						const double kWeight =
							kWeights_ == nullptr ? 1.0 : kWeights_[i];
						if (!std::isfinite(kWeight) || kWeight < 0.0)
							return false;
						if (kWeight == 0.0)
							continue;

						const Eigen::Vector3d kPoint(
							kData_(kIndex, 0), kData_(kIndex, 1), kData_(kIndex, 2));
						if (!kPoint.allFinite())
							return false;

						weightedMean += kWeight * kPoint;
						weightSum += kWeight;
					}

					if (!(weightSum > 0.0) || !std::isfinite(weightSum))
						return false;

					weightedMean /= weightSum;
					if (!weightedMean.allFinite())
						return false;

					Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
					for (size_t i = 0; i < kSampleNumber_; ++i)
					{
						const size_t kIndex = kSample_[i];
						const double kWeight =
							kWeights_ == nullptr ? 1.0 : kWeights_[i];
						if (kWeight == 0.0)
							continue;

						const Eigen::Vector3d kOffset =
							Eigen::Vector3d(kData_(kIndex, 0),
								kData_(kIndex, 1),
								kData_(kIndex, 2)) - weightedMean;
						covariance += kWeight * kOffset * kOffset.transpose();
					}
					covariance /= weightSum;
					if (!covariance.allFinite())
						return false;

					Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigenSolver(
						covariance);
					if (eigenSolver.info() != Eigen::Success)
						return false;

					// Largest eigenvalue -> principal direction.  An all-coincident
					// (or numerically zero-variance) sample has no defined line.
					const double kLargestEigenvalue = eigenSolver.eigenvalues()(2);
					if (!std::isfinite(kLargestEigenvalue) ||
						!(kLargestEigenvalue > 0.0))
						return false;

					const Eigen::Vector3d kDirection =
						eigenSolver.eigenvectors().col(2);
					if (!kDirection.allFinite() || !(kDirection.norm() > 0.0))
						return false;

					ModelMatrix descriptor;
					if (!line3d::makeDescriptor(weightedMean, kDirection, descriptor))
						return false;

					models_.emplace_back();
					models_.back().setData(descriptor);
					return true;
				}
			};
		}
	}
}
