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

#include <cmath>
#include <vector>

#include <Eigen/Core>

#include "abstract_estimator.h"
#include "line3d_utils.h"
#include "../models/model.h"
#include "../utils/types.h"

namespace superansac
{
	namespace estimator
	{
		// Estimator for a 3D line fitted to a set of XYZ points.
		//
		// Minimal model: 2 distinct points.
		// Non-minimal model: weighted PCA over >= 3 points.
		// Residual: squared orthogonal point-to-line distance.
		// The model descriptor is the 2x3 contract in line3d_utils.h.
		class Line3DEstimator : public Estimator
		{
		public:
			Line3DEstimator() {}
			~Line3DEstimator() {}

			bool isWeightingApplicable() const override
			{
				return true;
			}

			// AC-RANSAC is rejected for this estimator in the public API.  The
			// values are only present to satisfy the abstract interface.
			double multError() const override
			{
				return 1.0;
			}

			double logAlpha0(size_t w, size_t h,
				double scalingFactor = 0.5) const override
			{
				(void)w;
				(void)h;
				(void)scalingFactor;
				return 0.0;
			}

			// Orthogonal point-to-line distance in 3D has 3 point coordinates
			// minus 1 degree of freedom along the line = 2.  MAGSAC uses this to
			// select its chi-square/gamma model.
			size_t getDegreesOfFreedom() const override
			{
				return 2;
			}

			FORCE_INLINE bool estimateModel(
				const DataMatrix &kData_,
				const size_t *kSample_,
				std::vector<models::Model> *model_) const override
			{
				if (model_ == nullptr || minimalSolver == nullptr)
					return false;
				return minimalSolver->estimateModel(
					kData_, kSample_, minimalSolver->sampleSize(), *model_);
			}

			FORCE_INLINE bool estimateModelNonminimal(
				const DataMatrix &kData_,
				const size_t *kSample_,
				const size_t &kSampleNumber_,
				std::vector<models::Model> *model_,
				const double *kWeights_ = nullptr) const override
			{
				if (model_ == nullptr || nonMinimalSolver == nullptr)
					return false;
				if (kSampleNumber_ < nonMinimalSampleSize())
					return false;
				return nonMinimalSolver->estimateModel(
					kData_, kSample_, kSampleNumber_, *model_, kWeights_);
			}

			FORCE_INLINE double squaredResidual(
				const double *kPoint_,
				const models::Model &kModel_) const override
			{
				return line3d::squaredResidual(kPoint_, kModel_.getData());
			}

			FORCE_INLINE double residual(
				const double *kPoint_,
				const models::Model &kModel_) const override
			{
				return std::sqrt(std::max(0.0, squaredResidual(kPoint_, kModel_)));
			}

			void squaredResiduals(
				const DataMatrix &kData_,
				const models::Model &kModel_,
				const size_t kStartRow_,
				const size_t kCount_,
				double *kOut_) const override
			{
				if (kOut_ == nullptr)
					return;

				const ModelMatrix &kDescriptor = kModel_.getData();
				const size_t kCols = kData_.cols();
				const double *row = kData_.data() + kStartRow_ * kCols;
				for (size_t i = 0; i < kCount_; ++i, row += kCols)
					kOut_[i] = line3d::squaredResidual(row, kDescriptor);
			}

			FORCE_INLINE bool isValidSample(
				const DataMatrix &kData_,
				const size_t *kSample_) const override
			{
				if (kSample_ == nullptr || kData_.cols() < 3)
					return false;

				const size_t kIndex0 = kSample_[0];
				const size_t kIndex1 = kSample_[1];
				if (kIndex0 >= static_cast<size_t>(kData_.rows()) ||
					kIndex1 >= static_cast<size_t>(kData_.rows()) ||
					kIndex0 == kIndex1)
					return false;

				const Eigen::Vector3d kPoint0(
					kData_(kIndex0, 0), kData_(kIndex0, 1), kData_(kIndex0, 2));
				const Eigen::Vector3d kPoint1(
					kData_(kIndex1, 0), kData_(kIndex1, 1), kData_(kIndex1, 2));
				return line3d::areDistinct(kPoint0, kPoint1);
			}

			FORCE_INLINE bool isValidModel(
				const models::Model &kModel_) const override
			{
				return line3d::isValidDescriptor(kModel_.getData());
			}
		};
	}
}
