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
			// Minimal 3D-line solver: two distinct finite points define the line.
			// The emitted descriptor follows the contract documented in
			// line3d_utils.h (closest point to origin, canonical unit direction).
			class Line3DMinimalSolver : public AbstractSolver
			{
			public:
				Line3DMinimalSolver() {}
				~Line3DMinimalSolver() {}

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
					return 2;
				}

				FORCE_INLINE bool estimateModel(
					const DataMatrix &kData_,
					const size_t *kSample_,
					const size_t kSampleNumber_,
					std::vector<models::Model> &models_,
					const double *kWeights_ = nullptr) const override
				{
					(void)kWeights_;
					if (kSample_ == nullptr || kSampleNumber_ < 2)
						return false;
					if (kData_.cols() < 3)
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

					if (!line3d::areDistinct(kPoint0, kPoint1))
						return false;

					ModelMatrix descriptor;
					if (!line3d::makeDescriptor(kPoint0, kPoint1 - kPoint0, descriptor))
						return false;

					models_.emplace_back();
					models_.back().setData(descriptor);
					return true;
				}
			};
		}
	}
}
