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
#include <limits>

#include "../utils/macros.h"
#include "../utils/types.h"

namespace superansac
{
	namespace estimator
	{
		namespace line3d
		{
			// Immutable 3D-line descriptor contract used by every line estimator,
			// local optimizer and Python result:
			//
			//   row 0: the point on the line closest to the origin
			//   row 1: unit direction
			//
			// The direction sign is canonicalized so that its component with the
			// largest absolute value is positive.  This makes the descriptor
			// deterministic for a given line, but it is a representation choice,
			// not an oriented physical axis: the same geometric line may be
			// represented by d or -d under an arbitrary rotation.  All residuals
			// and inlier decisions are sign-invariant.
			//
			// A model is therefore a 2x3 matrix.  The residual is the squared
			// orthogonal (perpendicular) point-to-line distance:
			//
			//   q = p - a
			//   r2 = ||q - (q . d) d||^2
			//
			// where a is row 0 and d is row 1.

			static constexpr double kDirectionNormTolerance = 1e-6;
			static constexpr double kPointDirectionTolerance = 1e-6;

			// Relative coincidence tolerance for deciding whether two sample points
			// are distinct.  It scales with the coordinate magnitude so millimetre
			// and metre inputs behave the same way.
			FORCE_INLINE double coincidenceTolerance(
				const Eigen::Vector3d &kA_,
				const Eigen::Vector3d &kB_)
			{
				const double kScale = std::max(1.0, std::max(kA_.norm(), kB_.norm()));
				return 1e-12 * kScale;
			}

			FORCE_INLINE bool areDistinct(
				const Eigen::Vector3d &kA_,
				const Eigen::Vector3d &kB_)
			{
				if (!kA_.allFinite() || !kB_.allFinite())
					return false;
				return (kA_ - kB_).norm() > coincidenceTolerance(kA_, kB_);
			}

			// Canonicalize an arbitrary non-zero vector: normalize, then flip the
			// sign so the largest-|component| component is positive.  Returns a
			// zero vector when the input is degenerate or non-finite.
			FORCE_INLINE Eigen::Vector3d canonicalDirection(
				const Eigen::Vector3d &kDirection_)
			{
				if (!kDirection_.allFinite())
					return Eigen::Vector3d::Zero();

				const double kNorm = kDirection_.norm();
				if (!(kNorm > 0.0) || !std::isfinite(kNorm))
					return Eigen::Vector3d::Zero();

				Eigen::Vector3d canonical = kDirection_ / kNorm;
				Eigen::Index largestIndex = 0;
				canonical.cwiseAbs().maxCoeff(&largestIndex);
				if (canonical(largestIndex) < 0.0)
					canonical = -canonical;
				return canonical;
			}

			// Point on the line closest to the origin.
			FORCE_INLINE Eigen::Vector3d closestPointToOrigin(
				const Eigen::Vector3d &kPoint_,
				const Eigen::Vector3d &kUnitDirection_)
			{
				return kPoint_ - kPoint_.dot(kUnitDirection_) * kUnitDirection_;
			}

			// Build the canonical 2x3 descriptor from a point on the line and a
			// direction.  Returns false for degenerate or non-finite input.
			FORCE_INLINE bool makeDescriptor(
				const Eigen::Vector3d &kPoint_,
				const Eigen::Vector3d &kDirection_,
				ModelMatrix &kDescriptor_)
			{
				if (!kPoint_.allFinite() || !kDirection_.allFinite())
					return false;

				const Eigen::Vector3d kUnitDirection = canonicalDirection(kDirection_);
				if (kUnitDirection.isZero())
					return false;

				const Eigen::Vector3d kClosestPoint =
					closestPointToOrigin(kPoint_, kUnitDirection);
				if (!kClosestPoint.allFinite())
					return false;

				kDescriptor_.resize(2, 3);
				kDescriptor_.row(0) = kClosestPoint.transpose();
				kDescriptor_.row(1) = kUnitDirection.transpose();
				return true;
			}

			// Validate the immutable descriptor contract.  Used before a model is
			// scored so malformed local-optimizer output can never produce NaNs in
			// the scoring loop.
			FORCE_INLINE bool isValidDescriptor(const ModelMatrix &kDescriptor_)
			{
				if (kDescriptor_.rows() != 2 || kDescriptor_.cols() != 3)
					return false;
				if (!kDescriptor_.allFinite())
					return false;

				const Eigen::Vector3d kClosestPoint =
					kDescriptor_.row(0).transpose();
				const Eigen::Vector3d kDirection =
					kDescriptor_.row(1).transpose();

				const double kNorm = kDirection.norm();
				if (!(kNorm > 0.0) || !std::isfinite(kNorm))
					return false;
				if (std::abs(kNorm - 1.0) > kDirectionNormTolerance)
					return false;
				if (std::abs(kClosestPoint.dot(kDirection)) >
					kPointDirectionTolerance * std::max(1.0, kClosestPoint.norm()))
					return false;

				// The direction sign must follow the documented canonical rule.
				Eigen::Vector3d normalized = kDirection / kNorm;
				Eigen::Index largestIndex = 0;
				normalized.cwiseAbs().maxCoeff(&largestIndex);
				if (normalized(largestIndex) < 0.0)
					return false;
				return true;
			}

			FORCE_INLINE double squaredResidual(
				const double *kPoint_,
				const ModelMatrix &kDescriptor_)
			{
				if (kPoint_ == nullptr)
					return std::numeric_limits<double>::infinity();

				const Eigen::Vector3d kPoint(kPoint_[0], kPoint_[1], kPoint_[2]);
				if (!kPoint.allFinite())
					return std::numeric_limits<double>::infinity();

				const Eigen::Vector3d kClosestPoint =
					kDescriptor_.row(0).transpose();
				const Eigen::Vector3d kDirection =
					kDescriptor_.row(1).transpose();

				const Eigen::Vector3d kOffset = kPoint - kClosestPoint;
				const double kAlong = kOffset.dot(kDirection);
				const Eigen::Vector3d kPerpendicular =
					kOffset - kAlong * kDirection;
				const double kSquared = kPerpendicular.squaredNorm();
				return std::isfinite(kSquared) ?
					kSquared : std::numeric_limits<double>::infinity();
			}
		}
	}
}
