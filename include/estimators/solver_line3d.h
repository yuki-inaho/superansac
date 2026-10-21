#pragma once

#include "abstract_solver.h"
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace superansac { namespace estimator { namespace solver {
namespace line3d_detail {
// Bit inspection remains valid when the rest of the project uses -ffast-math.
inline bool finite(double value) {
    static_assert(sizeof(double) == sizeof(uint64_t) && std::numeric_limits<double>::is_iec559,
                  "Line3D requires IEEE-754 doubles");
    uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}
template <class Derived> bool finite(const Eigen::MatrixBase<Derived>& values) {
    for (Eigen::Index r = 0; r < values.rows(); ++r)
        for (Eigen::Index c = 0; c < values.cols(); ++c)
            if (!finite(values(r, c))) return false;
    return true;
}
inline void canonicalize(Eigen::Vector3d& direction) {
    Eigen::Index axis;
    direction.cwiseAbs().maxCoeff(&axis);
    if (direction[axis] < 0.0) direction = -direction;
}
inline bool distinct(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    const Eigen::Vector3d delta = b - a;
    if (!finite(delta)) return false;
    const double separation = delta.cwiseAbs().maxCoeff();
    const double magnitude = std::max(a.cwiseAbs().maxCoeff(), b.cwiseAbs().maxCoeff());
    return separation > 16.0 * std::numeric_limits<double>::epsilon() * magnitude;
}
inline void append(const Eigen::Vector3d& anchor, Eigen::Vector3d direction,
                   std::vector<models::Model>& models) {
    canonicalize(direction);
    ModelMatrix descriptor(2, 3);
    descriptor.row(0) = anchor.transpose();
    descriptor.row(1) = direction.transpose();
    models::Model model;
    model.setData(descriptor);
    models.emplace_back(model);
}
}

// A line is a 2x3 descriptor: an anchor followed by a unit direction.
// With a sample, weights are in sample order (the scoring getWeights contract).
// With sample == nullptr, weights are in data-row order.
class Line3DSolver : public AbstractSolver {
public:
    size_t sampleSize() const override { return 2; }
    bool estimateModel(const DataMatrix& data, const size_t* sample, size_t count,
                       std::vector<models::Model>& models,
                       const double* weights = nullptr) const override {
        if (data.cols() != 3 || count < 2 || (!sample && count > size_t(data.rows())))
            return false;
        double maxWeight = 0.0;
        size_t positiveCount = 0, first = 0;
        for (size_t i = 0; i < count; ++i) {
            const size_t idx = sample ? sample[i] : i;
            if (idx >= size_t(data.rows()) || !line3d_detail::finite(data.row(idx))) return false;
            const double weight = weights ? weights[i] : 1.0;
            if (!line3d_detail::finite(weight) || weight < 0.0) return false;
            if (weight > 0.0) {
                if (positiveCount++ == 0) first = idx;
                maxWeight = std::max(maxWeight, weight);
            }
        }
        if (positiveCount < 2) return false;
        const Eigen::Vector3d origin = data.row(first).transpose();
        double scale = 0.0;
        for (size_t i = 0; i < count; ++i) {
            if (weights && weights[i] == 0.0) continue;
            const Eigen::Vector3d point = data.row(sample ? sample[i] : i).transpose();
            const Eigen::Vector3d delta = point - origin;
            if (!line3d_detail::finite(delta)) return false;
            if (line3d_detail::distinct(origin, point))
                scale = std::max(scale, delta.cwiseAbs().maxCoeff());
        }
        if (!(scale > 0.0)) return false;
        if (count == 2 && !weights) {
            const Eigen::Vector3d other = data.row(sample ? sample[1] : 1).transpose();
            Eigen::Vector3d direction = (other - origin) / scale;
            direction.normalize();
            line3d_detail::append(origin + 0.5 * (other - origin), direction, models);
            return true;
        }
        // Scale centered coordinates and weights before forming covariance to
        // avoid squaring very large/small coordinates or summing huge weights.
        Eigen::Vector3d mean = Eigen::Vector3d::Zero();
        double totalWeight = 0.0;
        for (size_t i = 0; i < count; ++i) {
            const double weight = weights ? weights[i] / maxWeight : 1.0;
            if (weight == 0.0) continue;
            const Eigen::Vector3d point = data.row(sample ? sample[i] : i).transpose();
            mean += weight * ((point - origin) / scale);
            totalWeight += weight;
        }
        mean /= totalWeight;
        Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
        for (size_t i = 0; i < count; ++i) {
            const double weight = weights ? weights[i] / maxWeight : 1.0;
            if (weight == 0.0) continue;
            const Eigen::Vector3d point = data.row(sample ? sample[i] : i).transpose();
            const Eigen::Vector3d delta = (point - origin) / scale - mean;
            covariance.noalias() += (weight / totalWeight) * delta * delta.transpose();
        }
        if (!line3d_detail::finite(covariance)) return false;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(covariance);
        if (eig.info() != Eigen::Success || !line3d_detail::finite(eig.eigenvalues())) return false;
        const auto& eigenvalues = eig.eigenvalues();
        // No unique principal direction for identical or isotropic samples.
        if (!(eigenvalues[2] > 0.0) ||
            eigenvalues[2] - eigenvalues[1] <= 64.0 * std::numeric_limits<double>::epsilon() * eigenvalues[2])
            return false;
        const Eigen::Vector3d anchor = origin + scale * mean;
        const Eigen::Vector3d direction = eig.eigenvectors().col(2).normalized();
        if (!line3d_detail::finite(anchor) || !line3d_detail::finite(direction)) return false;
        line3d_detail::append(anchor, direction, models);
        return true;
    }
};
}}}
