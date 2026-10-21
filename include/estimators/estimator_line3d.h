#pragma once

#include "abstract_estimator.h"
#include "solver_line3d.h"
#include <stdexcept>

namespace superansac { namespace estimator {
class Line3DEstimator : public Estimator {
public:
    Line3DEstimator() {
        setMinimalSolver(new solver::Line3DSolver());
        setNonMinimalSolver(new solver::Line3DSolver());
    }
    bool isWeightingApplicable() const override { return true; }
    size_t getDegreesOfFreedom() const override { return 2; }
    double multError() const override { return 2.0; }
    double logAlpha0(size_t, size_t, double = 0.5) const override {
        throw std::logic_error("AC-RANSAC's image-area null model is not defined for Line3D.");
    }
    bool estimateModel(const DataMatrix& data, const size_t* sample,
                       std::vector<models::Model>* models) const override {
        return models && minimalSolver->estimateModel(data, sample, 2, *models);
    }
    bool estimateModelNonminimal(const DataMatrix& data, const size_t* sample,
                                 const size_t& count, std::vector<models::Model>* models,
                                 const double* weights = nullptr) const override {
        return models && nonMinimalSolver->estimateModel(data, sample, count, *models, weights);
    }
    bool isValidSample(const DataMatrix& data, const size_t* sample) const override {
        if (data.cols() != 3 || !sample || sample[0] >= size_t(data.rows()) ||
            sample[1] >= size_t(data.rows())) return false;
        const Eigen::Vector3d a = data.row(sample[0]).transpose();
        const Eigen::Vector3d b = data.row(sample[1]).transpose();
        return solver::line3d_detail::finite(a) && solver::line3d_detail::finite(b) &&
               solver::line3d_detail::distinct(a, b);
    }
    bool isValidModel(const models::Model& model) const override {
        const auto& m = model.getData();
        return m.rows() == 2 && m.cols() == 3 && solver::line3d_detail::finite(m) &&
               std::abs(m.row(1).squaredNorm() - 1.0) < 1e-10;
    }
    bool isValidModel(models::Model& model, const DataMatrix&, const size_t*, double,
                      bool& updated) const override {
        updated = false;
        return isValidModel(model);
    }
    double squaredResidual(const double* point, const models::Model& model) const override {
        const auto& m = model.getData();
        const Eigen::Vector3d delta(point[0] - m(0, 0), point[1] - m(0, 1), point[2] - m(0, 2));
        const Eigen::Vector3d direction = m.row(1).transpose();
        // Cross product avoids cancellation in ||delta||^2 - (delta.dot(d))^2.
        return delta.cross(direction).squaredNorm();
    }
    double residual(const double* point, const models::Model& model) const override {
        return std::sqrt(squaredResidual(point, model));
    }
};
}}
