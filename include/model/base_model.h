#ifndef HMKF_BASE_MODEL_H
#define HMKF_BASE_MODEL_H

#include <iostream>
#include <cmath>
#include <vector>
#include <memory>
#include <Eigen/Eigen>

#include "distribution/base_distribution.h"

struct StateInfo {
    Eigen::VectorXd mean;
    Eigen::MatrixXd covariance;
};

class BaseModel
{
public:
    BaseModel() = default;

    // dynamics model
    virtual Eigen::VectorXd propagate(const Eigen::VectorXd& x_curr,
                                      const Eigen::VectorXd& u_curr,
                                      const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map,
                                      const double dt) = 0;

    virtual Eigen::VectorXd propagate(const Eigen::VectorXd& x_curr,
                                      const Eigen::VectorXd& u_curr,
                                      const Eigen::VectorXd& system_noise,
                                      const double dt) = 0;

    // measurement model
    virtual Eigen::VectorXd measure(const Eigen::VectorXd& x_curr,
                                    const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map) = 0;

    virtual Eigen::VectorXd measure(const Eigen::VectorXd& x_curr, const Eigen::VectorXd& observation_noise) = 0;

    // get df/dx
    virtual Eigen::MatrixXd getStateMatrix(const Eigen::VectorXd& x_curr,
                                           const Eigen::VectorXd& u_curr,
                                           const double dt) = 0;

    virtual Eigen::MatrixXd getProcessNoiseMatrix(const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map) = 0;

    // get dh/dx
    virtual Eigen::MatrixXd getMeasurementMatrix(const Eigen::VectorXd& x_curr,
                                                 const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map) = 0;

    virtual Eigen::MatrixXd getMeasurementNoiseMatrix(const Eigen::VectorXd& x_curr,
                                                      const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map) = 0;

    // propagate dynamics moment
    virtual StateInfo propagateStateMoments(const StateInfo &state_info,
                                            const Eigen::VectorXd &control_inputs,
                                            const double dt,
                                            const std::map<int, std::shared_ptr<BaseDistribution>> &noise_map) = 0;

    virtual StateInfo getMeasurementMoments(const StateInfo &state_info,
                                            const std::map<int, std::shared_ptr<BaseDistribution>> &noise_map) = 0;

    virtual Eigen::MatrixXd getStateMeasurementMatrix(const StateInfo& state_info, const StateInfo& measurement_info,
                                                      const std::map<int, std::shared_ptr<BaseDistribution>> &noise_map) = 0;

protected:
};

#endif //HMKF_BASE_MODEL_H
