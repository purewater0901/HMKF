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
    BaseModel(const size_t state_dim,
              const size_t system_noise_dim,
              const size_t measurement_dim,
              const size_t measurement_noise_dim)
              : state_dim_(state_dim),
                system_noise_dim_(system_noise_dim),
                measurement_dim_(measurement_dim),
                measurement_noise_dim_(measurement_noise_dim)

    {
    };

    size_t getAugmentedSize() {return state_dim_ + system_noise_dim_ + measurement_noise_dim_;}

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

    virtual Eigen::MatrixXd getProcessNoiseMatrix(const Eigen::VectorXd& x_curr,
                                                  const Eigen::VectorXd& u_curr,
                                                  const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map,
                                                  const double dt) = 0;

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

    size_t state_dim_{0};
    size_t system_noise_dim_{0};
    size_t measurement_dim_{0};
    size_t measurement_noise_dim_{0};

protected:
};

#endif //HMKF_BASE_MODEL_H
