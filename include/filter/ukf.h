#ifndef HMKF_UKF_H
#define HMKF_UKF_H

#include <iostream>
#include <vector>
#include <memory>
#include <map>
#include <Eigen/Eigen>

#include "model/base_model.h"
#include "distribution/base_distribution.h"

class UKF
{
public:
    UKF(const std::shared_ptr<BaseModel>& vehicle_model,
        const double alpha_squared=1.0,
        const double beta=0.0,
        const double kappa=3.0);

    StateInfo predict(const StateInfo& state_info,
                      const Eigen::VectorXd& control_inputs,
                      const double dt,
                      const std::map<int, std::shared_ptr<BaseDistribution>>& system_noise_map,
                      const std::map<int, std::shared_ptr<BaseDistribution>>& measurement_noise_map);

    StateInfo update(const StateInfo& state_info,
                     const Eigen::VectorXd& measurement_values,
                     const std::map<int, std::shared_ptr<BaseDistribution>>& system_noise_map,
                     const std::map<int, std::shared_ptr<BaseDistribution>>& measurement_noise_map);

    StateInfo getMeasurementInfo(const StateInfo& state_info,
                                 const std::map<int, std::shared_ptr<BaseDistribution>>& system_noise_map,
                                 const std::map<int, std::shared_ptr<BaseDistribution>>& measurement_noise_map);

    StateInfo update(const StateInfo& state_info,
                     const Eigen::VectorXd& measurement_values,
                     const std::map<int, std::shared_ptr<BaseDistribution>>& system_noise_map,
                     const std::map<int, std::shared_ptr<BaseDistribution>>& measurement_noise_map,
                     const Eigen::Vector2d& landmark);

    StateInfo getMeasurementInfo(const StateInfo& state_info,
                                 const std::map<int, std::shared_ptr<BaseDistribution>>& system_noise_map,
                                 const std::map<int, std::shared_ptr<BaseDistribution>>& measurement_noise_map,
                                 const Eigen::Vector2d& landmark);

    std::shared_ptr<BaseModel> vehicle_model_;

    const int augmented_size_{0};
    const double alpha_squared_{1.0};
    const double beta_{0.0};
    const double kappa_{0.0};
    const double lambda_;

    double Sigma_WM0_;
    double Sigma_WC0_;
    double Sigma_WMI_;
    double Sigma_WCI_;
};

#endif //HMKF_UKF_H
