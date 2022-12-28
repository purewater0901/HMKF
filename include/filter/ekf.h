#ifndef HMKF_EKF_H
#define HMKF_EKF_H

#include <iostream>
#include <vector>
#include <memory>
#include <map>
#include <Eigen/Eigen>

#include "model/base_model.h"
#include "distribution/base_distribution.h"

class EKF
{
public:
    EKF(const std::shared_ptr<BaseModel>& vehicle_model) : vehicle_model_(vehicle_model) {}

    StateInfo predict(const StateInfo& state_info,
                      const Eigen::VectorXd& inputs,
                      const double dt,
                      const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map);

    StateInfo update(const StateInfo& state_info,
                     const Eigen::VectorXd& y,
                     const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map);

    std::shared_ptr<BaseModel> vehicle_model_;
};

#endif //HMKF_EKF_H
