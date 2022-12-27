#ifndef UNCERTAINTY_PROPAGATION_KINEMATIC_VEHICLE_EKF_H
#define UNCERTAINTY_PROPAGATION_KINEMATIC_VEHICLE_EKF_H

#include <iostream>
#include <vector>
#include <memory>
#include <map>
#include <Eigen/Eigen>

#include "model/kinematic_vehicle_model.h"
#include "distribution/base_distribution.h"

class KinematicVehicleEKF
{
public:
    KinematicVehicleEKF() = default;

    StateInfo predict(const StateInfo& state_info,
                      const Eigen::Vector2d& inputs,
                      const double dt,
                      const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map);

    StateInfo update(const StateInfo& state_info,
                     const Eigen::Vector3d& y,
                     const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map);
};

#endif //UNCERTAINTY_PROPAGATION_KINEMATIC_VEHICLE_EKF_H
