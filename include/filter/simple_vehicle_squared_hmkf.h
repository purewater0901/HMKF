#ifndef HMKF_SIMPLE_VEHICLE_SQUARED_HMKF_H
#define HMKF_SIMPLE_VEHICLE_SQUARED_HMKF_H

#include <iostream>
#include <memory>
#include <vector>
#include <Eigen/Eigen>

#include "distribution/base_distribution.h"
#include "model/simple_vehicle_squared_model.h"

class SimpleVehicleSquaredHMKF
{
public:
    SimpleVehicleSquaredHMKF(const std::shared_ptr<BaseModel>& vehicle_model);
    StateInfo predict(const StateInfo& state_info,
                      const Eigen::Vector2d & control_inputs,
                      const double dt,
                      const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map,
                      std::shared_ptr<SimpleVehicleSquared::HighOrderMoments>& high_order_moments);
    StateInfo update(const SimpleVehicleSquared::HighOrderMoments & predicted_moments,
                     const Eigen::VectorXd & observed_values,
                     const Eigen::Vector2d & landmark,
                     const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map);

    std::shared_ptr<BaseModel> vehicle_model_;
};

#endif //HMKF_SIMPLE_VEHICLE_SQUARED_HMKF_H
