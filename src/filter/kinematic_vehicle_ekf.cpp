#include "filter/kinematic_vehicle_ekf.h"
#include "utilities.h"

using namespace KinematicVehicle;

StateInfo KinematicVehicleEKF::predict(const StateInfo &state_info,
                                       const Eigen::Vector2d &inputs,
                                       const double dt,
                                       const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    KinematicVehicleModel vehicle_model_;
    const auto wv_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WV);
    const auto wyaw_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WYAW);

    // State mean prediction
    const auto A = vehicle_model_.getStateMatrix(state_info.mean, dt);
    const auto Q = vehicle_model_.getProcessNoiseMatrix(noise_map);

    StateInfo predicted_info;
    predicted_info.mean = vehicle_model_.propagate(state_info.mean, inputs, noise_map, dt);
    predicted_info.covariance = A * state_info.covariance * A.transpose() + Q;
    return predicted_info;
}

StateInfo KinematicVehicleEKF::update(const StateInfo& state_info,
                                      const Eigen::Vector3d& y,
                                      const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    KinematicVehicleModel vehicle_model_;
    const auto predicted_y = vehicle_model_.measure(state_info.mean, noise_map);

    Eigen::MatrixXd H = vehicle_model_.getMeasurementMatrix(state_info.mean);
    Eigen::Matrix3d R = vehicle_model_.getMeasurementNoiseMatrix(noise_map);

    const Eigen::Matrix3d S = H*state_info.covariance*H.transpose() + R;
    const auto K = state_info.covariance * H.transpose() * S.inverse();

    StateInfo updated_info;
    updated_info.mean = state_info.mean + K * (y - predicted_y);
    updated_info.covariance = state_info.covariance - K * H * state_info.covariance;

    return updated_info;
}
