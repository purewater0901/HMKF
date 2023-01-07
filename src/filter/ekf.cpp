#include "filter/ekf.h"

StateInfo EKF::predict(const StateInfo &state_info,
                       const Eigen::VectorXd &inputs,
                       const double dt,
                       const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    // State mean prediction
    const auto A = vehicle_model_->getStateMatrix(state_info.mean, inputs, dt);

    const auto Q = vehicle_model_->getProcessNoiseMatrix(state_info.mean, inputs, noise_map, dt);

    StateInfo predicted_info;
    predicted_info.mean = vehicle_model_->propagate(state_info.mean, inputs, noise_map, dt);
    predicted_info.covariance = A * state_info.covariance * A.transpose() + Q;
    return predicted_info;
}

StateInfo EKF::update(const StateInfo& state_info,
                      const Eigen::VectorXd& y,
                      const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    const auto predicted_y = vehicle_model_->measure(state_info.mean, noise_map);

    Eigen::MatrixXd H = vehicle_model_->getMeasurementMatrix(state_info.mean, noise_map);
    Eigen::MatrixXd R = vehicle_model_->getMeasurementNoiseMatrix(state_info.mean, noise_map);

    const Eigen::MatrixXd S = H*state_info.covariance*H.transpose() + R;
    const auto K = state_info.covariance * H.transpose() * S.inverse();

    StateInfo updated_info;
    updated_info.mean = state_info.mean + K * (y - predicted_y);
    updated_info.covariance = state_info.covariance - K * H * state_info.covariance;

    return updated_info;
}