#include "filter/mkf.h"

StateInfo MKF::predict(const StateInfo &state_info,
                       const Eigen::VectorXd &control_inputs,
                       const double dt,
                       const std::map<int, std::shared_ptr<BaseDistribution>> &noise_map)
{
    return vehicle_model_->propagateStateMoments(state_info, control_inputs, dt, noise_map);
}

StateInfo MKF::update(const StateInfo &state_info,
                      const Eigen::VectorXd &observed_values,
                      const std::map<int, std::shared_ptr<BaseDistribution>> &noise_map)
{
    const auto measurement_info = vehicle_model_->getMeasurementMoments(state_info, noise_map);
    const auto state_observation_cov = vehicle_model_->getStateMeasurementMatrix(state_info, measurement_info, noise_map);

    const auto& predicted_mean = state_info.mean;
    const auto& predicted_cov = state_info.covariance;
    const auto& measurement_mean = measurement_info.mean;
    const auto& measurement_cov = measurement_info.covariance;

    // Kalman Gain
    const auto K = state_observation_cov * measurement_cov.inverse();

    StateInfo updated_info;
    updated_info.mean = predicted_mean + K * (observed_values - measurement_mean);
    updated_info.covariance = predicted_cov - K * measurement_cov * K.transpose();

    return updated_info;
}

StateInfo MKF::update(const StateInfo & state_info,
                      const Eigen::VectorXd & observed_values,
                      const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map,
                      const Eigen::Vector2d& landmark)
{
    const auto measurement_info = vehicle_model_->getMeasurementMoments(state_info, noise_map, landmark);
    const auto state_observation_cov = vehicle_model_->getStateMeasurementMatrix(state_info, measurement_info, noise_map, landmark);

    const auto& predicted_mean = state_info.mean;
    const auto& predicted_cov = state_info.covariance;
    const auto& measurement_mean = measurement_info.mean;
    const auto& measurement_cov = measurement_info.covariance;

    // Kalman Gain
    const auto K = state_observation_cov * measurement_cov.inverse();

    StateInfo updated_info;
    updated_info.mean = predicted_mean + K * (observed_values - measurement_mean);
    updated_info.covariance = predicted_cov - K * measurement_cov * K.transpose();

    return updated_info;
}
