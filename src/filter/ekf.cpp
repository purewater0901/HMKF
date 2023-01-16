#include "filter/ekf.h"

StateInfo EKF::predict(const StateInfo &state_info,
                       const Eigen::VectorXd &inputs,
                       const double dt,
                       const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    // State mean prediction
    const auto A = vehicle_model_->getStateMatrix(state_info.mean, inputs, noise_map, dt);

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
    const auto meas_info = getMeasurementInfo(state_info, noise_map);
    const Eigen::VectorXd& predicted_y = meas_info.mean;
    const Eigen::MatrixXd& S = meas_info.covariance;

    Eigen::MatrixXd H = vehicle_model_->getMeasurementMatrix(state_info.mean, noise_map);

    const auto K = state_info.covariance * H.transpose() * S.inverse();

    StateInfo updated_info;
    updated_info.mean = state_info.mean + K * (y - predicted_y);
    updated_info.covariance = state_info.covariance - K * H * state_info.covariance;

    return updated_info;
}

StateInfo EKF::update(const StateInfo& state_info,
                      const Eigen::VectorXd& y,
                      const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map,
                      const Eigen::Vector2d& landmark)
{
    const auto meas_info = getMeasurementInfo(state_info, noise_map, landmark);
    const Eigen::VectorXd& predicted_y = meas_info.mean;
    const Eigen::MatrixXd& S = meas_info.covariance;

    Eigen::MatrixXd H = vehicle_model_->getMeasurementMatrix(state_info.mean, noise_map, landmark);

    const auto K = state_info.covariance * H.transpose() * S.inverse();

    StateInfo updated_info;
    updated_info.mean = state_info.mean + K * (y - predicted_y);
    updated_info.covariance = state_info.covariance - K * H * state_info.covariance;

    return updated_info;
}

StateInfo EKF::getMeasurementInfo(const StateInfo& state_info,
                                  const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    StateInfo measurement_info;
    measurement_info.mean = vehicle_model_->measure(state_info.mean, noise_map);

    Eigen::MatrixXd H = vehicle_model_->getMeasurementMatrix(state_info.mean, noise_map);
    Eigen::MatrixXd R = vehicle_model_->getMeasurementNoiseMatrix(state_info.mean, noise_map);

    measurement_info.covariance = H*state_info.covariance*H.transpose() + R;

    return measurement_info;
}

StateInfo EKF::getMeasurementInfo(const StateInfo& state_info,
                                  const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map,
                                  const Eigen::Vector2d& landmark)
{
    StateInfo measurement_info;
    measurement_info.mean = vehicle_model_->measureWithLandmark(state_info.mean, noise_map, landmark);

    Eigen::MatrixXd H = vehicle_model_->getMeasurementMatrix(state_info.mean, noise_map, landmark);
    Eigen::MatrixXd R = vehicle_model_->getMeasurementNoiseMatrix(state_info.mean, noise_map, landmark);

    measurement_info.covariance = H*state_info.covariance*H.transpose() + R;

    return measurement_info;
}
