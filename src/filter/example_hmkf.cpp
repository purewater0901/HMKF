#include "filter/example_hmkf.h"
#include "distribution/two_dimensional_normal_distribution.h"

using namespace Example;

Example::PredictedMoments ExampleHMKF::predict(const StateInfo &state,
                                               const Eigen::VectorXd &control_inputs,
                                               const double dt,
                                               const std::map<int, std::shared_ptr<BaseDistribution>> &noise_map)
{
    TwoDimensionalNormalDistribution dist(state.mean, state.covariance);

    const double xPow1 = dist.calc_moment(STATE::IDX::X, 1);
    const double yPow1 = dist.calc_moment(STATE::IDX::Y, 1);
    const double xPow2 = dist.calc_moment(STATE::IDX::X, 2); // x^2
    const double yPow2 = dist.calc_moment(STATE::IDX::Y, 2); // y^2
    const double xPow1_yPow1 = dist.calc_xy_cos_y_sin_y_moment(1, 1, 0, 0);
    const double xPow3 = dist.calc_moment(STATE::IDX::X, 3); // x^3
    const double yPow3 = dist.calc_moment(STATE::IDX::Y, 3); // y^3
    const double xPow2_yPow1 = dist.calc_xy_cos_y_sin_y_moment(2, 1, 0, 0);
    const double xPow1_yPow2 = dist.calc_xy_cos_y_sin_y_moment(1, 2, 0, 0);
    const double xPow4 = dist.calc_moment(STATE::IDX::X, 4); // x^4
    const double yPow4 = dist.calc_xy_cos_y_sin_y_moment(0, 4, 0, 0); // y)^4
    const double xPow2_yPow2 = dist.calc_xy_cos_y_sin_y_moment(2, 2, 0, 0);

    // Input
    const double v = control_inputs(INPUT::V);

    // System noise
    const auto wv_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WV);
    const auto wyaw_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WYAW);
    const double wvPow1 = wv_dist_ptr->calc_moment(1);
    const double wvPow2 = wv_dist_ptr->calc_moment(2);
    const double wvPow3 = wv_dist_ptr->calc_moment(3);
    const double wvPow4 = wv_dist_ptr->calc_moment(4);
    const double cwyawPow1 = wyaw_dist_ptr->calc_cos_moment(1);
    const double swyawPow1 = wyaw_dist_ptr->calc_sin_moment(1);
    const double cwyawPow2 = wyaw_dist_ptr->calc_cos_moment(2);
    const double swyawPow2 = wyaw_dist_ptr->calc_sin_moment(2);
    const double cwyawPow3 = wyaw_dist_ptr->calc_cos_moment(3);
    const double swyawPow3 = wyaw_dist_ptr->calc_sin_moment(3);
    const double cwyawPow4 = wyaw_dist_ptr->calc_cos_moment(4);
    const double swyawPow4 = wyaw_dist_ptr->calc_sin_moment(4);
    const double cwyawPow1_swyawPow1 = wyaw_dist_ptr->calc_cos_sin_moment(1, 1);
    const double cwyawPow2_swyawPow2 = wyaw_dist_ptr->calc_cos_sin_moment(2, 2);
    const double cwyawPow2_swyawPow1 = wyaw_dist_ptr->calc_cos_sin_moment(2, 1);
    const double cwyawPow1_swyawPow2 = wyaw_dist_ptr->calc_cos_sin_moment(1, 2);

    // moment propagation
    PredictedMoments next_moments;
    next_moments.xPow1 = xPow1 + (v+wvPow1)*cwyawPow1*dt;
    next_moments.yPow1 = yPow1 + (v+wvPow1)*swyawPow1*dt;
    next_moments.xPow2 = xPow2 + (v*v + wvPow2 + 2*v*wvPow1) * cwyawPow2 * dt * dt
                         + 2.0 * xPow1 * (v+wvPow1) * cwyawPow1 * dt;
    next_moments.yPow2 = yPow2 + (v*v + wvPow2 + 2*v*wvPow1) * swyawPow2 * dt * dt
                         + 2.0 * yPow1 * (v+wvPow1) * swyawPow1 * dt;
    next_moments.xPow1_yPow1 = xPow1_yPow1 + xPow1*(v+wvPow1)*swyawPow1*dt + yPow1*(v+wvPow1)*cwyawPow1*dt
                         + (v*v+wvPow2+2*v*wvPow1)*cwyawPow1_swyawPow1*dt*dt;
    next_moments.xPow3 = pow(dt, 3)*pow(v, 3)*cwyawPow3 + 3*pow(dt, 3)*pow(v, 2)*cwyawPow3*wvPow1 + 3*pow(dt, 3)*v*cwyawPow3*wvPow2 + pow(dt, 3)*cwyawPow3*wvPow3 + 3*pow(dt, 2)*pow(v, 2)*cwyawPow2*xPow1 + 6*pow(dt, 2)*v*cwyawPow2*wvPow1*xPow1 + 3*pow(dt, 2)*cwyawPow2*wvPow2*xPow1 + 3*dt*v*cwyawPow1*xPow2 + 3*dt*cwyawPow1*wvPow1*xPow2 + xPow3;
    next_moments.yPow3 = pow(dt, 3)*pow(v, 3)*swyawPow3 + 3*pow(dt, 3)*pow(v, 2)*swyawPow3*wvPow1 + 3*pow(dt, 3)*v*swyawPow3*wvPow2 + pow(dt, 3)*swyawPow3*wvPow3 + 3*pow(dt, 2)*pow(v, 2)*swyawPow2*yPow1 + 6*pow(dt, 2)*v*swyawPow2*wvPow1*yPow1 + 3*pow(dt, 2)*swyawPow2*wvPow2*yPow1 + 3*dt*v*swyawPow1*yPow2 + 3*dt*swyawPow1*wvPow1*yPow2 + yPow3;
    next_moments.xPow1_yPow2 = pow(dt, 3)*pow(v, 3)*cwyawPow1_swyawPow2 + 3*pow(dt, 3)*pow(v, 2)*cwyawPow1_swyawPow2*wvPow1 + 3*pow(dt, 3)*v*cwyawPow1_swyawPow2*wvPow2 + pow(dt, 3)*cwyawPow1_swyawPow2*wvPow3 + 2*pow(dt, 2)*pow(v, 2)*cwyawPow1_swyawPow1*yPow1 + pow(dt, 2)*pow(v, 2)*swyawPow2*xPow1 + 4*pow(dt, 2)*v*cwyawPow1_swyawPow1*wvPow1*yPow1 + 2*pow(dt, 2)*v*swyawPow2*wvPow1*xPow1 + 2*pow(dt, 2)*cwyawPow1_swyawPow1*wvPow2*yPow1 + pow(dt, 2)*swyawPow2*wvPow2*xPow1 + dt*v*cwyawPow1*yPow2 + 2*dt*v*swyawPow1*xPow1_yPow1 + dt*cwyawPow1*wvPow1*yPow2 + 2*dt*swyawPow1*wvPow1*xPow1_yPow1 + xPow1_yPow2;
    next_moments.xPow2_yPow1 = pow(dt, 3)*pow(v, 3)*cwyawPow2_swyawPow1 + 3*pow(dt, 3)*pow(v, 2)*cwyawPow2_swyawPow1*wvPow1 + 3*pow(dt, 3)*v*cwyawPow2_swyawPow1*wvPow2 + pow(dt, 3)*cwyawPow2_swyawPow1*wvPow3 + 2*pow(dt, 2)*pow(v, 2)*cwyawPow1_swyawPow1*xPow1 + pow(dt, 2)*pow(v, 2)*cwyawPow2*yPow1 + 4*pow(dt, 2)*v*cwyawPow1_swyawPow1*wvPow1*xPow1 + 2*pow(dt, 2)*v*cwyawPow2*wvPow1*yPow1 + 2*pow(dt, 2)*cwyawPow1_swyawPow1*wvPow2*xPow1 + pow(dt, 2)*cwyawPow2*wvPow2*yPow1 + 2*dt*v*cwyawPow1*xPow1_yPow1 + dt*v*swyawPow1*xPow2 + 2*dt*cwyawPow1*wvPow1*xPow1_yPow1 + dt*swyawPow1*wvPow1*xPow2 + xPow2_yPow1;
    next_moments.xPow4 = pow(dt, 4)*pow(v, 4)*cwyawPow4 + 4*pow(dt, 4)*pow(v, 3)*cwyawPow4*wvPow1 + 6*pow(dt, 4)*pow(v, 2)*cwyawPow4*wvPow2 + 4*pow(dt, 4)*v*cwyawPow4*wvPow3 + pow(dt, 4)*cwyawPow4*wvPow4 + 4*pow(dt, 3)*pow(v, 3)*cwyawPow3*xPow1 + 12*pow(dt, 3)*pow(v, 2)*cwyawPow3*wvPow1*xPow1 + 12*pow(dt, 3)*v*cwyawPow3*wvPow2*xPow1 + 4*pow(dt, 3)*cwyawPow3*wvPow3*xPow1 + 6*pow(dt, 2)*pow(v, 2)*cwyawPow2*xPow2 + 12*pow(dt, 2)*v*cwyawPow2*wvPow1*xPow2 + 6*pow(dt, 2)*cwyawPow2*wvPow2*xPow2 + 4*dt*v*cwyawPow1*xPow3 + 4*dt*cwyawPow1*wvPow1*xPow3 + xPow4;
    next_moments.yPow4 = pow(dt, 4)*pow(v, 4)*swyawPow4 + 4*pow(dt, 4)*pow(v, 3)*swyawPow4*wvPow1 + 6*pow(dt, 4)*pow(v, 2)*swyawPow4*wvPow2 + 4*pow(dt, 4)*v*swyawPow4*wvPow3 + pow(dt, 4)*swyawPow4*wvPow4 + 4*pow(dt, 3)*pow(v, 3)*swyawPow3*yPow1 + 12*pow(dt, 3)*pow(v, 2)*swyawPow3*wvPow1*yPow1 + 12*pow(dt, 3)*v*swyawPow3*wvPow2*yPow1 + 4*pow(dt, 3)*swyawPow3*wvPow3*yPow1 + 6*pow(dt, 2)*pow(v, 2)*swyawPow2*yPow2 + 12*pow(dt, 2)*v*swyawPow2*wvPow1*yPow2 + 6*pow(dt, 2)*swyawPow2*wvPow2*yPow2 + 4*dt*v*swyawPow1*yPow3 + 4*dt*swyawPow1*wvPow1*yPow3 + yPow4;
    next_moments.xPow2_yPow2 = pow(dt, 4)*pow(v, 4)*cwyawPow2_swyawPow2 + 4*pow(dt, 4)*pow(v, 3)*cwyawPow2_swyawPow2*wvPow1 + 6*pow(dt, 4)*pow(v, 2)*cwyawPow2_swyawPow2*wvPow2 + 4*pow(dt, 4)*v*cwyawPow2_swyawPow2*wvPow3 + pow(dt, 4)*cwyawPow2_swyawPow2*wvPow4 + 2*pow(dt, 3)*pow(v, 3)*cwyawPow1_swyawPow2*xPow1 + 2*pow(dt, 3)*pow(v, 3)*cwyawPow2_swyawPow1*yPow1 + 6*pow(dt, 3)*pow(v, 2)*cwyawPow1_swyawPow2*wvPow1*xPow1 + 6*pow(dt, 3)*pow(v, 2)*cwyawPow2_swyawPow1*wvPow1*yPow1 + 6*pow(dt, 3)*v*cwyawPow1_swyawPow2*wvPow2*xPow1 + 6*pow(dt, 3)*v*cwyawPow2_swyawPow1*wvPow2*yPow1 + 2*pow(dt, 3)*cwyawPow1_swyawPow2*wvPow3*xPow1 + 2*pow(dt, 3)*cwyawPow2_swyawPow1*wvPow3*yPow1 + 4*pow(dt, 2)*pow(v, 2)*cwyawPow1_swyawPow1*xPow1_yPow1 + pow(dt, 2)*pow(v, 2)*cwyawPow2*yPow2 + pow(dt, 2)*pow(v, 2)*swyawPow2*xPow2 + 8*pow(dt, 2)*v*cwyawPow1_swyawPow1*wvPow1*xPow1_yPow1 + 2*pow(dt, 2)*v*cwyawPow2*wvPow1*yPow2 + 2*pow(dt, 2)*v*swyawPow2*wvPow1*xPow2 + 4*pow(dt, 2)*cwyawPow1_swyawPow1*wvPow2*xPow1_yPow1 + pow(dt, 2)*cwyawPow2*wvPow2*yPow2 + pow(dt, 2)*swyawPow2*wvPow2*xPow2 + 2*dt*v*cwyawPow1*xPow1_yPow2 + 2*dt*v*swyawPow1*xPow2_yPow1 + 2*dt*cwyawPow1*wvPow1*xPow1_yPow2 + 2*dt*swyawPow1*wvPow1*xPow2_yPow1 + xPow2_yPow2;

    return next_moments;
}

StateInfo ExampleHMKF::update(const PredictedMoments & predicted_moments,
                              const Eigen::VectorXd & observed_values,
                              const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    // predicted moments
    const double& xPow1 = predicted_moments.xPow1;
    const double& yPow1 = predicted_moments.yPow1;
    const double& xPow2 = predicted_moments.xPow2;
    const double& yPow2 = predicted_moments.yPow2;
    const double& xPow1_yPow1 = predicted_moments.xPow1_yPow1;

    // Predicted mean and covariance
    Eigen::Vector2d predicted_mean = Eigen::Vector2d::Zero();
    Eigen::Matrix2d predicted_cov = Eigen::Matrix2d::Zero();
    predicted_mean(STATE::IDX::X) = xPow1;
    predicted_mean(STATE::IDX::Y) = yPow1;
    predicted_cov(STATE::IDX::X, STATE::IDX::X) = xPow2 - xPow1*xPow1;
    predicted_cov(STATE::IDX::Y, STATE::IDX::Y) = yPow2 - yPow1*yPow1;
    predicted_cov(STATE::IDX::X, STATE::IDX::Y) = xPow1_yPow1 - xPow1*yPow1;
    predicted_cov(STATE::IDX::Y, STATE::IDX::X) = predicted_cov(STATE::IDX::X, STATE::IDX::Y);

    // Measurement mean and covariance
    Eigen::VectorXd measurement_mean = Eigen::VectorXd::Zero(1);
    Eigen::MatrixXd measurement_cov = Eigen::MatrixXd::Zero(1, 1);

    const auto meas_moments = getMeasurementMoments(predicted_moments, noise_map);

    measurement_mean(MEASUREMENT::IDX::R) = meas_moments.rPow1;
    measurement_cov(MEASUREMENT::IDX::R, MEASUREMENT::IDX::R) = meas_moments.rPow2 - meas_moments.rPow1*meas_moments.rPow1;

    const auto state_observation_cov = getStateMeasurementMatrix(predicted_moments, meas_moments, noise_map);

    // Kalman Gain
    const auto K = state_observation_cov * measurement_cov.inverse();

    StateInfo updated_info;
    updated_info.mean = predicted_mean + K * (observed_values - measurement_mean);
    updated_info.covariance = predicted_cov - K * measurement_cov * K.transpose();

    return updated_info;
}

Example::MeasurementMoments ExampleHMKF::getMeasurementMoments(const Example::PredictedMoments & predicted_moments,
                                                               const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    const auto wr_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WR);

    // Measurement noise
    const double wrPow1 = wr_dist_ptr->calc_moment(1);
    const double wrPow2 = wr_dist_ptr->calc_moment(2);

    // predicted moments
    const double& xPow2 = predicted_moments.xPow2;
    const double& yPow2 = predicted_moments.yPow2;
    const double& xPow4 = predicted_moments.xPow4;
    const double& yPow4 = predicted_moments.yPow4;
    const double& xPow2_yPow2 = predicted_moments.xPow2_yPow2;

    Example::MeasurementMoments meas_moments;
    meas_moments.rPow1 = xPow2 + yPow2 + wrPow1;
    meas_moments.rPow2 = xPow4 + yPow4 + +wrPow2 + 2.0*xPow2_yPow2 + 2.0*xPow2*wrPow1 + 2.0*yPow2*wrPow1;

    return meas_moments;
}

Eigen::MatrixXd ExampleHMKF::getStateMeasurementMatrix(const Example::PredictedMoments& predicted_moments,
                                                       const Example::MeasurementMoments & measurement_moments,
                                                       const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    const auto wr_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WR);

    // predicted moments
    const double& xPow1 = predicted_moments.xPow1;
    const double& yPow1 = predicted_moments.yPow1;
    const double& xPow3 = predicted_moments.xPow3;
    const double& yPow3 = predicted_moments.yPow3;
    const double& xPow1_yPow2 = predicted_moments.xPow1_yPow2;
    const double& xPow2_yPow1 = predicted_moments.xPow2_yPow1;

    // Measurement noise
    const double wrPow1 = wr_dist_ptr->calc_moment(1);

    // measurement moments
    const double& mrPow1 = measurement_moments.rPow1;

    // x*(x^2 + y^2 + w_r) = x^3 + xy^2 + x*w_r
    // y*(x^2 + y^2 + w_r) = x^2y + y^3 + y*w_r
    Eigen::MatrixXd state_observation_cov = Eigen::MatrixXd::Zero(2, 1); // sigma = E[XY^T] - E[X]E[Y]^T
    state_observation_cov(STATE::IDX::X, MEASUREMENT::IDX::R) = xPow3 + xPow1_yPow2 + xPow1*wrPow1 - xPow1*mrPow1;
    state_observation_cov(STATE::IDX::Y, MEASUREMENT::IDX::R) = xPow2_yPow1 + yPow3 + yPow1*wrPow1 - yPow1*mrPow1;

    return state_observation_cov;
}
