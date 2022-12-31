#include "filter/example_hmkf.h"
#include "distribution/two_dimensional_normal_distribution.h"

using namespace Example;

Example::PredictedMoments ExampleHMKF::predict(const Example::StateInfo &state,
                                               const Eigen::Vector2d &control_inputs,
                                               const double dt,
                                               const std::map<int, std::shared_ptr<BaseDistribution>> &noise_map)
{
    const auto wx_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WX);
    const auto wyaw_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WYAW);

    TwoDimensionalNormalDistribution dist(state.mean, state.covariance);

    const double xPow1 = dist.calc_moment(STATE::IDX::X, 1);
    const double yawPow1 = dist.calc_moment(STATE::IDX::YAW, 1);
    const double cPow1 = dist.calc_xy_cos_y_sin_y_moment(0, 0, 1, 0); // cos(yaw)
    const double xPow2 = dist.calc_moment(STATE::IDX::X, 2); // x^2
    const double yawPow2 = dist.calc_moment(STATE::IDX::YAW, 2); // yaw^2
    const double cPow2 = dist.calc_xy_cos_y_sin_y_moment(0, 0, 2, 0); // cos(yaw)^2
    const double xPow1_yawPow1 = dist.calc_xy_cos_y_sin_y_moment(1, 1, 0, 0); // xyaw
    const double cPow1_xPow1 = dist.calc_xy_cos_y_sin_y_moment(1, 0, 1, 0); // x*cos(yaw)
    const double cPow1_yawPow1 = dist.calc_xy_cos_y_sin_y_moment(0, 1, 1, 0); // yaw*cos(yaw)
    const double cPow1_yawPow2 = dist.calc_xy_cos_y_sin_y_moment(0, 1, 2, 0); // yaw*cos(yaw)*cos(yaw)
    const double xPow3 = dist.calc_moment(STATE::IDX::X, 3); // x^3
    const double cPow3 = dist.calc_xy_cos_y_sin_y_moment(0, 0, 3, 0); // cos(yaw)^3
    const double cPow1_xPow2 = dist.calc_xy_cos_y_sin_y_moment(2, 0, 1, 0); // xx*cos(yaw)
    const double cPow2_xPow1 = dist.calc_xy_cos_y_sin_y_moment(1, 0, 2, 0); // x*cos(yaw)*cos(yaw)
    const double xPow2_yawPow1 = dist.calc_xy_cos_y_sin_y_moment(2, 1, 0, 0); // xx*yaw
    const double xPow1_yawPow1_cPow1 = dist.calc_xy_cos_y_sin_y_moment(1, 1, 1, 0); // x*yaw*cos(yaw)
    const double xPow4 = dist.calc_moment(STATE::IDX::X, 4); // x^4
    const double cPow4 = dist.calc_xy_cos_y_sin_y_moment(0, 0, 4, 0); // cos(yaw)^4
    const double xPow3_cPow1 = dist.calc_xy_cos_y_sin_y_moment(3, 0, 1, 0);
    const double xPow1_cPow3 = dist.calc_xy_cos_y_sin_y_moment(1, 0, 3, 0);
    const double xPow2_cPow2 = dist.calc_xy_cos_y_sin_y_moment(2, 0, 2, 0);

    // Input
    const double v = control_inputs(INPUT::V);
    const double u = control_inputs(INPUT::U);

    // System noise
    const double wxPow1 = wx_dist_ptr->calc_moment(1);
    const double wyawPow1 = wyaw_dist_ptr->calc_moment(1);
    const double wxPow2 = wx_dist_ptr->calc_moment(2);
    const double wyawPow2 = wyaw_dist_ptr->calc_moment(2);
    const double wxPow3 = wx_dist_ptr->calc_moment(3);
    const double wxPow4 = wx_dist_ptr->calc_moment(4);

    // moment propagation
    PredictedMoments next_moments;
    next_moments.xPow1 = xPow1 + v*cPow1 + wxPow1;
    next_moments.yawPow1 = yawPow1 + u + wyawPow1;
    next_moments.xPow2 = xPow2 + v*v*cPow2 + wxPow2 + 2*v*cPow1_xPow1 + 2*xPow1*wxPow1 + 2*v*wxPow1*cPow1;
    next_moments.yawPow2 = yawPow2 + u*u + wyawPow2 + 2*u*yawPow1 + 2*yawPow1*wyawPow1 + 2*u*wyawPow1;
    next_moments.xPow1_yawPow1 = xPow1_yawPow1 + u*xPow1 + wyawPow1*xPow1
                                + v*cPow1_yawPow1 + v*u*cPow1 + v*wyawPow1*cPow1
                                + wxPow1*yawPow1 + wxPow1*u + wxPow1*wyawPow1;
    next_moments.xPow3 = v*v*v*cPow3 + 3*v*v*wxPow1*cPow2 + 3*v*v*cPow2_xPow1
                        + 3*v*wxPow2*cPow1 + 6*v*wxPow1*cPow1_xPow1 +3*v*cPow1_xPow2
                        + wxPow3 + 3*wxPow2*xPow1 + 3*wxPow1*xPow2 + xPow3;
    next_moments.xPow2_yawPow1 = cPow1_yawPow2*v*v + 2*v*wxPow1*cPow1_yawPow1
                               + 2*v*xPow1_yawPow1_cPow1 + yawPow1*wxPow2
                               + 2*xPow1_yawPow1*wxPow1 + xPow2_yawPow1
                               + u*v*v*cPow2 + 2*u*v*wxPow1*cPow1
                               + 2*u*v*cPow1_xPow1 + u*wxPow2
                               + 2*u*wxPow1*xPow1 + u*xPow2
                               + v*v*wyawPow1*cPow2 + 2*v*wxPow1*wyawPow1*cPow1
                               + 2*v*wyawPow1*cPow1_xPow1 + wxPow2*wyawPow1
                               + 2*wxPow1*wyawPow1*xPow1 + wyawPow1*xPow2;
    next_moments.xPow4 = std::pow(v,4) * cPow4 + 4*std::pow(v,3)*wxPow1*cPow3
                         + 4*std::pow(v,3) * xPow1_cPow3 + 6*v*v*wxPow2*cPow2
                         + 12*v*v*wxPow1*cPow2_xPow1 + 6*v*v*xPow2_cPow2
                         + 4*v*wxPow3*cPow1 + 12*v*wxPow2*cPow1_xPow1
                         + 12*v*wxPow1*cPow1_xPow2 + 4*v*xPow3_cPow1
                         + wxPow4 + 4*wxPow3*xPow1 + 6*wxPow2*xPow2
                         + 4*wxPow1*xPow3+xPow4;

    return next_moments;
}

StateInfo ExampleHMKF::update(const PredictedMoments & predicted_moments,
                              const Eigen::VectorXd & observed_values,
                              const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    const auto wr_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WR);
    const auto wyaw_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WYAW);

    // predicted moments
    const double& xPow1 = predicted_moments.xPow1;
    const double& yawPow1 = predicted_moments.yawPow1;
    const double& xPow2 = predicted_moments.xPow2;
    const double& yawPow2 = predicted_moments.yawPow2;
    const double& xPow1_yawPow1 = predicted_moments.xPow1_yawPow1;
    const double& xPow3 = predicted_moments.xPow3;
    const double& xPow2_yawPow1 = predicted_moments.xPow2_yawPow1;
    const double& xPow4 = predicted_moments.xPow4;

    // Measurement noise
    const double wrPow1 = wr_dist_ptr->calc_moment(1);
    const double wyawPow1 = wyaw_dist_ptr->calc_moment(1);
    const double wrPow2 = wr_dist_ptr->calc_moment(2);
    const double wyawPow2 = wyaw_dist_ptr->calc_moment(2);

    // Predicted mean and covariance
    Eigen::Vector2d predicted_mean = Eigen::Vector2d::Zero();
    Eigen::Matrix2d predicted_cov = Eigen::Matrix2d::Zero();
    predicted_mean(STATE::IDX::X) = xPow1;
    predicted_mean(STATE::IDX::YAW) = yawPow1;
    predicted_cov(STATE::IDX::X, STATE::IDX::X) = xPow2 - xPow1*xPow1;
    predicted_cov(STATE::IDX::YAW, STATE::IDX::YAW) = yawPow2 - yawPow1*yawPow1;
    predicted_cov(STATE::IDX::X, STATE::IDX::YAW) = xPow1_yawPow1 - xPow1*yawPow1;
    predicted_cov(STATE::IDX::YAW, STATE::IDX::X) = predicted_cov(STATE::IDX::X, STATE::IDX::YAW);

    // Measurement mean and covariance
    Eigen::Vector2d measurement_mean = Eigen::Vector2d::Zero();
    Eigen::Matrix2d measurement_cov = Eigen::Matrix2d::Zero();

    const double mrPow1 = xPow2 + wrPow1;
    const double myawPow1 = yawPow1 + wyawPow1;
    const double mrPow2 = xPow4 + 2.0*xPow2*wrPow1 + wrPow2;
    const double myawPow2 = yawPow2 + +2.0*yawPow1*wyawPow1+ wyawPow2;
    const double mrPow1_yawPow1 = xPow2_yawPow1 + xPow2*wyawPow1 + wrPow1*wyawPow1 + wrPow1*wyawPow1;

    measurement_mean(MEASUREMENT::IDX::R) = mrPow1;
    measurement_mean(MEASUREMENT::IDX::YAW) = myawPow1;
    measurement_cov(MEASUREMENT::IDX::R, MEASUREMENT::IDX::R) = mrPow2 - mrPow1*mrPow1;
    measurement_cov(MEASUREMENT::IDX::YAW, MEASUREMENT::IDX::YAW) = myawPow2 - myawPow1*myawPow1;
    measurement_cov(MEASUREMENT::IDX::R, MEASUREMENT::IDX::YAW) = mrPow1_yawPow1 - mrPow1*myawPow1;
    measurement_cov(MEASUREMENT::IDX::YAW, MEASUREMENT::IDX::R) = measurement_cov(MEASUREMENT::IDX::R, MEASUREMENT::IDX::YAW);

    Eigen::Matrix2d state_observation_cov = Eigen::Matrix2d::Zero(); // sigma = E[XY^T] - E[X]E[Y]^T
    state_observation_cov(STATE::IDX::X, MEASUREMENT::IDX::R) = xPow3 + xPow1*wrPow1 - xPow1*mrPow1;
    state_observation_cov(STATE::IDX::X, MEASUREMENT::IDX::YAW) = xPow1_yawPow1 + xPow1*wyawPow1 - xPow1*myawPow1;
    state_observation_cov(STATE::IDX::YAW, MEASUREMENT::IDX::R) = xPow2_yawPow1 + yawPow1*wrPow1 - yawPow1*mrPow1;
    state_observation_cov(STATE::IDX::YAW, MEASUREMENT::IDX::YAW) = yawPow2 + yawPow1*wyawPow1 - yawPow1*myawPow1;

    // Kalman Gain
    const auto K = state_observation_cov * measurement_cov.inverse();

    StateInfo updated_info;
    updated_info.mean = predicted_mean + K * (observed_values - measurement_mean);
    updated_info.covariance = predicted_cov - K * measurement_cov * K.transpose();

    return updated_info;
}
