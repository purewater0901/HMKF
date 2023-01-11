#include "model/normal_vehicle_model.h"

#include "distribution/three_dimensional_normal_distribution.h"

using namespace NormalVehicle;

Eigen::VectorXd NormalVehicleModel::propagate(const Eigen::VectorXd& x_curr,
                                              const Eigen::VectorXd& u_curr,
                                              const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map,
                                              const double dt)
{
    const auto wx_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WX);
    const auto wy_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WY);
    const auto wyaw_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WYAW);

    Eigen::VectorXd system_noise = Eigen::VectorXd::Zero(3);
    system_noise(SYSTEM_NOISE::IDX::WX) = wx_dist_ptr->calc_mean();
    system_noise(SYSTEM_NOISE::IDX::WY) = wy_dist_ptr->calc_mean();
    system_noise(SYSTEM_NOISE::IDX::WYAW) = wyaw_dist_ptr->calc_mean();

    return propagate(x_curr, u_curr, system_noise, dt);
}

Eigen::VectorXd NormalVehicleModel::propagate(const Eigen::VectorXd& x_curr,
                                              const Eigen::VectorXd& u_curr,
                                              const Eigen::VectorXd& system_noise,
                                              const double dt)
{
    /*  == Nonlinear model ==
     *
     * x_{k+1}   = x_k + v_k * cos(yaw_k) * dt + wx
     * y_{k+1}   = y_k + v_k * sin(yaw_k) * dt + wy
     * yaw_{k+1} = yaw_k + u_k * dt + wyaw
     *
     */

    Eigen::VectorXd x_next = Eigen::VectorXd::Zero(3);
    x_next(STATE::IDX::X) = x_curr(STATE::IDX::X) + u_curr(INPUT::IDX::V) * std::cos(x_curr(STATE::IDX::YAW)) * dt
                          + system_noise(SYSTEM_NOISE::IDX::WX);
    x_next(STATE::IDX::Y) = x_curr(STATE::IDX::Y) + u_curr(INPUT::IDX::V) * std::sin(x_curr(STATE::IDX::YAW)) * dt
                          + system_noise(SYSTEM_NOISE::IDX::WY);
    x_next(STATE::IDX::YAW) = x_curr(STATE::IDX::YAW) + u_curr(INPUT::IDX::U) * dt + system_noise(SYSTEM_NOISE::IDX::WYAW);

    return x_next;
}

Eigen::VectorXd NormalVehicleModel::measure(const Eigen::VectorXd& x_curr,
                                            const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    const auto wr_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WR);
    const auto wyaw_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WYAW);

    Eigen::VectorXd measurement_noise = Eigen::VectorXd::Zero(2);
    measurement_noise(MEASUREMENT_NOISE::IDX::WR) = wr_dist_ptr->calc_mean();
    measurement_noise(MEASUREMENT_NOISE::IDX::WYAW) = wyaw_dist_ptr->calc_mean();

    return measure(x_curr, measurement_noise);
}

Eigen::VectorXd NormalVehicleModel::measure(const Eigen::VectorXd& x_curr,
                                            const Eigen::VectorXd& measurement_noise)
{
    /*  == Nonlinear model ==
    *
    * r = x^4 + y^4 + wr
    * yaw_k = yaw_k + w_yaw
    *
    */

    Eigen::VectorXd y_next = Eigen::VectorXd::Zero(2);
    y_next(MEASUREMENT::IDX::R) = std::pow(x_curr(STATE::IDX::X), 4) + std::pow(x_curr(STATE::IDX::Y), 4)
                                  + measurement_noise(MEASUREMENT_NOISE::IDX::WR);
    y_next(MEASUREMENT::IDX::YAW) = x_curr(STATE::IDX::YAW) + measurement_noise(MEASUREMENT_NOISE::IDX::WYAW);

    return y_next;
}

Eigen::MatrixXd NormalVehicleModel::getStateMatrix(const Eigen::VectorXd& x_curr,
                                                   const Eigen::VectorXd& u_curr,
                                                   const double dt)
{
    /*  == Nonlinear model ==
     *
     * x_{k+1}   = x_k + v_k * cos(yaw_k) * dt + wx
     * y_{k+1}   = y_k + v_k * sin(yaw_k) * dt + wy
     * yaw_{k+1} = yaw_k + u_k * dt + wyaw
     *
     * dx/dx = 1.0 dy/dx = 0.0 dyaw/dx = 0.0
     * dy/dx = 0.0 dy/dy = 1.0 dyaw/dy = 0.0
     * dx/dyaw = -v_k * sin(yaw_k) * dt
     * dy/dyaw = vk * cos(yaw_k) * dt
     * dyaw/dyaw = 1.0
     */

    const double& v_k = u_curr(INPUT::IDX::V);
    const double& yaw_k = x_curr(STATE::IDX::YAW);
    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(3, 3);
    A(STATE::IDX::X, STATE::IDX::YAW) =  -v_k * std::sin(yaw_k) * dt;
    A(STATE::IDX::Y, STATE::IDX::YAW) =   v_k * std::cos(yaw_k) * dt;

    return A;
}

Eigen::MatrixXd NormalVehicleModel::getProcessNoiseMatrix(const Eigen::VectorXd& x_curr,
                                                          const Eigen::VectorXd& u_curr,
                                                          const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map,
                                                          const double dt)
{
    const auto wx_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WX);
    const auto wy_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WY);
    const auto wyaw_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WYAW);

    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(3, 3);
    Q(SYSTEM_NOISE::IDX::WX, SYSTEM_NOISE::IDX::WX) = wx_dist_ptr->calc_variance();
    Q(SYSTEM_NOISE::IDX::WY, SYSTEM_NOISE::IDX::WY) = wy_dist_ptr->calc_variance();
    Q(SYSTEM_NOISE::IDX::WYAW, SYSTEM_NOISE::IDX::WYAW) = wyaw_dist_ptr->calc_variance();

    return Q;
}

Eigen::MatrixXd NormalVehicleModel::getMeasurementMatrix(const Eigen::VectorXd& x_curr,
                                                         const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    /*  == Nonlinear model ==
     *
     * r = x^4 + y^4 + wr
     * yaw_k = yaw_k + w_yaw
     * dr/dx = 4*x^3 dr/dy = 4y^3
     * dyaw/dyaw = 1.0
     */
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 3);
    H(MEASUREMENT::IDX::R, STATE::IDX::X) = 4.0 * std::pow(x_curr(STATE::IDX::X), 3);
    H(MEASUREMENT::IDX::R, STATE::IDX::Y) = 4.0 * std::pow(x_curr(STATE::IDX::Y), 3);
    H(MEASUREMENT::IDX::YAW, STATE::IDX::YAW) = 1.0;

    return H;
}

Eigen::MatrixXd NormalVehicleModel::getMeasurementNoiseMatrix(const Eigen::VectorXd& x_curr,
                                                              const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    const auto wr_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WR);
    const auto wyaw_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WYAW);

    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(2, 2);
    R(MEASUREMENT_NOISE::IDX::WR, MEASUREMENT_NOISE::IDX::WR) = wr_dist_ptr->calc_variance();
    R(MEASUREMENT_NOISE::IDX::WYAW, MEASUREMENT_NOISE::IDX::WYAW) = wyaw_dist_ptr->calc_variance();

    return R;
}

StateInfo NormalVehicleModel::propagateStateMoments(const StateInfo &state_info,
                                                    const Eigen::VectorXd &control_inputs,
                                                    const double dt,
                                                    const std::map<int, std::shared_ptr<BaseDistribution>> &noise_map)
{
    // Step1. Approximate to Gaussian Distribution
    const auto state_mean = state_info.mean;
    const auto state_cov = state_info.covariance;
    ThreeDimensionalNormalDistribution dist(state_info.mean, state_info.covariance);

    // Step2. State Moment
    const double xPow1 = dist.calc_moment(STATE::IDX::X, 1); // x
    const double yPow1 = dist.calc_moment(STATE::IDX::Y, 1); // y
    const double cPow1 = dist.calc_cos_moment(STATE::IDX::YAW, 1); // cos(yaw)
    const double sPow1 = dist.calc_sin_moment(STATE::IDX::YAW, 1); // sin(yaw)
    const double yawPow1 = dist.calc_moment(STATE::IDX::YAW, 1); // yaw
    const double xPow2 = dist.calc_moment(STATE::IDX::X, 2); // x^2
    const double yPow2 = dist.calc_moment(STATE::IDX::Y, 2); // y^2
    const double cPow2 = dist.calc_cos_moment(STATE::IDX::YAW, 2); // cos(yaw)^2
    const double sPow2 = dist.calc_sin_moment(STATE::IDX::YAW, 2); // sin(yaw)^2
    const double yawPow2 = dist.calc_moment(STATE::IDX::YAW, 2); // yaw^2
    const double xPow1_yPow1 = dist.calc_cross_second_moment(STATE::IDX::X, STATE::IDX::Y); // xy
    const double cPow1_xPow1 = dist.calc_x_cos_z_moment(STATE::IDX::X, STATE::IDX::YAW); // x*cos(yaw)
    const double sPow1_xPow1 = dist.calc_x_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW); // x*sin(yaw)
    const double cPow1_yPow1 = dist.calc_x_cos_z_moment(STATE::IDX::Y, STATE::IDX::YAW); // y*cos(yaw)
    const double sPow1_yPow1 = dist.calc_x_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW); // y*sin(yaw)
    const double cPow1_sPow1 = dist.calc_cos_sin_moment(STATE::IDX::YAW, 1, 1); // cos(yaw)*sin(yaw)
    const double xPow1_yawPow1 = dist.calc_cross_second_moment(STATE::IDX::X, STATE::IDX::YAW); // x*yaw
    const double yPow1_yawPow1 = dist.calc_cross_second_moment(STATE::IDX::Y, STATE::IDX::YAW); // y*yaw
    const double cPow1_yawPow1 = dist.calc_x_cos_x_moment(STATE::IDX::YAW, 1, 1); // yaw*cos(yaw)
    const double sPow1_yawPow1 = dist.calc_x_sin_x_moment(STATE::IDX::YAW, 1, 1); // yaw*sin(yaw)

    // Step3. Control Input
    const double v = control_inputs(INPUT::IDX::V) * dt;
    const double u = control_inputs(INPUT::IDX::U) * dt;

    // Step4. System Noise
    const auto wx_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WX);
    const auto wy_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WY);
    const auto wyaw_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WYAW);
    const double wxPow1 = wx_dist_ptr->calc_moment(1);
    const double wyPow1 = wy_dist_ptr->calc_moment(1);
    const double wyawPow1 = wyaw_dist_ptr->calc_moment(1);
    const double wxPow2 = wx_dist_ptr->calc_moment(2);
    const double wyPow2 = wy_dist_ptr->calc_moment(2);
    const double wyawPow2 = wyaw_dist_ptr->calc_moment(2);

    // Dynamics updates.
    const double next_xPow1 = xPow1 + v*cPow1 + wxPow1;
    const double next_yPow1 = yPow1 + v*sPow1 + wyPow1;
    const double next_yawPow1 = yawPow1 + u + wyawPow1;

    const double next_xPow2 = xPow2 + v*v*cPow2 + wxPow2 + 2*v*cPow1*wxPow1 + 2*v*cPow1_xPow1 + 2*wxPow1*xPow1;
    const double next_yPow2 = yPow2 + v*v*sPow2 + wyPow2 + 2*v*sPow1*wyPow1 + 2*v*sPow1_yPow1 + 2*wyPow1*yPow1;
    const double next_yawPow2 = yawPow2 + u*u + wyawPow2 + 2*u*wyawPow1 + 2*u*yawPow1 + 2*wyawPow1*yawPow1;
    const double next_xPow1_yPow1 = xPow1_yPow1 + v*sPow1_xPow1 + xPow1*wyPow1
                                  + v*cPow1_yPow1 + v*v*cPow1_sPow1 + v*cPow1*wyPow1
                                  + wxPow1*yPow1 + v*sPow1*wxPow1 + wxPow1*wyPow1;
    const double next_xPow1_yawPow1 = u*v*cPow1 + u*wxPow1 + u*xPow1 + v*cPow1*wyawPow1 + v*cPow1_yawPow1
                                    + wxPow1*wyawPow1 + wxPow1*yawPow1 + wyawPow1*xPow1 + xPow1_yawPow1;
    const double next_yPow1_yawPow1 = u*v*sPow1 + u*wyPow1 + u*yPow1 + v*sPow1*wyawPow1 + v*sPow1_yawPow1
                                    + wyPow1*wyawPow1 + wyPow1*yawPow1 + wyawPow1*yPow1 + yPow1_yawPow1;

    StateInfo next_state;
    next_state.mean = Eigen::VectorXd::Zero(3);
    next_state.covariance = Eigen::MatrixXd::Zero(3, 3);

    next_state.mean(STATE::IDX::X) = next_xPow1;
    next_state.mean(STATE::IDX::Y) = next_yPow1;
    next_state.mean(STATE::IDX::YAW)= next_yawPow1;
    next_state.covariance(STATE::IDX::X, STATE::IDX::X) = next_xPow2 - next_xPow1 * next_xPow1;
    next_state.covariance(STATE::IDX::Y, STATE::IDX::Y) = next_yPow2 - next_yPow1 * next_yPow1;
    next_state.covariance(STATE::IDX::YAW, STATE::IDX::YAW) = next_yawPow2 - next_yawPow1 * next_yawPow1;
    next_state.covariance(STATE::IDX::X, STATE::IDX::Y) = next_xPow1_yPow1 - next_xPow1 * next_yPow1;
    next_state.covariance(STATE::IDX::X, STATE::IDX::YAW) = next_xPow1_yawPow1 - next_xPow1 * next_yawPow1;
    next_state.covariance(STATE::IDX::Y, STATE::IDX::YAW) = next_yPow1_yawPow1 - next_yPow1 * next_yawPow1;
    next_state.covariance(STATE::IDX::Y, STATE::IDX::X) = next_state.covariance(STATE::IDX::X, STATE::IDX::Y);
    next_state.covariance(STATE::IDX::YAW, STATE::IDX::X) = next_state.covariance(STATE::IDX::X, STATE::IDX::YAW);
    next_state.covariance(STATE::IDX::YAW, STATE::IDX::Y) = next_state.covariance(STATE::IDX::Y, STATE::IDX::YAW);

    return next_state;

}

StateInfo NormalVehicleModel::getMeasurementMoments(const StateInfo &state_info,
                                                    const std::map<int, std::shared_ptr<BaseDistribution>> &noise_map)
{
    const auto predicted_mean = state_info.mean;
    const auto predicted_cov = state_info.covariance;
    ThreeDimensionalNormalDistribution dist(state_info.mean, state_info.covariance);

    const double yawPow1 = dist.calc_moment(STATE::IDX::YAW, 1);
    const double yawPow2 = dist.calc_moment(STATE::IDX::YAW, 2);
    const double xPow4 = dist.calc_moment(STATE::IDX::X, 4);
    const double yPow4 = dist.calc_moment(STATE::IDX::Y, 4);
    const double xPow4_yawPow1 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW, 4, 1, 0, 0);
    const double yPow4_yawPow1 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW, 4, 1, 0, 0);
    const double xPow8 = dist.calc_moment(STATE::IDX::X, 8);
    const double yPow8 = dist.calc_moment(STATE::IDX::Y, 8);
    const double xPow4_yPow4 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::Y, 4, 4, 0, 0);

    // Observation Noise
    const auto wr_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WR);
    const auto wyaw_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WYAW);
    const double mrPow1 = wr_dist_ptr->calc_moment(1);
    const double mrPow2 = wr_dist_ptr->calc_moment(2);
    const double myawPow1 = wyaw_dist_ptr->calc_moment(1);
    const double myawPow2 = wyaw_dist_ptr->calc_moment(2);

    // measurement update
    const double measurement_rPow1 = xPow4 + yPow4 + mrPow1;
    const double measurement_rPow2 = xPow8 + yPow8 + mrPow2 + 2.0*xPow4_yPow4 + 2.0*xPow4*mrPow1 + 2.0*yPow4*mrPow1;
    const double measurement_yawPow1 = yawPow1 + myawPow1;
    const double measurement_yawPow2 = yawPow2 + myawPow2 + 2.0*yawPow1*myawPow1;
    const double measurement_rPow1_yawPow1 = xPow4_yawPow1 + yPow4_yawPow1 + yawPow1*mrPow1 + xPow4*myawPow1
                                           + yPow4*myawPow1 + mrPow1*myawPow1;

    StateInfo measurement_info;
    measurement_info.mean = Eigen::VectorXd::Zero(2);
    measurement_info.covariance = Eigen::MatrixXd::Zero(2, 2);

    measurement_info.mean(MEASUREMENT::IDX::R) = measurement_rPow1;
    measurement_info.mean(MEASUREMENT::IDX::YAW) = measurement_yawPow1;
    measurement_info.covariance(MEASUREMENT::IDX::R, MEASUREMENT::IDX::R) = measurement_rPow2 - measurement_rPow1*measurement_rPow1;
    measurement_info.covariance(MEASUREMENT::IDX::YAW, MEASUREMENT::IDX::YAW) = measurement_yawPow2 - measurement_yawPow1*measurement_yawPow1;
    measurement_info.covariance(MEASUREMENT::IDX::R, MEASUREMENT::IDX::YAW) = measurement_rPow1_yawPow1 - measurement_rPow1*measurement_yawPow1;
    measurement_info.covariance(MEASUREMENT::IDX::YAW, MEASUREMENT::IDX::R) = measurement_info.covariance(MEASUREMENT::IDX::R, MEASUREMENT::IDX::YAW);

    return measurement_info;
}

Eigen::MatrixXd NormalVehicleModel::getStateMeasurementMatrix(const StateInfo& state_info,
                                                              const StateInfo& measurement_info,
                                                              const std::map<int, std::shared_ptr<BaseDistribution>> &noise_map)
{
    const auto predicted_mean = state_info.mean;
    const auto measurement_mean = measurement_info.mean;
    ThreeDimensionalNormalDistribution dist(state_info.mean, state_info.covariance);

    // Observation Noise
    const auto wr_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WR);
    const auto wyaw_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WYAW);
    const double mrPow1 = wr_dist_ptr->calc_moment(1);
    const double myawPow1 = wyaw_dist_ptr->calc_moment(1);

    // State Moment
    const double xPow1 = dist.calc_moment(STATE::IDX::X, 1);
    const double yPow1 = dist.calc_moment(STATE::IDX::Y, 1);
    const double yawPow1 = dist.calc_moment(STATE::IDX::YAW, 1);
    const double xPow1_yawPow1 = dist.calc_cross_second_moment(STATE::IDX::X, STATE::IDX::YAW);
    const double yPow1_yawPow1 = dist.calc_cross_second_moment(STATE::IDX::Y, STATE::IDX::YAW);
    const double yawPow2 = dist.calc_moment(STATE::IDX::YAW, 2);
    const double xPow5 = dist.calc_moment(STATE::IDX::X, 5);
    const double yPow5 = dist.calc_moment(STATE::IDX::Y, 5);
    const double xPow1_yPow4 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::Y, 1, 4, 0, 0);
    const double xPow4_yPow1 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::Y, 4, 1, 0, 0);
    const double xPow4_yawPow1 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW, 4, 1, 0, 0);
    const double yPow4_yawPow1 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW, 4, 1, 0, 0);

    Eigen::MatrixXd state_observation_cov(3, 2); // sigma = E[XY^T] - E[X]E[Y]^T
    state_observation_cov(STATE::IDX::X, MEASUREMENT::IDX::R)
            = xPow5 + xPow1_yPow4 + xPow1 * mrPow1
              - predicted_mean(STATE::IDX::X) * measurement_mean(MEASUREMENT::IDX::R); // xp * (xp^4 + yp^4 + mr)
    state_observation_cov(STATE::IDX::X, MEASUREMENT::IDX::YAW)
            = xPow1_yawPow1 + xPow1 * myawPow1
              - predicted_mean(STATE::IDX::X) * measurement_mean(MEASUREMENT::IDX::YAW); // x_p * (yaw + myaw)
    state_observation_cov(STATE::IDX::Y, MEASUREMENT::IDX::R)
            = xPow4_yPow1 + yPow5  + yPow1 * mrPow1
              - predicted_mean(STATE::IDX::Y) * measurement_mean(MEASUREMENT::IDX::R); // yp * (xp^4 + yp^4 + mr)
    state_observation_cov(STATE::IDX::Y, MEASUREMENT::IDX::YAW)
            = yPow1_yawPow1 + yPow1 * myawPow1
              - predicted_mean(STATE::IDX::Y) * measurement_mean(MEASUREMENT::IDX::YAW); // y_p * (yaw + myaw)
    state_observation_cov(STATE::IDX::YAW, MEASUREMENT::IDX::R)
            = xPow4_yawPow1 + yPow4_yawPow1 + yawPow1 * mrPow1
              - predicted_mean(STATE::IDX::YAW) * measurement_mean(MEASUREMENT::IDX::R); // yaw_p * (x_p^4 + y_p^4 + mr)
    state_observation_cov(STATE::IDX::YAW, MEASUREMENT::IDX::YAW)
            = yawPow2 + yawPow1 * myawPow1
              - predicted_mean(STATE::IDX::YAW) * measurement_mean(MEASUREMENT::IDX::YAW); // yaw_p * (yaw_p + myaw)

    return state_observation_cov;
}
