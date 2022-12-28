#include "model/mobile_robot_model.h"

#include "distribution/four_dimensional_normal_distribution.h"

using namespace MobileRobot;

Eigen::VectorXd MobileRobotModel::propagate(const Eigen::VectorXd& x_curr,
                                            const Eigen::VectorXd& u_curr,
                                            const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map,
                                            const double dt)
{
    const auto wv_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WV);
    const auto wyaw_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WYAW);
    Eigen::VectorXd system_noise = Eigen::VectorXd::Zero(2);
    system_noise(SYSTEM_NOISE::IDX::WV) = wv_dist_ptr->calc_mean();
    system_noise(SYSTEM_NOISE::IDX::WYAW) = wyaw_dist_ptr->calc_mean();

    return propagate(x_curr, u_curr, system_noise, dt);
}

Eigen::VectorXd MobileRobotModel::propagate(const Eigen::VectorXd& x_curr,
                                            const Eigen::VectorXd& u_curr,
                                            const Eigen::VectorXd& system_noise,
                                            const double dt)
{
    /*  == Nonlinear model ==
     *
     * x_{k+1}   = x_k + v_k+wv * cos(yaw_k) * dt
     * y_{k+1}   = y_k + v_k * sin(yaw_k) * dt
     * v_{k+1}   = v_k + a_k * dt + w_v
     * yaw_{k+1} = yaw_k + u_k * dt + w_yaw
     *
     */
    Eigen::VectorXd x_next = Eigen::VectorXd::Zero(4);
    x_next(STATE::IDX::X) = x_curr(STATE::IDX::X) + x_curr(STATE::IDX::V) * std::cos(x_curr(STATE::IDX::YAW)) * dt;
    x_next(STATE::IDX::Y) = x_curr(STATE::IDX::Y) + x_curr(STATE::IDX::V) * std::sin(x_curr(STATE::IDX::YAW)) * dt;
    x_next(STATE::IDX::V) = x_curr(STATE::IDX::V) + u_curr(INPUT::IDX::A) + system_noise(SYSTEM_NOISE::IDX::WV);
    x_next(STATE::IDX::YAW) = x_curr(STATE::IDX::YAW) + u_curr(INPUT::IDX::U) + system_noise(SYSTEM_NOISE::IDX::WYAW);

    return x_next;
}

Eigen::VectorXd MobileRobotModel::measure(const Eigen::VectorXd& x_curr,
                                          const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    const auto wx_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WX);
    const auto wy_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WY);
    const auto wvc_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WVC);
    const auto wyaw_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WYAW);

    Eigen::VectorXd measurement_noise = Eigen::VectorXd::Zero(4);
    measurement_noise(MEASUREMENT_NOISE::IDX::WX) = wx_dist_ptr->calc_mean();
    measurement_noise(MEASUREMENT_NOISE::IDX::WY) = wy_dist_ptr->calc_mean();
    measurement_noise(MEASUREMENT_NOISE::IDX::WVC) = wvc_dist_ptr->calc_mean();
    measurement_noise(MEASUREMENT_NOISE::IDX::WYAW) = wyaw_dist_ptr->calc_mean();

    return measure(x_curr, measurement_noise);
}

Eigen::VectorXd MobileRobotModel::measure(const Eigen::VectorXd& x_curr, const Eigen::VectorXd& measurement_noise)
{
    /*  == Nonlinear model ==
    *
    * x = x + v * wx
    * y = y + v * wy
    * vc = (v + wv) * cos(yaw_k + wyaw)
    *
    */

    Eigen::VectorXd y_next = Eigen::VectorXd::Zero(3);
    y_next(OBSERVATION::IDX::X) = x_curr(STATE::IDX::X) + x_curr(STATE::IDX::V) * measurement_noise(MEASUREMENT_NOISE::IDX::WX);
    y_next(OBSERVATION::IDX::Y) = x_curr(STATE::IDX::Y) + x_curr(STATE::IDX::V) * measurement_noise(MEASUREMENT_NOISE::IDX::WY);
    y_next(OBSERVATION::IDX::VC) = (x_curr(STATE::IDX::V) + measurement_noise(MEASUREMENT_NOISE::IDX::WVC))
                                   * std::cos(x_curr(STATE::IDX::YAW) + measurement_noise(MEASUREMENT_NOISE::IDX::WYAW));

    return y_next;
}

Eigen::MatrixXd MobileRobotModel::getStateMatrix(const Eigen::VectorXd& x_curr,
                                                 const Eigen::VectorXd& u_curr,
                                                 const double dt)
{
    const double& v_k = x_curr(STATE::IDX::V);
    const double& yaw_k = x_curr(STATE::IDX::YAW);
    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(4, 4);
    A(STATE::IDX::X, STATE::IDX::V) = std::cos(yaw_k) * dt;
    A(STATE::IDX::X, STATE::IDX::YAW) =  -v_k * std::sin(yaw_k) * dt;
    A(STATE::IDX::Y, STATE::IDX::V) = std::sin(yaw_k) * dt;
    A(STATE::IDX::Y, STATE::IDX::YAW) =   v_k * std::cos(yaw_k) * dt;

    return A;
}

Eigen::MatrixXd MobileRobotModel::getProcessNoiseMatrix(const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    const auto wv_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WV);
    const auto wyaw_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WYAW);
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(4, 4);
    Q(STATE::IDX::V, STATE::IDX::V) = wv_dist_ptr->calc_variance();
    Q(STATE::IDX::YAW, STATE::IDX::YAW) = wyaw_dist_ptr->calc_variance();

    return Q;
}

Eigen::MatrixXd MobileRobotModel::getMeasurementMatrix(const Eigen::VectorXd& x_curr,
                                                       const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    const auto wx_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WX);
    const auto wy_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WY);
    const auto wvc_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WVC);
    const auto wyaw_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WYAW);

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, 4);
    H(OBSERVATION::IDX::X, STATE::IDX::X) = 1.0;
    H(OBSERVATION::IDX::X, STATE::IDX::V) = wx_dist_ptr->calc_mean();
    H(OBSERVATION::IDX::Y, STATE::IDX::Y) = 1.0;
    H(OBSERVATION::IDX::Y, STATE::IDX::V) = wy_dist_ptr->calc_mean();
    H(OBSERVATION::IDX::VC, STATE::IDX::V) = std::cos(x_curr(STATE::IDX::YAW) + wyaw_dist_ptr->calc_mean());
    H(OBSERVATION::IDX::VC, STATE::IDX::YAW) = -(x_curr(STATE::IDX::V) + wvc_dist_ptr->calc_mean())
                                               * std::sin(x_curr(STATE::IDX::YAW) + wyaw_dist_ptr->calc_mean());
    return H;
}

Eigen::MatrixXd MobileRobotModel::getMeasurementNoiseMatrix(const Eigen::VectorXd& x_curr,
                                                            const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    const auto wx_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WX);
    const auto wy_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WY);
    const auto wvc_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WVC);
    const auto wyaw_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WYAW);

    const double& yaw_k = x_curr(STATE::IDX::YAW);

    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(4, 4);
    R(MEASUREMENT_NOISE::IDX::WX, MEASUREMENT_NOISE::IDX::WX) = wx_dist_ptr->calc_variance();
    R(MEASUREMENT_NOISE::IDX::WY, MEASUREMENT_NOISE::IDX::WY) = wy_dist_ptr->calc_variance();
    R(MEASUREMENT_NOISE::IDX::WVC, MEASUREMENT_NOISE::IDX::WVC) = wvc_dist_ptr->calc_variance();
    R(MEASUREMENT_NOISE::IDX::WYAW, MEASUREMENT_NOISE::IDX::WYAW) = wyaw_dist_ptr->calc_variance();

    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(3, 4);
    M(OBSERVATION::IDX::X, MEASUREMENT_NOISE::IDX::WX) = x_curr(STATE::IDX::V);
    M(OBSERVATION::IDX::Y, MEASUREMENT_NOISE::IDX::WY) = x_curr(STATE::IDX::V);
    M(OBSERVATION::IDX::VC, MEASUREMENT_NOISE::IDX::WVC) = std::cos(yaw_k + wyaw_dist_ptr->calc_mean());
    M(OBSERVATION::IDX::VC, MEASUREMENT_NOISE::IDX::WYAW) = -(x_curr(STATE::IDX::V) + wvc_dist_ptr->calc_mean())
                                                            * std::sin(yaw_k + wyaw_dist_ptr->calc_mean());

    return M*R*M.transpose();
}

StateInfo MobileRobotModel::propagateStateMoments(const StateInfo &state_info,
                                                  const Eigen::VectorXd &control_inputs,
                                                  const double dt,
                                                  const std::map<int, std::shared_ptr<BaseDistribution>> &noise_map)
{
    // Step1. Approximate to Gaussian Distribution
    const auto state_mean = state_info.mean;
    const auto state_cov = state_info.covariance;
    FourDimensionalNormalDistribution dist(state_info.mean, state_info.covariance);

    // Step2. State Moment
    const double xPow1 = dist.calc_moment(STATE::IDX::X, 1);
    const double yPow1 = dist.calc_moment(STATE::IDX::Y, 1);
    const double vPow1 = dist.calc_moment(STATE::IDX::V, 1);
    const double yawPow1 = dist.calc_moment(STATE::IDX::YAW, 1);
    const double cPow1 = dist.calc_cos_moment(STATE::IDX::YAW, 1);
    const double sPow1 = dist.calc_sin_moment(STATE::IDX::YAW, 1);

    const double xPow2 = dist.calc_moment(STATE::IDX::X, 2);
    const double yPow2 = dist.calc_moment(STATE::IDX::Y, 2);
    const double vPow2 = dist.calc_moment(STATE::IDX::V, 2);
    const double yawPow2 = dist.calc_moment(STATE::IDX::YAW, 2);
    const double cPow2 = dist.calc_cos_moment(STATE::IDX::YAW, 2);
    const double sPow2 = dist.calc_sin_moment(STATE::IDX::YAW, 2);
    const double xPow1_yPow1 = dist.calc_cross_second_moment(STATE::IDX::X, STATE::IDX::Y);
    const double xPow1_yawPow1 = dist.calc_cross_second_moment(STATE::IDX::X, STATE::IDX::YAW);
    const double yPow1_yawPow1 = dist.calc_cross_second_moment(STATE::IDX::Y, STATE::IDX::YAW);
    const double vPow1_xPow1 = dist.calc_cross_second_moment(STATE::IDX::V, STATE::IDX::X);
    const double vPow1_yPow1 = dist.calc_cross_second_moment(STATE::IDX::V, STATE::IDX::Y);
    const double vPow1_yawPow1 = dist.calc_cross_second_moment(STATE::IDX::V, STATE::IDX::YAW);
    const double vPow1_cPow1 = dist.calc_x_cos_z_moment(STATE::IDX::V, STATE::IDX::YAW);
    const double vPow1_sPow1 = dist.calc_x_sin_z_moment(STATE::IDX::V, STATE::IDX::YAW);
    const double cPow1_xPow1 = dist.calc_x_cos_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    const double sPow1_xPow1 = dist.calc_x_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    const double cPow1_yPow1 = dist.calc_x_cos_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    const double sPow1_yPow1 = dist.calc_x_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    const double cPow1_yawPow1 = dist.calc_x_cos_x_moment(STATE::IDX::YAW, 1, 1);
    const double sPow1_yawPow1 = dist.calc_x_sin_x_moment(STATE::IDX::YAW, 1, 1);
    const double cPow1_sPow1 = dist.calc_cos_sin_moment(STATE::IDX::YAW, 1, 1);

    const double vPow1_cPow2 = dist.calc_x_cos_y_cos_y_moment(STATE::IDX::V, STATE::IDX::YAW);
    const double vPow1_sPow2 = dist.calc_x_sin_y_sin_y_moment(STATE::IDX::V, STATE::IDX::YAW);
    const double vPow2_cPow1 = dist.calc_xx_cos_z_moment(STATE::IDX::V, STATE::IDX::YAW);
    const double vPow2_sPow1 = dist.calc_xx_sin_z_moment(STATE::IDX::V, STATE::IDX::YAW);
    const double vPow1_cPow1_xPow1 = dist.calc_xy_cos_z_moment(STATE::IDX::X, STATE::IDX::V, STATE::IDX::YAW);
    const double vPow1_cPow1_yPow1 = dist.calc_xy_cos_z_moment(STATE::IDX::Y, STATE::IDX::V, STATE::IDX::YAW);
    const double vPow1_cPow1_yawPow1 = dist.calc_xy_cos_y_moment(STATE::IDX::V, STATE::IDX::YAW);
    const double vPow1_sPow1_xPow1 = dist.calc_xy_sin_z_moment(STATE::IDX::X, STATE::IDX::V, STATE::IDX::YAW);
    const double vPow1_sPow1_yPow1 = dist.calc_xy_sin_z_moment(STATE::IDX::Y, STATE::IDX::V, STATE::IDX::YAW);
    const double vPow1_sPow1_yawPow1 = dist.calc_xy_sin_y_moment(STATE::IDX::V, STATE::IDX::YAW);
    const double vPow1_cPow1_sPow1 = dist.calc_x_cos_y_sin_y_moment(STATE::IDX::V, STATE::IDX::YAW);

    const double vPow2_cPow2 = dist.calc_xx_cos_y_cos_y_moment(STATE::IDX::V, STATE::IDX::YAW);
    const double vPow2_sPow2 = dist.calc_xx_sin_y_sin_y_moment(STATE::IDX::V, STATE::IDX::YAW);
    const double vPow2_cPow1_sPow1 = dist.calc_xx_cos_y_sin_y_moment(STATE::IDX::V, STATE::IDX::YAW);

    // Step3. Control Input
    const double a = control_inputs(INPUT::IDX::A);
    const double u = control_inputs(INPUT::IDX::U);
    const double cu = std::cos(u);
    const double su = std::sin(u);

    // Step4. System Noise
    const auto wv_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WV);
    const auto wyaw_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WYAW);
    const double wvPow1 = wv_dist_ptr->calc_moment(1);
    const double wyawPow1 = wyaw_dist_ptr->calc_moment(1);
    const double cyawPow1 = wyaw_dist_ptr->calc_cos_moment(1);
    const double syawPow1 = wyaw_dist_ptr->calc_sin_moment(1);

    const double wvPow2 = wv_dist_ptr->calc_moment(2);
    const double wyawPow2 = wyaw_dist_ptr->calc_moment(2);
    const double cyawPow2 = wyaw_dist_ptr->calc_cos_moment(2);
    const double syawPow2 = wyaw_dist_ptr->calc_sin_moment(2);
    const double cyawPow1_syawPow1 = wyaw_dist_ptr->calc_cos_sin_moment(1, 1);

    // Dynamics updates.
    const double next_xPow1 = dt*vPow1_cPow1 + xPow1;
    const double next_yPow1 = dt*vPow1_sPow1 + yPow1;
    const double next_vPow1 = 1.0*a + vPow1 + wvPow1;
    const double next_yawPow1 = 1.0*u + wyawPow1 + yawPow1;
    const double next_cPow1 = cu*cPow1*cyawPow1 - cu*sPow1*syawPow1 - su*cPow1*syawPow1 - su*cyawPow1*sPow1;
    const double next_sPow1 = cu*cPow1*syawPow1 + cu*cyawPow1*sPow1 + su*cPow1*cyawPow1 - su*sPow1*syawPow1;

    const double next_xPow2 = pow(dt, 2)*vPow2_cPow2 + 2*dt*vPow1_cPow1_xPow1 + xPow2;
    const double next_yPow2 = pow(dt, 2)*vPow2_sPow2 + 2*dt*vPow1_sPow1_yPow1 + yPow2;
    const double next_vPow2 = 1.0*pow(a, 2) + 2*a*vPow1 + 2*a*wvPow1 + 2*vPow1*wvPow1 + vPow2 + wvPow2;
    const double next_yawPow2 = 1.0*pow(u, 2) + 2*u*wyawPow1 + 2*u*yawPow1 + 2*wyawPow1*yawPow1 + wyawPow2 + yawPow2;
    const double next_cPow2 = pow(cu, 2)*cPow2*cyawPow2 + pow(cu, 2)*sPow2*syawPow2 - 2*cu*su*cPow1_sPow1*cyawPow2 + 2*cu*su*cPow1_sPow1*syawPow2 - 2*cu*su*cPow2*cyawPow1_syawPow1 + 2*cu*su*cyawPow1_syawPow1*sPow2 + pow(su, 2)*cPow2*syawPow2 + pow(su, 2)*cyawPow2*sPow2 + cPow1_sPow1*cyawPow1_syawPow1*(-2*pow(cu, 2) + 2*pow(su, 2));
    const double next_sPow2 = pow(cu, 2)*cPow2*syawPow2 + pow(cu, 2)*cyawPow2*sPow2 + 2*cu*su*cPow1_sPow1*cyawPow2 - 2*cu*su*cPow1_sPow1*syawPow2 + 2*cu*su*cPow2*cyawPow1_syawPow1 - 2*cu*su*cyawPow1_syawPow1*sPow2 + pow(su, 2)*cPow2*cyawPow2 + pow(su, 2)*sPow2*syawPow2 + cPow1_sPow1*cyawPow1_syawPow1*(2*pow(cu, 2) - 2*pow(su, 2));
    const double next_vPow1_cPow1 = a*cu*cPow1*cyawPow1 - a*cu*sPow1*syawPow1 - a*su*cPow1*syawPow1 - a*su*cyawPow1*sPow1 + cu*cPow1*cyawPow1*wvPow1 + cu*cyawPow1*vPow1_cPow1 - cu*sPow1*syawPow1*wvPow1 - cu*syawPow1*vPow1_sPow1 - su*cPow1*syawPow1*wvPow1 - su*cyawPow1*sPow1*wvPow1 - su*cyawPow1*vPow1_sPow1 - su*syawPow1*vPow1_cPow1;
    const double next_vPow1_sPow1 = a*cu*cPow1*syawPow1 + a*cu*cyawPow1*sPow1 + a*su*cPow1*cyawPow1 - a*su*sPow1*syawPow1 + cu*cPow1*syawPow1*wvPow1 + cu*cyawPow1*sPow1*wvPow1 + cu*cyawPow1*vPow1_sPow1 + cu*syawPow1*vPow1_cPow1 + su*cPow1*cyawPow1*wvPow1 + su*cyawPow1*vPow1_cPow1 - su*sPow1*syawPow1*wvPow1 - su*syawPow1*vPow1_sPow1;
    const double next_vPow2_cPow2 = pow(a, 2)*pow(cu, 2)*cPow2*cyawPow2 + pow(a, 2)*pow(cu, 2)*sPow2*syawPow2 - 2*pow(a, 2)*cu*su*cPow1_sPow1*cyawPow2 + 2*pow(a, 2)*cu*su*cPow1_sPow1*syawPow2 - 2*pow(a, 2)*cu*su*cPow2*cyawPow1_syawPow1 + 2*pow(a, 2)*cu*su*cyawPow1_syawPow1*sPow2 + pow(a, 2)*pow(su, 2)*cPow2*syawPow2 + pow(a, 2)*pow(su, 2)*cyawPow2*sPow2 + 2*a*pow(cu, 2)*vPow1_cPow2*cyawPow2 + 2*a*pow(cu, 2)*cPow2*cyawPow2*wvPow1 + 2*a*pow(cu, 2)*vPow1_sPow2*syawPow2 + 2*a*pow(cu, 2)*sPow2*syawPow2*wvPow1 - 4*a*cu*su*cPow1_sPow1*cyawPow2*wvPow1 + 4*a*cu*su*cPow1_sPow1*syawPow2*wvPow1 - 4*a*cu*su*vPow1_cPow2*cyawPow1_syawPow1 - 2*a*cu*su*vPow1_cPow1_sPow1*cyawPow2 + 2*a*cu*su*vPow1_cPow1_sPow1*syawPow2 - 4*a*cu*su*cPow2*cyawPow1_syawPow1*wvPow1 + 4*a*cu*su*cyawPow1_syawPow1*vPow1_sPow2 + 4*a*cu*su*cyawPow1_syawPow1*sPow2*wvPow1 - 2*a*cu*su*cyawPow2*vPow1_cPow1_sPow1 + 2*a*cu*su*vPow1_cPow1_sPow1*syawPow2 + 2*a*pow(su, 2)*vPow1_cPow2*syawPow2 + 2*a*pow(su, 2)*cPow2*syawPow2*wvPow1 + 2*a*pow(su, 2)*cyawPow2*vPow1_sPow2 + 2*a*pow(su, 2)*cyawPow2*sPow2*wvPow1 + 2*pow(cu, 2)*vPow1_cPow2*cyawPow2*wvPow1 + pow(cu, 2)*cPow2*cyawPow2*wvPow2 + pow(cu, 2)*cyawPow2*vPow2_cPow2 + 2*pow(cu, 2)*vPow1_sPow2*syawPow2*wvPow1 + pow(cu, 2)*sPow2*syawPow2*wvPow2 + pow(cu, 2)*syawPow2*vPow2_sPow2 - 2*cu*su*cPow1_sPow1*cyawPow2*wvPow2 + 2*cu*su*cPow1_sPow1*syawPow2*wvPow2 - 4*cu*su*vPow1_cPow2*cyawPow1_syawPow1*wvPow1 - 2*cu*su*vPow1_cPow1_sPow1*cyawPow2*wvPow1 + 2*cu*su*vPow1_cPow1_sPow1*syawPow2*wvPow1 - 2*cu*su*cPow2*cyawPow1_syawPow1*wvPow2 + 4*cu*su*cyawPow1_syawPow1*vPow1_sPow2*wvPow1 + 2*cu*su*cyawPow1_syawPow1*sPow2*wvPow2 - 2*cu*su*cyawPow1_syawPow1*vPow2_cPow2 + 2*cu*su*cyawPow1_syawPow1*vPow2_sPow2 - 2*cu*su*cyawPow2*vPow1_cPow1_sPow1*wvPow1 - 2*cu*su*cyawPow2*vPow2_cPow1_sPow1 + 2*cu*su*vPow1_cPow1_sPow1*syawPow2*wvPow1 + 2*cu*su*syawPow2*vPow2_cPow1_sPow1 + 2*pow(su, 2)*vPow1_cPow2*syawPow2*wvPow1 + pow(su, 2)*cPow2*syawPow2*wvPow2 + 2*pow(su, 2)*cyawPow2*vPow1_sPow2*wvPow1 + pow(su, 2)*cyawPow2*sPow2*wvPow2 + pow(su, 2)*cyawPow2*vPow2_sPow2 + pow(su, 2)*syawPow2*vPow2_cPow2 + cPow1_sPow1*cyawPow1_syawPow1*wvPow1*(-4*a*pow(cu, 2) + 4*a*pow(su, 2)) + cPow1_sPow1*cyawPow1_syawPow1*wvPow2*(-2*pow(cu, 2) + 2*pow(su, 2)) + cPow1_sPow1*cyawPow1_syawPow1*(-2*pow(a, 2)*pow(cu, 2) + 2*pow(a, 2)*pow(su, 2)) + vPow1_cPow1_sPow1*cyawPow1_syawPow1*wvPow1*(-2*pow(cu, 2) + 2*pow(su, 2)) + vPow1_cPow1_sPow1*cyawPow1_syawPow1*(-2*a*pow(cu, 2) + 2*a*pow(su, 2)) + cyawPow1_syawPow1*vPow1_cPow1_sPow1*wvPow1*(-2*pow(cu, 2) + 2*pow(su, 2)) + cyawPow1_syawPow1*vPow1_cPow1_sPow1*(-2*a*pow(cu, 2) + 2*a*pow(su, 2)) + cyawPow1_syawPow1*vPow2_cPow1_sPow1*(-2*pow(cu, 2) + 2*pow(su, 2));
    const double next_vPow2_sPow2 = pow(a, 2)*pow(cu, 2)*cPow2*syawPow2 + pow(a, 2)*pow(cu, 2)*cyawPow2*sPow2 + 2*pow(a, 2)*cu*su*cPow1_sPow1*cyawPow2 - 2*pow(a, 2)*cu*su*cPow1_sPow1*syawPow2 + 2*pow(a, 2)*cu*su*cPow2*cyawPow1_syawPow1 - 2*pow(a, 2)*cu*su*cyawPow1_syawPow1*sPow2 + pow(a, 2)*pow(su, 2)*cPow2*cyawPow2 + pow(a, 2)*pow(su, 2)*sPow2*syawPow2 + 2*a*pow(cu, 2)*vPow1_cPow2*syawPow2 + 2*a*pow(cu, 2)*cPow2*syawPow2*wvPow1 + 2*a*pow(cu, 2)*cyawPow2*vPow1_sPow2 + 2*a*pow(cu, 2)*cyawPow2*sPow2*wvPow1 + 4*a*cu*su*cPow1_sPow1*cyawPow2*wvPow1 - 4*a*cu*su*cPow1_sPow1*syawPow2*wvPow1 + 4*a*cu*su*vPow1_cPow2*cyawPow1_syawPow1 + 2*a*cu*su*vPow1_cPow1_sPow1*cyawPow2 - 2*a*cu*su*vPow1_cPow1_sPow1*syawPow2 + 4*a*cu*su*cPow2*cyawPow1_syawPow1*wvPow1 - 4*a*cu*su*cyawPow1_syawPow1*vPow1_sPow2 - 4*a*cu*su*cyawPow1_syawPow1*sPow2*wvPow1 + 2*a*cu*su*cyawPow2*vPow1_cPow1_sPow1 - 2*a*cu*su*vPow1_cPow1_sPow1*syawPow2 + 2*a*pow(su, 2)*vPow1_cPow2*cyawPow2 + 2*a*pow(su, 2)*cPow2*cyawPow2*wvPow1 + 2*a*pow(su, 2)*vPow1_sPow2*syawPow2 + 2*a*pow(su, 2)*sPow2*syawPow2*wvPow1 + 2*pow(cu, 2)*vPow1_cPow2*syawPow2*wvPow1 + pow(cu, 2)*cPow2*syawPow2*wvPow2 + 2*pow(cu, 2)*cyawPow2*vPow1_sPow2*wvPow1 + pow(cu, 2)*cyawPow2*sPow2*wvPow2 + pow(cu, 2)*cyawPow2*vPow2_sPow2 + pow(cu, 2)*syawPow2*vPow2_cPow2 + 2*cu*su*cPow1_sPow1*cyawPow2*wvPow2 - 2*cu*su*cPow1_sPow1*syawPow2*wvPow2 + 4*cu*su*vPow1_cPow2*cyawPow1_syawPow1*wvPow1 + 2*cu*su*vPow1_cPow1_sPow1*cyawPow2*wvPow1 - 2*cu*su*vPow1_cPow1_sPow1*syawPow2*wvPow1 + 2*cu*su*cPow2*cyawPow1_syawPow1*wvPow2 - 4*cu*su*cyawPow1_syawPow1*vPow1_sPow2*wvPow1 - 2*cu*su*cyawPow1_syawPow1*sPow2*wvPow2 + 2*cu*su*cyawPow1_syawPow1*vPow2_cPow2 - 2*cu*su*cyawPow1_syawPow1*vPow2_sPow2 + 2*cu*su*cyawPow2*vPow1_cPow1_sPow1*wvPow1 + 2*cu*su*cyawPow2*vPow2_cPow1_sPow1 - 2*cu*su*vPow1_cPow1_sPow1*syawPow2*wvPow1 - 2*cu*su*syawPow2*vPow2_cPow1_sPow1 + 2*pow(su, 2)*vPow1_cPow2*cyawPow2*wvPow1 + pow(su, 2)*cPow2*cyawPow2*wvPow2 + pow(su, 2)*cyawPow2*vPow2_cPow2 + 2*pow(su, 2)*vPow1_sPow2*syawPow2*wvPow1 + pow(su, 2)*sPow2*syawPow2*wvPow2 + pow(su, 2)*syawPow2*vPow2_sPow2 + cPow1_sPow1*cyawPow1_syawPow1*wvPow1*(4*a*pow(cu, 2) - 4*a*pow(su, 2)) + cPow1_sPow1*cyawPow1_syawPow1*wvPow2*(2*pow(cu, 2) - 2*pow(su, 2)) + cPow1_sPow1*cyawPow1_syawPow1*(2*pow(a, 2)*pow(cu, 2) - 2*pow(a, 2)*pow(su, 2)) + vPow1_cPow1_sPow1*cyawPow1_syawPow1*wvPow1*(2*pow(cu, 2) - 2*pow(su, 2)) + vPow1_cPow1_sPow1*cyawPow1_syawPow1*(2*a*pow(cu, 2) - 2*a*pow(su, 2)) + cyawPow1_syawPow1*vPow1_cPow1_sPow1*wvPow1*(2*pow(cu, 2) - 2*pow(su, 2)) + cyawPow1_syawPow1*vPow1_cPow1_sPow1*(2*a*pow(cu, 2) - 2*a*pow(su, 2)) + cyawPow1_syawPow1*vPow2_cPow1_sPow1*(2*pow(cu, 2) - 2*pow(su, 2));
    const double next_vPow1_cPow1_xPow1 = a*cu*dt*vPow1_cPow2*cyawPow1 - a*cu*dt*vPow1_cPow1_sPow1*syawPow1 + a*cu*cPow1_xPow1*cyawPow1 - a*cu*sPow1_xPow1*syawPow1 - a*dt*su*vPow1_cPow2*syawPow1 - a*dt*su*cyawPow1*vPow1_cPow1_sPow1 - a*su*cPow1_xPow1*syawPow1 - a*su*cyawPow1*sPow1_xPow1 + cu*dt*vPow1_cPow2*cyawPow1*wvPow1 + cu*dt*cyawPow1*vPow2_cPow2 - cu*dt*vPow1_cPow1_sPow1*syawPow1*wvPow1 - cu*dt*syawPow1*vPow2_cPow1_sPow1 + cu*cPow1_xPow1*cyawPow1*wvPow1 + cu*cyawPow1*vPow1_cPow1_xPow1 - cu*sPow1_xPow1*syawPow1*wvPow1 - cu*syawPow1*vPow1_sPow1_xPow1 - dt*su*vPow1_cPow2*syawPow1*wvPow1 - dt*su*cyawPow1*vPow1_cPow1_sPow1*wvPow1 - dt*su*cyawPow1*vPow2_cPow1_sPow1 - dt*su*syawPow1*vPow2_cPow2 - su*cPow1_xPow1*syawPow1*wvPow1 - su*cyawPow1*sPow1_xPow1*wvPow1 - su*cyawPow1*vPow1_sPow1_xPow1 - su*syawPow1*vPow1_cPow1_xPow1;
    const double next_cPow1_xPow1 = cu*dt*vPow1_cPow2*cyawPow1 - cu*dt*vPow1_cPow1_sPow1*syawPow1 + cu*cPow1_xPow1*cyawPow1 - cu*sPow1_xPow1*syawPow1 - dt*su*vPow1_cPow2*syawPow1 - dt*su*cyawPow1*vPow1_cPow1_sPow1 - su*cPow1_xPow1*syawPow1 - su*cyawPow1*sPow1_xPow1;
    const double next_sPow1_xPow1 = cu*dt*vPow1_cPow2*syawPow1 + cu*dt*cyawPow1*vPow1_cPow1_sPow1 + cu*cPow1_xPow1*syawPow1 + cu*cyawPow1*sPow1_xPow1 + dt*su*vPow1_cPow2*cyawPow1 - dt*su*vPow1_cPow1_sPow1*syawPow1 + su*cPow1_xPow1*cyawPow1 - su*sPow1_xPow1*syawPow1;
    const double next_vPow1_sPow1_xPow1 = a*cu*dt*vPow1_cPow2*syawPow1 + a*cu*dt*cyawPow1*vPow1_cPow1_sPow1 + a*cu*cPow1_xPow1*syawPow1 + a*cu*cyawPow1*sPow1_xPow1 + a*dt*su*vPow1_cPow2*cyawPow1 - a*dt*su*vPow1_cPow1_sPow1*syawPow1 + a*su*cPow1_xPow1*cyawPow1 - a*su*sPow1_xPow1*syawPow1 + cu*dt*vPow1_cPow2*syawPow1*wvPow1 + cu*dt*cyawPow1*vPow1_cPow1_sPow1*wvPow1 + cu*dt*cyawPow1*vPow2_cPow1_sPow1 + cu*dt*syawPow1*vPow2_cPow2 + cu*cPow1_xPow1*syawPow1*wvPow1 + cu*cyawPow1*sPow1_xPow1*wvPow1 + cu*cyawPow1*vPow1_sPow1_xPow1 + cu*syawPow1*vPow1_cPow1_xPow1 + dt*su*vPow1_cPow2*cyawPow1*wvPow1 + dt*su*cyawPow1*vPow2_cPow2 - dt*su*vPow1_cPow1_sPow1*syawPow1*wvPow1 - dt*su*syawPow1*vPow2_cPow1_sPow1 + su*cPow1_xPow1*cyawPow1*wvPow1 + su*cyawPow1*vPow1_cPow1_xPow1 - su*sPow1_xPow1*syawPow1*wvPow1 - su*syawPow1*vPow1_sPow1_xPow1;
    const double next_vPow1_sPow1_yPow1 = a*cu*dt*vPow1_cPow1_sPow1*syawPow1 + a*cu*dt*cyawPow1*vPow1_sPow2 + a*cu*cPow1_yPow1*syawPow1 + a*cu*cyawPow1*sPow1_yPow1 + a*dt*su*vPow1_cPow1_sPow1*cyawPow1 - a*dt*su*vPow1_sPow2*syawPow1 + a*su*cPow1_yPow1*cyawPow1 - a*su*sPow1_yPow1*syawPow1 + cu*dt*vPow1_cPow1_sPow1*syawPow1*wvPow1 + cu*dt*cyawPow1*vPow1_sPow2*wvPow1 + cu*dt*cyawPow1*vPow2_sPow2 + cu*dt*syawPow1*vPow2_cPow1_sPow1 + cu*cPow1_yPow1*syawPow1*wvPow1 + cu*cyawPow1*sPow1_yPow1*wvPow1 + cu*cyawPow1*vPow1_sPow1_yPow1 + cu*syawPow1*vPow1_cPow1_yPow1 + dt*su*vPow1_cPow1_sPow1*cyawPow1*wvPow1 + dt*su*cyawPow1*vPow2_cPow1_sPow1 - dt*su*vPow1_sPow2*syawPow1*wvPow1 - dt*su*syawPow1*vPow2_sPow2 + su*cPow1_yPow1*cyawPow1*wvPow1 + su*cyawPow1*vPow1_cPow1_yPow1 - su*sPow1_yPow1*syawPow1*wvPow1 - su*syawPow1*vPow1_sPow1_yPow1;
    const double next_cPow1_yPow1 = cu*dt*vPow1_cPow1_sPow1*cyawPow1 - cu*dt*vPow1_sPow2*syawPow1 + cu*cPow1_yPow1*cyawPow1 - cu*sPow1_yPow1*syawPow1 - dt*su*vPow1_cPow1_sPow1*syawPow1 - dt*su*cyawPow1*vPow1_sPow2 - su*cPow1_yPow1*syawPow1 - su*cyawPow1*sPow1_yPow1;
    const double next_sPow1_yPow1 = cu*dt*vPow1_cPow1_sPow1*syawPow1 + cu*dt*cyawPow1*vPow1_sPow2 + cu*cPow1_yPow1*syawPow1 + cu*cyawPow1*sPow1_yPow1 + dt*su*vPow1_cPow1_sPow1*cyawPow1 - dt*su*vPow1_sPow2*syawPow1 + su*cPow1_yPow1*cyawPow1 - su*sPow1_yPow1*syawPow1;
    const double next_vPow1_cPow1_yPow1 = a*cu*dt*vPow1_cPow1_sPow1*cyawPow1 - a*cu*dt*vPow1_sPow2*syawPow1 + a*cu*cPow1_yPow1*cyawPow1 - a*cu*sPow1_yPow1*syawPow1 - a*dt*su*vPow1_cPow1_sPow1*syawPow1 - a*dt*su*cyawPow1*vPow1_sPow2 - a*su*cPow1_yPow1*syawPow1 - a*su*cyawPow1*sPow1_yPow1 + cu*dt*vPow1_cPow1_sPow1*cyawPow1*wvPow1 + cu*dt*cyawPow1*vPow2_cPow1_sPow1 - cu*dt*vPow1_sPow2*syawPow1*wvPow1 - cu*dt*syawPow1*vPow2_sPow2 + cu*cPow1_yPow1*cyawPow1*wvPow1 + cu*cyawPow1*vPow1_cPow1_yPow1 - cu*sPow1_yPow1*syawPow1*wvPow1 - cu*syawPow1*vPow1_sPow1_yPow1 - dt*su*vPow1_cPow1_sPow1*syawPow1*wvPow1 - dt*su*cyawPow1*vPow1_sPow2*wvPow1 - dt*su*cyawPow1*vPow2_sPow2 - dt*su*syawPow1*vPow2_cPow1_sPow1 - su*cPow1_yPow1*syawPow1*wvPow1 - su*cyawPow1*sPow1_yPow1*wvPow1 - su*cyawPow1*vPow1_sPow1_yPow1 - su*syawPow1*vPow1_cPow1_yPow1;
    const double next_xPow1_yPow1 = pow(dt, 2)*vPow2_cPow1_sPow1 + dt*vPow1_cPow1_yPow1 + dt*vPow1_sPow1_xPow1 + xPow1_yPow1;
    const double next_vPow2_cPow1_sPow1 = -4*pow(a, 2)*cu*su*cPow1_sPow1*cyawPow1_syawPow1 + pow(a, 2)*cu*su*cPow2*cyawPow2 - pow(a, 2)*cu*su*cPow2*syawPow2 - pow(a, 2)*cu*su*cyawPow2*sPow2 + pow(a, 2)*cu*su*sPow2*syawPow2 - 8*a*cu*su*cPow1_sPow1*cyawPow1_syawPow1*wvPow1 + 2*a*cu*su*vPow1_cPow2*cyawPow2 - 2*a*cu*su*vPow1_cPow2*syawPow2 - 4*a*cu*su*vPow1_cPow1_sPow1*cyawPow1_syawPow1 + 2*a*cu*su*cPow2*cyawPow2*wvPow1 - 2*a*cu*su*cPow2*syawPow2*wvPow1 - 4*a*cu*su*cyawPow1_syawPow1*vPow1_cPow1_sPow1 - 2*a*cu*su*cyawPow2*vPow1_sPow2 - 2*a*cu*su*cyawPow2*sPow2*wvPow1 + 2*a*cu*su*vPow1_sPow2*syawPow2 + 2*a*cu*su*sPow2*syawPow2*wvPow1 - 4*cu*su*cPow1_sPow1*cyawPow1_syawPow1*wvPow2 + 2*cu*su*vPow1_cPow2*cyawPow2*wvPow1 - 2*cu*su*vPow1_cPow2*syawPow2*wvPow1 - 4*cu*su*vPow1_cPow1_sPow1*cyawPow1_syawPow1*wvPow1 + cu*su*cPow2*cyawPow2*wvPow2 - cu*su*cPow2*syawPow2*wvPow2 - 4*cu*su*cyawPow1_syawPow1*vPow1_cPow1_sPow1*wvPow1 - 4*cu*su*cyawPow1_syawPow1*vPow2_cPow1_sPow1 - 2*cu*su*cyawPow2*vPow1_sPow2*wvPow1 - cu*su*cyawPow2*sPow2*wvPow2 + cu*su*cyawPow2*vPow2_cPow2 - cu*su*cyawPow2*vPow2_sPow2 + 2*cu*su*vPow1_sPow2*syawPow2*wvPow1 + cu*su*sPow2*syawPow2*wvPow2 - cu*su*syawPow2*vPow2_cPow2 + cu*su*syawPow2*vPow2_sPow2 + cPow1_sPow1*cyawPow2*wvPow1*(2*a*pow(cu, 2) - 2*a*pow(su, 2)) + cPow1_sPow1*cyawPow2*wvPow2*(pow(cu, 2) - pow(su, 2)) + cPow1_sPow1*cyawPow2*(pow(a, 2)*pow(cu, 2) - pow(a, 2)*pow(su, 2)) + cPow1_sPow1*syawPow2*wvPow1*(-2*a*pow(cu, 2) + 2*a*pow(su, 2)) + cPow1_sPow1*syawPow2*wvPow2*(-pow(cu, 2) + pow(su, 2)) + cPow1_sPow1*syawPow2*(-pow(a, 2)*pow(cu, 2) + pow(a, 2)*pow(su, 2)) + vPow1_cPow2*cyawPow1_syawPow1*wvPow1*(2*pow(cu, 2) - 2*pow(su, 2)) + vPow1_cPow2*cyawPow1_syawPow1*(2*a*pow(cu, 2) - 2*a*pow(su, 2)) + vPow1_cPow1_sPow1*cyawPow2*wvPow1*(pow(cu, 2) - pow(su, 2)) + vPow1_cPow1_sPow1*cyawPow2*(a*pow(cu, 2) - a*pow(su, 2)) + vPow1_cPow1_sPow1*syawPow2*wvPow1*(-pow(cu, 2) + pow(su, 2)) + vPow1_cPow1_sPow1*syawPow2*(-a*pow(cu, 2) + a*pow(su, 2)) + cPow2*cyawPow1_syawPow1*wvPow1*(2*a*pow(cu, 2) - 2*a*pow(su, 2)) + cPow2*cyawPow1_syawPow1*wvPow2*(pow(cu, 2) - pow(su, 2)) + cPow2*cyawPow1_syawPow1*(pow(a, 2)*pow(cu, 2) - pow(a, 2)*pow(su, 2)) + cyawPow1_syawPow1*vPow1_sPow2*wvPow1*(-2*pow(cu, 2) + 2*pow(su, 2)) + cyawPow1_syawPow1*vPow1_sPow2*(-2*a*pow(cu, 2) + 2*a*pow(su, 2)) + cyawPow1_syawPow1*sPow2*wvPow1*(-2*a*pow(cu, 2) + 2*a*pow(su, 2)) + cyawPow1_syawPow1*sPow2*wvPow2*(-pow(cu, 2) + pow(su, 2)) + cyawPow1_syawPow1*sPow2*(-pow(a, 2)*pow(cu, 2) + pow(a, 2)*pow(su, 2)) + cyawPow1_syawPow1*vPow2_cPow2*(pow(cu, 2) - pow(su, 2)) + cyawPow1_syawPow1*vPow2_sPow2*(-pow(cu, 2) + pow(su, 2)) + cyawPow2*vPow1_cPow1_sPow1*wvPow1*(pow(cu, 2) - pow(su, 2)) + cyawPow2*vPow1_cPow1_sPow1*(a*pow(cu, 2) - a*pow(su, 2)) + cyawPow2*vPow2_cPow1_sPow1*(pow(cu, 2) - pow(su, 2)) + vPow1_cPow1_sPow1*syawPow2*wvPow1*(-pow(cu, 2) + pow(su, 2)) + vPow1_cPow1_sPow1*syawPow2*(-a*pow(cu, 2) + a*pow(su, 2)) + syawPow2*vPow2_cPow1_sPow1*(-pow(cu, 2) + pow(su, 2));
    const double next_vPow1_xPow1 = a*dt*vPow1_cPow1 + a*xPow1 + dt*vPow2_cPow1 + dt*vPow1_cPow1*wvPow1 + vPow1_xPow1 + wvPow1*xPow1;
    const double next_vPow2_cPow1 = pow(a, 2)*cu*cPow1*cyawPow1 - pow(a, 2)*cu*sPow1*syawPow1 - pow(a, 2)*su*cPow1*syawPow1 - pow(a, 2)*su*cyawPow1*sPow1 + 2*a*cu*cPow1*cyawPow1*wvPow1 + a*cu*vPow1_cPow1*cyawPow1 + a*cu*cyawPow1*vPow1_cPow1 - 2*a*cu*sPow1*syawPow1*wvPow1 - a*cu*vPow1_sPow1*syawPow1 - a*cu*syawPow1*vPow1_sPow1 - 2*a*su*cPow1*syawPow1*wvPow1 - a*su*vPow1_cPow1*syawPow1 - 2*a*su*cyawPow1*sPow1*wvPow1 - a*su*cyawPow1*vPow1_sPow1 - a*su*cyawPow1*vPow1_sPow1 - a*su*syawPow1*vPow1_cPow1 + cu*cPow1*cyawPow1*wvPow2 + cu*vPow1_cPow1*cyawPow1*wvPow1 + cu*cyawPow1*vPow2_cPow1 + cu*cyawPow1*vPow1_cPow1*wvPow1 - cu*sPow1*syawPow1*wvPow2 - cu*vPow1_sPow1*syawPow1*wvPow1 - cu*syawPow1*vPow2_sPow1 - cu*syawPow1*vPow1_sPow1*wvPow1 - su*cPow1*syawPow1*wvPow2 - su*vPow1_cPow1*syawPow1*wvPow1 - su*cyawPow1*sPow1*wvPow2 - su*cyawPow1*vPow1_sPow1*wvPow1 - su*cyawPow1*vPow2_sPow1 - su*cyawPow1*vPow1_sPow1*wvPow1 - su*syawPow1*vPow2_cPow1 - su*syawPow1*vPow1_cPow1*wvPow1;
    const double next_xPow1_yawPow1 = dt*u*vPow1_cPow1 + dt*vPow1_cPow1*wyawPow1 + dt*vPow1_cPow1_yawPow1 + u*xPow1 + wyawPow1*xPow1 + xPow1_yawPow1;
    const double next_vPow1_cPow1_yawPow1 = a*cu*u*cPow1*cyawPow1 - a*cu*u*sPow1*syawPow1 + a*cu*cPow1*cyawPow1*wyawPow1 + a*cu*cPow1_yawPow1*cyawPow1 - a*cu*sPow1*syawPow1*wyawPow1 - a*cu*sPow1_yawPow1*syawPow1 - a*su*u*cPow1*syawPow1 - a*su*u*cyawPow1*sPow1 - a*su*cPow1*syawPow1*wyawPow1 - a*su*cPow1_yawPow1*syawPow1 - a*su*cyawPow1*sPow1*wyawPow1 - a*su*cyawPow1*sPow1_yawPow1 + cu*u*cPow1*cyawPow1*wvPow1 + cu*u*cyawPow1*vPow1_cPow1 - cu*u*sPow1*syawPow1*wvPow1 - cu*u*syawPow1*vPow1_sPow1 + cu*cPow1*cyawPow1*wvPow1*wyawPow1 + cu*cPow1_yawPow1*cyawPow1*wvPow1 + cu*cyawPow1*vPow1_cPow1*wyawPow1 + cu*cyawPow1*vPow1_cPow1_yawPow1 - cu*sPow1*syawPow1*wvPow1*wyawPow1 - cu*sPow1_yawPow1*syawPow1*wvPow1 - cu*syawPow1*vPow1_sPow1*wyawPow1 - cu*syawPow1*vPow1_sPow1_yawPow1 - su*u*cPow1*syawPow1*wvPow1 - su*u*cyawPow1*sPow1*wvPow1 - su*u*cyawPow1*vPow1_sPow1 - su*u*syawPow1*vPow1_cPow1 - su*cPow1*syawPow1*wvPow1*wyawPow1 - su*cPow1_yawPow1*syawPow1*wvPow1 - su*cyawPow1*sPow1*wvPow1*wyawPow1 - su*cyawPow1*sPow1_yawPow1*wvPow1 - su*cyawPow1*vPow1_sPow1*wyawPow1 - su*cyawPow1*vPow1_sPow1_yawPow1 - su*syawPow1*vPow1_cPow1*wyawPow1 - su*syawPow1*vPow1_cPow1_yawPow1;
    const double next_vPow1_yPow1 = a*dt*vPow1_sPow1 + a*yPow1 + dt*vPow2_sPow1 + dt*vPow1_sPow1*wvPow1 + vPow1_yPow1 + wvPow1*yPow1;
    const double next_vPow2_sPow1 = pow(a, 2)*cu*cPow1*syawPow1 + pow(a, 2)*cu*cyawPow1*sPow1 + pow(a, 2)*su*cPow1*cyawPow1 - pow(a, 2)*su*sPow1*syawPow1 + 2*a*cu*cPow1*syawPow1*wvPow1 + a*cu*vPow1_cPow1*syawPow1 + 2*a*cu*cyawPow1*sPow1*wvPow1 + a*cu*cyawPow1*vPow1_sPow1 + a*cu*cyawPow1*vPow1_sPow1 + a*cu*syawPow1*vPow1_cPow1 + 2*a*su*cPow1*cyawPow1*wvPow1 + a*su*vPow1_cPow1*cyawPow1 + a*su*cyawPow1*vPow1_cPow1 - 2*a*su*sPow1*syawPow1*wvPow1 - a*su*vPow1_sPow1*syawPow1 - a*su*syawPow1*vPow1_sPow1 + cu*cPow1*syawPow1*wvPow2 + cu*vPow1_cPow1*syawPow1*wvPow1 + cu*cyawPow1*sPow1*wvPow2 + cu*cyawPow1*vPow1_sPow1*wvPow1 + cu*cyawPow1*vPow2_sPow1 + cu*cyawPow1*vPow1_sPow1*wvPow1 + cu*syawPow1*vPow2_cPow1 + cu*syawPow1*vPow1_cPow1*wvPow1 + su*cPow1*cyawPow1*wvPow2 + su*vPow1_cPow1*cyawPow1*wvPow1 + su*cyawPow1*vPow2_cPow1 + su*cyawPow1*vPow1_cPow1*wvPow1 - su*sPow1*syawPow1*wvPow2 - su*vPow1_sPow1*syawPow1*wvPow1 - su*syawPow1*vPow2_sPow1 - su*syawPow1*vPow1_sPow1*wvPow1;
    const double next_yPow1_yawPow1 = dt*u*vPow1_sPow1 + dt*vPow1_sPow1*wyawPow1 + dt*vPow1_sPow1_yawPow1 + u*yPow1 + wyawPow1*yPow1 + yPow1_yawPow1;
    const double next_vPow1_sPow1_yawPow1 = a*cu*u*cPow1*syawPow1 + a*cu*u*cyawPow1*sPow1 + a*cu*cPow1*syawPow1*wyawPow1 + a*cu*cPow1_yawPow1*syawPow1 + a*cu*cyawPow1*sPow1*wyawPow1 + a*cu*cyawPow1*sPow1_yawPow1 + a*su*u*cPow1*cyawPow1 - a*su*u*sPow1*syawPow1 + a*su*cPow1*cyawPow1*wyawPow1 + a*su*cPow1_yawPow1*cyawPow1 - a*su*sPow1*syawPow1*wyawPow1 - a*su*sPow1_yawPow1*syawPow1 + cu*u*cPow1*syawPow1*wvPow1 + cu*u*cyawPow1*sPow1*wvPow1 + cu*u*cyawPow1*vPow1_sPow1 + cu*u*syawPow1*vPow1_cPow1 + cu*cPow1*syawPow1*wvPow1*wyawPow1 + cu*cPow1_yawPow1*syawPow1*wvPow1 + cu*cyawPow1*sPow1*wvPow1*wyawPow1 + cu*cyawPow1*sPow1_yawPow1*wvPow1 + cu*cyawPow1*vPow1_sPow1*wyawPow1 + cu*cyawPow1*vPow1_sPow1_yawPow1 + cu*syawPow1*vPow1_cPow1*wyawPow1 + cu*syawPow1*vPow1_cPow1_yawPow1 + su*u*cPow1*cyawPow1*wvPow1 + su*u*cyawPow1*vPow1_cPow1 - su*u*sPow1*syawPow1*wvPow1 - su*u*syawPow1*vPow1_sPow1 + su*cPow1*cyawPow1*wvPow1*wyawPow1 + su*cPow1_yawPow1*cyawPow1*wvPow1 + su*cyawPow1*vPow1_cPow1*wyawPow1 + su*cyawPow1*vPow1_cPow1_yawPow1 - su*sPow1*syawPow1*wvPow1*wyawPow1 - su*sPow1_yawPow1*syawPow1*wvPow1 - su*syawPow1*vPow1_sPow1*wyawPow1 - su*syawPow1*vPow1_sPow1_yawPow1;
    const double next_vPow1_yawPow1 = 1.0*a*u + a*wyawPow1 + a*yawPow1 + u*vPow1 + u*wvPow1 + vPow1*wyawPow1 + vPow1_yawPow1 + wvPow1*wyawPow1 + wvPow1*yawPow1;
    const double next_cPow1_sPow1 = -4*cu*su*cPow1_sPow1*cyawPow1_syawPow1 + cu*su*cPow2*cyawPow2 - cu*su*cPow2*syawPow2 - cu*su*cyawPow2*sPow2 + cu*su*sPow2*syawPow2 + cPow1_sPow1*cyawPow2*(pow(cu, 2) - pow(su, 2)) + cPow1_sPow1*syawPow2*(-pow(cu, 2) + pow(su, 2)) + cPow2*cyawPow1_syawPow1*(pow(cu, 2) - pow(su, 2)) + cyawPow1_syawPow1*sPow2*(-pow(cu, 2) + pow(su, 2));
    const double next_vPow1_cPow2 = a*pow(cu, 2)*cPow2*cyawPow2 + a*pow(cu, 2)*sPow2*syawPow2 - 2*a*cu*su*cPow1_sPow1*cyawPow2 + 2*a*cu*su*cPow1_sPow1*syawPow2 - 2*a*cu*su*cPow2*cyawPow1_syawPow1 + 2*a*cu*su*cyawPow1_syawPow1*sPow2 + a*pow(su, 2)*cPow2*syawPow2 + a*pow(su, 2)*cyawPow2*sPow2 + pow(cu, 2)*vPow1_cPow2*cyawPow2 + pow(cu, 2)*cPow2*cyawPow2*wvPow1 + pow(cu, 2)*vPow1_sPow2*syawPow2 + pow(cu, 2)*sPow2*syawPow2*wvPow1 - 2*cu*su*cPow1_sPow1*cyawPow2*wvPow1 + 2*cu*su*cPow1_sPow1*syawPow2*wvPow1 - 2*cu*su*vPow1_cPow2*cyawPow1_syawPow1 - cu*su*vPow1_cPow1_sPow1*cyawPow2 + cu*su*vPow1_cPow1_sPow1*syawPow2 - 2*cu*su*cPow2*cyawPow1_syawPow1*wvPow1 + 2*cu*su*cyawPow1_syawPow1*vPow1_sPow2 + 2*cu*su*cyawPow1_syawPow1*sPow2*wvPow1 - cu*su*cyawPow2*vPow1_cPow1_sPow1 + cu*su*vPow1_cPow1_sPow1*syawPow2 + pow(su, 2)*vPow1_cPow2*syawPow2 + pow(su, 2)*cPow2*syawPow2*wvPow1 + pow(su, 2)*cyawPow2*vPow1_sPow2 + pow(su, 2)*cyawPow2*sPow2*wvPow1 + cPow1_sPow1*cyawPow1_syawPow1*wvPow1*(-2*pow(cu, 2) + 2*pow(su, 2)) + cPow1_sPow1*cyawPow1_syawPow1*(-2*a*pow(cu, 2) + 2*a*pow(su, 2)) + vPow1_cPow1_sPow1*cyawPow1_syawPow1*(-pow(cu, 2) + pow(su, 2)) + cyawPow1_syawPow1*vPow1_cPow1_sPow1*(-pow(cu, 2) + pow(su, 2));
    const double next_vPow1_cPow1_sPow1 = -4*a*cu*su*cPow1_sPow1*cyawPow1_syawPow1 + a*cu*su*cPow2*cyawPow2 - a*cu*su*cPow2*syawPow2 - a*cu*su*cyawPow2*sPow2 + a*cu*su*sPow2*syawPow2 - pow(cu, 2)*vPow1_cPow1_sPow1*syawPow2 + pow(cu, 2)*cyawPow2*vPow1_cPow1_sPow1 - 4*cu*su*cPow1_sPow1*cyawPow1_syawPow1*wvPow1 + cu*su*vPow1_cPow2*cyawPow2 - cu*su*vPow1_cPow2*syawPow2 - 2*cu*su*vPow1_cPow1_sPow1*cyawPow1_syawPow1 + cu*su*cPow2*cyawPow2*wvPow1 - cu*su*cPow2*syawPow2*wvPow1 - 2*cu*su*cyawPow1_syawPow1*vPow1_cPow1_sPow1 - cu*su*cyawPow2*vPow1_sPow2 - cu*su*cyawPow2*sPow2*wvPow1 + cu*su*vPow1_sPow2*syawPow2 + cu*su*sPow2*syawPow2*wvPow1 - pow(su, 2)*vPow1_cPow1_sPow1*cyawPow2 + pow(su, 2)*vPow1_cPow1_sPow1*syawPow2 + cPow1_sPow1*cyawPow2*wvPow1*(pow(cu, 2) - pow(su, 2)) + cPow1_sPow1*cyawPow2*(a*pow(cu, 2) - a*pow(su, 2)) + cPow1_sPow1*syawPow2*wvPow1*(-pow(cu, 2) + pow(su, 2)) + cPow1_sPow1*syawPow2*(-a*pow(cu, 2) + a*pow(su, 2)) + vPow1_cPow2*cyawPow1_syawPow1*(pow(cu, 2) - pow(su, 2)) + cPow2*cyawPow1_syawPow1*wvPow1*(pow(cu, 2) - pow(su, 2)) + cPow2*cyawPow1_syawPow1*(a*pow(cu, 2) - a*pow(su, 2)) + cyawPow1_syawPow1*vPow1_sPow2*(-pow(cu, 2) + pow(su, 2)) + cyawPow1_syawPow1*sPow2*wvPow1*(-pow(cu, 2) + pow(su, 2)) + cyawPow1_syawPow1*sPow2*(-a*pow(cu, 2) + a*pow(su, 2));
    const double next_vPow1_sPow2 = a*pow(cu, 2)*cPow2*syawPow2 + a*pow(cu, 2)*cyawPow2*sPow2 + 2*a*cu*su*cPow1_sPow1*cyawPow2 - 2*a*cu*su*cPow1_sPow1*syawPow2 + 2*a*cu*su*cPow2*cyawPow1_syawPow1 - 2*a*cu*su*cyawPow1_syawPow1*sPow2 + a*pow(su, 2)*cPow2*cyawPow2 + a*pow(su, 2)*sPow2*syawPow2 + pow(cu, 2)*vPow1_cPow2*syawPow2 + pow(cu, 2)*cPow2*syawPow2*wvPow1 + pow(cu, 2)*cyawPow2*vPow1_sPow2 + pow(cu, 2)*cyawPow2*sPow2*wvPow1 + 2*cu*su*cPow1_sPow1*cyawPow2*wvPow1 - 2*cu*su*cPow1_sPow1*syawPow2*wvPow1 + 2*cu*su*vPow1_cPow2*cyawPow1_syawPow1 + cu*su*vPow1_cPow1_sPow1*cyawPow2 - cu*su*vPow1_cPow1_sPow1*syawPow2 + 2*cu*su*cPow2*cyawPow1_syawPow1*wvPow1 - 2*cu*su*cyawPow1_syawPow1*vPow1_sPow2 - 2*cu*su*cyawPow1_syawPow1*sPow2*wvPow1 + cu*su*cyawPow2*vPow1_cPow1_sPow1 - cu*su*vPow1_cPow1_sPow1*syawPow2 + pow(su, 2)*vPow1_cPow2*cyawPow2 + pow(su, 2)*cPow2*cyawPow2*wvPow1 + pow(su, 2)*vPow1_sPow2*syawPow2 + pow(su, 2)*sPow2*syawPow2*wvPow1 + cPow1_sPow1*cyawPow1_syawPow1*wvPow1*(2*pow(cu, 2) - 2*pow(su, 2)) + cPow1_sPow1*cyawPow1_syawPow1*(2*a*pow(cu, 2) - 2*a*pow(su, 2)) + vPow1_cPow1_sPow1*cyawPow1_syawPow1*(pow(cu, 2) - pow(su, 2)) + cyawPow1_syawPow1*vPow1_cPow1_sPow1*(pow(cu, 2) - pow(su, 2));
    const double next_cPow1_yawPow1 = cu*u*cPow1*cyawPow1 - cu*u*sPow1*syawPow1 + cu*cPow1*cyawPow1*wyawPow1 + cu*cPow1_yawPow1*cyawPow1 - cu*sPow1*syawPow1*wyawPow1 - cu*sPow1_yawPow1*syawPow1 - su*u*cPow1*syawPow1 - su*u*cyawPow1*sPow1 - su*cPow1*syawPow1*wyawPow1 - su*cPow1_yawPow1*syawPow1 - su*cyawPow1*sPow1*wyawPow1 - su*cyawPow1*sPow1_yawPow1;
    const double next_sPow1_yawPow1 = cu*u*cPow1*syawPow1 + cu*u*cyawPow1*sPow1 + cu*cPow1*syawPow1*wyawPow1 + cu*cPow1_yawPow1*syawPow1 + cu*cyawPow1*sPow1*wyawPow1 + cu*cyawPow1*sPow1_yawPow1 + su*u*cPow1*cyawPow1 - su*u*sPow1*syawPow1 + su*cPow1*cyawPow1*wyawPow1 + su*cPow1_yawPow1*cyawPow1 - su*sPow1*syawPow1*wyawPow1 - su*sPow1_yawPow1*syawPow1;

    StateInfo next_state;
    next_state.mean = Eigen::VectorXd::Zero(4);
    next_state.covariance= Eigen::MatrixXd::Zero(4, 4);

    next_state.mean(STATE::IDX::X) = next_xPow1;
    next_state.mean(STATE::IDX::Y) = next_yPow1;
    next_state.mean(STATE::IDX::V) = next_vPow1;
    next_state.mean(STATE::IDX::YAW)= next_yawPow1;

    next_state.covariance(STATE::IDX::X, STATE::IDX::X) = next_xPow2 - next_xPow1 * next_xPow1;
    next_state.covariance(STATE::IDX::Y, STATE::IDX::Y) = next_yPow2 - next_yPow1 * next_yPow1;
    next_state.covariance(STATE::IDX::V, STATE::IDX::V) = next_vPow2 - next_vPow1 * next_vPow1;
    next_state.covariance(STATE::IDX::YAW, STATE::IDX::YAW) = next_yawPow2 - next_yawPow1 * next_yawPow1;

    next_state.covariance(STATE::IDX::X, STATE::IDX::Y) = next_xPow1_yPow1 - next_xPow1 * next_yPow1;
    next_state.covariance(STATE::IDX::X, STATE::IDX::V) = next_vPow1_xPow1 - next_xPow1 * next_vPow1;
    next_state.covariance(STATE::IDX::X, STATE::IDX::YAW) = next_xPow1_yawPow1 - next_xPow1 * next_yawPow1;

    next_state.covariance(STATE::IDX::Y, STATE::IDX::X) = next_state.covariance(STATE::IDX::X, STATE::IDX::Y);
    next_state.covariance(STATE::IDX::Y, STATE::IDX::V) = next_vPow1_yPow1 - next_yPow1 * next_vPow1;
    next_state.covariance(STATE::IDX::Y, STATE::IDX::YAW) = next_yPow1_yawPow1 - next_yPow1 * next_yawPow1;

    next_state.covariance(STATE::IDX::V, STATE::IDX::X) = next_state.covariance(STATE::IDX::X, STATE::IDX::V);
    next_state.covariance(STATE::IDX::V, STATE::IDX::Y) = next_state.covariance(STATE::IDX::Y, STATE::IDX::V);
    next_state.covariance(STATE::IDX::V, STATE::IDX::YAW) = next_vPow1_yawPow1 - next_vPow1 * next_yawPow1;

    next_state.covariance(STATE::IDX::YAW, STATE::IDX::X) = next_state.covariance(STATE::IDX::X, STATE::IDX::YAW);
    next_state.covariance(STATE::IDX::YAW, STATE::IDX::Y) = next_state.covariance(STATE::IDX::Y, STATE::IDX::YAW);
    next_state.covariance(STATE::IDX::YAW, STATE::IDX::V) = next_state.covariance(STATE::IDX::V, STATE::IDX::YAW);

    return next_state;
}

StateInfo MobileRobotModel::getMeasurementMoments(const StateInfo &state_info,
                                                  const std::map<int, std::shared_ptr<BaseDistribution>> &noise_map)
{
    FourDimensionalNormalDistribution dist(state_info.mean, state_info.covariance);
    const double xPow1 = dist.calc_moment(STATE::IDX::X, 1);
    const double yPow1 = dist.calc_moment(STATE::IDX::Y, 1);
    const double vPow1 = dist.calc_moment(STATE::IDX::V, 1);
    const double cyawPow1 = dist.calc_cos_moment(STATE::IDX::YAW, 1);
    const double syawPow1 = dist.calc_sin_moment(STATE::IDX::YAW, 1);

    const double xPow2 = dist.calc_moment(STATE::IDX::X, 2);
    const double yPow2 = dist.calc_moment(STATE::IDX::Y, 2);
    const double vPow2 = dist.calc_moment(STATE::IDX::V, 2);
    const double cyawPow2 = dist.calc_cos_moment(STATE::IDX::YAW, 2);
    const double syawPow2 = dist.calc_sin_moment(STATE::IDX::YAW, 2);
    const double xPow1_cyawPow1 = dist.calc_x_cos_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    const double xPow1_syawPow1 = dist.calc_x_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    const double yPow1_cyawPow1 = dist.calc_x_cos_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    const double yPow1_syawPow1 = dist.calc_x_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    const double vPow1_cyawPow1 = dist.calc_x_cos_z_moment(STATE::IDX::V, STATE::IDX::YAW);
    const double vPow1_syawPow1 = dist.calc_x_sin_z_moment(STATE::IDX::V, STATE::IDX::YAW);
    const double xPow1_yPow1 = dist.calc_cross_second_moment(STATE::IDX::X, STATE::IDX::Y);
    const double xPow1_vPow1 = dist.calc_cross_second_moment(STATE::IDX::X, STATE::IDX::V);
    const double yPow1_vPow1 = dist.calc_cross_second_moment(STATE::IDX::Y, STATE::IDX::V);
    const double cyawPow1_syawPow1 = dist.calc_cos_sin_moment(STATE::IDX::YAW, 1, 1);

    const double vPow2_cyawPow1 = dist.calc_xx_cos_z_moment(STATE::IDX::V, STATE::IDX::YAW);
    const double vPow2_syawPow1 = dist.calc_xx_sin_z_moment(STATE::IDX::V, STATE::IDX::YAW);
    const double vPow1_cyawPow2 = dist.calc_x_cos_y_cos_y_moment(STATE::IDX::V, STATE::IDX::YAW);
    const double vPow1_syawPow2 = dist.calc_x_sin_y_sin_y_moment(STATE::IDX::V, STATE::IDX::YAW);
    const double xPow1_vPow1_syawPow1 = dist.calc_xy_sin_z_moment(STATE::IDX::X, STATE::IDX::V, STATE::IDX::YAW);
    const double xPow1_vPow1_cyawPow1 = dist.calc_xy_cos_z_moment(STATE::IDX::X, STATE::IDX::V, STATE::IDX::YAW);
    const double yPow1_vPow1_syawPow1 = dist.calc_xy_sin_z_moment(STATE::IDX::Y, STATE::IDX::V, STATE::IDX::YAW);
    const double yPow1_vPow1_cyawPow1 = dist.calc_xy_cos_z_moment(STATE::IDX::Y, STATE::IDX::V, STATE::IDX::YAW);
    const double vPow1_cyawPow1_syawPow1 = dist.calc_x_cos_y_sin_y_moment(STATE::IDX::V, STATE::IDX::YAW);

    const double vPow2_cyawPow2 = dist.calc_xx_cos_y_cos_y_moment(STATE::IDX::V, STATE::IDX::YAW);
    const double vPow2_syawPow2 = dist.calc_xx_sin_y_sin_y_moment(STATE::IDX::V, STATE::IDX::YAW);
    const double vPow2_cyawPow1_syawPow1 = dist.calc_xx_cos_y_sin_y_moment(STATE::IDX::V, STATE::IDX::YAW);

    // Observation Noise
    const auto wx_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WX);
    const auto wy_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WY);
    const auto wv_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WVC);
    const auto wyaw_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WYAW);

    const double wxPow1 = wx_dist_ptr->calc_moment(1);
    const double wyPow1 = wy_dist_ptr->calc_moment(1);
    const double wvPow1 = wv_dist_ptr->calc_moment(1);
    const double cwyawPow1 = wyaw_dist_ptr->calc_cos_moment(1);
    const double swyawPow1 = wyaw_dist_ptr->calc_sin_moment(1);
    const double wxPow2 = wx_dist_ptr->calc_moment(2);
    const double wyPow2 = wy_dist_ptr->calc_moment(2);
    const double wvPow2 = wv_dist_ptr->calc_moment(2);
    const double cwyawPow2 = wyaw_dist_ptr->calc_cos_moment(2);
    const double swyawPow2 = wyaw_dist_ptr->calc_sin_moment(2);
    const double cwyawPow1_swyawPow1 = wyaw_dist_ptr->calc_cos_sin_moment(1, 1);


    // Calculate Observation Moments
    const double measurement_xPow1 = xPow1 + vPow1 * wxPow1;
    const double measurement_yPow1 = yPow1 + vPow1 * wyPow1;
    const double measurement_vcPow1 = vPow1_cyawPow1 * cwyawPow1 - vPow1_syawPow1 * swyawPow1
                                 + wvPow1 * cyawPow1 * cwyawPow1 - wvPow1 * syawPow1 * swyawPow1;
    const double measurement_xPow2 = xPow2 + vPow2 * wxPow2 + 2.0 * wxPow1 * xPow1_vPow1;
    const double measurement_yPow2 = yPow2 + vPow2 * wyPow2 + 2.0 * wyPow1 * yPow1_vPow1;
    const double measurement_vcPow2 = vPow2_syawPow2 * swyawPow2
                                - 2 * vPow2_cyawPow1_syawPow1 * cwyawPow1_swyawPow1
                                + vPow2_cyawPow2 * cwyawPow2
                                + 2 * vPow1_syawPow2 * wvPow1 * swyawPow2
                                - 4 * vPow1_cyawPow1_syawPow1 * cwyawPow1_swyawPow1 * wvPow1
                                + 2 * vPow1_cyawPow2 * wvPow1 * cwyawPow2
                                + wvPow2 * syawPow2 * swyawPow2
                                - 2 * wvPow2 * cyawPow1_syawPow1 * cwyawPow1_swyawPow1
                                + wvPow2 * cyawPow2 * cwyawPow2;
    const double measurement_xPow1_yPow1 = vPow2 * wxPow1 * wyPow1 + yPow1_vPow1 * wxPow1 + xPow1_vPow1 * wyPow1 + xPow1_yPow1;
    const double measurement_xPow1_vcPow1 = - vPow2_syawPow1 * swyawPow1 * wxPow1
                                       + vPow2_cyawPow1 * cwyawPow1 * wxPow1
                                       - vPow1_syawPow1 * swyawPow1 * wxPow1 * wvPow1
                                       + vPow1_cyawPow1 * cwyawPow1 * wxPow1 * wvPow1
                                       - xPow1_vPow1_syawPow1 * swyawPow1
                                       + xPow1_vPow1_cyawPow1 * cwyawPow1
                                       - xPow1_syawPow1 * swyawPow1 * wvPow1
                                       + xPow1_cyawPow1 * cwyawPow1 * wvPow1;
    const double measurement_yPow1_vcPow1 = -vPow2_syawPow1 * swyawPow1 * wyPow1 +
                                        vPow2_cyawPow1 * cwyawPow1 * wyPow1 -
                                        vPow1_syawPow1 * swyawPow1 * wyPow1 * wvPow1 +
                                        vPow1_cyawPow1 * cwyawPow1 * wyPow1 * wvPow1 -
                                        yPow1_vPow1_syawPow1 * swyawPow1 +
                                        yPow1_vPow1_cyawPow1 * cwyawPow1 -
                                        yPow1_syawPow1 * swyawPow1 * wvPow1 +
                                        yPow1_cyawPow1 * cwyawPow1 * wvPow1;

    StateInfo measurement_state;
    measurement_state.mean = Eigen::VectorXd::Zero(3);
    measurement_state.covariance = Eigen::MatrixXd::Zero(3, 3);

    measurement_state.mean(OBSERVATION::IDX::X) = measurement_xPow1;
    measurement_state.mean(OBSERVATION::IDX::Y) = measurement_yPow1;
    measurement_state.mean(OBSERVATION::IDX::VC) = measurement_vcPow1;

    measurement_state.covariance(OBSERVATION::IDX::X, OBSERVATION::IDX::X) =
            measurement_xPow2 - measurement_xPow1 * measurement_xPow1;
    measurement_state.covariance(OBSERVATION::IDX::Y, OBSERVATION::IDX::Y) =
            measurement_yPow2 - measurement_yPow1 * measurement_yPow1;
    measurement_state.covariance(OBSERVATION::IDX::VC, OBSERVATION::IDX::VC) =
            measurement_vcPow2 - measurement_vcPow1 * measurement_vcPow1;

    measurement_state.covariance(OBSERVATION::IDX::X, OBSERVATION::IDX::Y) =
            measurement_xPow1_yPow1 - measurement_xPow1 * measurement_yPow1;
    measurement_state.covariance(OBSERVATION::IDX::X, OBSERVATION::IDX::VC) =
            measurement_xPow1_vcPow1 - measurement_xPow1 * measurement_vcPow1;
    measurement_state.covariance(OBSERVATION::IDX::Y, OBSERVATION::IDX::VC) =
            measurement_yPow1_vcPow1 - measurement_yPow1 * measurement_vcPow1;

    measurement_state.covariance(OBSERVATION::IDX::Y, OBSERVATION::IDX::X) = measurement_state.covariance(OBSERVATION::IDX::X, OBSERVATION::IDX::Y);
    measurement_state.covariance(OBSERVATION::IDX::VC, OBSERVATION::IDX::X) = measurement_state.covariance(OBSERVATION::IDX::X, OBSERVATION::IDX::VC);
    measurement_state.covariance(OBSERVATION::IDX::VC, OBSERVATION::IDX::Y) = measurement_state.covariance(OBSERVATION::IDX::Y, OBSERVATION::IDX::VC);

    return measurement_state;
}

Eigen::MatrixXd MobileRobotModel::getStateMeasurementMatrix(const StateInfo& state_info,
                                                            const StateInfo& measurement_info,
                                                            const std::map<int, std::shared_ptr<BaseDistribution>> &noise_map)
{
    FourDimensionalNormalDistribution dist(state_info.mean, state_info.covariance);
    const auto predicted_mean = state_info.mean;
    const auto measurement_mean = measurement_info.mean;

    const auto wx_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WX);
    const auto wy_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WY);
    const auto wv_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WVC);
    const auto wyaw_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WYAW);
    const double wxPow1 = wx_dist_ptr->calc_moment(1);
    const double wyPow1 = wy_dist_ptr->calc_moment(1);
    const double wvPow1 = wv_dist_ptr->calc_moment(1);
    const double cwyawPow1 = wyaw_dist_ptr->calc_cos_moment(1);
    const double swyawPow1 = wyaw_dist_ptr->calc_sin_moment(1);

    Eigen::MatrixXd state_observation_cov(4, 3); // sigma = E[XY^T] - E[X]E[Y]^T

    state_observation_cov(STATE::IDX::X, OBSERVATION::IDX::X)
            = dist.calc_moment(STATE::IDX::X, 2) + dist.calc_cross_second_moment(STATE::IDX::X, STATE::IDX::V) * wxPow1
              - predicted_mean(STATE::IDX::X) * measurement_mean(OBSERVATION::IDX::X); // xp * (xp + vp * wx)
    state_observation_cov(STATE::IDX::X, OBSERVATION::IDX::Y)
            = dist.calc_cross_second_moment(STATE::IDX::X, STATE::IDX::Y)
              + dist.calc_cross_second_moment(STATE::IDX::X, STATE::IDX::V) * wyPow1 +
              - predicted_mean(STATE::IDX::X) * measurement_mean(OBSERVATION::IDX::Y); // xp * (yp + vp * wy)
    state_observation_cov(STATE::IDX::X, OBSERVATION::IDX::VC)
            = -dist.calc_xy_sin_z_moment(STATE::IDX::X, STATE::IDX::V, STATE::IDX::YAW) * swyawPow1
              +dist.calc_xy_cos_z_moment(STATE::IDX::X, STATE::IDX::V, STATE::IDX::YAW) * cwyawPow1
              -dist.calc_x_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW) * wvPow1 * swyawPow1
              +dist.calc_x_cos_z_moment(STATE::IDX::X, STATE::IDX::YAW) * wvPow1 * cwyawPow1
              - predicted_mean(STATE::IDX::X) * measurement_mean(OBSERVATION::IDX::VC); // xp * ((vp + wv)*cos(theta + wtheta))

    state_observation_cov(STATE::IDX::Y, OBSERVATION::IDX::X)
            = dist.calc_cross_second_moment(STATE::IDX::X, STATE::IDX::Y)
              + dist.calc_cross_second_moment(STATE::IDX::Y, STATE::IDX::V) * wxPow1
              - predicted_mean(STATE::IDX::Y) * measurement_mean(OBSERVATION::IDX::X); // yp * (xp + vp * wx)
    state_observation_cov(STATE::IDX::Y, OBSERVATION::IDX::Y)
            = dist.calc_moment(STATE::IDX::Y, 2)
              + dist.calc_cross_second_moment(STATE::IDX::Y, STATE::IDX::V) * wyPow1
              - predicted_mean(STATE::IDX::Y) * measurement_mean(OBSERVATION::IDX::Y); // yp * (yp + vp * wy)
    state_observation_cov(STATE::IDX::Y, OBSERVATION::IDX::VC)
            = -dist.calc_xy_sin_z_moment(STATE::IDX::Y, STATE::IDX::V, STATE::IDX::YAW) * swyawPow1
              +dist.calc_xy_cos_z_moment(STATE::IDX::Y, STATE::IDX::V, STATE::IDX::YAW) * cwyawPow1
              -dist.calc_x_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW) * wvPow1 * swyawPow1
              +dist.calc_x_cos_z_moment(STATE::IDX::Y, STATE::IDX::YAW) * wvPow1 * cwyawPow1
              - predicted_mean(STATE::IDX::Y) * measurement_mean(OBSERVATION::IDX::VC); // yp * ((vp + wv)*cos(theta + wtheta))


    state_observation_cov(STATE::IDX::V, OBSERVATION::IDX::X)
            = dist.calc_moment(STATE::IDX::V, 2) * wxPow1 + dist.calc_cross_second_moment(STATE::IDX::X, STATE::IDX::V)
              - predicted_mean(STATE::IDX::V) * measurement_mean(OBSERVATION::IDX::X);
    state_observation_cov(STATE::IDX::V, OBSERVATION::IDX::Y)
            = dist.calc_moment(STATE::IDX::V, 2) * wyPow1 + dist.calc_cross_second_moment(STATE::IDX::Y, STATE::IDX::V)
              - predicted_mean(STATE::IDX::V) * measurement_mean(OBSERVATION::IDX::Y);
    state_observation_cov(STATE::IDX::V, OBSERVATION::IDX::VC)
            = - dist.calc_xx_sin_z_moment(STATE::IDX::V, STATE::IDX::YAW) * swyawPow1
              + dist.calc_xx_cos_z_moment(STATE::IDX::V, STATE::IDX::YAW) * cwyawPow1
              - dist.calc_x_sin_z_moment(STATE::IDX::V, STATE::IDX::YAW) * wvPow1 * swyawPow1
              + dist.calc_x_cos_z_moment(STATE::IDX::V, STATE::IDX::YAW) * wvPow1 * cwyawPow1
              - predicted_mean(STATE::IDX::V) * measurement_mean(OBSERVATION::IDX::VC);

    state_observation_cov(STATE::IDX::YAW, OBSERVATION::IDX::X)
            = dist.calc_cross_second_moment(STATE::IDX::V, STATE::IDX::YAW) * wxPow1 +
              dist.calc_cross_second_moment(STATE::IDX::X, STATE::IDX::YAW) +
              - predicted_mean(STATE::IDX::YAW) * measurement_mean(OBSERVATION::IDX::X); // yaw_p * (x_p + wx*v_p)
    state_observation_cov(STATE::IDX::YAW, OBSERVATION::IDX::Y)
            = dist.calc_cross_second_moment(STATE::IDX::V, STATE::IDX::YAW) * wyPow1 +
              dist.calc_cross_second_moment(STATE::IDX::Y, STATE::IDX::YAW) +
              - predicted_mean(STATE::IDX::YAW) * measurement_mean(OBSERVATION::IDX::Y); // yaw_p * (x_p + wx*v_p)
    state_observation_cov(STATE::IDX::YAW, OBSERVATION::IDX::VC)
            = -dist.calc_xy_sin_y_moment(STATE::IDX::V, STATE::IDX::YAW) * swyawPow1
              + dist.calc_xy_cos_y_moment(STATE::IDX::V, STATE::IDX::YAW) * cwyawPow1
              - dist.calc_x_sin_x_moment(STATE::IDX::YAW, 1, 1) * wvPow1 * swyawPow1
              + dist.calc_x_cos_x_moment(STATE::IDX::YAW, 1, 1) * wvPow1 * cwyawPow1
              - predicted_mean(STATE::IDX::YAW) * measurement_mean(OBSERVATION::IDX::VC);

    return state_observation_cov;
}
