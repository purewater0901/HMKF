#include "model/example_model.h"
#include "distribution/two_dimensional_normal_distribution.h"

using namespace Example;

Eigen::VectorXd ExampleVehicleModel::propagate(const Eigen::VectorXd& x_curr,
                                                 const Eigen::VectorXd& u_curr,
                                                 const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map,
                                                 const double dt)
{
    /*  == Nonlinear model ==
    *
    * x_{k+1}   = x_k + v_k * cos(yaw_k) * dt + wx
    * yaw_{k+1} = yaw_k + u_k * dt + wyaw
    *
    */

    const auto wx_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WX);
    const auto wyaw_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WYAW);

    Eigen::VectorXd system_noise = Eigen::VectorXd::Zero(2);
    system_noise(0) = wx_dist_ptr->calc_mean();
    system_noise(1) = wyaw_dist_ptr->calc_mean();

    return propagate(x_curr, u_curr, system_noise, dt);
}

Eigen::VectorXd ExampleVehicleModel::propagate(const Eigen::VectorXd& x_curr,
                                               const Eigen::VectorXd& u_curr,
                                               const Eigen::VectorXd& system_noise,
                                               const double dt)
{
    const double yaw_k = x_curr(STATE::IDX::YAW);
    const double& v = u_curr(INPUT::IDX::V);
    const double& wx = system_noise(SYSTEM_NOISE::IDX::WX);
    const double& wyaw = system_noise(SYSTEM_NOISE::IDX::WYAW);

    Eigen::VectorXd x_next = Eigen::VectorXd::Zero(2);
    x_next(STATE::IDX::X) = x_curr(STATE::IDX::X) + v * std::cos(yaw_k) + wx;
    x_next(STATE::IDX::YAW) = yaw_k + u_curr(INPUT::IDX::U) + wyaw;

    return x_next;
}

Eigen::VectorXd ExampleVehicleModel::measure(const Eigen::VectorXd& x_curr,
                                               const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    // Covariance Update
    /*  == Nonlinear model ==
     *
     * r = x^2 + mr
     * yaw_k = yaw_k * mywa
     *
     */

    const auto mr_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WR);
    const auto myaw_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WYAW);

    Eigen::VectorXd measurement_noise = Eigen::VectorXd::Zero(2);
    measurement_noise(MEASUREMENT_NOISE::IDX::WR) = mr_dist_ptr->calc_mean();
    measurement_noise(MEASUREMENT_NOISE::IDX::WYAW) = myaw_dist_ptr->calc_mean();

    return measure(x_curr, measurement_noise);
}

Eigen::VectorXd ExampleVehicleModel::measure(const Eigen::VectorXd& x_curr, const Eigen::VectorXd& measurement_noise)
{

    Eigen::VectorXd y = Eigen::VectorXd::Zero(2);
    y(MEASUREMENT::IDX::R) = x_curr(STATE::IDX::X) * x_curr(STATE::IDX::X) + measurement_noise(MEASUREMENT_NOISE::IDX::WR);
    y(MEASUREMENT::IDX::YAW) = x_curr(STATE::IDX::YAW) + measurement_noise(MEASUREMENT_NOISE::IDX::WYAW);

    return y;
}

Eigen::MatrixXd ExampleVehicleModel::getStateMatrix(const Eigen::VectorXd& x_curr,
                                                    const Eigen::VectorXd& u_curr,
                                                    const double dt)
{
    const double& v = u_curr(INPUT::IDX::V);
    const double& yaw_k = x_curr(STATE::IDX::YAW);
    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(2, 2);
    A(STATE::IDX::X, STATE::IDX::YAW) =  -v * std::sin(yaw_k);

    return A;
}

Eigen::MatrixXd ExampleVehicleModel::getProcessNoiseMatrix(const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    const auto wx_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WX);
    const auto wyaw_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WYAW);

    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(2, 2);
    Q(SYSTEM_NOISE::IDX::WX, SYSTEM_NOISE::IDX::WX) = wx_dist_ptr->calc_variance();
    Q(SYSTEM_NOISE::IDX::WYAW, SYSTEM_NOISE::IDX::WYAW) = wyaw_dist_ptr->calc_variance();
    return Q;
}

Eigen::MatrixXd ExampleVehicleModel::getMeasurementMatrix(const Eigen::VectorXd& x_curr,
                                                          const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    // Covariance Update
    /*  == Nonlinear model ==
     *
     * r = x^2 + mr(dr/dx = 2*x, dr/dyaw = 0.0)
     * yaw_k = yaw_k + myaw(dyaw/dyaw = 1.0)
     *
     */

    const double& yaw_k = x_curr(STATE::IDX::YAW);

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 2);
    H(MEASUREMENT::IDX::R, STATE::IDX::X) = 2.0 * x_curr(STATE::IDX::X);
    H(MEASUREMENT::IDX::YAW, STATE::IDX::YAW) = 1.0;

    return H;
}

Eigen::MatrixXd ExampleVehicleModel::getMeasurementNoiseMatrix(const Eigen::VectorXd& x_curr,
                                                                 const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    const auto mr_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WR);
    const auto myaw_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WYAW);

    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(2, 2);
    R(MEASUREMENT_NOISE::IDX::WR, MEASUREMENT_NOISE::IDX::WR) = mr_dist_ptr->calc_variance();
    R(MEASUREMENT_NOISE::IDX::WYAW, MEASUREMENT_NOISE::IDX::WYAW) = myaw_dist_ptr->calc_variance();

    return R;
}

StateInfo ExampleVehicleModel::propagateStateMoments(const StateInfo &state_info,
                                                       const Eigen::VectorXd &control_inputs,
                                                       const double dt,
                                                       const std::map<int, std::shared_ptr<BaseDistribution>> &noise_map)
{
    // Step1. Approximate to Gaussian Distribution
    const auto state_mean = state_info.mean;
    const auto state_cov = state_info.covariance;
    TwoDimensionalNormalDistribution dist(state_info.mean, state_info.covariance);

    // Step2. State Moment
    const double xPow1 = dist.calc_moment(STATE::IDX::X, 1);
    const double yawPow1 = dist.calc_moment(STATE::IDX::YAW, 1);
    const double cPow1 = dist.calc_xy_cos_y_sin_y_moment(0, 0, 1, 0); // cos(yaw)
    const double xPow2 = dist.calc_moment(STATE::IDX::X, 2); // x^2
    const double yawPow2 = dist.calc_moment(STATE::IDX::YAW, 2); // yaw^2
    const double cPow2 = dist.calc_xy_cos_y_sin_y_moment(0, 0, 2, 0); // cos(yaw)^2
    const double xPow1_yawPow1 = dist.calc_xy_cos_y_sin_y_moment(1, 1, 0, 0); // xyaw
    const double cPow1_xPow1 = dist.calc_xy_cos_y_sin_y_moment(1, 0, 1, 0); // x*cos(yaw)
    const double cPow1_yawPow1 = dist.calc_xy_cos_y_sin_y_moment(0, 1, 1, 0); // yaw*cos(yaw)

    // Input
    const double v = control_inputs(INPUT::V);
    const double u = control_inputs(INPUT::U);

    // System noise
    const auto wx_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WX);
    const auto wyaw_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WYAW);
    const double wxPow1 = wx_dist_ptr->calc_moment(1);
    const double wyawPow1 = wyaw_dist_ptr->calc_moment(1);
    const double wxPow2 = wx_dist_ptr->calc_moment(2);
    const double wyawPow2 = wyaw_dist_ptr->calc_moment(2);

    // moment propagation
    const double next_xPow1 = xPow1 + v*cPow1 + wxPow1;
    const double next_yawPow1 = yawPow1 + u + wyawPow1;
    const double next_xPow2 = xPow2 + v*v*cPow2 + wxPow2 + 2*v*cPow1_xPow1 + 2*xPow1*wxPow1 + 2*v*wxPow1*cPow1;
    const double next_yawPow2 = yawPow2 + u*u + wyawPow2 + 2*u*yawPow1 + 2*yawPow1*wyawPow1 + 2*u*wyawPow1;
    const double next_xPow1_yawPow1 = xPow1_yawPow1 + u*xPow1 + wyawPow1*xPow1
                                 + v*cPow1_yawPow1 + v*u*cPow1 + v*wyawPow1*cPow1
                                 + wxPow1*yawPow1 + wxPow1*u + wxPow1*wyawPow1;

    StateInfo next_state_info;
    next_state_info.mean = Eigen::VectorXd::Zero(2);
    next_state_info.covariance = Eigen::MatrixXd::Zero(2, 2);

    next_state_info.mean(STATE::IDX::X) = next_xPow1;
    next_state_info.mean(STATE::IDX::YAW)= next_yawPow1;
    next_state_info.covariance(STATE::IDX::X, STATE::IDX::X) = next_xPow2 - next_xPow1 * next_xPow1;
    next_state_info.covariance(STATE::IDX::YAW, STATE::IDX::YAW) = next_yawPow2 - next_yawPow1 * next_yawPow1;
    next_state_info.covariance(STATE::IDX::X, STATE::IDX::YAW) = next_xPow1_yawPow1 - next_xPow1 * next_yawPow1;
    next_state_info.covariance(STATE::IDX::YAW, STATE::IDX::X) = next_state_info.covariance(STATE::IDX::X, STATE::IDX::YAW);

    return next_state_info;
}

StateInfo ExampleVehicleModel::getMeasurementMoments(const StateInfo &state_info,
                                                     const std::map<int, std::shared_ptr<BaseDistribution>> &noise_map)
{
    TwoDimensionalNormalDistribution dist(state_info.mean, state_info.covariance);

    const auto wr_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WR);
    const auto wyaw_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WYAW);

    // Measurement noise
    const double wrPow1 = wr_dist_ptr->calc_moment(1);
    const double wyawPow1 = wyaw_dist_ptr->calc_moment(1);
    const double wrPow2 = wr_dist_ptr->calc_moment(2);
    const double wyawPow2 = wyaw_dist_ptr->calc_moment(2);

    // predicted moments
    const double& xPow1 = dist.calc_xy_cos_y_sin_y_moment(1, 0, 0, 0);
    const double& yawPow1 = dist.calc_xy_cos_y_sin_y_moment(0, 1, 0, 0);
    const double& xPow2 = dist.calc_xy_cos_y_sin_y_moment(2, 0, 0, 0);
    const double& yawPow2 = dist.calc_xy_cos_y_sin_y_moment(0, 2, 0, 0);
    const double& xPow1_yawPow1 = dist.calc_xy_cos_y_sin_y_moment(1, 1, 0, 0);
    const double& xPow3 = dist.calc_xy_cos_y_sin_y_moment(3, 0, 0, 0);
    const double& xPow2_yawPow1 = dist.calc_xy_cos_y_sin_y_moment(2, 1, 0, 0);
    const double& xPow4 = dist.calc_xy_cos_y_sin_y_moment(4, 0, 0, 0);

    const double measurement_rPow1 = xPow2 + wrPow1;
    const double measurement_yawPow1 = yawPow1 + wyawPow1;
    const double measurement_rPow2 = xPow4 + 2.0*xPow2*wrPow1 + wrPow2;
    const double measurement_yawPow2 = yawPow2 + +2.0*yawPow1*wyawPow1+ wyawPow2;
    const double measurement_rPow1_yawPow1 = xPow2_yawPow1 + xPow2*wyawPow1 + wrPow1*yawPow1 + wrPow1*wyawPow1;

    StateInfo measurement_state;
    measurement_state.mean = Eigen::VectorXd::Zero(2);
    measurement_state.covariance = Eigen::MatrixXd::Zero(2, 2);

    measurement_state.mean(MEASUREMENT::IDX::R) = measurement_rPow1;
    measurement_state.mean(MEASUREMENT::IDX::YAW) = measurement_yawPow1;

    measurement_state.covariance(MEASUREMENT::IDX::R, MEASUREMENT::IDX::R) =
            measurement_rPow2 - measurement_rPow1 * measurement_rPow1;
    measurement_state.covariance(MEASUREMENT::IDX::YAW, MEASUREMENT::IDX::YAW) =
            measurement_yawPow2 - measurement_yawPow1 * measurement_yawPow1;

    measurement_state.covariance(MEASUREMENT::IDX::R, MEASUREMENT::IDX::YAW) =
            measurement_rPow1_yawPow1 - measurement_rPow1 * measurement_yawPow1;

    measurement_state.covariance(MEASUREMENT::IDX::YAW, MEASUREMENT::IDX::R) =
            measurement_state.covariance(MEASUREMENT::IDX::R, MEASUREMENT::IDX::YAW);

    return measurement_state;
}

Eigen::MatrixXd ExampleVehicleModel::getStateMeasurementMatrix(const StateInfo& state_info, const StateInfo& measurement_info,
                                                                 const std::map<int, std::shared_ptr<BaseDistribution>> &noise_map)
{
    TwoDimensionalNormalDistribution dist(state_info.mean, state_info.covariance);

    const auto wr_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WR);
    const auto wyaw_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WYAW);

    // predicted moments
    const double& xPow1 = dist.calc_xy_cos_y_sin_y_moment(1, 0, 0, 0);
    const double& yawPow1 = dist.calc_xy_cos_y_sin_y_moment(0, 1, 0, 0);
    const double& xPow2 = dist.calc_xy_cos_y_sin_y_moment(2, 0, 0, 0);
    const double& yawPow2 = dist.calc_xy_cos_y_sin_y_moment(0, 2, 0, 0);
    const double& xPow1_yawPow1 = dist.calc_xy_cos_y_sin_y_moment(1, 1, 0, 0);
    const double& xPow3 = dist.calc_xy_cos_y_sin_y_moment(3, 0, 0, 0);
    const double& xPow2_yawPow1 = dist.calc_xy_cos_y_sin_y_moment(2, 1, 0, 0);
    const double& xPow4 = dist.calc_xy_cos_y_sin_y_moment(4, 0, 0, 0);

    // Measurement noise
    const double wrPow1 = wr_dist_ptr->calc_moment(1);
    const double wyawPow1 = wyaw_dist_ptr->calc_moment(1);

    // measurement moments
    const double& mrPow1 = measurement_info.mean(MEASUREMENT::IDX::R);
    const double& myawPow1 = measurement_info.mean(MEASUREMENT::IDX::YAW);

    // x*(x*x + w_r) = x**3 + x*w_r
    Eigen::MatrixXd state_observation_cov = Eigen::MatrixXd::Zero(2, 2); // sigma = E[XY^T] - E[X]E[Y]^T
    state_observation_cov(STATE::IDX::X, MEASUREMENT::IDX::R) = xPow3 + xPow1*wrPow1 - xPow1*mrPow1;
    state_observation_cov(STATE::IDX::X, MEASUREMENT::IDX::YAW) = xPow1_yawPow1 + xPow1*wyawPow1 - xPow1*myawPow1;
    state_observation_cov(STATE::IDX::YAW, MEASUREMENT::IDX::R) = xPow2_yawPow1 + yawPow1*wrPow1 - yawPow1*mrPow1;
    state_observation_cov(STATE::IDX::YAW, MEASUREMENT::IDX::YAW) = yawPow2 + yawPow1*wyawPow1 - yawPow1*myawPow1;

    return state_observation_cov;
}
