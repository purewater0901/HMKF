#include "model/squared_example_model.h"

#include "distribution/two_dimensional_normal_distribution.h"

using namespace SquaredExample;

Eigen::VectorXd ExampleSquaredVehicleModel::propagate(const Eigen::VectorXd& x_curr,
                                               const Eigen::VectorXd& u_curr,
                                               const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map,
                                               const double dt)
{
    /*  == Nonlinear model ==
    *
    * x_{k+1} = x_k + (v_k + w_v) * cos(w_yaw) * dt
    * y_{k+1} = y_k + (v_k + w_v) * sin(w_yaw) * dt
    *
    */

    const auto wv_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WV);
    const auto wyaw_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WYAW);

    Eigen::VectorXd system_noise = Eigen::VectorXd::Zero(2);
    system_noise(0) = wv_dist_ptr->calc_mean();
    system_noise(1) = wyaw_dist_ptr->calc_mean();

    return propagate(x_curr, u_curr, system_noise, dt);
}

Eigen::VectorXd ExampleSquaredVehicleModel::propagate(const Eigen::VectorXd& x_curr,
                                               const Eigen::VectorXd& u_curr,
                                               const Eigen::VectorXd& system_noise,
                                               const double dt)
{
    const double& v = u_curr(INPUT::IDX::V);
    const double& wv = system_noise(SYSTEM_NOISE::IDX::WV);
    const double& wyaw = system_noise(SYSTEM_NOISE::IDX::WYAW);

    Eigen::VectorXd x_next = Eigen::VectorXd::Zero(2);
    x_next(STATE::IDX::X) = x_curr(STATE::IDX::X) + (v + wv) * std::cos(wyaw)*dt;
    x_next(STATE::IDX::Y) = x_curr(STATE::IDX::X) + (v + wv) * std::sin(wyaw)*dt;

    return x_next;
}

Eigen::VectorXd ExampleSquaredVehicleModel::measure(const Eigen::VectorXd& x_curr,
                                             const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    // Covariance Update
    /*  == Nonlinear model ==
     *
     * r = x^4 + y^4 + mr
     *
     */

    const auto mr_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WR);

    Eigen::VectorXd measurement_noise = Eigen::VectorXd::Zero(1);
    measurement_noise(MEASUREMENT_NOISE::IDX::WR) = mr_dist_ptr->calc_mean();

    return measure(x_curr, measurement_noise);
}

Eigen::VectorXd ExampleSquaredVehicleModel::measure(const Eigen::VectorXd& x_curr, const Eigen::VectorXd& measurement_noise)
{

    Eigen::VectorXd y = Eigen::VectorXd::Zero(1);
    y(MEASUREMENT::IDX::R) = std::pow(x_curr(STATE::IDX::X), 4) + std::pow(x_curr(STATE::IDX::Y), 4)
                             + measurement_noise(MEASUREMENT_NOISE::IDX::WR);

    return y;
}

Eigen::MatrixXd ExampleSquaredVehicleModel::getStateMatrix(const Eigen::VectorXd& x_curr,
                                                    const Eigen::VectorXd& u_curr,
                                                    const double dt)
{
    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(2, 2);
    return A;
}

Eigen::MatrixXd ExampleSquaredVehicleModel::getProcessNoiseMatrix(const Eigen::VectorXd& x_curr,
                                                           const Eigen::VectorXd& u_curr,
                                                           const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map,
                                                           const double dt)
{
    const auto wv_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WV);
    const auto wyaw_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WYAW);

    const double mean_wv = wv_dist_ptr->calc_mean();
    const double mean_wyaw = wyaw_dist_ptr->calc_mean();

    const double v = u_curr(INPUT::IDX::V);

    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(2, 2);
    Q(SYSTEM_NOISE::IDX::WV, SYSTEM_NOISE::IDX::WV) = wv_dist_ptr->calc_variance();
    Q(SYSTEM_NOISE::IDX::WYAW, SYSTEM_NOISE::IDX::WYAW) = wyaw_dist_ptr->calc_variance();
    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(2, 2);
    L(STATE::IDX::X, SYSTEM_NOISE::IDX::WV) = std::cos(mean_wyaw) * dt;
    L(STATE::IDX::Y, SYSTEM_NOISE::IDX::WV) = std::sin(mean_wyaw) * dt;
    L(STATE::IDX::X, SYSTEM_NOISE::IDX::WYAW) = -(v+mean_wv)*std::cos(mean_wyaw)*dt;
    L(STATE::IDX::Y, SYSTEM_NOISE::IDX::WYAW) =  (v+mean_wv)*std::sin(mean_wyaw)*dt;

    return L*Q*L.transpose();
}

Eigen::MatrixXd ExampleSquaredVehicleModel::getMeasurementMatrix(const Eigen::VectorXd& x_curr,
                                                                 const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    // Covariance Update
    /*  == Nonlinear model ==
     *
     * r = x^4 + y^4 + mr(dr/dx = 4*x^3, dr/dy = 4*y^3)
     *
     */

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(1, 2);
    H(MEASUREMENT::IDX::R, STATE::IDX::X) = 4.0 * std::pow(x_curr(STATE::IDX::X), 3);
    H(MEASUREMENT::IDX::R, STATE::IDX::Y) = 4.0 * std::pow(x_curr(STATE::IDX::Y), 3);

    return H;
}

Eigen::MatrixXd ExampleSquaredVehicleModel::getMeasurementNoiseMatrix(const Eigen::VectorXd& x_curr,
                                                               const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    const auto mr_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WR);

    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(1, 1);
    R(MEASUREMENT_NOISE::IDX::WR, MEASUREMENT_NOISE::IDX::WR) = mr_dist_ptr->calc_variance();

    return R;
}

StateInfo ExampleSquaredVehicleModel::propagateStateMoments(const StateInfo &state_info,
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
    const double yPow1 = dist.calc_moment(STATE::IDX::Y, 1);
    const double xPow2 = dist.calc_moment(STATE::IDX::X, 2); // x^2
    const double yPow2 = dist.calc_moment(STATE::IDX::Y, 2); // y^2
    const double xPow1_yPow1 = dist.calc_xy_cos_y_sin_y_moment(1, 1, 0, 0); // xy

    // Input
    const double v = control_inputs(INPUT::V);

    // System noise
    const auto wv_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WV);
    const auto wyaw_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WYAW);
    const double wvPow1 = wv_dist_ptr->calc_moment(1);
    const double wvPow2 = wv_dist_ptr->calc_moment(2);
    const double cwyawPow1 = wyaw_dist_ptr->calc_cos_moment(1);
    const double swyawPow1 = wyaw_dist_ptr->calc_sin_moment(1);
    const double cwyawPow2 = wyaw_dist_ptr->calc_cos_moment(2);
    const double swyawPow2 = wyaw_dist_ptr->calc_sin_moment(2);
    const double cwyawPow1_swyawPow1 = wyaw_dist_ptr->calc_cos_sin_moment(1, 1);

    // moment propagation
    const double next_xPow1 = xPow1 + (v+wvPow1)*cwyawPow1*dt;
    const double next_yPow1 = yPow1 + (v+wvPow1)*swyawPow1*dt;
    const double next_xPow2 = xPow2 + (v*v + wvPow2 + 2*v*wvPow1) * cwyawPow2 * dt * dt
                              + 2.0 * xPow1 * (v+wvPow1) * cwyawPow1 * dt;
    const double next_yPow2 = yPow2 + (v*v + wvPow2 + 2*v*wvPow1) * swyawPow2 * dt * dt
                              + 2.0 * yPow1 * (v+wvPow1) * swyawPow1 * dt;
    const double next_xPow1_yPow1 = xPow1_yPow1 + xPow1*(v+wvPow1)*swyawPow1*dt + yPow1*(v+wvPow1)*cwyawPow1*dt
                                    + (v*v+wvPow2+2*v*wvPow1)*cwyawPow1_swyawPow1*dt*dt;

    StateInfo next_state_info;
    next_state_info.mean = Eigen::VectorXd::Zero(2);
    next_state_info.covariance = Eigen::MatrixXd::Zero(2, 2);

    next_state_info.mean(STATE::IDX::X) = next_xPow1;
    next_state_info.mean(STATE::IDX::Y) = next_yPow1;
    next_state_info.covariance(STATE::IDX::X, STATE::IDX::X) = next_xPow2 - next_xPow1 * next_xPow1;
    next_state_info.covariance(STATE::IDX::Y, STATE::IDX::Y) = next_yPow2 - next_yPow1 * next_yPow1;
    next_state_info.covariance(STATE::IDX::X, STATE::IDX::Y) = next_xPow1_yPow1 - next_xPow1 * next_yPow1;
    next_state_info.covariance(STATE::IDX::Y, STATE::IDX::X) = next_state_info.covariance(STATE::IDX::X, STATE::IDX::Y);

    return next_state_info;
}

StateInfo ExampleSquaredVehicleModel::getMeasurementMoments(const StateInfo &state_info,
                                                     const std::map<int, std::shared_ptr<BaseDistribution>> &noise_map)
{
    TwoDimensionalNormalDistribution dist(state_info.mean, state_info.covariance);

    const auto wr_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WR);

    // Measurement noise
    const double wrPow1 = wr_dist_ptr->calc_moment(1);
    const double wrPow2 = wr_dist_ptr->calc_moment(2);

    // predicted moments
    const double xPow4 = dist.calc_xy_cos_y_sin_y_moment(4, 0, 0, 0);
    const double yPow4 = dist.calc_xy_cos_y_sin_y_moment(0, 4, 0, 0);
    const double xPow8 = dist.calc_xy_cos_y_sin_y_moment(8, 0, 0, 0);
    const double yPow8 = dist.calc_xy_cos_y_sin_y_moment(0, 8, 0, 0);
    const double xPow4_yPow4 = dist.calc_xy_cos_y_sin_y_moment(4, 4, 0, 0);

    const double measurement_rPow1 = xPow4 + yPow4 + wrPow1;
    const double measurement_rPow2 = xPow8 + yPow8 + wrPow2 + 2.0*xPow4_yPow4 + 2.0*xPow4*wrPow1 + 2.0*yPow4*wrPow1;

    StateInfo measurement_state;
    measurement_state.mean = Eigen::VectorXd::Zero(1);
    measurement_state.covariance = Eigen::MatrixXd::Zero(1, 1);

    measurement_state.mean(MEASUREMENT::IDX::R) = measurement_rPow1;

    measurement_state.covariance(MEASUREMENT::IDX::R, MEASUREMENT::IDX::R) =
            measurement_rPow2 - measurement_rPow1 * measurement_rPow1;

    return measurement_state;
}

Eigen::MatrixXd ExampleSquaredVehicleModel::getStateMeasurementMatrix(const StateInfo& state_info, const StateInfo& measurement_info,
                                                               const std::map<int, std::shared_ptr<BaseDistribution>> &noise_map)
{
    TwoDimensionalNormalDistribution dist(state_info.mean, state_info.covariance);

    const auto wr_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WR);

    // predicted moments
    const double& xPow1 = dist.calc_xy_cos_y_sin_y_moment(1, 0, 0, 0);
    const double& yPow1 = dist.calc_xy_cos_y_sin_y_moment(0, 1, 0, 0);
    const double& xPow5 = dist.calc_xy_cos_y_sin_y_moment(5, 0, 0, 0);
    const double& yPow5 = dist.calc_xy_cos_y_sin_y_moment(0, 5, 0, 0);
    const double& xPow1_yPow4 = dist.calc_xy_cos_y_sin_y_moment(1, 4, 0, 0);
    const double& xPow4_yPow1 = dist.calc_xy_cos_y_sin_y_moment(4, 1, 0, 0);

    // Measurement noise
    const double wrPow1 = wr_dist_ptr->calc_moment(1);

    // measurement moments
    const double& mrPow1 = measurement_info.mean(MEASUREMENT::IDX::R);

    // x*(x^4 + y^4 + w_r) = x^5 + xy^4 + x*w_r
    // y*(x^4 + y^4 + w_r) = x^4y + y^5 + y*w_r
    Eigen::MatrixXd state_observation_cov = Eigen::MatrixXd::Zero(2, 1); // sigma = E[XY^T] - E[X]E[Y]^T
    state_observation_cov(STATE::IDX::X, MEASUREMENT::IDX::R) = xPow5 + xPow1_yPow4 + xPow1*wrPow1 - xPow1*mrPow1;
    state_observation_cov(STATE::IDX::Y, MEASUREMENT::IDX::R) = xPow4_yPow1 + yPow5 + yPow1*wrPow1 - yPow1*mrPow1;

    return state_observation_cov;
}