#include "model/simple_vehicle_model.h"
#include "distribution/three_dimensional_normal_distribution.h"

using namespace SimpleVehicle;

Eigen::VectorXd SimpleVehicleModel::propagate(const Eigen::VectorXd& x_curr,
                                              const Eigen::VectorXd& u_curr,
                                              const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map,
                                              const double dt)
{
    const auto wv_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WV);
    const auto wu_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WU);

    Eigen::VectorXd system_noise = Eigen::VectorXd::Zero(2);
    system_noise(SYSTEM_NOISE::IDX::WV) = wv_dist_ptr->calc_mean();
    system_noise(SYSTEM_NOISE::IDX::WU) = wu_dist_ptr->calc_mean();

    return propagate(x_curr, u_curr, system_noise, dt);
}

Eigen::VectorXd SimpleVehicleModel::propagate(const Eigen::VectorXd &x_curr,
                                              const Eigen::VectorXd &u_curr,
                                              const Eigen::VectorXd &system_noise,
                                              const double dt)
{
    /*
     * dynamics model
     * x(k+1) = x(k) + (v+w_v)*cos(theta)*dt
     * y(k+1) = y(k) + (v+w_v)*sin(theta)*dt
     * theta(k+1) = theta(k) + (u+wu)*dt
     */

    Eigen::Vector3d x_next;
    x_next(STATE::IDX::X) = x_curr(STATE::IDX::X) +
            (u_curr(INPUT::IDX::V) + system_noise(SYSTEM_NOISE::IDX::WV)) * std::cos(x_curr(STATE::IDX::YAW));
    x_next(STATE::IDX::Y) = x_curr(STATE::IDX::Y) +
            (u_curr(INPUT::IDX::V) + system_noise(SYSTEM_NOISE::IDX::WV)) * std::sin(x_curr(STATE::IDX::YAW));
    x_next(STATE::IDX::YAW) = x_curr(STATE::IDX::YAW)
            + (u_curr(INPUT::IDX::U) + system_noise(SYSTEM_NOISE::IDX::WU));

    return x_next;
}

// measurement model
Eigen::VectorXd SimpleVehicleModel::measure(const Eigen::VectorXd& x_curr,
                                            const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    throw std::runtime_error("[SimpleVehicleModel]: Undefined Function. Need landmark in the arugment");
}

Eigen::VectorXd SimpleVehicleModel::measure(const Eigen::VectorXd& x_curr, const Eigen::VectorXd& observation_noise)
{
throw std::runtime_error("[SimpleVehicleModel]: Undefined Function. Need landmark in the arugment");
}

Eigen::VectorXd SimpleVehicleModel::measureWithLandmark(const Eigen::VectorXd& x_curr,
                                                        const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map,
                                                        const Eigen::Vector2d& landmark)
{
    const auto mr_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WR);
    const auto ma_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WA);

    Eigen::VectorXd meas_noise = Eigen::VectorXd::Zero(2);
    meas_noise(MEASUREMENT_NOISE::IDX::WR) = mr_dist_ptr->calc_mean();
    meas_noise(MEASUREMENT_NOISE::IDX::WA) = ma_dist_ptr->calc_mean();

    return measureWithLandmark(x_curr, meas_noise, landmark);
}

Eigen::VectorXd SimpleVehicleModel::measureWithLandmark(const Eigen::VectorXd& x_curr,
                                                        const Eigen::VectorXd& observation_noise,
                                                        const Eigen::Vector2d& landmark)
{
    const double& x = x_curr(STATE::IDX::X);
    const double& y = x_curr(STATE::IDX::Y);
    const double& yaw = x_curr(STATE::IDX::YAW);
    const double& x_land = landmark(0);
    const double& y_land = landmark(1);
    const double& wr = observation_noise(MEASUREMENT_NOISE::IDX::WR); // length noise
    const double& wa = observation_noise(MEASUREMENT_NOISE::IDX::WA); // bearing noise
    const double ha = (x_land - x) * std::cos(yaw) + (y_land - y) * std::sin(yaw);
    const double hb = (y_land - y) * std::cos(yaw) - (x_land - x) * std::sin(yaw);

    Eigen::VectorXd y_next = Eigen::VectorXd::Zero(2);
    y_next(MEASUREMENT::IDX::RCOS) =  wr * std::cos(wa) * ha - wr * std::sin(wa) * hb;
    y_next(MEASUREMENT::IDX::RSIN) =  wr * std::cos(wa) * hb + wr * std::sin(wa) * ha;

    return y_next;
}

Eigen::MatrixXd SimpleVehicleModel::getStateMatrix(const Eigen::VectorXd& x_curr,
                                                   const Eigen::VectorXd& u_curr,
                                                   const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map,
                                                   const double dt)
{
    /*  == Nonlinear model ==
     *
     * x_{k+1}   = x_k + (v_k+wv) * cos(yaw_k) * dt
     * y_{k+1}   = y_k + (v_k+wv) * sin(yaw_k) * dt
     * yaw_{k+1} = yaw_k + (u_k+wu) * dt
     *
     * dx/dx = 1.0 dx/dyaw = -(v_k + wv) * sin(yaw_k) * dt
     * dy/dy = 1.0 dy/dyaw = (v_k + wv) * cos(yaw_k) * dt
     * dyaw/dyaw = 1.0
     */
    const double& x = x_curr(STATE::IDX::X);
    const double& y = x_curr(STATE::IDX::Y);
    const double& yaw = x_curr(STATE::IDX::YAW);
    const double& v_k = u_curr(0);
    const auto wv_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WV);
    const double wv = wv_dist_ptr->calc_mean();

    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(3, 3);
    A(STATE::IDX::X, STATE::IDX::YAW) =  -(v_k + wv) * std::sin(yaw);
    A(STATE::IDX::Y, STATE::IDX::YAW) =   (v_k + wv) * std::cos(yaw);

    return A;
}


Eigen::MatrixXd SimpleVehicleModel::getProcessNoiseMatrix(const Eigen::VectorXd& x_curr,
                                                          const Eigen::VectorXd& u_curr,
                                                          const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map,
                                                          const double dt)
{
    /*  == Nonlinear model ==
    *
    * x_{k+1}   = x_k + (v_k+wv) * cos(yaw_k) * dt
    * y_{k+1}   = y_k + (v_k+wv) * sin(yaw_k) * dt
    * yaw_{k+1} = yaw_k + (u_k+wu) * dt
    *
    * dx/dwv = cos(yaw_k) * dt
    * dy/dwv = sin(yaw_k) * dt
     * dyaw/dwu = dt
    * dyaw/dyaw = 1.0
    */

    const auto wv_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WV);
    const auto wu_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WU);

    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(2, 2);
    Q << wv_dist_ptr->calc_variance()/(dt*dt), 0.0,
         0.0, wu_dist_ptr->calc_variance()/(dt*dt);

    const double& yaw = x_curr(STATE::IDX::YAW);
    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(3, 2);
    L(STATE::IDX::X, SYSTEM_NOISE::IDX::WV) = std::cos(yaw) * dt;
    L(STATE::IDX::Y, SYSTEM_NOISE::IDX::WV) = std::sin(yaw) * dt;
    L(STATE::IDX::YAW, SYSTEM_NOISE::IDX::WU) = dt;

    return L*Q*L.transpose();
}

Eigen::MatrixXd SimpleVehicleModel::getMeasurementMatrix(const Eigen::VectorXd& x_curr,
                                                         const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    throw std::runtime_error("[SimpleVehicleModel getMeasurementMatrix]: Undefined Function. Need landmark in the arugment");
}

Eigen::MatrixXd SimpleVehicleModel::getMeasurementNoiseMatrix(const Eigen::VectorXd& x_curr,
                                                              const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    throw std::runtime_error("[SimpleVehicleModel getMeasurementNoiseMatrix]: Undefined Function. Need landmark in the arugment");
}

Eigen::MatrixXd SimpleVehicleModel::getMeasurementMatrix(const Eigen::VectorXd& x_curr,
                                                         const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map,
                                                         const Eigen::Vector2d& landmark)
{
    const auto wr_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WR);
    const auto wa_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WA);

    // Update state mean
    const double mean_wr = wr_dist_ptr->calc_mean();
    const double mean_wa = wa_dist_ptr->calc_mean();

    // Covariance Update
    /*  == Nonlinear model ==
     *
     * rcos = (x_land - x) * cos(yaw) + (y_land - y) * sin(yaw) + mrcos
     * rsin = (y_land - y) * cos(yaw) - (x_land - x) * sin(yaw) + mrsin
     *
     */

    const double& x = x_curr(STATE::IDX::X);
    const double& y = x_curr(STATE::IDX::Y);
    const double& yaw = x_curr(STATE::IDX::YAW);
    const double& x_land = landmark(0);
    const double& y_land = landmark(1);

    const double drcos_bearing_dx = -std::cos(yaw);
    const double drcos_bearing_dy = -std::sin(yaw);
    const double drcos_bearing_dyaw = -(x_land - x) * std::sin(yaw) + (y_land - y) * std::cos(yaw);
    const double drsin_bearing_dx = std::sin(yaw);
    const double drsin_bearing_dy = -std::cos(yaw);
    const double drsin_bearing_dyaw = -(y_land - y) * std::sin(yaw) - (x_land - x) * std::cos(yaw);

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 3);
    H(MEASUREMENT::IDX::RCOS, STATE::IDX::X) = mean_wr * std::cos(mean_wa) * drcos_bearing_dx - mean_wr * std::sin(mean_wa) * drsin_bearing_dx;
    H(MEASUREMENT::IDX::RCOS, STATE::IDX::Y) = mean_wr * std::cos(mean_wa) * drcos_bearing_dy - mean_wr * std::sin(mean_wa) * drsin_bearing_dy;
    H(MEASUREMENT::IDX::RCOS, STATE::IDX::YAW) = mean_wr * std::cos(mean_wa) * drcos_bearing_dyaw - mean_wr * std::sin(mean_wa) * drsin_bearing_dyaw;
    H(MEASUREMENT::IDX::RSIN, STATE::IDX::X) = mean_wr * std::cos(mean_wa) * drsin_bearing_dx + mean_wr * std::sin(mean_wa) * drcos_bearing_dx;
    H(MEASUREMENT::IDX::RSIN, STATE::IDX::Y) = mean_wr * std::cos(mean_wa) * drsin_bearing_dy + mean_wr * std::sin(mean_wa) * drcos_bearing_dy;
    H(MEASUREMENT::IDX::RSIN, STATE::IDX::YAW) = mean_wr * std::cos(mean_wa) * drsin_bearing_dyaw + mean_wr * std::sin(mean_wa) * drcos_bearing_dyaw;

    return H;
}

Eigen::MatrixXd SimpleVehicleModel::getMeasurementNoiseMatrix(const Eigen::VectorXd& x_curr,
                                                              const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map,
                                                              const Eigen::Vector2d& landmark)
{
    const auto mr_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WR);
    const auto ma_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WA);
    const double mean_mr = mr_dist_ptr->calc_mean();
    const double mean_ma = ma_dist_ptr->calc_mean();

    const double& x = x_curr(STATE::IDX::X);
    const double& y = x_curr(STATE::IDX::Y);
    const double& yaw = x_curr(STATE::IDX::YAW);
    const double& x_land = landmark(0);
    const double& y_land = landmark(1);
    const double rcos_bearing = (x_land - x) * std::cos(yaw) + (y_land - y) * std::sin(yaw);
    const double rsin_bearing = (y_land - y) * std::cos(yaw) - (x_land - x) * std::sin(yaw);

    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(2, 2);
    R(MEASUREMENT_NOISE::IDX::WR, MEASUREMENT_NOISE::IDX::WR) = mr_dist_ptr->calc_variance();
    R(MEASUREMENT_NOISE::IDX::WA, MEASUREMENT_NOISE::IDX::WA) = ma_dist_ptr->calc_variance();

    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(2, 2);
    L(MEASUREMENT::IDX::RCOS, MEASUREMENT_NOISE::IDX::WR) = std::cos(mean_ma) * rcos_bearing - std::sin(mean_ma) * rsin_bearing;
    L(MEASUREMENT::IDX::RSIN, MEASUREMENT_NOISE::IDX::WR) = std::cos(mean_ma) * rsin_bearing + std::sin(mean_ma) * rcos_bearing;
    L(MEASUREMENT::IDX::RCOS, MEASUREMENT_NOISE::IDX::WA) = -mean_mr * std::sin(mean_ma) * rcos_bearing - mean_mr * std::cos(mean_ma) * rsin_bearing;
    L(MEASUREMENT::IDX::RSIN, MEASUREMENT_NOISE::IDX::WA) = -mean_mr * std::sin(mean_ma) * rsin_bearing + mean_mr * std::cos(mean_ma) * rcos_bearing;

    return L*R*L.transpose();
}

StateInfo SimpleVehicleModel::propagateStateMoments(const StateInfo &state_info,
                                                    const Eigen::VectorXd &control_inputs,
                                                    const double dt,
                                                    const std::map<int, std::shared_ptr<BaseDistribution>> &noise_map)
{
    const auto& state_mean = state_info.mean;
    const auto& state_cov = state_info.covariance;
    ThreeDimensionalNormalDistribution dist(state_info.mean, state_info.covariance);

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

    const auto wv_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WV);
    const auto wu_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WU);
    const double wvPow1 = wv_dist_ptr->calc_moment(1);
    const double wvPow2 = wv_dist_ptr->calc_moment(2);
    const double wuPow1 = wu_dist_ptr->calc_moment(1);
    const double wuPow2 = wu_dist_ptr->calc_moment(2);

    const double &v = control_inputs(INPUT::IDX::V);
    const double &u = control_inputs(INPUT::IDX::U);
    const double &cu = std::cos(u);
    const double &su = std::sin(u);

    // Dynamics updates.
    const double next_xPow1 = v*cPow1 + cPow1*wvPow1 + xPow1;
    const double next_yPow1 = v*sPow1 + sPow1*wvPow1 + yPow1;
    const double next_yawPow1 = u + yawPow1 + wuPow1;
    const double next_xPow2 = pow(v, 2)*cPow2 + 2*v*cPow1_xPow1 + 2*v*cPow2*wvPow1 + 2*cPow1_xPow1*wvPow1 + cPow2*wvPow2 + xPow2;
    const double next_yPow2 = pow(v, 2)*sPow2 + 2*v*sPow1_yPow1 + 2*v*sPow2*wvPow1 + 2*sPow1_yPow1*wvPow1 + sPow2*wvPow2 + yPow2;
    const double next_yawPow2 = pow(u, 2) + 2*u*yawPow1 + 2*u*wuPow1 + 2*yawPow1*wuPow1 + yawPow2 + wuPow2;
    const double next_xPow1_yPow1 = pow(v, 2)*cPow1_sPow1 + 2*v*cPow1_sPow1*wvPow1 + v*cPow1_yPow1 + v*sPow1_xPow1 + cPow1_sPow1*wvPow2 + cPow1_yPow1*wvPow1 + sPow1_xPow1*wvPow1 + xPow1_yPow1;
    const double next_xPow1_yawPow1 = u*v*cPow1 + u*cPow1*wvPow1 + u*xPow1 + v*cPow1*wuPow1 + v*cPow1_yawPow1 + cPow1*wuPow1*wvPow1 + cPow1_yawPow1*wvPow1 + xPow1_yawPow1 + wuPow1*xPow1;
    const double next_yPow1_yawPow1 = u*v*sPow1 + u*sPow1*wvPow1 + u*yPow1 + v*sPow1*wuPow1 + v*sPow1_yawPow1 + sPow1*wuPow1*wvPow1 + sPow1_yawPow1*wvPow1 + yPow1_yawPow1 + wuPow1*yPow1;

    StateInfo result;
    result.mean = Eigen::VectorXd::Zero(3);
    result.covariance = Eigen::MatrixXd::Zero(3, 3);

    result.mean(STATE::IDX::X) = next_xPow1;
    result.mean(STATE::IDX::Y) = next_yPow1;
    result.mean(STATE::IDX::YAW) = next_yawPow1;
    result.covariance(STATE::IDX::X, STATE::IDX::X) = next_xPow2 - next_xPow1*next_xPow1;
    result.covariance(STATE::IDX::Y, STATE::IDX::Y) = next_yPow2 - next_yPow1*next_yPow1;
    result.covariance(STATE::IDX::YAW, STATE::IDX::YAW) = next_yawPow2 - next_yawPow1*next_yawPow1;
    result.covariance(STATE::IDX::X, STATE::IDX::Y) = next_xPow1_yPow1 - next_xPow1*next_yPow1;
    result.covariance(STATE::IDX::X, STATE::IDX::YAW) = next_xPow1_yawPow1 - next_xPow1*next_yawPow1;
    result.covariance(STATE::IDX::Y, STATE::IDX::YAW) = next_yPow1_yawPow1 - next_yPow1*next_yawPow1;
    result.covariance(STATE::IDX::Y, STATE::IDX::X) = result.covariance(STATE::IDX::X, STATE::IDX::Y);
    result.covariance(STATE::IDX::YAW, STATE::IDX::X) = result.covariance(STATE::IDX::X, STATE::IDX::YAW);
    result.covariance(STATE::IDX::YAW, STATE::IDX::Y) = result.covariance(STATE::IDX::Y, STATE::IDX::YAW);

    return result;
}

StateInfo SimpleVehicleModel::getMeasurementMoments(const StateInfo &state_info,
                                                    const std::map<int, std::shared_ptr<BaseDistribution>> &noise_map)
{
    throw std::runtime_error("[SimpleVehicleModel getMeasurementMoments]: Undefined Function. Need landmark in the argument");
}

Eigen::MatrixXd SimpleVehicleModel::getStateMeasurementMatrix(const StateInfo& state_info,
                                                              const StateInfo& measurement_info,
                                                              const std::map<int, std::shared_ptr<BaseDistribution>> &noise_map)
{
    throw std::runtime_error("[SimpleVehicleModel getStateMeasurementMoments]: Undefined Function. Need landmark in the argument");
}

StateInfo SimpleVehicleModel::getMeasurementMoments(const StateInfo &state_info,
                                                    const std::map<int, std::shared_ptr<BaseDistribution>> &noise_map,
                                                    const Eigen::Vector2d& landmark)
{
    const auto predicted_mean = state_info.mean;
    const auto predicted_cov = state_info.covariance;

    ThreeDimensionalNormalDistribution dist(predicted_mean, predicted_cov);
    const double cPow1= dist.calc_cos_moment(STATE::IDX::YAW, 1);
    const double sPow1= dist.calc_sin_moment(STATE::IDX::YAW, 1);

    const double cPow2= dist.calc_cos_moment(STATE::IDX::YAW, 2);
    const double sPow2= dist.calc_sin_moment(STATE::IDX::YAW, 2);
    const double xPow1_cPow1 = dist.calc_x_cos_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    const double yPow1_cPow1 = dist.calc_x_cos_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    const double xPow1_sPow1 = dist.calc_x_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    const double yPow1_sPow1 = dist.calc_x_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    const double cPow1_sPow1 = dist.calc_cos_sin_moment(STATE::IDX::YAW, 1, 1);

    const double xPow1_cPow2 = dist.calc_x_cos_z_cos_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    const double yPow1_cPow2 = dist.calc_x_cos_z_cos_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    const double xPow1_sPow2 = dist.calc_x_sin_z_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    const double yPow1_sPow2 = dist.calc_x_sin_z_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    const double xPow1_cPow1_sPow1 = dist.calc_x_cos_z_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    const double yPow1_cPow1_sPow1 = dist.calc_x_cos_z_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW);

    const double xPow2_cPow2 = dist.calc_xx_cos_z_cos_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    const double yPow2_cPow2 = dist.calc_xx_cos_z_cos_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    const double xPow2_sPow2 = dist.calc_xx_sin_z_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    const double yPow2_sPow2 = dist.calc_xx_sin_z_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    const double xPow2_cPow1_sPow1 = dist.calc_xx_cos_z_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    const double yPow2_cPow1_sPow1 = dist.calc_xx_cos_z_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    const double xPow1_yPow1_cPow2 = dist.calc_xy_cos_z_cos_z_moment();
    const double xPow1_yPow1_sPow2 = dist.calc_xy_sin_z_sin_z_moment();
    const double xPow1_yPow1_cPow1_sPow1 = dist.calc_xy_cos_z_sin_z_moment();

    const auto wr_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WR);
    const auto wa_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WA);
    const double wrPow1 = wr_dist_ptr->calc_moment(1);
    const double wrPow2 = wr_dist_ptr->calc_moment(2);
    const double cwaPow1 = wa_dist_ptr->calc_cos_moment(1);
    const double swaPow1 = wa_dist_ptr->calc_sin_moment(1);
    const double cwaPow2 = wa_dist_ptr->calc_cos_moment(2);
    const double swaPow2 = wa_dist_ptr->calc_sin_moment(2);
    const double cwaPow1_swaPow1 = wa_dist_ptr->calc_cos_sin_moment(1, 1);

    // Landmark
    const double &x_land = landmark(0);
    const double &y_land = landmark(1);

    const double caPow1 = x_land * cPow1 - xPow1_cPow1 + y_land * sPow1 - yPow1_sPow1;
    const double saPow1 = y_land * cPow1 - yPow1_cPow1 - x_land * sPow1 + xPow1_sPow1;
    const double caPow2 = std::pow(x_land, 2) * cPow2 + xPow2_cPow2 - 2.0 * x_land * xPow1_cPow2
                          + std::pow(y_land, 2) * sPow2 + yPow2_sPow2 - 2.0 * y_land * yPow1_sPow2
                          + 2.0 * x_land * y_land * cPow1_sPow1 - 2.0 * x_land * yPow1_cPow1_sPow1
                          - 2.0 * y_land * xPow1_cPow1_sPow1 + 2.0 * xPow1_yPow1_cPow1_sPow1;
    const double saPow2 = std::pow(y_land, 2) * cPow2 + yPow2_cPow2 - 2.0 * y_land * yPow1_cPow2
                          + std::pow(x_land, 2) * sPow2 + xPow2_sPow2 - 2.0 * x_land * xPow1_sPow2
                          - 2.0 * x_land * y_land * cPow1_sPow1 + 2.0 * x_land * yPow1_cPow1_sPow1
                          + 2.0 * y_land * xPow1_cPow1_sPow1 - 2.0 * xPow1_yPow1_cPow1_sPow1;
    const double caPow1_saPow1 =  x_land * y_land * cPow2 + xPow1_yPow1_cPow2 - x_land * yPow1_cPow2 - y_land * xPow1_cPow2
                                  - std::pow(x_land, 2) * cPow1_sPow1 - xPow2_cPow1_sPow1 + 2.0 * x_land * xPow1_cPow1_sPow1
                                  + std::pow(y_land, 2) * cPow1_sPow1 + yPow2_cPow1_sPow1 - 2.0 * y_land * yPow1_cPow1_sPow1
                                  - x_land * y_land * sPow2 - xPow1_yPow1_sPow2 + x_land * yPow1_sPow2 + y_land * xPow1_sPow2;

    const double rcosPow1 = wrPow1 * cwaPow1 * caPow1 - wrPow1 * swaPow1 * saPow1;
    const double rsinPow1 = wrPow1 * cwaPow1 * saPow1 + wrPow1 * swaPow1 * caPow1;

    const double rcosPow2 = wrPow2 * cwaPow2 * caPow2 + wrPow2 * swaPow2 * saPow2 - 2.0 * wrPow2 * cwaPow1_swaPow1 * caPow1_saPow1;
    const double rsinPow2 = wrPow2 * cwaPow2 * saPow2 + wrPow2 * swaPow2 * caPow2 + 2.0 * wrPow2 * cwaPow1_swaPow1 * caPow1_saPow1;
    const double rcosPow1_rsinPow1 = wrPow2 * (cwaPow2 - swaPow2) * caPow1_saPow1 - wrPow2 * cwaPow1_swaPow1 * (caPow2 - saPow2);

    StateInfo meas_info;
    meas_info.mean = Eigen::VectorXd::Zero(2);
    meas_info.covariance = Eigen::MatrixXd::Zero(2, 2);
    meas_info.mean(MEASUREMENT::IDX::RCOS) = rcosPow1;
    meas_info.mean(MEASUREMENT::IDX::RSIN) = rsinPow1;
    meas_info.covariance(MEASUREMENT::IDX::RCOS, MEASUREMENT::IDX::RCOS) = rcosPow2 - rcosPow1*rcosPow1;
    meas_info.covariance(MEASUREMENT::IDX::RSIN, MEASUREMENT::IDX::RSIN) = rsinPow2 - rsinPow1*rsinPow1;
    meas_info.covariance(MEASUREMENT::IDX::RCOS, MEASUREMENT::IDX::RSIN) = rcosPow1_rsinPow1 - rcosPow1*rsinPow1;
    meas_info.covariance(MEASUREMENT::IDX::RSIN, MEASUREMENT::IDX::RCOS) = meas_info.covariance(MEASUREMENT::IDX::RCOS, MEASUREMENT::IDX::RSIN);

    return meas_info;
}

Eigen::MatrixXd SimpleVehicleModel::getStateMeasurementMatrix(const StateInfo& state_info,
                                                              const StateInfo& measurement_info,
                                                              const std::map<int, std::shared_ptr<BaseDistribution>> &noise_map,
                                                              const Eigen::Vector2d& landmark)
{
    const auto predicted_mean = state_info.mean;
    const auto predicted_cov = state_info.covariance;
    const auto measurement_mean = measurement_info.mean;
    ThreeDimensionalNormalDistribution dist(predicted_mean, predicted_cov);

    const auto wr_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WR);
    const auto wa_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WA);
    const double wrPow1 = wr_dist_ptr->calc_moment(1);
    const double cwaPow1 = wa_dist_ptr->calc_cos_moment(1);
    const double swaPow1 = wa_dist_ptr->calc_sin_moment(1);

    const double& x_land = landmark(0);
    const double& y_land = landmark(1);

    const double xPow1_cPow1 = dist.calc_x_cos_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    const double xPow1_sPow1 = dist.calc_x_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    const double xPow2_cPow1 = dist.calc_xx_cos_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    const double xPow2_sPow1 = dist.calc_xx_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    const double yPow1_cPow1 = dist.calc_x_cos_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    const double yPow1_sPow1 = dist.calc_x_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    const double yPow2_cPow1 = dist.calc_xx_cos_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    const double yPow2_sPow1 = dist.calc_xx_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    const double xPow1_yPow1_cPow1 = dist.calc_xy_cos_z_moment();
    const double xPow1_yPow1_sPow1 = dist.calc_xy_sin_z_moment();
    const double yawPow1_cPow1 = dist.calc_x_cos_x_moment(STATE::IDX::YAW, 1, 1);
    const double yawPow1_sPow1 = dist.calc_x_sin_x_moment(STATE::IDX::YAW, 1, 1);
    const double xPow1_yawPow1_cPow1 = dist.calc_xy_cos_y_moment(STATE::IDX::X, STATE::IDX::YAW);
    const double yPow1_yawPow1_cPow1 = dist.calc_xy_cos_y_moment(STATE::IDX::Y, STATE::IDX::YAW);
    const double xPow1_yawPow1_sPow1 = dist.calc_xy_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW);
    const double yPow1_yawPow1_sPow1 = dist.calc_xy_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW);

    const double xPow1_caPow1 = x_land * xPow1_cPow1 - xPow2_cPow1 + y_land * xPow1_sPow1 - xPow1_yPow1_sPow1;
    const double xPow1_saPow1 = y_land * xPow1_cPow1 - xPow1_yPow1_cPow1 - x_land * xPow1_sPow1 + xPow2_sPow1;
    const double yPow1_caPow1 = x_land * yPow1_cPow1 - xPow1_yPow1_cPow1 + y_land * yPow1_sPow1 - yPow2_sPow1;
    const double yPow1_saPow1 = y_land * yPow1_cPow1 - yPow2_cPow1 - x_land * yPow1_sPow1 + xPow1_yPow1_sPow1;
    const double yawPow1_caPow1 = x_land * yawPow1_cPow1 - xPow1_yawPow1_cPow1 + y_land * yawPow1_sPow1 - yPow1_yawPow1_sPow1;
    const double yawPow1_saPow1 = y_land * yawPow1_cPow1 - yPow1_yawPow1_cPow1 - x_land * yawPow1_sPow1 + xPow1_yawPow1_sPow1;

    Eigen::MatrixXd state_observation_cov(3, 2); // sigma = E[XY^T] - E[X]E[Y]^T
    state_observation_cov(STATE::IDX::X, MEASUREMENT::IDX::RCOS)
            = wrPow1 * cwaPow1 * xPow1_caPow1 - wrPow1 * swaPow1 * xPow1_saPow1
              - predicted_mean(STATE::IDX::X) * measurement_mean(MEASUREMENT::IDX::RCOS);
    state_observation_cov(STATE::IDX::X, MEASUREMENT::IDX::RSIN)
            = wrPow1 * cwaPow1 * xPow1_saPow1 + wrPow1 * swaPow1 * xPow1_caPow1
              - predicted_mean(STATE::IDX::X) * measurement_mean(MEASUREMENT::IDX::RSIN); // x_p * yaw

    state_observation_cov(STATE::IDX::Y, MEASUREMENT::IDX::RCOS)
            =  wrPow1 * cwaPow1 * yPow1_caPow1 - wrPow1 * swaPow1 * yPow1_saPow1
               - predicted_mean(STATE::IDX::Y) * measurement_mean(MEASUREMENT::IDX::RCOS); // yp * (xp^2 + yp^2)
    state_observation_cov(STATE::IDX::Y, MEASUREMENT::IDX::RSIN)
            =  wrPow1 * cwaPow1 * yPow1_saPow1 + wrPow1 * swaPow1 * yPow1_caPow1
               - predicted_mean(STATE::IDX::Y) * measurement_mean(MEASUREMENT::IDX::RSIN); // y_p * yaw

    state_observation_cov(STATE::IDX::YAW, MEASUREMENT::IDX::RCOS)
            =   wrPow1 * cwaPow1 * yawPow1_caPow1 - wrPow1 * swaPow1 * yawPow1_saPow1
                - predicted_mean(STATE::IDX::YAW) * measurement_mean(MEASUREMENT::IDX::RCOS);
    state_observation_cov(STATE::IDX::YAW, MEASUREMENT::IDX::RSIN)
            =    wrPow1 * cwaPow1 * yawPow1_saPow1 + wrPow1 * swaPow1 * yawPow1_caPow1
                 - predicted_mean(STATE::IDX::YAW) * measurement_mean(MEASUREMENT::IDX::RSIN);

    return state_observation_cov;
}
