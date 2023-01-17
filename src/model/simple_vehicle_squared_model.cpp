#include "model/simple_vehicle_squared_model.h"
#include "distribution/three_dimensional_normal_distribution.h"

using namespace SimpleVehicleSquared;

Eigen::VectorXd SimpleVehicleSquaredModel::propagate(const Eigen::VectorXd& x_curr,
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

Eigen::VectorXd SimpleVehicleSquaredModel::propagate(const Eigen::VectorXd &x_curr,
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
    x_next(STATE::IDX::X) = x_curr(STATE::IDX::X) + (u_curr(INPUT::IDX::V) + system_noise(SYSTEM_NOISE::IDX::WV))* std::cos(x_curr(STATE::IDX::YAW));
    x_next(STATE::IDX::Y) = x_curr(STATE::IDX::Y) + (u_curr(INPUT::IDX::V) + system_noise(SYSTEM_NOISE::IDX::WV)) * std::sin(x_curr(STATE::IDX::YAW));
    x_next(STATE::IDX::YAW) = x_curr(STATE::IDX::YAW) + (u_curr(INPUT::IDX::U) + system_noise(SYSTEM_NOISE::IDX::WU));

    return x_next;
}

// measurement model
Eigen::VectorXd SimpleVehicleSquaredModel::measure(const Eigen::VectorXd& x_curr,
                                                   const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    throw std::runtime_error("[SimpleVehicleSquaredModel]: Undefined Function. Need landmark in the arugment");
}

Eigen::VectorXd SimpleVehicleSquaredModel::measure(const Eigen::VectorXd& x_curr, const Eigen::VectorXd& observation_noise)
{
    throw std::runtime_error("[SimpleVehicleSquaredModel]: Undefined Function. Need landmark in the arugment");
}

Eigen::VectorXd SimpleVehicleSquaredModel::measureWithLandmark(const Eigen::VectorXd& x_curr,
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

Eigen::VectorXd SimpleVehicleSquaredModel::measureWithLandmark(const Eigen::VectorXd& x_curr,
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
    const double rcos_bearing = (x_land - x) * std::cos(yaw) + (y_land - y) * std::sin(yaw);
    const double rsin_bearing = (y_land - y) * std::cos(yaw) - (x_land - x) * std::sin(yaw);

    Eigen::VectorXd y_next = Eigen::VectorXd::Zero(2);
    const double rcos = wr * std::cos(wa) * rcos_bearing - wr * std::sin(wa) * rsin_bearing;
    const double rsin = wr * std::cos(wa) * rsin_bearing + wr * std::sin(wa) * rcos_bearing;
    y_next(MEASUREMENT::IDX::RCOS) = rcos * rcos;
    y_next(MEASUREMENT::IDX::RSIN) = rsin * rsin;

    return y_next;
}

Eigen::MatrixXd SimpleVehicleSquaredModel::getStateMatrix(const Eigen::VectorXd& x_curr,
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


Eigen::MatrixXd SimpleVehicleSquaredModel::getProcessNoiseMatrix(const Eigen::VectorXd& x_curr,
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

Eigen::MatrixXd SimpleVehicleSquaredModel::getMeasurementMatrix(const Eigen::VectorXd& x_curr,
                                                                const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    throw std::runtime_error("[SimpleVehicleSquaredModel getMeasurementMatrix]: Undefined Function. Need landmark in the arugment");
}

Eigen::MatrixXd SimpleVehicleSquaredModel::getMeasurementNoiseMatrix(const Eigen::VectorXd& x_curr,
                                                                     const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    throw std::runtime_error("[SimpleVehicleSquaredModel getMeasurementNoiseMatrix]: Undefined Function. Need landmark in the arugment");
}

Eigen::MatrixXd SimpleVehicleSquaredModel::getMeasurementMatrix(const Eigen::VectorXd& x_curr,
                                                                const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map,
                                                                const Eigen::Vector2d& landmark)
{
    const auto wr_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WR);
    const auto wa_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WA);
    const double wr = wr_dist_ptr->calc_mean();
    const double wa = wa_dist_ptr->calc_mean();

    // Covariance Update
    /*  == Nonlinear model ==
     *
     * ha = (x_land - x) * cPow1 + (y_land - y) * sPow1 + mrcos
     * hb = (y_land - y) * cPow1 - (x_land - x) * sPow1 + mrsin
     * y1 = ha*vr*cos(v_phi) - hb*vr*sin(v_phi)
     * y2 = ha*vr*cos(v_phi) - hb*vr*sin(v_phi)
     * Y1 = y1 * y1
     * Y2 = y2 * y2
     *
     */

    const double& x = x_curr(STATE::IDX::X);
    const double& y = x_curr(STATE::IDX::Y);
    const double& yaw = x_curr(STATE::IDX::YAW);
    const double& x_land = landmark(0);
    const double& y_land = landmark(1);

    const double rcos_bearing = (x_land - x) * std::cos(yaw) + (y_land - y) * std::sin(yaw);
    const double rsin_bearing = (y_land - y) * std::cos(yaw) - (x_land - x) * std::sin(yaw);
    const double y1 = wr * std::cos(wa) * rcos_bearing - wr * std::sin(wa) * rsin_bearing;
    const double y2 = wr * std::cos(wa) * rsin_bearing + wr * std::sin(wa) * rcos_bearing;

    const double drcos_bearing_dx = -std::cos(yaw);
    const double drcos_bearing_dy = -std::sin(yaw);
    const double drcos_bearing_dyaw = -(x_land - x) * std::sin(yaw) + (y_land - y) * std::cos(yaw);
    const double drsin_bearing_dx = std::sin(yaw);
    const double drsin_bearing_dy = -std::cos(yaw);
    const double drsin_bearing_dyaw = -(y_land - y) * std::sin(yaw) - (x_land - x) * std::cos(yaw);

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 3);
    const double dy1_dx = wr * std::cos(wa) * drcos_bearing_dx - wr * std::sin(wa) * drsin_bearing_dx;
    const double dy1_dy = wr * std::cos(wa) * drcos_bearing_dy - wr * std::sin(wa) * drsin_bearing_dy;
    const double dy1_dyaw = wr * std::cos(wa) * drcos_bearing_dyaw - wr * std::sin(wa) * drsin_bearing_dyaw;
    const double dy2_dx = wr * std::cos(wa) * drsin_bearing_dx + wr * std::sin(wa) * drcos_bearing_dx;
    const double dy2_dy = wr * std::cos(wa) * drsin_bearing_dy + wr * std::sin(wa) * drcos_bearing_dy;
    const double dy2_dyaw = wr * std::cos(wa) * drsin_bearing_dyaw + wr * std::sin(wa) * drcos_bearing_dyaw;
    H(MEASUREMENT::IDX::RCOS, STATE::IDX::X) = 2.0 * y1 * dy1_dx;
    H(MEASUREMENT::IDX::RCOS, STATE::IDX::Y) = 2.0 * y1 * dy1_dy;
    H(MEASUREMENT::IDX::RCOS, STATE::IDX::YAW) = 2.0 * y1 * dy1_dyaw;
    H(MEASUREMENT::IDX::RSIN, STATE::IDX::X) = 2.0 * y2 * dy2_dx;
    H(MEASUREMENT::IDX::RSIN, STATE::IDX::Y) = 2.0 * y2 * dy2_dy;
    H(MEASUREMENT::IDX::RSIN, STATE::IDX::YAW) = 2.0 * y2 * dy2_dyaw;

    return H;
}

Eigen::MatrixXd SimpleVehicleSquaredModel::getMeasurementNoiseMatrix(const Eigen::VectorXd& x_curr,
                                                                     const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map,
                                                                     const Eigen::Vector2d& landmark)
{
    const auto mr_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WR);
    const auto ma_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WA);
    const double mr = mr_dist_ptr->calc_mean();
    const double ma = ma_dist_ptr->calc_mean();

    const double& x = x_curr(STATE::IDX::X);
    const double& y = x_curr(STATE::IDX::Y);
    const double& yaw = x_curr(STATE::IDX::YAW);
    const double& x_land = landmark(0);
    const double& y_land = landmark(1);
    const double rcos_bearing = (x_land - x) * std::cos(yaw) + (y_land - y) * std::sin(yaw);
    const double rsin_bearing = (y_land - y) * std::cos(yaw) - (x_land - x) * std::sin(yaw);
    const double y1 = mr * std::cos(ma) * rcos_bearing - mr * std::sin(ma) * rsin_bearing;
    const double y2 = mr * std::cos(ma) * rsin_bearing + mr * std::sin(ma) * rcos_bearing;

    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(2, 2);
    R(MEASUREMENT_NOISE::IDX::WR, MEASUREMENT_NOISE::IDX::WR) = mr_dist_ptr->calc_variance();
    R(MEASUREMENT_NOISE::IDX::WA, MEASUREMENT_NOISE::IDX::WA) = ma_dist_ptr->calc_variance();

    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(2, 2);
    const double dy1_dmr = std::cos(ma) * rcos_bearing - std::sin(ma) * rsin_bearing;
    const double dy2_dmr = std::cos(ma) * rsin_bearing + std::sin(ma) * rcos_bearing;
    const double dy1_dma = -mr * std::sin(ma) * rcos_bearing - mr * std::cos(ma) * rsin_bearing;
    const double dy2_dma = -mr * std::sin(ma) * rsin_bearing + mr * std::cos(ma) * rcos_bearing;
    L(MEASUREMENT::IDX::RCOS, MEASUREMENT_NOISE::IDX::WR) = 2.0 * y1 * dy1_dmr;
    L(MEASUREMENT::IDX::RSIN, MEASUREMENT_NOISE::IDX::WR) = 2.0 * y2 * dy2_dmr;
    L(MEASUREMENT::IDX::RCOS, MEASUREMENT_NOISE::IDX::WA) = 2.0 * y1 * dy1_dma;
    L(MEASUREMENT::IDX::RSIN, MEASUREMENT_NOISE::IDX::WA) = 2.0 * y2 * dy2_dma;

    return L*R*L.transpose();
}

StateInfo SimpleVehicleSquaredModel::propagateStateMoments(const StateInfo &state_info,
                                                           const Eigen::VectorXd &control_inputs,
                                                           const double dt,
                                                           const std::map<int, std::shared_ptr<BaseDistribution>> &noise_map)
{
    const auto& state_mean = state_info.mean;
    const auto& state_cov = state_info.covariance;
    ThreeDimensionalNormalDistribution dist(state_info.mean, state_info.covariance);

    const double xPow1 = dist.calc_moment(STATE::IDX::X, 1); // x
    const double yPow1 = dist.calc_moment(STATE::IDX::Y, 1); // y
    const double cPow1 = dist.calc_cos_moment(STATE::IDX::YAW, 1); // cPow1
    const double sPow1 = dist.calc_sin_moment(STATE::IDX::YAW, 1); // sPow1
    const double yawPow1 = dist.calc_moment(STATE::IDX::YAW, 1); // yaw
    const double xPow2 = dist.calc_moment(STATE::IDX::X, 2); // x^2
    const double yPow2 = dist.calc_moment(STATE::IDX::Y, 2); // y^2
    const double cPow2 = dist.calc_cos_moment(STATE::IDX::YAW, 2); // cPow1^2
    const double sPow2 = dist.calc_sin_moment(STATE::IDX::YAW, 2); // sPow1^2
    const double yawPow2 = dist.calc_moment(STATE::IDX::YAW, 2); // yaw^2
    const double xPow1_yPow1 = dist.calc_cross_second_moment(STATE::IDX::X, STATE::IDX::Y); // xy
    const double cPow1_xPow1 = dist.calc_x_cos_z_moment(STATE::IDX::X, STATE::IDX::YAW); // x*cPow1
    const double sPow1_xPow1 = dist.calc_x_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW); // x*sPow1
    const double cPow1_yPow1 = dist.calc_x_cos_z_moment(STATE::IDX::Y, STATE::IDX::YAW); // y*cPow1
    const double sPow1_yPow1 = dist.calc_x_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW); // y*sPow1
    const double cPow1_sPow1 = dist.calc_cos_sin_moment(STATE::IDX::YAW, 1, 1); // cPow1*sPow1
    const double xPow1_yawPow1 = dist.calc_cross_second_moment(STATE::IDX::X, STATE::IDX::YAW); // x*yaw
    const double yPow1_yawPow1 = dist.calc_cross_second_moment(STATE::IDX::Y, STATE::IDX::YAW); // y*yaw
    const double cPow1_yawPow1 = dist.calc_x_cos_x_moment(STATE::IDX::YAW, 1, 1); // yaw*cPow1
    const double sPow1_yawPow1 = dist.calc_x_sin_x_moment(STATE::IDX::YAW, 1, 1); // yaw*sPow1

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

StateInfo SimpleVehicleSquaredModel::getMeasurementMoments(const StateInfo &state_info,
                                                           const std::map<int, std::shared_ptr<BaseDistribution>> &noise_map)
{
    throw std::runtime_error("[SimpleVehicleSquaredModel getMeasurementMoments]: Undefined Function. Need landmark in the argument");
}

Eigen::MatrixXd SimpleVehicleSquaredModel::getStateMeasurementMatrix(const StateInfo& state_info,
                                                                     const StateInfo& measurement_info,
                                                                     const std::map<int, std::shared_ptr<BaseDistribution>> &noise_map)
{
    throw std::runtime_error("[SimpleVehicleSquaredModel getStateMeasurementMoments]: Undefined Function. Need landmark in the argument");
}

StateInfo SimpleVehicleSquaredModel::getMeasurementMoments(const StateInfo &state_info,
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

    const double cPow4 = dist.calc_cos_moment(STATE::IDX::YAW, 4);
    const double sPow4 = dist.calc_sin_moment(STATE::IDX::YAW, 4);
    const double cPow1_sPow3 = dist.calc_cos_sin_moment(STATE::IDX::YAW, 1, 3);
    const double cPow3_sPow1 = dist.calc_cos_sin_moment(STATE::IDX::YAW, 3, 1);
    const double cPow2_sPow2 = dist.calc_cos_sin_moment(STATE::IDX::YAW, 2, 2);
    const double xPow2_cPow2 = dist.calc_xx_cos_z_cos_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    const double yPow2_cPow2 = dist.calc_xx_cos_z_cos_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    const double xPow2_sPow2 = dist.calc_xx_sin_z_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    const double yPow2_sPow2 = dist.calc_xx_sin_z_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    const double xPow2_cPow1_sPow1 = dist.calc_xx_cos_z_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    const double yPow2_cPow1_sPow1 = dist.calc_xx_cos_z_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    const double xPow1_yPow1_cPow2 = dist.calc_xy_cos_z_cos_z_moment();
    const double xPow1_yPow1_sPow2 = dist.calc_xy_sin_z_sin_z_moment();
    const double xPow1_yPow1_cPow1_sPow1 = dist.calc_xy_cos_z_sin_z_moment();

    const double xPow1_cPow4 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW, 1, 0, 4, 0);
    const double yPow1_sPow4 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW, 1, 0, 0, 4);
    const double xPow1_cPow3_sPow1 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW, 1, 0, 3, 1);
    const double xPow1_cPow2_sPow2 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW, 1, 0, 2, 2);
    const double xPow1_cPow1_sPow3 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW, 1, 0, 1, 3);
    const double yPow1_cPow3_sPow1 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW, 1, 0, 3, 1);
    const double yPow1_cPow2_sPow2 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW, 1, 0, 2, 2);
    const double yPow1_cPow1_sPow3 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW, 1, 0, 1, 3);

    const double xPow2_cPow4 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW, 2, 0, 4, 0);
    const double yPow2_sPow4 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW, 2, 0, 0, 4);
    const double xPow2_cPow2_sPow2 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW, 2, 0, 2, 2);
    const double yPow2_cPow2_sPow2 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW, 2, 0, 2, 2);
    const double xPow2_cPow3_sPow1 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW, 2, 0, 3, 1);
    const double yPow2_cPow1_sPow3 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW, 2, 0, 1, 3);
    const double xPow1_yPow1_cPow3_sPow1 = dist.calc_xy_cos_z_sin_z_moment(1, 1, 3, 1);
    const double xPow1_yPow1_cPow2_sPow2 = dist.calc_xy_cos_z_sin_z_moment(1, 1, 2, 2);
    const double xPow1_yPow1_cPow1_sPow3 = dist.calc_xy_cos_z_sin_z_moment(1, 1, 1, 3);

    const double xPow3_cPow4 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW, 3, 0, 4, 0);
    const double yPow3_sPow4 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW, 3, 0, 0, 4);
    const double xPow3_cPow3_sPow1 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW, 3, 0, 3, 1);
    const double yPow3_cPow1_sPow3 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW, 3, 0, 1, 3);
    const double xPow2_yPow1_cPow3_sPow1 = dist.calc_xy_cos_z_sin_z_moment(2, 1, 3, 1);
    const double xPow2_yPow1_cPow2_sPow2 = dist.calc_xy_cos_z_sin_z_moment(2, 1, 2, 2);
    const double xPow1_yPow2_cPow2_sPow2 = dist.calc_xy_cos_z_sin_z_moment(1, 2, 2, 2);
    const double xPow1_yPow2_cPow1_sPow3 = dist.calc_xy_cos_z_sin_z_moment(1, 2, 1, 3);

    const double xPow4_cPow4 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW, 4, 0, 4, 0);
    const double yPow4_sPow4 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW, 4, 0, 0, 4);
    const double xPow3_yPow1_cPow3_sPow1 = dist.calc_xy_cos_z_sin_z_moment(3, 1, 3, 1);
    const double xPow2_yPow2_cPow2_sPow2 = dist.calc_xy_cos_z_sin_z_moment(2, 2, 2, 2);
    const double xPow1_yPow3_cPow1_sPow3 = dist.calc_xy_cos_z_sin_z_moment(1, 3, 1, 3);

    const auto wr_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WR);
    const auto wa_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WA);
    const double wrPow1 = wr_dist_ptr->calc_moment(1);
    const double wrPow2 = wr_dist_ptr->calc_moment(2);
    const double wrPow4 = wr_dist_ptr->calc_moment(4);
    const double cwaPow1 = wa_dist_ptr->calc_cos_moment(1);
    const double swaPow1 = wa_dist_ptr->calc_sin_moment(1);
    const double cwaPow2 = wa_dist_ptr->calc_cos_moment(2);
    const double swaPow2 = wa_dist_ptr->calc_sin_moment(2);
    const double cwaPow4 = wa_dist_ptr->calc_cos_moment(4);
    const double swaPow4 = wa_dist_ptr->calc_sin_moment(4);
    const double cwaPow1_swaPow1 = wa_dist_ptr->calc_cos_sin_moment(1, 1);
    const double cwaPow3_swaPow1 = wa_dist_ptr->calc_cos_sin_moment(3, 1);
    const double cwaPow1_swaPow3 = wa_dist_ptr->calc_cos_sin_moment(1, 3);
    const double cwaPow2_swaPow2 = wa_dist_ptr->calc_cos_sin_moment(2, 2);

    // Landmark
    const double &x_land = landmark(0);
    const double &y_land = landmark(1);

    const double haPow1 = x_land * cPow1 - xPow1_cPow1 + y_land * sPow1 - yPow1_sPow1;
    const double hbPow1 = y_land * cPow1 - yPow1_cPow1 - x_land * sPow1 + xPow1_sPow1;
    const double haPow2 = std::pow(x_land, 2) * cPow2 + xPow2_cPow2 - 2.0 * x_land * xPow1_cPow2
                          + std::pow(y_land, 2) * sPow2 + yPow2_sPow2 - 2.0 * y_land * yPow1_sPow2
                          + 2.0 * x_land * y_land * cPow1_sPow1 - 2.0 * x_land * yPow1_cPow1_sPow1
                          - 2.0 * y_land * xPow1_cPow1_sPow1 + 2.0 * xPow1_yPow1_cPow1_sPow1;
    const double hbPow2 = std::pow(y_land, 2) * cPow2 + yPow2_cPow2 - 2.0 * y_land * yPow1_cPow2
                          + std::pow(x_land, 2) * sPow2 + xPow2_sPow2 - 2.0 * x_land * xPow1_sPow2
                          - 2.0 * x_land * y_land * cPow1_sPow1 + 2.0 * x_land * yPow1_cPow1_sPow1
                          + 2.0 * y_land * xPow1_cPow1_sPow1 - 2.0 * xPow1_yPow1_cPow1_sPow1;
    const double haPow1_hbPow1 =  x_land * y_land * cPow2 + xPow1_yPow1_cPow2 - x_land * yPow1_cPow2 - y_land * xPow1_cPow2
                                  - std::pow(x_land, 2) * cPow1_sPow1 - xPow2_cPow1_sPow1 + 2.0 * x_land * xPow1_cPow1_sPow1
                                  + std::pow(y_land, 2) * cPow1_sPow1 + yPow2_cPow1_sPow1 - 2.0 * y_land * yPow1_cPow1_sPow1
                                  - x_land * y_land * sPow2 - xPow1_yPow1_sPow2 + x_land * yPow1_sPow2 + y_land * xPow1_sPow2;
    /*
    const double haPow4 = xPow4_cPow4 - 4 * x_land * xPow3_cPow4 - 4 * xPow3_yPow1_cPow3_sPow1 + 4 * y_land * xPow3_cPow3_sPow1
                        + 6 * xPow2_cPow4 * pow(x_land, 2) + 12 * x_land * xPow2_yPow1_cPow3_sPow1
                        - 12 * x_land * y_land * xPow2_cPow3_sPow1 + 6 * xPow2_yPow2_cPow2_sPow2
                        - 12 * y_land * xPow2_yPow1_cPow2_sPow2 + 6 * pow(y_land, 2) * xPow2_cPow2_sPow2
                        - 4 * pow(x_land, 3) * xPow1_cPow4 - 12 * pow(x_land, 2) * xPow1_yPow1_cPow3_sPow1
                        + 12 * pow(x_land, 2) * y_land * xPow1_cPow3_sPow1 - 12 * x_land * xPow1_yPow2_cPow2_sPow2
                        + 24 * x_land * y_land * xPow1_yPow1_cPow2_sPow2 - 12 * x_land * pow(y_land, 2) * xPow1_cPow2_sPow2
                        - 4 * xPow1_yPow3_cPow1_sPow3 + 12 * y_land * xPow1_yPow2_cPow1_sPow3
                        - 12 * pow(y_land, 2) * xPow1_yPow1_cPow1_sPow3 + 4 * pow(y_land, 3) * xPow1_cPow1_sPow3
                        + pow(x_land, 4) * cPow4 + 4 * pow(x_land, 3) * yPow1_cPow3_sPow1
                        - 4 * pow(x_land, 3) * y_land * cPow3_sPow1 + 6 * pow(x_land, 2) * yPow2_cPow2_sPow2
                        - 12 * pow(x_land, 2) * y_land * yPow1_cPow2_sPow2 + 6 * pow(x_land, 2) * pow(y_land, 2) * cPow2_sPow2
                        + 4 * x_land * yPow3_cPow1_sPow3 - 12 * x_land * y_land * yPow2_cPow1_sPow3
                        + 12 * x_land * pow(y_land, 2) * yPow1_cPow1_sPow3 - 4 * x_land * pow(y_land, 3) * cPow1_sPow3
                        + yPow4_sPow4 - 4 * y_land * yPow3_sPow4 + 6 * pow(y_land, 2) * yPow2_sPow4
                        - 4 * pow(y_land, 3) * yPow1_sPow4 + pow(y_land, 4) * sPow4;
                        */
    const double haPow4 = pow(x_land, 4)*cPow4 + 4*pow(x_land, 3)*y_land*cPow3_sPow1 - 4*pow(x_land, 3)*yPow1_cPow3_sPow1 - 4*pow(x_land, 3)*xPow1_cPow4 + 6*pow(x_land, 2)*pow(y_land, 2)*cPow2_sPow2 - 12*pow(x_land, 2)*y_land*yPow1_cPow2_sPow2 - 12*pow(x_land, 2)*y_land*xPow1_cPow3_sPow1 + 6*pow(x_land, 2)*yPow2_cPow2_sPow2 + 12*pow(x_land, 2)*xPow1_yPow1_cPow3_sPow1 + 6*pow(x_land, 2)*xPow2_cPow4 + 4*x_land*pow(y_land, 3)*cPow1_sPow3 - 12*x_land*pow(y_land, 2)*yPow1_cPow1_sPow3 - 12*x_land*pow(y_land, 2)*xPow1_cPow2_sPow2 + 12*x_land*y_land*yPow2_cPow1_sPow3 + 24*x_land*y_land*xPow1_yPow1_cPow2_sPow2 + 12*x_land*y_land*xPow2_cPow3_sPow1 - 4*x_land*yPow3_cPow1_sPow3 - 12*x_land*xPow1_yPow2_cPow2_sPow2 - 12*x_land*xPow2_yPow1_cPow3_sPow1 - 4*x_land*cPow4_xPow3 + pow(y_land, 4)*sPow4 - 4*pow(y_land, 3)*cPow1_sPow3_xPow1 - 4*pow(y_land, 3)*yPow1_sPow4 + 12*pow(y_land, 2)*cPow1_sPow3_xPow1_yPow1 + 6*pow(y_land, 2)*cPow2_sPow2_xPow2 + 6*pow(y_land, 2)*yPow2_sPow4 - 12*y_land*cPow1_sPow3_xPow1_yPow2 - 12*y_land*cPow2_sPow2_xPow2_yPow1 - 4*y_land*cPow3_sPow1_xPow3 - 4*y_land*sPow4_yPow3 + 4*cPow1_sPow3_xPow1_yPow3 + 6*xPow2_yPow2_cPow2_sPow2 + 4*xPow3_yPow1_cPow3_sPow1 + xPow4_cPow4 + yPow4_sPow4;
    const double hbPow4 = pow(x_land, 4)*sPow4 - 4*pow(x_land, 3)*y_land*cPow1_sPow3 + 4*pow(x_land, 3)*yPow1_cPow1_sPow3 - 4*pow(x_land, 3)*xPow1_sPow4 + 6*pow(x_land, 2)*pow(y_land, 2)*cPow2_sPow2 + 12*pow(x_land, 2)*y_land*xPow1_cPow1_sPow3 - 12*pow(x_land, 2)*y_land*yPow1_cPow2_sPow2 - 12*pow(x_land, 2)*xPow1_yPow1_cPow1_sPow3 + 6*pow(x_land, 2)*yPow2_cPow2_sPow2 + 6*pow(x_land, 2)*xPow2_sPow4 - 4*x_land*pow(y_land, 3)*cPow3_sPow1 - 12*x_land*pow(y_land, 2)*xPow1_cPow2_sPow2 + 12*x_land*pow(y_land, 2)*yPow1_cPow3_sPow1 - 12*x_land*y_land*xPow2_cPow1_sPow3 + 24*x_land*y_land*xPow1_yPow1_cPow2_sPow2 - 12*x_land*y_land*yPow2_cPow3_sPow1 + 12*x_land*xPow2_yPow1_cPow1_sPow3 - 12*x_land*xPow1_yPow2_cPow2_sPow2 + 4*x_land*yPow3_cPow3_sPow1 - 4*x_land*sPow4_xPow3 + pow(y_land, 4)*cPow4 + 4*pow(y_land, 3)*cPow3_sPow1_xPow1 - 4*pow(y_land, 3)*yPow1_cPow4 + 6*pow(y_land, 2)*cPow2_sPow2_xPow2 - 12*pow(y_land, 2)*cPow3_sPow1_xPow1_yPow1 + 6*pow(y_land, 2)*cPow4_yPow2 + 4*y_land*cPow1_sPow3_xPow3 - 12*y_land*cPow2_sPow2_xPow2_yPow1 + 12*y_land*cPow3_sPow1_xPow1_yPow2 - 4*y_land*cPow4_yPow3 - 4*cPow1_sPow3_xPow3_yPow1 + 6*xPow2_yPow2_cPow2_sPow2 - 4*cPow3_sPow1_xPow1_yPow3 + yPow4_cPow4 + xPow4_sPow4;
    const double haPow2_hbPow2 = pow(x_land, 2)*pow(y_land, 2)*cPow4 + pow(x_land, 2)*pow(y_land, 2)*sPow4 - 2*pow(x_land, 2)*y_land*yPow1_cPow4 - 2*pow(x_land, 2)*y_land*yPow1_sPow4 + pow(x_land, 2)*cPow4_yPow2 + pow(x_land, 2)*yPow2_sPow4 - 2*x_land*pow(y_land, 2)*cPow4_xPow1 - 2*x_land*pow(y_land, 2)*xPow1_sPow4 + 6*x_land*y_land*xPow2_cPow1_sPow3 - 6*x_land*y_land*yPow2_cPow1_sPow3 - 16*x_land*y_land*cPow2_sPow2_xPow1_yPow1 - 6*x_land*y_land*xPow2_cPow3_sPow1 + 6*x_land*y_land*yPow2_cPow3_sPow1 + 4*x_land*y_land*cPow4_xPow1_yPow1 + 4*x_land*y_land*xPow1_yPow1_sPow4 - 6*x_land*xPow2_cPow1_sPow3_yPow1 + 2*x_land*cPow1_sPow3_yPow3 + 8*x_land*cPow2_sPow2_xPow1_yPow2 - 4*x_land*cPow2_sPow2_xPow3 + 6*x_land*xPow2_cPow3_sPow1_yPow1 - 2*x_land*cPow3_sPow1_yPow3 - 2*x_land*cPow4_xPow1_yPow2 - 2*x_land*xPow1_yPow2_sPow4 + pow(y_land, 2)*cPow4_xPow2 + pow(y_land, 2)*sPow4_xPow2 + 6*y_land*cPow1_sPow3_xPow1_yPow2 - 2*y_land*cPow1_sPow3_xPow3 + 8*y_land*cPow2_sPow2_xPow2_yPow1 - 4*y_land*cPow2_sPow2_yPow3 - 6*y_land*cPow3_sPow1_xPow1_yPow2 + 2*y_land*cPow3_sPow1_xPow3 - 2*y_land*cPow4_xPow2_yPow1 - 2*y_land*sPow4_xPow2_yPow1 + cPow1_sPow3*(2*pow(x_land, 3)*y_land - 2*x_land*pow(y_land, 3)) + cPow1_sPow3_xPow1*(-6*pow(x_land, 2)*y_land + 2*pow(y_land, 3)) + cPow1_sPow3_xPow1_yPow1*(6*pow(x_land, 2) - 6*pow(y_land, 2)) - 2*cPow1_sPow3_xPow1_yPow3 + 2*cPow1_sPow3_xPow3_yPow1 + cPow1_sPow3_yPow1*(-2*pow(x_land, 3) + 6*x_land*pow(y_land, 2)) + cPow2_sPow2*(pow(x_land, 4) - 4*pow(x_land, 2)*pow(y_land, 2) + pow(y_land, 4)) + cPow2_sPow2_xPow1*(-4*pow(x_land, 3) + 8*x_land*pow(y_land, 2)) + cPow2_sPow2_xPow2*(6*pow(x_land, 2) - 4*pow(y_land, 2)) - 4*xPow2_yPow2_cPow2_sPow2 + cPow2_sPow2_xPow4 + cPow2_sPow2_yPow1*(8*pow(x_land, 2)*y_land - 4*pow(y_land, 3)) + cPow2_sPow2_yPow2*(-4*pow(x_land, 2) + 6*pow(y_land, 2)) + cPow2_sPow2_yPow4 + cPow3_sPow1*(-2*pow(x_land, 3)*y_land + 2*x_land*pow(y_land, 3)) + cPow3_sPow1_xPow1*(6*pow(x_land, 2)*y_land - 2*pow(y_land, 3)) + cPow3_sPow1_xPow1_yPow1*(-6*pow(x_land, 2) + 6*pow(y_land, 2)) + 2*cPow3_sPow1_xPow1_yPow3 - 2*xPow3_yPow1_cPow3_sPow1 + cPow3_sPow1_yPow1*(2*pow(x_land, 3) - 6*x_land*pow(y_land, 2)) + cPow4_xPow2_yPow2 + sPow4_xPow2_yPow2;
    const double haPow3_hbPow1 = pow(x_land, 3)*y_land*cPow4 - pow(x_land, 3)*yPow1_cPow4 - 6*pow(x_land, 2)*y_land*cPow3_sPow1_yPow1 - 3*pow(x_land, 2)*y_land*cPow4_xPow1 + 3*pow(x_land, 2)*yPow2_cPow3_sPow1 + 3*pow(x_land, 2)*cPow4_xPow1_yPow1 - x_land*pow(y_land, 3)*sPow4 + 6*x_land*pow(y_land, 2)*cPow1_sPow3_xPow1 + 3*x_land*pow(y_land, 2)*yPow1_sPow4 - 12*x_land*y_land*cPow1_sPow3_xPow1_yPow1 - 9*x_land*y_land*cPow2_sPow2_xPow2 + 9*x_land*y_land*cPow2_sPow2_yPow2 + 12*x_land*y_land*cPow3_sPow1_xPow1_yPow1 + 3*x_land*y_land*cPow4_xPow2 - 3*x_land*y_land*yPow2_sPow4 + 6*x_land*cPow1_sPow3_xPow1_yPow2 + 9*x_land*cPow2_sPow2_xPow2_yPow1 - 3*x_land*cPow2_sPow2_yPow3 - 6*x_land*cPow3_sPow1_xPow1_yPow2 + 4*x_land*cPow3_sPow1_xPow3 - 3*x_land*cPow4_xPow2_yPow1 + x_land*sPow4_yPow3 + pow(y_land, 3)*xPow1_sPow4 - 3*pow(y_land, 2)*xPow2_cPow1_sPow3 - 3*pow(y_land, 2)*xPow1_yPow1_sPow4 + 6*y_land*xPow2_cPow1_sPow3_yPow1 - 4*y_land*cPow1_sPow3_yPow3 - 9*y_land*cPow2_sPow2_xPow1_yPow2 + 3*y_land*cPow2_sPow2_xPow3 - 6*y_land*xPow2_cPow3_sPow1_yPow1 - y_land*cPow4_xPow3 + 3*y_land*xPow1_yPow2_sPow4 + cPow1_sPow3*(-3*pow(x_land, 2)*pow(y_land, 2) + pow(y_land, 4)) - 3*xPow2_cPow1_sPow3_yPow2 + cPow1_sPow3_yPow1*(6*pow(x_land, 2)*y_land - 4*pow(y_land, 3)) + yPow2_cPow1_sPow3*(-3*pow(x_land, 2) + 6*pow(y_land, 2)) + cPow1_sPow3_yPow4 + cPow2_sPow2*(-3*pow(x_land, 3)*y_land + 3*x_land*pow(y_land, 3)) + cPow2_sPow2_xPow1*(9*pow(x_land, 2)*y_land - 3*pow(y_land, 3)) + cPow2_sPow2_xPow1_yPow1*(-9*pow(x_land, 2) + 9*pow(y_land, 2)) + 3*cPow2_sPow2_xPow1_yPow3 - 3*cPow2_sPow2_xPow3_yPow1 + cPow2_sPow2_yPow1*(3*pow(x_land, 3) - 9*x_land*pow(y_land, 2)) + cPow3_sPow1*(-pow(x_land, 4) + 3*pow(x_land, 2)*pow(y_land, 2)) + cPow3_sPow1_xPow1*(4*pow(x_land, 3) - 6*x_land*pow(y_land, 2)) + xPow2_cPow3_sPow1*(-6*pow(x_land, 2) + 3*pow(y_land, 2)) + 3*xPow2_cPow3_sPow1_yPow2 - cPow3_sPow1_xPow4 + cPow4_xPow3_yPow1 - xPow1_sPow4_yPow3;
    const double haPow1_hbPow3 = -pow(x_land, 3)*y_land*sPow4 + pow(x_land, 3)*yPow1_sPow4 - 6*pow(x_land, 2)*y_land*cPow1_sPow3_yPow1 + 3*pow(x_land, 2)*y_land*xPow1_sPow4 + 3*pow(x_land, 2)*yPow2_cPow1_sPow3 - 3*pow(x_land, 2)*xPow1_yPow1_sPow4 + x_land*pow(y_land, 3)*cPow4 + 6*x_land*pow(y_land, 2)*cPow3_sPow1_xPow1 - 3*x_land*pow(y_land, 2)*yPow1_cPow4 + 12*x_land*y_land*cPow1_sPow3_xPow1_yPow1 + 9*x_land*y_land*cPow2_sPow2_xPow2 - 9*x_land*y_land*cPow2_sPow2_yPow2 - 12*x_land*y_land*cPow3_sPow1_xPow1_yPow1 + 3*x_land*y_land*cPow4_yPow2 - 3*x_land*y_land*sPow4_xPow2 - 6*x_land*cPow1_sPow3_xPow1_yPow2 + 4*x_land*cPow1_sPow3_xPow3 - 9*x_land*cPow2_sPow2_xPow2_yPow1 + 3*x_land*cPow2_sPow2_yPow3 + 6*x_land*cPow3_sPow1_xPow1_yPow2 - x_land*cPow4_yPow3 + 3*x_land*sPow4_xPow2_yPow1 - pow(y_land, 3)*cPow4_xPow1 - 3*pow(y_land, 2)*xPow2_cPow3_sPow1 + 3*pow(y_land, 2)*cPow4_xPow1_yPow1 - 6*y_land*xPow2_cPow1_sPow3_yPow1 + 9*y_land*cPow2_sPow2_xPow1_yPow2 - 3*y_land*cPow2_sPow2_xPow3 + 6*y_land*xPow2_cPow3_sPow1_yPow1 - 4*y_land*cPow3_sPow1_yPow3 - 3*y_land*cPow4_xPow1_yPow2 + y_land*sPow4_xPow3 + cPow1_sPow3*(-pow(x_land, 4) + 3*pow(x_land, 2)*pow(y_land, 2)) + cPow1_sPow3_xPow1*(4*pow(x_land, 3) - 6*x_land*pow(y_land, 2)) + xPow2_cPow1_sPow3*(-6*pow(x_land, 2) + 3*pow(y_land, 2)) + 3*xPow2_cPow1_sPow3_yPow2 - cPow1_sPow3_xPow4 + cPow2_sPow2*(3*pow(x_land, 3)*y_land - 3*x_land*pow(y_land, 3)) + cPow2_sPow2_xPow1*(-9*pow(x_land, 2)*y_land + 3*pow(y_land, 3)) + cPow2_sPow2_xPow1_yPow1*(9*pow(x_land, 2) - 9*pow(y_land, 2)) - 3*cPow2_sPow2_xPow1_yPow3 + 3*cPow2_sPow2_xPow3_yPow1 + cPow2_sPow2_yPow1*(-3*pow(x_land, 3) + 9*x_land*pow(y_land, 2)) + cPow3_sPow1*(-3*pow(x_land, 2)*pow(y_land, 2) + pow(y_land, 4)) - 3*xPow2_cPow3_sPow1_yPow2 + cPow3_sPow1_yPow1*(6*pow(x_land, 2)*y_land - 4*pow(y_land, 3)) + yPow2_cPow3_sPow1*(-3*pow(x_land, 2) + 6*pow(y_land, 2)) + cPow3_sPow1_yPow4 + cPow4_xPow1_yPow3 - sPow4_xPow3_yPow1;

    const double rcosPow2 = wrPow2 * cwaPow2 * haPow2 + wrPow2 * swaPow2 * hbPow2 - 2.0 * wrPow2 * cwaPow1_swaPow1 * haPow1_hbPow1;
    const double rsinPow2 = wrPow2 * cwaPow2 * hbPow2 + wrPow2 * swaPow2 * haPow2 + 2.0 * wrPow2 * cwaPow1_swaPow1 * haPow1_hbPow1;
    const double rcosPow4 = wrPow4 * cwaPow4 * haPow4 + wrPow4 * swaPow4 * hbPow4 + 6.0 * wrPow4 * cwaPow2_swaPow2 * haPow2_hbPow2
                            - 4.0 * wrPow4 * cwaPow3_swaPow1 * haPow3_hbPow1
                            - 4.0 * wrPow4 * cwaPow1_swaPow3 * haPow1_hbPow3;
    const double rsinPow4 = wrPow4 * cwaPow4 * hbPow4 + wrPow4 * swaPow4 * haPow4 + 6.0 * wrPow4 * cwaPow2_swaPow2 * haPow2_hbPow2
                            + 4.0 * wrPow4 * cwaPow3_swaPow1 * haPow1_hbPow3
                            + 4.0 * wrPow4 * cwaPow1_swaPow3 * haPow3_hbPow1;
    const double rcosPow2_rsinPow2 = haPow4 * wrPow4 * cwaPow2_swaPow2
                                   - 2 * haPow3_hbPow1 * wrPow4 * cwaPow1_swaPow3
                                   + 2 * haPow3_hbPow1 * wrPow4 * cwaPow3_swaPow1
                                   + haPow2_hbPow2 * wrPow4 * swaPow4
                                   - 4 * haPow2_hbPow2 * wrPow4 * cwaPow2_swaPow2
                                   + haPow2_hbPow2 * wrPow4 * cwaPow4
                                   + 2 * haPow1_hbPow3 * wrPow4 * cwaPow1_swaPow3
                                   - 2 * haPow1_hbPow3 * wrPow4 * cwaPow3_swaPow1
                                   + hbPow4 * wrPow4 * cwaPow2_swaPow2;

    StateInfo meas_info;
    meas_info.mean = Eigen::VectorXd::Zero(2);
    meas_info.covariance = Eigen::MatrixXd::Zero(2, 2);
    meas_info.mean(MEASUREMENT::IDX::RCOS) = rcosPow2;
    meas_info.mean(MEASUREMENT::IDX::RSIN) = rsinPow2;
    meas_info.covariance(MEASUREMENT::IDX::RCOS, MEASUREMENT::IDX::RCOS) = rcosPow4 - rcosPow2*rcosPow2;
    meas_info.covariance(MEASUREMENT::IDX::RSIN, MEASUREMENT::IDX::RSIN) = rsinPow4 - rsinPow2*rsinPow2;
    meas_info.covariance(MEASUREMENT::IDX::RCOS, MEASUREMENT::IDX::RSIN) = rcosPow2_rsinPow2 - rcosPow2*rsinPow2;
    meas_info.covariance(MEASUREMENT::IDX::RSIN, MEASUREMENT::IDX::RCOS) = meas_info.covariance(MEASUREMENT::IDX::RCOS, MEASUREMENT::IDX::RSIN);

    return meas_info;
}

Eigen::MatrixXd SimpleVehicleSquaredModel::getStateMeasurementMatrix(const StateInfo& state_info,
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

