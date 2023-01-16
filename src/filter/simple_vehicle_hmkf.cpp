#include "filter/simple_vehicle_hmkf.h"
#include "distribution/three_dimensional_normal_distribution.h"

using namespace SimpleVehicle;

SimpleVehicleHMKF::SimpleVehicleHMKF(const std::shared_ptr<BaseModel>& vehicle_model) : vehicle_model_(vehicle_model)
{
}

StateInfo SimpleVehicleHMKF::predict(const StateInfo& state_info,
                                     const Eigen::Vector2d & control_inputs,
                                     const double dt,
                                     const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map,
                                     std::shared_ptr<SimpleVehicleModel::HighOrderMoments>& high_order_moments)
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
    const double cPow3 = dist.calc_cos_moment(STATE::IDX::YAW, 3);
    const double sPow3 = dist.calc_sin_moment(STATE::IDX::YAW, 3);
    const double cPow2_xPow1 = dist.calc_x_cos_z_cos_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    const double sPow2_xPow1 = dist.calc_x_sin_z_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    const double cPow2_yPow1 = dist.calc_x_cos_z_cos_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    const double sPow2_yPow1 = dist.calc_x_sin_z_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    const double cPow1_sPow1_xPow1 = dist.calc_x_cos_z_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    const double cPow1_sPow1_yPow1 = dist.calc_x_cos_z_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    const double cPow1_xPow2 = dist.calc_xx_cos_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    const double sPow1_xPow2 = dist.calc_xx_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    const double cPow1_yPow2 = dist.calc_xx_cos_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    const double sPow1_yPow2 = dist.calc_xx_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    const double cPow1_xPow1_yPow1 = dist.calc_xy_cos_z_moment();
    const double sPow1_xPow1_yPow1 = dist.calc_xy_sin_z_moment();
    const double cPow1_sPow1_yawPow1 = dist.calc_x_cos_x_sin_x_moment(STATE::IDX::YAW, 1, 1, 1);
    const double cPow2_yawPow1 = dist.calc_x_cos_x_moment(STATE::IDX::YAW, 1, 2);
    const double sPow2_yawPow1 = dist.calc_x_sin_x_moment(STATE::IDX::YAW, 1, 2);
    const double xPow1_cPow1_yawPow1 = dist.calc_xy_cos_y_moment(STATE::IDX::X, STATE::IDX::YAW);
    const double xPow1_sPow1_yawPow1 = dist.calc_xy_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW);
    const double yPow1_cPow1_yawPow1 = dist.calc_xy_cos_y_moment(STATE::IDX::Y, STATE::IDX::YAW);
    const double yPow1_sPow1_yawPow1 = dist.calc_xy_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW);
    const double cPow2_sPow1 = dist.calc_cos_sin_moment(STATE::IDX::YAW, 2, 1);
    const double cPow1_sPow2 = dist.calc_cos_sin_moment(STATE::IDX::YAW, 1, 2);

    const double cPow4 = dist.calc_cos_moment(STATE::IDX::YAW, 4);
    const double sPow4 = dist.calc_sin_moment(STATE::IDX::YAW, 4);
    const double cPow3_xPow1 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW, 1, 0, 3, 0);
    const double cPow3_yPow1 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW, 1, 0, 3, 0);
    const double sPow3_xPow1 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW, 1, 0, 0, 3);
    const double sPow3_yPow1 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW, 1, 0, 0, 3);
    const double cPow2_xPow2 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW, 2, 0, 2, 0);
    const double cPow2_yPow2 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW, 2, 0, 2, 0);
    const double sPow2_xPow2 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW, 2, 0, 0, 2);
    const double sPow2_yPow2 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW, 2, 0, 0, 2);
    const double cPow1_sPow1_xPow2 = dist.calc_xx_cos_z_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    const double cPow1_sPow1_yPow2 = dist.calc_xx_cos_z_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    const double cPow1_sPow2_xPow1 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW, 1, 0, 1, 2);
    const double cPow1_sPow2_yPow1 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW, 1, 0, 1, 2);
    const double cPow2_sPow1_xPow1 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW, 1, 0, 2, 1);
    const double cPow2_sPow1_yPow1 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW, 1, 0, 2, 1);
    const double cPow1_sPow1_xPow1_yPow1 = dist.calc_xy_cos_z_sin_z_moment();
    const double cPow2_xPow1_yPow1 = dist.calc_xy_cos_z_cos_z_moment();
    const double sPow2_xPow1_yPow1 = dist.calc_xy_sin_z_sin_z_moment();
    const double cPow3_sPow1 = dist.calc_cos_sin_moment(STATE::IDX::YAW, 3, 1);
    const double cPow1_sPow3 = dist.calc_cos_sin_moment(STATE::IDX::YAW, 1, 3);
    const double cPow2_sPow2 = dist.calc_cos_sin_moment(STATE::IDX::YAW, 2, 2);

    // Step3. Control Input
    const double& v = control_inputs(INPUT::IDX::V);
    const double& u = control_inputs(INPUT::IDX::U);
    const double& cu = std::cos(u);
    const double& su = std::sin(u);

    // Step4. System Noise
    const auto wv_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WV);
    const auto wu_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WU);
    const double wvPow1 = wv_dist_ptr->calc_moment(1);
    const double wvPow2 = wv_dist_ptr->calc_moment(2);
    const double wuPow1 = wu_dist_ptr->calc_moment(1);
    const double wuPow2 = wu_dist_ptr->calc_moment(2);
    const double cwuPow1 = wu_dist_ptr->calc_cos_moment(1);
    const double swuPow1 = wu_dist_ptr->calc_sin_moment(1);
    const double swuPow2 = wu_dist_ptr->calc_sin_moment(2);
    const double cwuPow2 = wu_dist_ptr->calc_cos_moment(2);
    const double cwuPow1_swuPow1 = wu_dist_ptr->calc_cos_sin_moment(1, 1);
    const double wuPow1_cwuPow1 = wu_dist_ptr->calc_x_cos_moment(1, 1);
    const double wuPow1_swuPow1 = wu_dist_ptr->calc_x_sin_moment(1, 1);

    SimpleVehicleModel::HighOrderMoments result;
    result.xPow1 = v*cPow1 + cPow1*wvPow1 + xPow1;
    result.yPow1 = v*sPow1 + sPow1*wvPow1 + yPow1;
    result.yawPow1 = u + yawPow1 + wuPow1;
    result.cPow1 = cu*cPow1*cwuPow1 - cu*sPow1*swuPow1 - su*cPow1*swuPow1 - su*cwuPow1*sPow1;
    result.sPow1 = cu*cPow1*swuPow1 + cu*cwuPow1*sPow1 + su*cPow1*cwuPow1 - su*sPow1*swuPow1;

    result.xPow2 = pow(v, 2)*cPow2 + 2*v*cPow1_xPow1 + 2*v*cPow2*wvPow1 + 2*cPow1_xPow1*wvPow1 + cPow2*wvPow2 + xPow2;
    result.yPow2 = pow(v, 2)*sPow2 + 2*v*sPow1_yPow1 + 2*v*sPow2*wvPow1 + 2*sPow1_yPow1*wvPow1 + sPow2*wvPow2 + yPow2;
    result.yawPow2 = pow(u, 2) + 2*u*yawPow1 + 2*u*wuPow1 + 2*yawPow1*wuPow1 + yawPow2 + wuPow2;
    result.cPow2 = pow(cu, 2)*cPow2*cwuPow2 + pow(cu, 2)*sPow2*swuPow2 - 2*cu*su*cPow1_sPow1*cwuPow2 + 2*cu*su*cPow1_sPow1*swuPow2 - 2*cu*su*cPow2*cwuPow1_swuPow1 + 2*cu*su*cwuPow1_swuPow1*sPow2 + pow(su, 2)*cPow2*swuPow2 + pow(su, 2)*cwuPow2*sPow2 + cPow1_sPow1*cwuPow1_swuPow1*(-2*pow(cu, 2) + 2*pow(su, 2));
    result.sPow2 = pow(cu, 2)*cPow2*swuPow2 + pow(cu, 2)*cwuPow2*sPow2 + 2*cu*su*cPow1_sPow1*cwuPow2 - 2*cu*su*cPow1_sPow1*swuPow2 + 2*cu*su*cPow2*cwuPow1_swuPow1 - 2*cu*su*cwuPow1_swuPow1*sPow2 + pow(su, 2)*cPow2*cwuPow2 + pow(su, 2)*sPow2*swuPow2 + cPow1_sPow1*cwuPow1_swuPow1*(2*pow(cu, 2) - 2*pow(su, 2));
    result.xPow1_yPow1 = pow(v, 2)*cPow1_sPow1 + 2*v*cPow1_sPow1*wvPow1 + v*cPow1_yPow1 + v*sPow1_xPow1 + cPow1_sPow1*wvPow2 + cPow1_yPow1*wvPow1 + sPow1_xPow1*wvPow1 + xPow1_yPow1;
    result.xPow1_yawPow1 = u*v*cPow1 + u*cPow1*wvPow1 + u*xPow1 + v*cPow1*wuPow1 + v*cPow1_yawPow1 + cPow1*wuPow1*wvPow1 + cPow1_yawPow1*wvPow1 + xPow1_yawPow1 + wuPow1*xPow1;
    result.yPow1_yawPow1 = u*v*sPow1 + u*sPow1*wvPow1 + u*yPow1 + v*sPow1*wuPow1 + v*sPow1_yawPow1 + sPow1*wuPow1*wvPow1 + sPow1_yawPow1*wvPow1 + yPow1_yawPow1 + wuPow1*yPow1;
    result.xPow1_cPow1 = -cu*v*cPow1_sPow1*swuPow1 + cu*v*cPow2*cwuPow1 - cu*cPow1_sPow1*swuPow1*wvPow1 + cu*cPow1_xPow1*cwuPow1 + cu*cPow2*cwuPow1*wvPow1 - cu*sPow1_xPow1*swuPow1 - su*v*cPow1_sPow1*cwuPow1 - su*v*cPow2*swuPow1 - su*cPow1_sPow1*cwuPow1*wvPow1 - su*cPow1_xPow1*swuPow1 - su*cPow2*swuPow1*wvPow1 - su*cwuPow1*sPow1_xPow1;
    result.yPow1_cPow1 = cu*v*cPow1_sPow1*cwuPow1 - cu*v*sPow2*swuPow1 + cu*cPow1_sPow1*cwuPow1*wvPow1 + cu*cPow1_yPow1*cwuPow1 - cu*sPow1_yPow1*swuPow1 - cu*sPow2*swuPow1*wvPow1 - su*v*cPow1_sPow1*swuPow1 - su*v*cwuPow1*sPow2 - su*cPow1_sPow1*swuPow1*wvPow1 - su*cPow1_yPow1*swuPow1 - su*cwuPow1*sPow1_yPow1 - su*cwuPow1*sPow2*wvPow1;
    result.xPow1_sPow1 = cu*v*cPow1_sPow1*cwuPow1 + cu*v*cPow2*swuPow1 + cu*cPow1_sPow1*cwuPow1*wvPow1 + cu*cPow1_xPow1*swuPow1 + cu*cPow2*swuPow1*wvPow1 + cu*cwuPow1*sPow1_xPow1 - su*v*cPow1_sPow1*swuPow1 + su*v*cPow2*cwuPow1 - su*cPow1_sPow1*swuPow1*wvPow1 + su*cPow1_xPow1*cwuPow1 + su*cPow2*cwuPow1*wvPow1 - su*sPow1_xPow1*swuPow1;
    result.yPow1_sPow1 = cu*v*cPow1_sPow1*swuPow1 + cu*v*cwuPow1*sPow2 + cu*cPow1_sPow1*swuPow1*wvPow1 + cu*cPow1_yPow1*swuPow1 + cu*cwuPow1*sPow1_yPow1 + cu*cwuPow1*sPow2*wvPow1 + su*v*cPow1_sPow1*cwuPow1 - su*v*sPow2*swuPow1 + su*cPow1_sPow1*cwuPow1*wvPow1 + su*cPow1_yPow1*cwuPow1 - su*sPow1_yPow1*swuPow1 - su*sPow2*swuPow1*wvPow1;
    result.cPow1_sPow1 = -4*cu*su*cPow1_sPow1*cwuPow1_swuPow1 + cu*su*cPow2*cwuPow2 - cu*su*cPow2*swuPow2 - cu*su*cwuPow2*sPow2 + cu*su*sPow2*swuPow2 + cPow1_sPow1*cwuPow2*(pow(cu, 2) - pow(su, 2)) + cPow1_sPow1*swuPow2*(-pow(cu, 2) + pow(su, 2)) + cPow2*cwuPow1_swuPow1*(pow(cu, 2) - pow(su, 2)) + cwuPow1_swuPow1*sPow2*(-pow(cu, 2) + pow(su, 2));
    result.yawPow1_cPow1 = cu*u*cPow1*cwuPow1 - cu*u*sPow1*swuPow1 + cu*cPow1*wuPow1_cwuPow1+ cu*cPow1_yawPow1*cwuPow1 - cu*sPow1*wuPow1_swuPow1- cu*sPow1_yawPow1*swuPow1 - su*u*cPow1*swuPow1 - su*u*cwuPow1*sPow1 - su*cPow1*wuPow1_swuPow1 - su*cPow1_yawPow1*swuPow1 - su*wuPow1_cwuPow1*sPow1 - su*cwuPow1*sPow1_yawPow1;
    result.yawPow1_sPow1 = cu*u*cPow1*swuPow1 + cu*u*cwuPow1*sPow1 + cu*cPow1*wuPow1_swuPow1 + cu*cPow1_yawPow1*swuPow1 + cu*wuPow1_cwuPow1*sPow1 + cu*cwuPow1*sPow1_yawPow1 + su*u*cPow1*cwuPow1 - su*u*sPow1*swuPow1 + su*cPow1*wuPow1_cwuPow1 + su*cPow1_yawPow1*cwuPow1 - su*sPow1*wuPow1_swuPow1 - su*sPow1_yawPow1*swuPow1;

    result.xPow1_cPow2 = pow(cu, 2)*v*cPow1_sPow2*swuPow2 + pow(cu, 2)*v*cPow3*cwuPow2 + pow(cu, 2)*cPow1_sPow2*swuPow2*wvPow1 + pow(cu, 2)*cPow2_xPow1*cwuPow2 + pow(cu, 2)*cPow3*cwuPow2*wvPow1 + pow(cu, 2)*sPow2_xPow1*swuPow2 + 2*cu*su*v*cPow1_sPow2*cwuPow1_swuPow1 - 2*cu*su*v*cPow2_sPow1*cwuPow2 + 2*cu*su*v*cPow2_sPow1*swuPow2 - 2*cu*su*v*cPow3*cwuPow1_swuPow1 - 2*cu*su*cPow1_sPow1_xPow1*cwuPow2 + 2*cu*su*cPow1_sPow1_xPow1*swuPow2 + 2*cu*su*cPow1_sPow2*cwuPow1_swuPow1*wvPow1 - 2*cu*su*cPow2_sPow1*cwuPow2*wvPow1 + 2*cu*su*cPow2_sPow1*swuPow2*wvPow1 - 2*cu*su*cPow2_xPow1*cwuPow1_swuPow1 - 2*cu*su*cPow3*cwuPow1_swuPow1*wvPow1 + 2*cu*su*cwuPow1_swuPow1*sPow2_xPow1 + pow(su, 2)*v*cPow1_sPow2*cwuPow2 + pow(su, 2)*v*cPow3*swuPow2 + pow(su, 2)*cPow1_sPow2*cwuPow2*wvPow1 + pow(su, 2)*cPow2_xPow1*swuPow2 + pow(su, 2)*cPow3*swuPow2*wvPow1 + pow(su, 2)*cwuPow2*sPow2_xPow1 + cPow1_sPow1_xPow1*cwuPow1_swuPow1*(-2*pow(cu, 2) + 2*pow(su, 2)) + cPow2_sPow1*cwuPow1_swuPow1*wvPow1*(-2*pow(cu, 2) + 2*pow(su, 2)) + cPow2_sPow1*cwuPow1_swuPow1*(-2*pow(cu, 2)*v + 2*pow(su, 2)*v);
    result.yPow1_cPow2 = pow(cu, 2)*v*cPow2_sPow1*cwuPow2 + pow(cu, 2)*v*sPow3*swuPow2 + pow(cu, 2)*cPow2_sPow1*cwuPow2*wvPow1 + pow(cu, 2)*cPow2_yPow1*cwuPow2 + pow(cu, 2)*sPow2_yPow1*swuPow2 + pow(cu, 2)*sPow3*swuPow2*wvPow1 - 2*cu*su*v*cPow1_sPow2*cwuPow2 + 2*cu*su*v*cPow1_sPow2*swuPow2 - 2*cu*su*v*cPow2_sPow1*cwuPow1_swuPow1 + 2*cu*su*v*cwuPow1_swuPow1*sPow3 - 2*cu*su*cPow1_sPow1_yPow1*cwuPow2 + 2*cu*su*cPow1_sPow1_yPow1*swuPow2 - 2*cu*su*cPow1_sPow2*cwuPow2*wvPow1 + 2*cu*su*cPow1_sPow2*swuPow2*wvPow1 - 2*cu*su*cPow2_sPow1*cwuPow1_swuPow1*wvPow1 - 2*cu*su*cPow2_yPow1*cwuPow1_swuPow1 + 2*cu*su*cwuPow1_swuPow1*sPow2_yPow1 + 2*cu*su*cwuPow1_swuPow1*sPow3*wvPow1 + pow(su, 2)*v*cPow2_sPow1*swuPow2 + pow(su, 2)*v*cwuPow2*sPow3 + pow(su, 2)*cPow2_sPow1*swuPow2*wvPow1 + pow(su, 2)*cPow2_yPow1*swuPow2 + pow(su, 2)*cwuPow2*sPow2_yPow1 + pow(su, 2)*cwuPow2*sPow3*wvPow1 + cPow1_sPow1_yPow1*cwuPow1_swuPow1*(-2*pow(cu, 2) + 2*pow(su, 2)) + cPow1_sPow2*cwuPow1_swuPow1*wvPow1*(-2*pow(cu, 2) + 2*pow(su, 2)) + cPow1_sPow2*cwuPow1_swuPow1*(-2*pow(cu, 2)*v + 2*pow(su, 2)*v);
    result.xPow1_sPow2 = pow(cu, 2)*v*cPow1_sPow2*cwuPow2 + pow(cu, 2)*v*cPow3*swuPow2 + pow(cu, 2)*cPow1_sPow2*cwuPow2*wvPow1 + pow(cu, 2)*cPow2_xPow1*swuPow2 + pow(cu, 2)*cPow3*swuPow2*wvPow1 + pow(cu, 2)*cwuPow2*sPow2_xPow1 - 2*cu*su*v*cPow1_sPow2*cwuPow1_swuPow1 + 2*cu*su*v*cPow2_sPow1*cwuPow2 - 2*cu*su*v*cPow2_sPow1*swuPow2 + 2*cu*su*v*cPow3*cwuPow1_swuPow1 + 2*cu*su*cPow1_sPow1_xPow1*cwuPow2 - 2*cu*su*cPow1_sPow1_xPow1*swuPow2 - 2*cu*su*cPow1_sPow2*cwuPow1_swuPow1*wvPow1 + 2*cu*su*cPow2_sPow1*cwuPow2*wvPow1 - 2*cu*su*cPow2_sPow1*swuPow2*wvPow1 + 2*cu*su*cPow2_xPow1*cwuPow1_swuPow1 + 2*cu*su*cPow3*cwuPow1_swuPow1*wvPow1 - 2*cu*su*cwuPow1_swuPow1*sPow2_xPow1 + pow(su, 2)*v*cPow1_sPow2*swuPow2 + pow(su, 2)*v*cPow3*cwuPow2 + pow(su, 2)*cPow1_sPow2*swuPow2*wvPow1 + pow(su, 2)*cPow2_xPow1*cwuPow2 + pow(su, 2)*cPow3*cwuPow2*wvPow1 + pow(su, 2)*sPow2_xPow1*swuPow2 + cPow1_sPow1_xPow1*cwuPow1_swuPow1*(2*pow(cu, 2) - 2*pow(su, 2)) + cPow2_sPow1*cwuPow1_swuPow1*wvPow1*(2*pow(cu, 2) - 2*pow(su, 2)) + cPow2_sPow1*cwuPow1_swuPow1*(2*pow(cu, 2)*v - 2*pow(su, 2)*v);
    result.yPow1_sPow2 = pow(cu, 2)*v*cPow2_sPow1*swuPow2 + pow(cu, 2)*v*cwuPow2*sPow3 + pow(cu, 2)*cPow2_sPow1*swuPow2*wvPow1 + pow(cu, 2)*cPow2_yPow1*swuPow2 + pow(cu, 2)*cwuPow2*sPow2_yPow1 + pow(cu, 2)*cwuPow2*sPow3*wvPow1 + 2*cu*su*v*cPow1_sPow2*cwuPow2 - 2*cu*su*v*cPow1_sPow2*swuPow2 + 2*cu*su*v*cPow2_sPow1*cwuPow1_swuPow1 - 2*cu*su*v*cwuPow1_swuPow1*sPow3 + 2*cu*su*cPow1_sPow1_yPow1*cwuPow2 - 2*cu*su*cPow1_sPow1_yPow1*swuPow2 + 2*cu*su*cPow1_sPow2*cwuPow2*wvPow1 - 2*cu*su*cPow1_sPow2*swuPow2*wvPow1 + 2*cu*su*cPow2_sPow1*cwuPow1_swuPow1*wvPow1 + 2*cu*su*cPow2_yPow1*cwuPow1_swuPow1 - 2*cu*su*cwuPow1_swuPow1*sPow2_yPow1 - 2*cu*su*cwuPow1_swuPow1*sPow3*wvPow1 + pow(su, 2)*v*cPow2_sPow1*cwuPow2 + pow(su, 2)*v*sPow3*swuPow2 + pow(su, 2)*cPow2_sPow1*cwuPow2*wvPow1 + pow(su, 2)*cPow2_yPow1*cwuPow2 + pow(su, 2)*sPow2_yPow1*swuPow2 + pow(su, 2)*sPow3*swuPow2*wvPow1 + cPow1_sPow1_yPow1*cwuPow1_swuPow1*(2*pow(cu, 2) - 2*pow(su, 2)) + cPow1_sPow2*cwuPow1_swuPow1*wvPow1*(2*pow(cu, 2) - 2*pow(su, 2)) + cPow1_sPow2*cwuPow1_swuPow1*(2*pow(cu, 2)*v - 2*pow(su, 2)*v);
    result.xPow2_cPow1 = -cu*pow(v, 2)*cPow2_sPow1*swuPow1 + cu*pow(v, 2)*cPow3*cwuPow1 - 2*cu*v*cPow1_sPow1_xPow1*swuPow1 - 2*cu*v*cPow2_sPow1*swuPow1*wvPow1 + 2*cu*v*cPow2_xPow1*cwuPow1 + 2*cu*v*cPow3*cwuPow1*wvPow1 - 2*cu*cPow1_sPow1_xPow1*swuPow1*wvPow1 + cu*cPow1_xPow2*cwuPow1 - cu*cPow2_sPow1*swuPow1*wvPow2 + 2*cu*cPow2_xPow1*cwuPow1*wvPow1 + cu*cPow3*cwuPow1*wvPow2 - cu*sPow1_xPow2*swuPow1 - su*pow(v, 2)*cPow2_sPow1*cwuPow1 - su*pow(v, 2)*cPow3*swuPow1 - 2*su*v*cPow1_sPow1_xPow1*cwuPow1 - 2*su*v*cPow2_sPow1*cwuPow1*wvPow1 - 2*su*v*cPow2_xPow1*swuPow1 - 2*su*v*cPow3*swuPow1*wvPow1 - 2*su*cPow1_sPow1_xPow1*cwuPow1*wvPow1 - su*cPow1_xPow2*swuPow1 - su*cPow2_sPow1*cwuPow1*wvPow2 - 2*su*cPow2_xPow1*swuPow1*wvPow1 - su*cPow3*swuPow1*wvPow2 - su*cwuPow1*sPow1_xPow2;
    result.yPow2_cPow1 = cu*pow(v, 2)*cPow1_sPow2*cwuPow1 - cu*pow(v, 2)*sPow3*swuPow1 + 2*cu*v*cPow1_sPow1_yPow1*cwuPow1 + 2*cu*v*cPow1_sPow2*cwuPow1*wvPow1 - 2*cu*v*sPow2_yPow1*swuPow1 - 2*cu*v*sPow3*swuPow1*wvPow1 + 2*cu*cPow1_sPow1_yPow1*cwuPow1*wvPow1 + cu*cPow1_sPow2*cwuPow1*wvPow2 + cu*cPow1_yPow2*cwuPow1 - cu*sPow1_yPow2*swuPow1 - 2*cu*sPow2_yPow1*swuPow1*wvPow1 - cu*sPow3*swuPow1*wvPow2 - su*pow(v, 2)*cPow1_sPow2*swuPow1 - su*pow(v, 2)*cwuPow1*sPow3 - 2*su*v*cPow1_sPow1_yPow1*swuPow1 - 2*su*v*cPow1_sPow2*swuPow1*wvPow1 - 2*su*v*cwuPow1*sPow2_yPow1 - 2*su*v*cwuPow1*sPow3*wvPow1 - 2*su*cPow1_sPow1_yPow1*swuPow1*wvPow1 - su*cPow1_sPow2*swuPow1*wvPow2 - su*cPow1_yPow2*swuPow1 - su*cwuPow1*sPow1_yPow2 - 2*su*cwuPow1*sPow2_yPow1*wvPow1 - su*cwuPow1*sPow3*wvPow2;
    result.xPow2_sPow1 = cu*pow(v, 2)*cPow2_sPow1*cwuPow1 + cu*pow(v, 2)*cPow3*swuPow1 + 2*cu*v*cPow1_sPow1_xPow1*cwuPow1 + 2*cu*v*cPow2_sPow1*cwuPow1*wvPow1 + 2*cu*v*cPow2_xPow1*swuPow1 + 2*cu*v*cPow3*swuPow1*wvPow1 + 2*cu*cPow1_sPow1_xPow1*cwuPow1*wvPow1 + cu*cPow1_xPow2*swuPow1 + cu*cPow2_sPow1*cwuPow1*wvPow2 + 2*cu*cPow2_xPow1*swuPow1*wvPow1 + cu*cPow3*swuPow1*wvPow2 + cu*cwuPow1*sPow1_xPow2 - su*pow(v, 2)*cPow2_sPow1*swuPow1 + su*pow(v, 2)*cPow3*cwuPow1 - 2*su*v*cPow1_sPow1_xPow1*swuPow1 - 2*su*v*cPow2_sPow1*swuPow1*wvPow1 + 2*su*v*cPow2_xPow1*cwuPow1 + 2*su*v*cPow3*cwuPow1*wvPow1 - 2*su*cPow1_sPow1_xPow1*swuPow1*wvPow1 + su*cPow1_xPow2*cwuPow1 - su*cPow2_sPow1*swuPow1*wvPow2 + 2*su*cPow2_xPow1*cwuPow1*wvPow1 + su*cPow3*cwuPow1*wvPow2 - su*sPow1_xPow2*swuPow1;
    result.yPow2_sPow1 = cu*pow(v, 2)*cPow1_sPow2*swuPow1 + cu*pow(v, 2)*cwuPow1*sPow3 + 2*cu*v*cPow1_sPow1_yPow1*swuPow1 + 2*cu*v*cPow1_sPow2*swuPow1*wvPow1 + 2*cu*v*cwuPow1*sPow2_yPow1 + 2*cu*v*cwuPow1*sPow3*wvPow1 + 2*cu*cPow1_sPow1_yPow1*swuPow1*wvPow1 + cu*cPow1_sPow2*swuPow1*wvPow2 + cu*cPow1_yPow2*swuPow1 + cu*cwuPow1*sPow1_yPow2 + 2*cu*cwuPow1*sPow2_yPow1*wvPow1 + cu*cwuPow1*sPow3*wvPow2 + su*pow(v, 2)*cPow1_sPow2*cwuPow1 - su*pow(v, 2)*sPow3*swuPow1 + 2*su*v*cPow1_sPow1_yPow1*cwuPow1 + 2*su*v*cPow1_sPow2*cwuPow1*wvPow1 - 2*su*v*sPow2_yPow1*swuPow1 - 2*su*v*sPow3*swuPow1*wvPow1 + 2*su*cPow1_sPow1_yPow1*cwuPow1*wvPow1 + su*cPow1_sPow2*cwuPow1*wvPow2 + su*cPow1_yPow2*cwuPow1 - su*sPow1_yPow2*swuPow1 - 2*su*sPow2_yPow1*swuPow1*wvPow1 - su*sPow3*swuPow1*wvPow2;
    result.xPow1_yPow1_cPow1 = -cu*pow(v, 2)*cPow1_sPow2*swuPow1 + cu*pow(v, 2)*cPow2_sPow1*cwuPow1 + cu*v*cPow1_sPow1_xPow1*cwuPow1 - cu*v*cPow1_sPow1_yPow1*swuPow1 - 2*cu*v*cPow1_sPow2*swuPow1*wvPow1 + 2*cu*v*cPow2_sPow1*cwuPow1*wvPow1 + cu*v*cPow2_yPow1*cwuPow1 - cu*v*sPow2_xPow1*swuPow1 + cu*cPow1_sPow1_xPow1*cwuPow1*wvPow1 - cu*cPow1_sPow1_yPow1*swuPow1*wvPow1 - cu*cPow1_sPow2*swuPow1*wvPow2 + cu*cPow1_xPow1_yPow1*cwuPow1 + cu*cPow2_sPow1*cwuPow1*wvPow2 + cu*cPow2_yPow1*cwuPow1*wvPow1 - cu*sPow1_xPow1_yPow1*swuPow1 - cu*sPow2_xPow1*swuPow1*wvPow1 - su*pow(v, 2)*cPow1_sPow2*cwuPow1 - su*pow(v, 2)*cPow2_sPow1*swuPow1 - su*v*cPow1_sPow1_xPow1*swuPow1 - su*v*cPow1_sPow1_yPow1*cwuPow1 - 2*su*v*cPow1_sPow2*cwuPow1*wvPow1 - 2*su*v*cPow2_sPow1*swuPow1*wvPow1 - su*v*cPow2_yPow1*swuPow1 - su*v*cwuPow1*sPow2_xPow1 - su*cPow1_sPow1_xPow1*swuPow1*wvPow1 - su*cPow1_sPow1_yPow1*cwuPow1*wvPow1 - su*cPow1_sPow2*cwuPow1*wvPow2 - su*cPow1_xPow1_yPow1*swuPow1 - su*cPow2_sPow1*swuPow1*wvPow2 - su*cPow2_yPow1*swuPow1*wvPow1 - su*cwuPow1*sPow1_xPow1_yPow1 - su*cwuPow1*sPow2_xPow1*wvPow1;
    result.xPow1_yPow1_sPow1 = cu*pow(v, 2)*cPow1_sPow2*cwuPow1 + cu*pow(v, 2)*cPow2_sPow1*swuPow1 + cu*v*cPow1_sPow1_xPow1*swuPow1 + cu*v*cPow1_sPow1_yPow1*cwuPow1 + 2*cu*v*cPow1_sPow2*cwuPow1*wvPow1 + 2*cu*v*cPow2_sPow1*swuPow1*wvPow1 + cu*v*cPow2_yPow1*swuPow1 + cu*v*cwuPow1*sPow2_xPow1 + cu*cPow1_sPow1_xPow1*swuPow1*wvPow1 + cu*cPow1_sPow1_yPow1*cwuPow1*wvPow1 + cu*cPow1_sPow2*cwuPow1*wvPow2 + cu*cPow1_xPow1_yPow1*swuPow1 + cu*cPow2_sPow1*swuPow1*wvPow2 + cu*cPow2_yPow1*swuPow1*wvPow1 + cu*cwuPow1*sPow1_xPow1_yPow1 + cu*cwuPow1*sPow2_xPow1*wvPow1 - su*pow(v, 2)*cPow1_sPow2*swuPow1 + su*pow(v, 2)*cPow2_sPow1*cwuPow1 + su*v*cPow1_sPow1_xPow1*cwuPow1 - su*v*cPow1_sPow1_yPow1*swuPow1 - 2*su*v*cPow1_sPow2*swuPow1*wvPow1 + 2*su*v*cPow2_sPow1*cwuPow1*wvPow1 + su*v*cPow2_yPow1*cwuPow1 - su*v*sPow2_xPow1*swuPow1 + su*cPow1_sPow1_xPow1*cwuPow1*wvPow1 - su*cPow1_sPow1_yPow1*swuPow1*wvPow1 - su*cPow1_sPow2*swuPow1*wvPow2 + su*cPow1_xPow1_yPow1*cwuPow1 + su*cPow2_sPow1*cwuPow1*wvPow2 + su*cPow2_yPow1*cwuPow1*wvPow1 - su*sPow1_xPow1_yPow1*swuPow1 - su*sPow2_xPow1*swuPow1*wvPow1;
    result.xPow1_cPow1_sPow1 = -cu*su*v*cPow1_sPow2*cwuPow2 + cu*su*v*cPow1_sPow2*swuPow2 - 4*cu*su*v*cPow2_sPow1*cwuPow1_swuPow1 + cu*su*v*cPow3*cwuPow2 - cu*su*v*cPow3*swuPow2 - 4*cu*su*cPow1_sPow1_xPow1*cwuPow1_swuPow1 - cu*su*cPow1_sPow2*cwuPow2*wvPow1 + cu*su*cPow1_sPow2*swuPow2*wvPow1 - 4*cu*su*cPow2_sPow1*cwuPow1_swuPow1*wvPow1 + cu*su*cPow2_xPow1*cwuPow2 - cu*su*cPow2_xPow1*swuPow2 + cu*su*cPow3*cwuPow2*wvPow1 - cu*su*cPow3*swuPow2*wvPow1 - cu*su*cwuPow2*sPow2_xPow1 + cu*su*sPow2_xPow1*swuPow2 + cPow1_sPow1_xPow1*cwuPow2*(pow(cu, 2) - pow(su, 2)) + cPow1_sPow1_xPow1*swuPow2*(-pow(cu, 2) + pow(su, 2)) + cPow1_sPow2*cwuPow1_swuPow1*wvPow1*(-pow(cu, 2) + pow(su, 2)) + cPow1_sPow2*cwuPow1_swuPow1*(-pow(cu, 2)*v + pow(su, 2)*v) + cPow2_sPow1*cwuPow2*wvPow1*(pow(cu, 2) - pow(su, 2)) + cPow2_sPow1*cwuPow2*(pow(cu, 2)*v - pow(su, 2)*v) + cPow2_sPow1*swuPow2*wvPow1*(-pow(cu, 2) + pow(su, 2)) + cPow2_sPow1*swuPow2*(-pow(cu, 2)*v + pow(su, 2)*v) + cPow2_xPow1*cwuPow1_swuPow1*(pow(cu, 2) - pow(su, 2)) + cPow3*cwuPow1_swuPow1*wvPow1*(pow(cu, 2) - pow(su, 2)) + cPow3*cwuPow1_swuPow1*(pow(cu, 2)*v - pow(su, 2)*v) + cwuPow1_swuPow1*sPow2_xPow1*(-pow(cu, 2) + pow(su, 2));
    result.yPow1_cPow1_sPow1 = -4*cu*su*v*cPow1_sPow2*cwuPow1_swuPow1 + cu*su*v*cPow2_sPow1*cwuPow2 - cu*su*v*cPow2_sPow1*swuPow2 - cu*su*v*cwuPow2*sPow3 + cu*su*v*sPow3*swuPow2 - 4*cu*su*cPow1_sPow1_yPow1*cwuPow1_swuPow1 - 4*cu*su*cPow1_sPow2*cwuPow1_swuPow1*wvPow1 + cu*su*cPow2_sPow1*cwuPow2*wvPow1 - cu*su*cPow2_sPow1*swuPow2*wvPow1 + cu*su*cPow2_yPow1*cwuPow2 - cu*su*cPow2_yPow1*swuPow2 - cu*su*cwuPow2*sPow2_yPow1 - cu*su*cwuPow2*sPow3*wvPow1 + cu*su*sPow2_yPow1*swuPow2 + cu*su*sPow3*swuPow2*wvPow1 + cPow1_sPow1_yPow1*cwuPow2*(pow(cu, 2) - pow(su, 2)) + cPow1_sPow1_yPow1*swuPow2*(-pow(cu, 2) + pow(su, 2)) + cPow1_sPow2*cwuPow2*wvPow1*(pow(cu, 2) - pow(su, 2)) + cPow1_sPow2*cwuPow2*(pow(cu, 2)*v - pow(su, 2)*v) + cPow1_sPow2*swuPow2*wvPow1*(-pow(cu, 2) + pow(su, 2)) + cPow1_sPow2*swuPow2*(-pow(cu, 2)*v + pow(su, 2)*v) + cPow2_sPow1*cwuPow1_swuPow1*wvPow1*(pow(cu, 2) - pow(su, 2)) + cPow2_sPow1*cwuPow1_swuPow1*(pow(cu, 2)*v - pow(su, 2)*v) + cPow2_yPow1*cwuPow1_swuPow1*(pow(cu, 2) - pow(su, 2)) + cwuPow1_swuPow1*sPow2_yPow1*(-pow(cu, 2) + pow(su, 2)) + cwuPow1_swuPow1*sPow3*wvPow1*(-pow(cu, 2) + pow(su, 2)) + cwuPow1_swuPow1*sPow3*(-pow(cu, 2)*v + pow(su, 2)*v);
    result.xPow1_yawPow1_cPow1 = -cu*u*v*cPow1_sPow1*swuPow1 + cu*u*v*cPow2*cwuPow1 - cu*u*cPow1_sPow1*swuPow1*wvPow1 + cu*u*cPow1_xPow1*cwuPow1 + cu*u*cPow2*cwuPow1*wvPow1 - cu*u*sPow1_xPow1*swuPow1 - cu*v*cPow1_sPow1*wuPow1_swuPow1 - cu*v*cPow1_sPow1_yawPow1*swuPow1 + cu*v*cPow2*wuPow1_cwuPow1 + cu*v*cPow2_yawPow1*cwuPow1 - cu*cPow1_sPow1*wuPow1_swuPow1*wvPow1 - cu*cPow1_sPow1_yawPow1*swuPow1*wvPow1 + cu*cPow1_xPow1*wuPow1_cwuPow1 + cu*xPow1_cPow1_yawPow1*cwuPow1 + cu*cPow2*wuPow1_cwuPow1*wvPow1 + cu*cPow2_yawPow1*cwuPow1*wvPow1 - cu*sPow1_xPow1*wuPow1_swuPow1 - cu*xPow1_sPow1_yawPow1*swuPow1 - su*u*v*cPow1_sPow1*cwuPow1 - su*u*v*cPow2*swuPow1 - su*u*cPow1_sPow1*cwuPow1*wvPow1 - su*u*cPow1_xPow1*swuPow1 - su*u*cPow2*swuPow1*wvPow1 - su*u*cwuPow1*sPow1_xPow1 - su*v*cPow1_sPow1*wuPow1_cwuPow1 - su*v*cPow1_sPow1_yawPow1*cwuPow1 - su*v*cPow2*wuPow1_swuPow1 - su*v*cPow2_yawPow1*swuPow1 - su*cPow1_sPow1*wuPow1_cwuPow1*wvPow1 - su*cPow1_sPow1_yawPow1*cwuPow1*wvPow1 - su*cPow1_xPow1*wuPow1_swuPow1 - su*xPow1_cPow1_yawPow1*swuPow1 - su*cPow2*wuPow1_swuPow1*wvPow1 - su*cPow2_yawPow1*swuPow1*wvPow1 - su*cwuPow1*xPow1_sPow1_yawPow1 - su*wuPow1_cwuPow1*sPow1_xPow1;
    result.xPow1_yawPow1_sPow1 = cu*u*v*cPow1_sPow1*cwuPow1 + cu*u*v*cPow2*swuPow1 + cu*u*cPow1_sPow1*cwuPow1*wvPow1 + cu*u*cPow1_xPow1*swuPow1 + cu*u*cPow2*swuPow1*wvPow1 + cu*u*cwuPow1*sPow1_xPow1 + cu*v*cPow1_sPow1*wuPow1_cwuPow1 + cu*v*cPow1_sPow1_yawPow1*cwuPow1 + cu*v*cPow2*wuPow1_swuPow1 + cu*v*cPow2_yawPow1*swuPow1 + cu*cPow1_sPow1*wuPow1_cwuPow1*wvPow1 + cu*cPow1_sPow1_yawPow1*cwuPow1*wvPow1 + cu*cPow1_xPow1*wuPow1_swuPow1 + cu*xPow1_cPow1_yawPow1*swuPow1 + cu*cPow2*wuPow1_swuPow1*wvPow1 + cu*cPow2_yawPow1*swuPow1*wvPow1 + cu*cwuPow1*xPow1_sPow1_yawPow1 + cu*wuPow1_cwuPow1*sPow1_xPow1 - su*u*v*cPow1_sPow1*swuPow1 + su*u*v*cPow2*cwuPow1 - su*u*cPow1_sPow1*swuPow1*wvPow1 + su*u*cPow1_xPow1*cwuPow1 + su*u*cPow2*cwuPow1*wvPow1 - su*u*sPow1_xPow1*swuPow1 - su*v*cPow1_sPow1*wuPow1_swuPow1 - su*v*cPow1_sPow1_yawPow1*swuPow1 + su*v*cPow2*wuPow1_cwuPow1 + su*v*cPow2_yawPow1*cwuPow1 - su*cPow1_sPow1*wuPow1_swuPow1*wvPow1 - su*cPow1_sPow1_yawPow1*swuPow1*wvPow1 + su*cPow1_xPow1*wuPow1_cwuPow1 + su*xPow1_cPow1_yawPow1*cwuPow1 + su*cPow2*wuPow1_cwuPow1*wvPow1 + su*cPow2_yawPow1*cwuPow1*wvPow1 - su*sPow1_xPow1*wuPow1_swuPow1 - su*xPow1_sPow1_yawPow1*swuPow1;
    result.yPow1_yawPow1_cPow1 = cu*u*v*cPow1_sPow1*cwuPow1 - cu*u*v*sPow2*swuPow1 + cu*u*cPow1_sPow1*cwuPow1*wvPow1 + cu*u*cPow1_yPow1*cwuPow1 - cu*u*sPow1_yPow1*swuPow1 - cu*u*sPow2*swuPow1*wvPow1 + cu*v*cPow1_sPow1*wuPow1_cwuPow1 + cu*v*cPow1_sPow1_yawPow1*cwuPow1 - cu*v*sPow2*wuPow1_swuPow1 - cu*v*sPow2_yawPow1*swuPow1 + cu*cPow1_sPow1*wuPow1_cwuPow1*wvPow1 + cu*cPow1_sPow1_yawPow1*cwuPow1*wvPow1 + cu*cPow1_yPow1*wuPow1_cwuPow1 + cu*yPow1_cPow1_yawPow1*cwuPow1 - cu*sPow1_yPow1*wuPow1_swuPow1 - cu*yPow1_sPow1_yawPow1*swuPow1 - cu*sPow2*wuPow1_swuPow1*wvPow1 - cu*sPow2_yawPow1*swuPow1*wvPow1 - su*u*v*cPow1_sPow1*swuPow1 - su*u*v*cwuPow1*sPow2 - su*u*cPow1_sPow1*swuPow1*wvPow1 - su*u*cPow1_yPow1*swuPow1 - su*u*cwuPow1*sPow1_yPow1 - su*u*cwuPow1*sPow2*wvPow1 - su*v*cPow1_sPow1*wuPow1_swuPow1 - su*v*cPow1_sPow1_yawPow1*swuPow1 - su*v*cwuPow1*sPow2_yawPow1 - su*v*wuPow1_cwuPow1*sPow2 - su*cPow1_sPow1*wuPow1_swuPow1*wvPow1 - su*cPow1_sPow1_yawPow1*swuPow1*wvPow1 - su*cPow1_yPow1*wuPow1_swuPow1 - su*yPow1_cPow1_yawPow1*swuPow1 - su*cwuPow1*yPow1_sPow1_yawPow1 - su*cwuPow1*sPow2_yawPow1*wvPow1 - su*wuPow1_cwuPow1*sPow1_yPow1 - su*wuPow1_cwuPow1*sPow2*wvPow1;
    result.yPow1_yawPow1_sPow1 = cu*u*v*cPow1_sPow1*swuPow1 + cu*u*v*cwuPow1*sPow2 + cu*u*cPow1_sPow1*swuPow1*wvPow1 + cu*u*cPow1_yPow1*swuPow1 + cu*u*cwuPow1*sPow1_yPow1 + cu*u*cwuPow1*sPow2*wvPow1 + cu*v*cPow1_sPow1*wuPow1_swuPow1 + cu*v*cPow1_sPow1_yawPow1*swuPow1 + cu*v*cwuPow1*sPow2_yawPow1 + cu*v*wuPow1_cwuPow1*sPow2 + cu*cPow1_sPow1*wuPow1_swuPow1*wvPow1 + cu*cPow1_sPow1_yawPow1*swuPow1*wvPow1 + cu*cPow1_yPow1*wuPow1_swuPow1 + cu*yPow1_cPow1_yawPow1*swuPow1 + cu*cwuPow1*yPow1_sPow1_yawPow1 + cu*cwuPow1*sPow2_yawPow1*wvPow1 + cu*wuPow1_cwuPow1*sPow1_yPow1 + cu*wuPow1_cwuPow1*sPow2*wvPow1 + su*u*v*cPow1_sPow1*cwuPow1 - su*u*v*sPow2*swuPow1 + su*u*cPow1_sPow1*cwuPow1*wvPow1 + su*u*cPow1_yPow1*cwuPow1 - su*u*sPow1_yPow1*swuPow1 - su*u*sPow2*swuPow1*wvPow1 + su*v*cPow1_sPow1*wuPow1_cwuPow1 + su*v*cPow1_sPow1_yawPow1*cwuPow1 - su*v*sPow2*wuPow1_swuPow1 - su*v*sPow2_yawPow1*swuPow1 + su*cPow1_sPow1*wuPow1_cwuPow1*wvPow1 + su*cPow1_sPow1_yawPow1*cwuPow1*wvPow1 + su*cPow1_yPow1*wuPow1_cwuPow1 + su*yPow1_cPow1_yawPow1*cwuPow1 - su*sPow1_yPow1*wuPow1_swuPow1 - su*yPow1_sPow1_yawPow1*swuPow1 - su*sPow2*wuPow1_swuPow1*wvPow1 - su*sPow2_yawPow1*swuPow1*wvPow1;

    result.xPow2_cPow2 = pow(cu, 2)*pow(v, 2)*cPow2_sPow2*swuPow2 + pow(cu, 2)*pow(v, 2)*cPow4*cwuPow2 + 2*pow(cu, 2)*v*cPow1_sPow2_xPow1*swuPow2 + 2*pow(cu, 2)*v*cPow2_sPow2*swuPow2*wvPow1 + 2*pow(cu, 2)*v*cPow3_xPow1*cwuPow2 + 2*pow(cu, 2)*v*cPow4*cwuPow2*wvPow1 + 2*pow(cu, 2)*cPow1_sPow2_xPow1*swuPow2*wvPow1 + pow(cu, 2)*cPow2_sPow2*swuPow2*wvPow2 + pow(cu, 2)*cPow2_xPow2*cwuPow2 + 2*pow(cu, 2)*cPow3_xPow1*cwuPow2*wvPow1 + pow(cu, 2)*cPow4*cwuPow2*wvPow2 + pow(cu, 2)*sPow2_xPow2*swuPow2 + 2*cu*su*pow(v, 2)*cPow2_sPow2*cwuPow1_swuPow1 - 2*cu*su*pow(v, 2)*cPow3_sPow1*cwuPow2 + 2*cu*su*pow(v, 2)*cPow3_sPow1*swuPow2 - 2*cu*su*pow(v, 2)*cPow4*cwuPow1_swuPow1 + 4*cu*su*v*cPow1_sPow2_xPow1*cwuPow1_swuPow1 - 4*cu*su*v*cPow2_sPow1_xPow1*cwuPow2 + 4*cu*su*v*cPow2_sPow1_xPow1*swuPow2 + 4*cu*su*v*cPow2_sPow2*cwuPow1_swuPow1*wvPow1 - 4*cu*su*v*cPow3_sPow1*cwuPow2*wvPow1 + 4*cu*su*v*cPow3_sPow1*swuPow2*wvPow1 - 4*cu*su*v*cPow3_xPow1*cwuPow1_swuPow1 - 4*cu*su*v*cPow4*cwuPow1_swuPow1*wvPow1 - 2*cu*su*cPow1_sPow1_xPow2*cwuPow2 + 2*cu*su*cPow1_sPow1_xPow2*swuPow2 + 4*cu*su*cPow1_sPow2_xPow1*cwuPow1_swuPow1*wvPow1 - 4*cu*su*cPow2_sPow1_xPow1*cwuPow2*wvPow1 + 4*cu*su*cPow2_sPow1_xPow1*swuPow2*wvPow1 + 2*cu*su*cPow2_sPow2*cwuPow1_swuPow1*wvPow2 - 2*cu*su*cPow2_xPow2*cwuPow1_swuPow1 - 2*cu*su*cPow3_sPow1*cwuPow2*wvPow2 + 2*cu*su*cPow3_sPow1*swuPow2*wvPow2 - 4*cu*su*cPow3_xPow1*cwuPow1_swuPow1*wvPow1 - 2*cu*su*cPow4*cwuPow1_swuPow1*wvPow2 + 2*cu*su*cwuPow1_swuPow1*sPow2_xPow2 + pow(su, 2)*pow(v, 2)*cPow2_sPow2*cwuPow2 + pow(su, 2)*pow(v, 2)*cPow4*swuPow2 + 2*pow(su, 2)*v*cPow1_sPow2_xPow1*cwuPow2 + 2*pow(su, 2)*v*cPow2_sPow2*cwuPow2*wvPow1 + 2*pow(su, 2)*v*cPow3_xPow1*swuPow2 + 2*pow(su, 2)*v*cPow4*swuPow2*wvPow1 + 2*pow(su, 2)*cPow1_sPow2_xPow1*cwuPow2*wvPow1 + pow(su, 2)*cPow2_sPow2*cwuPow2*wvPow2 + pow(su, 2)*cPow2_xPow2*swuPow2 + 2*pow(su, 2)*cPow3_xPow1*swuPow2*wvPow1 + pow(su, 2)*cPow4*swuPow2*wvPow2 + pow(su, 2)*cwuPow2*sPow2_xPow2 + cPow1_sPow1_xPow2*cwuPow1_swuPow1*(-2*pow(cu, 2) + 2*pow(su, 2)) + cPow2_sPow1_xPow1*cwuPow1_swuPow1*wvPow1*(-4*pow(cu, 2) + 4*pow(su, 2)) + cPow2_sPow1_xPow1*cwuPow1_swuPow1*(-4*pow(cu, 2)*v + 4*pow(su, 2)*v) + cPow3_sPow1*cwuPow1_swuPow1*wvPow1*(-4*pow(cu, 2)*v + 4*pow(su, 2)*v) + cPow3_sPow1*cwuPow1_swuPow1*wvPow2*(-2*pow(cu, 2) + 2*pow(su, 2)) + cPow3_sPow1*cwuPow1_swuPow1*(-2*pow(cu, 2)*pow(v, 2) + 2*pow(su, 2)*pow(v, 2));
    result.yPow2_cPow2 = pow(cu, 2)*pow(v, 2)*cPow2_sPow2*cwuPow2 + pow(cu, 2)*pow(v, 2)*sPow4*swuPow2 + 2*pow(cu, 2)*v*cPow2_sPow1_yPow1*cwuPow2 + 2*pow(cu, 2)*v*cPow2_sPow2*cwuPow2*wvPow1 + 2*pow(cu, 2)*v*sPow3_yPow1*swuPow2 + 2*pow(cu, 2)*v*sPow4*swuPow2*wvPow1 + 2*pow(cu, 2)*cPow2_sPow1_yPow1*cwuPow2*wvPow1 + pow(cu, 2)*cPow2_sPow2*cwuPow2*wvPow2 + pow(cu, 2)*cPow2_yPow2*cwuPow2 + pow(cu, 2)*sPow2_yPow2*swuPow2 + 2*pow(cu, 2)*sPow3_yPow1*swuPow2*wvPow1 + pow(cu, 2)*sPow4*swuPow2*wvPow2 - 2*cu*su*pow(v, 2)*cPow1_sPow3*cwuPow2 + 2*cu*su*pow(v, 2)*cPow1_sPow3*swuPow2 - 2*cu*su*pow(v, 2)*cPow2_sPow2*cwuPow1_swuPow1 + 2*cu*su*pow(v, 2)*cwuPow1_swuPow1*sPow4 - 4*cu*su*v*cPow1_sPow2_yPow1*cwuPow2 + 4*cu*su*v*cPow1_sPow2_yPow1*swuPow2 - 4*cu*su*v*cPow1_sPow3*cwuPow2*wvPow1 + 4*cu*su*v*cPow1_sPow3*swuPow2*wvPow1 - 4*cu*su*v*cPow2_sPow1_yPow1*cwuPow1_swuPow1 - 4*cu*su*v*cPow2_sPow2*cwuPow1_swuPow1*wvPow1 + 4*cu*su*v*cwuPow1_swuPow1*sPow3_yPow1 + 4*cu*su*v*cwuPow1_swuPow1*sPow4*wvPow1 - 2*cu*su*cPow1_sPow1_yPow2*cwuPow2 + 2*cu*su*cPow1_sPow1_yPow2*swuPow2 - 4*cu*su*cPow1_sPow2_yPow1*cwuPow2*wvPow1 + 4*cu*su*cPow1_sPow2_yPow1*swuPow2*wvPow1 - 2*cu*su*cPow1_sPow3*cwuPow2*wvPow2 + 2*cu*su*cPow1_sPow3*swuPow2*wvPow2 - 4*cu*su*cPow2_sPow1_yPow1*cwuPow1_swuPow1*wvPow1 - 2*cu*su*cPow2_sPow2*cwuPow1_swuPow1*wvPow2 - 2*cu*su*cPow2_yPow2*cwuPow1_swuPow1 + 2*cu*su*cwuPow1_swuPow1*sPow2_yPow2 + 4*cu*su*cwuPow1_swuPow1*sPow3_yPow1*wvPow1 + 2*cu*su*cwuPow1_swuPow1*sPow4*wvPow2 + pow(su, 2)*pow(v, 2)*cPow2_sPow2*swuPow2 + pow(su, 2)*pow(v, 2)*cwuPow2*sPow4 + 2*pow(su, 2)*v*cPow2_sPow1_yPow1*swuPow2 + 2*pow(su, 2)*v*cPow2_sPow2*swuPow2*wvPow1 + 2*pow(su, 2)*v*cwuPow2*sPow3_yPow1 + 2*pow(su, 2)*v*cwuPow2*sPow4*wvPow1 + 2*pow(su, 2)*cPow2_sPow1_yPow1*swuPow2*wvPow1 + pow(su, 2)*cPow2_sPow2*swuPow2*wvPow2 + pow(su, 2)*cPow2_yPow2*swuPow2 + pow(su, 2)*cwuPow2*sPow2_yPow2 + 2*pow(su, 2)*cwuPow2*sPow3_yPow1*wvPow1 + pow(su, 2)*cwuPow2*sPow4*wvPow2 + cPow1_sPow1_yPow2*cwuPow1_swuPow1*(-2*pow(cu, 2) + 2*pow(su, 2)) + cPow1_sPow2_yPow1*cwuPow1_swuPow1*wvPow1*(-4*pow(cu, 2) + 4*pow(su, 2)) + cPow1_sPow2_yPow1*cwuPow1_swuPow1*(-4*pow(cu, 2)*v + 4*pow(su, 2)*v) + cPow1_sPow3*cwuPow1_swuPow1*wvPow1*(-4*pow(cu, 2)*v + 4*pow(su, 2)*v) + cPow1_sPow3*cwuPow1_swuPow1*wvPow2*(-2*pow(cu, 2) + 2*pow(su, 2)) + cPow1_sPow3*cwuPow1_swuPow1*(-2*pow(cu, 2)*pow(v, 2) + 2*pow(su, 2)*pow(v, 2));
    result.xPow2_sPow2 = pow(cu, 2)*pow(v, 2)*cPow2_sPow2*cwuPow2 + pow(cu, 2)*pow(v, 2)*cPow4*swuPow2 + 2*pow(cu, 2)*v*cPow1_sPow2_xPow1*cwuPow2 + 2*pow(cu, 2)*v*cPow2_sPow2*cwuPow2*wvPow1 + 2*pow(cu, 2)*v*cPow3_xPow1*swuPow2 + 2*pow(cu, 2)*v*cPow4*swuPow2*wvPow1 + 2*pow(cu, 2)*cPow1_sPow2_xPow1*cwuPow2*wvPow1 + pow(cu, 2)*cPow2_sPow2*cwuPow2*wvPow2 + pow(cu, 2)*cPow2_xPow2*swuPow2 + 2*pow(cu, 2)*cPow3_xPow1*swuPow2*wvPow1 + pow(cu, 2)*cPow4*swuPow2*wvPow2 + pow(cu, 2)*cwuPow2*sPow2_xPow2 - 2*cu*su*pow(v, 2)*cPow2_sPow2*cwuPow1_swuPow1 + 2*cu*su*pow(v, 2)*cPow3_sPow1*cwuPow2 - 2*cu*su*pow(v, 2)*cPow3_sPow1*swuPow2 + 2*cu*su*pow(v, 2)*cPow4*cwuPow1_swuPow1 - 4*cu*su*v*cPow1_sPow2_xPow1*cwuPow1_swuPow1 + 4*cu*su*v*cPow2_sPow1_xPow1*cwuPow2 - 4*cu*su*v*cPow2_sPow1_xPow1*swuPow2 - 4*cu*su*v*cPow2_sPow2*cwuPow1_swuPow1*wvPow1 + 4*cu*su*v*cPow3_sPow1*cwuPow2*wvPow1 - 4*cu*su*v*cPow3_sPow1*swuPow2*wvPow1 + 4*cu*su*v*cPow3_xPow1*cwuPow1_swuPow1 + 4*cu*su*v*cPow4*cwuPow1_swuPow1*wvPow1 + 2*cu*su*cPow1_sPow1_xPow2*cwuPow2 - 2*cu*su*cPow1_sPow1_xPow2*swuPow2 - 4*cu*su*cPow1_sPow2_xPow1*cwuPow1_swuPow1*wvPow1 + 4*cu*su*cPow2_sPow1_xPow1*cwuPow2*wvPow1 - 4*cu*su*cPow2_sPow1_xPow1*swuPow2*wvPow1 - 2*cu*su*cPow2_sPow2*cwuPow1_swuPow1*wvPow2 + 2*cu*su*cPow2_xPow2*cwuPow1_swuPow1 + 2*cu*su*cPow3_sPow1*cwuPow2*wvPow2 - 2*cu*su*cPow3_sPow1*swuPow2*wvPow2 + 4*cu*su*cPow3_xPow1*cwuPow1_swuPow1*wvPow1 + 2*cu*su*cPow4*cwuPow1_swuPow1*wvPow2 - 2*cu*su*cwuPow1_swuPow1*sPow2_xPow2 + pow(su, 2)*pow(v, 2)*cPow2_sPow2*swuPow2 + pow(su, 2)*pow(v, 2)*cPow4*cwuPow2 + 2*pow(su, 2)*v*cPow1_sPow2_xPow1*swuPow2 + 2*pow(su, 2)*v*cPow2_sPow2*swuPow2*wvPow1 + 2*pow(su, 2)*v*cPow3_xPow1*cwuPow2 + 2*pow(su, 2)*v*cPow4*cwuPow2*wvPow1 + 2*pow(su, 2)*cPow1_sPow2_xPow1*swuPow2*wvPow1 + pow(su, 2)*cPow2_sPow2*swuPow2*wvPow2 + pow(su, 2)*cPow2_xPow2*cwuPow2 + 2*pow(su, 2)*cPow3_xPow1*cwuPow2*wvPow1 + pow(su, 2)*cPow4*cwuPow2*wvPow2 + pow(su, 2)*sPow2_xPow2*swuPow2 + cPow1_sPow1_xPow2*cwuPow1_swuPow1*(2*pow(cu, 2) - 2*pow(su, 2)) + cPow2_sPow1_xPow1*cwuPow1_swuPow1*wvPow1*(4*pow(cu, 2) - 4*pow(su, 2)) + cPow2_sPow1_xPow1*cwuPow1_swuPow1*(4*pow(cu, 2)*v - 4*pow(su, 2)*v) + cPow3_sPow1*cwuPow1_swuPow1*wvPow1*(4*pow(cu, 2)*v - 4*pow(su, 2)*v) + cPow3_sPow1*cwuPow1_swuPow1*wvPow2*(2*pow(cu, 2) - 2*pow(su, 2)) + cPow3_sPow1*cwuPow1_swuPow1*(2*pow(cu, 2)*pow(v, 2) - 2*pow(su, 2)*pow(v, 2));
    result.yPow2_sPow2 = pow(cu, 2)*pow(v, 2)*cPow2_sPow2*swuPow2 + pow(cu, 2)*pow(v, 2)*cwuPow2*sPow4 + 2*pow(cu, 2)*v*cPow2_sPow1_yPow1*swuPow2 + 2*pow(cu, 2)*v*cPow2_sPow2*swuPow2*wvPow1 + 2*pow(cu, 2)*v*cwuPow2*sPow3_yPow1 + 2*pow(cu, 2)*v*cwuPow2*sPow4*wvPow1 + 2*pow(cu, 2)*cPow2_sPow1_yPow1*swuPow2*wvPow1 + pow(cu, 2)*cPow2_sPow2*swuPow2*wvPow2 + pow(cu, 2)*cPow2_yPow2*swuPow2 + pow(cu, 2)*cwuPow2*sPow2_yPow2 + 2*pow(cu, 2)*cwuPow2*sPow3_yPow1*wvPow1 + pow(cu, 2)*cwuPow2*sPow4*wvPow2 + 2*cu*su*pow(v, 2)*cPow1_sPow3*cwuPow2 - 2*cu*su*pow(v, 2)*cPow1_sPow3*swuPow2 + 2*cu*su*pow(v, 2)*cPow2_sPow2*cwuPow1_swuPow1 - 2*cu*su*pow(v, 2)*cwuPow1_swuPow1*sPow4 + 4*cu*su*v*cPow1_sPow2_yPow1*cwuPow2 - 4*cu*su*v*cPow1_sPow2_yPow1*swuPow2 + 4*cu*su*v*cPow1_sPow3*cwuPow2*wvPow1 - 4*cu*su*v*cPow1_sPow3*swuPow2*wvPow1 + 4*cu*su*v*cPow2_sPow1_yPow1*cwuPow1_swuPow1 + 4*cu*su*v*cPow2_sPow2*cwuPow1_swuPow1*wvPow1 - 4*cu*su*v*cwuPow1_swuPow1*sPow3_yPow1 - 4*cu*su*v*cwuPow1_swuPow1*sPow4*wvPow1 + 2*cu*su*cPow1_sPow1_yPow2*cwuPow2 - 2*cu*su*cPow1_sPow1_yPow2*swuPow2 + 4*cu*su*cPow1_sPow2_yPow1*cwuPow2*wvPow1 - 4*cu*su*cPow1_sPow2_yPow1*swuPow2*wvPow1 + 2*cu*su*cPow1_sPow3*cwuPow2*wvPow2 - 2*cu*su*cPow1_sPow3*swuPow2*wvPow2 + 4*cu*su*cPow2_sPow1_yPow1*cwuPow1_swuPow1*wvPow1 + 2*cu*su*cPow2_sPow2*cwuPow1_swuPow1*wvPow2 + 2*cu*su*cPow2_yPow2*cwuPow1_swuPow1 - 2*cu*su*cwuPow1_swuPow1*sPow2_yPow2 - 4*cu*su*cwuPow1_swuPow1*sPow3_yPow1*wvPow1 - 2*cu*su*cwuPow1_swuPow1*sPow4*wvPow2 + pow(su, 2)*pow(v, 2)*cPow2_sPow2*cwuPow2 + pow(su, 2)*pow(v, 2)*sPow4*swuPow2 + 2*pow(su, 2)*v*cPow2_sPow1_yPow1*cwuPow2 + 2*pow(su, 2)*v*cPow2_sPow2*cwuPow2*wvPow1 + 2*pow(su, 2)*v*sPow3_yPow1*swuPow2 + 2*pow(su, 2)*v*sPow4*swuPow2*wvPow1 + 2*pow(su, 2)*cPow2_sPow1_yPow1*cwuPow2*wvPow1 + pow(su, 2)*cPow2_sPow2*cwuPow2*wvPow2 + pow(su, 2)*cPow2_yPow2*cwuPow2 + pow(su, 2)*sPow2_yPow2*swuPow2 + 2*pow(su, 2)*sPow3_yPow1*swuPow2*wvPow1 + pow(su, 2)*sPow4*swuPow2*wvPow2 + cPow1_sPow1_yPow2*cwuPow1_swuPow1*(2*pow(cu, 2) - 2*pow(su, 2)) + cPow1_sPow2_yPow1*cwuPow1_swuPow1*wvPow1*(4*pow(cu, 2) - 4*pow(su, 2)) + cPow1_sPow2_yPow1*cwuPow1_swuPow1*(4*pow(cu, 2)*v - 4*pow(su, 2)*v) + cPow1_sPow3*cwuPow1_swuPow1*wvPow1*(4*pow(cu, 2)*v - 4*pow(su, 2)*v) + cPow1_sPow3*cwuPow1_swuPow1*wvPow2*(2*pow(cu, 2) - 2*pow(su, 2)) + cPow1_sPow3*cwuPow1_swuPow1*(2*pow(cu, 2)*pow(v, 2) - 2*pow(su, 2)*pow(v, 2));
    result.xPow1_yPow1_cPow2 = pow(cu, 2)*pow(v, 2)*cPow1_sPow3*swuPow2 + pow(cu, 2)*pow(v, 2)*cPow3_sPow1*cwuPow2 + pow(cu, 2)*v*cPow1_sPow2_yPow1*swuPow2 + 2*pow(cu, 2)*v*cPow1_sPow3*swuPow2*wvPow1 + pow(cu, 2)*v*cPow2_sPow1_xPow1*cwuPow2 + 2*pow(cu, 2)*v*cPow3_sPow1*cwuPow2*wvPow1 + pow(cu, 2)*v*cPow3_yPow1*cwuPow2 + pow(cu, 2)*v*sPow3_xPow1*swuPow2 + pow(cu, 2)*cPow1_sPow2_yPow1*swuPow2*wvPow1 + pow(cu, 2)*cPow1_sPow3*swuPow2*wvPow2 + pow(cu, 2)*cPow2_sPow1_xPow1*cwuPow2*wvPow1 + pow(cu, 2)*cPow2_xPow1_yPow1*cwuPow2 + pow(cu, 2)*cPow3_sPow1*cwuPow2*wvPow2 + pow(cu, 2)*cPow3_yPow1*cwuPow2*wvPow1 + pow(cu, 2)*sPow2_xPow1_yPow1*swuPow2 + pow(cu, 2)*sPow3_xPow1*swuPow2*wvPow1 + 2*cu*su*pow(v, 2)*cPow1_sPow3*cwuPow1_swuPow1 - 2*cu*su*pow(v, 2)*cPow2_sPow2*cwuPow2 + 2*cu*su*pow(v, 2)*cPow2_sPow2*swuPow2 - 2*cu*su*pow(v, 2)*cPow3_sPow1*cwuPow1_swuPow1 - 2*cu*su*v*cPow1_sPow2_xPow1*cwuPow2 + 2*cu*su*v*cPow1_sPow2_xPow1*swuPow2 + 2*cu*su*v*cPow1_sPow2_yPow1*cwuPow1_swuPow1 + 4*cu*su*v*cPow1_sPow3*cwuPow1_swuPow1*wvPow1 - 2*cu*su*v*cPow2_sPow1_xPow1*cwuPow1_swuPow1 - 2*cu*su*v*cPow2_sPow1_yPow1*cwuPow2 + 2*cu*su*v*cPow2_sPow1_yPow1*swuPow2 - 4*cu*su*v*cPow2_sPow2*cwuPow2*wvPow1 + 4*cu*su*v*cPow2_sPow2*swuPow2*wvPow1 - 4*cu*su*v*cPow3_sPow1*cwuPow1_swuPow1*wvPow1 - 2*cu*su*v*cPow3_yPow1*cwuPow1_swuPow1 + 2*cu*su*v*cwuPow1_swuPow1*sPow3_xPow1 - 2*cu*su*cPow1_sPow1_xPow1_yPow1*cwuPow2 + 2*cu*su*cPow1_sPow1_xPow1_yPow1*swuPow2 - 2*cu*su*cPow1_sPow2_xPow1*cwuPow2*wvPow1 + 2*cu*su*cPow1_sPow2_xPow1*swuPow2*wvPow1 + 2*cu*su*cPow1_sPow2_yPow1*cwuPow1_swuPow1*wvPow1 + 2*cu*su*cPow1_sPow3*cwuPow1_swuPow1*wvPow2 - 2*cu*su*cPow2_sPow1_xPow1*cwuPow1_swuPow1*wvPow1 - 2*cu*su*cPow2_sPow1_yPow1*cwuPow2*wvPow1 + 2*cu*su*cPow2_sPow1_yPow1*swuPow2*wvPow1 - 2*cu*su*cPow2_sPow2*cwuPow2*wvPow2 + 2*cu*su*cPow2_sPow2*swuPow2*wvPow2 - 2*cu*su*cPow2_xPow1_yPow1*cwuPow1_swuPow1 - 2*cu*su*cPow3_sPow1*cwuPow1_swuPow1*wvPow2 - 2*cu*su*cPow3_yPow1*cwuPow1_swuPow1*wvPow1 + 2*cu*su*cwuPow1_swuPow1*sPow2_xPow1_yPow1 + 2*cu*su*cwuPow1_swuPow1*sPow3_xPow1*wvPow1 + pow(su, 2)*pow(v, 2)*cPow1_sPow3*cwuPow2 + pow(su, 2)*pow(v, 2)*cPow3_sPow1*swuPow2 + pow(su, 2)*v*cPow1_sPow2_yPow1*cwuPow2 + 2*pow(su, 2)*v*cPow1_sPow3*cwuPow2*wvPow1 + pow(su, 2)*v*cPow2_sPow1_xPow1*swuPow2 + 2*pow(su, 2)*v*cPow3_sPow1*swuPow2*wvPow1 + pow(su, 2)*v*cPow3_yPow1*swuPow2 + pow(su, 2)*v*cwuPow2*sPow3_xPow1 + pow(su, 2)*cPow1_sPow2_yPow1*cwuPow2*wvPow1 + pow(su, 2)*cPow1_sPow3*cwuPow2*wvPow2 + pow(su, 2)*cPow2_sPow1_xPow1*swuPow2*wvPow1 + pow(su, 2)*cPow2_xPow1_yPow1*swuPow2 + pow(su, 2)*cPow3_sPow1*swuPow2*wvPow2 + pow(su, 2)*cPow3_yPow1*swuPow2*wvPow1 + pow(su, 2)*cwuPow2*sPow2_xPow1_yPow1 + pow(su, 2)*cwuPow2*sPow3_xPow1*wvPow1 + cPow1_sPow1_xPow1_yPow1*cwuPow1_swuPow1*(-2*pow(cu, 2) + 2*pow(su, 2)) + cPow1_sPow2_xPow1*cwuPow1_swuPow1*wvPow1*(-2*pow(cu, 2) + 2*pow(su, 2)) + cPow1_sPow2_xPow1*cwuPow1_swuPow1*(-2*pow(cu, 2)*v + 2*pow(su, 2)*v) + cPow2_sPow1_yPow1*cwuPow1_swuPow1*wvPow1*(-2*pow(cu, 2) + 2*pow(su, 2)) + cPow2_sPow1_yPow1*cwuPow1_swuPow1*(-2*pow(cu, 2)*v + 2*pow(su, 2)*v) + cPow2_sPow2*cwuPow1_swuPow1*wvPow1*(-4*pow(cu, 2)*v + 4*pow(su, 2)*v) + cPow2_sPow2*cwuPow1_swuPow1*wvPow2*(-2*pow(cu, 2) + 2*pow(su, 2)) + cPow2_sPow2*cwuPow1_swuPow1*(-2*pow(cu, 2)*pow(v, 2) + 2*pow(su, 2)*pow(v, 2));
    result.xPow1_yPow1_sPow2 = pow(cu, 2)*pow(v, 2)*cPow1_sPow3*cwuPow2 + pow(cu, 2)*pow(v, 2)*cPow3_sPow1*swuPow2 + pow(cu, 2)*v*cPow1_sPow2_yPow1*cwuPow2 + 2*pow(cu, 2)*v*cPow1_sPow3*cwuPow2*wvPow1 + pow(cu, 2)*v*cPow2_sPow1_xPow1*swuPow2 + 2*pow(cu, 2)*v*cPow3_sPow1*swuPow2*wvPow1 + pow(cu, 2)*v*cPow3_yPow1*swuPow2 + pow(cu, 2)*v*cwuPow2*sPow3_xPow1 + pow(cu, 2)*cPow1_sPow2_yPow1*cwuPow2*wvPow1 + pow(cu, 2)*cPow1_sPow3*cwuPow2*wvPow2 + pow(cu, 2)*cPow2_sPow1_xPow1*swuPow2*wvPow1 + pow(cu, 2)*cPow2_xPow1_yPow1*swuPow2 + pow(cu, 2)*cPow3_sPow1*swuPow2*wvPow2 + pow(cu, 2)*cPow3_yPow1*swuPow2*wvPow1 + pow(cu, 2)*cwuPow2*sPow2_xPow1_yPow1 + pow(cu, 2)*cwuPow2*sPow3_xPow1*wvPow1 - 2*cu*su*pow(v, 2)*cPow1_sPow3*cwuPow1_swuPow1 + 2*cu*su*pow(v, 2)*cPow2_sPow2*cwuPow2 - 2*cu*su*pow(v, 2)*cPow2_sPow2*swuPow2 + 2*cu*su*pow(v, 2)*cPow3_sPow1*cwuPow1_swuPow1 + 2*cu*su*v*cPow1_sPow2_xPow1*cwuPow2 - 2*cu*su*v*cPow1_sPow2_xPow1*swuPow2 - 2*cu*su*v*cPow1_sPow2_yPow1*cwuPow1_swuPow1 - 4*cu*su*v*cPow1_sPow3*cwuPow1_swuPow1*wvPow1 + 2*cu*su*v*cPow2_sPow1_xPow1*cwuPow1_swuPow1 + 2*cu*su*v*cPow2_sPow1_yPow1*cwuPow2 - 2*cu*su*v*cPow2_sPow1_yPow1*swuPow2 + 4*cu*su*v*cPow2_sPow2*cwuPow2*wvPow1 - 4*cu*su*v*cPow2_sPow2*swuPow2*wvPow1 + 4*cu*su*v*cPow3_sPow1*cwuPow1_swuPow1*wvPow1 + 2*cu*su*v*cPow3_yPow1*cwuPow1_swuPow1 - 2*cu*su*v*cwuPow1_swuPow1*sPow3_xPow1 + 2*cu*su*cPow1_sPow1_xPow1_yPow1*cwuPow2 - 2*cu*su*cPow1_sPow1_xPow1_yPow1*swuPow2 + 2*cu*su*cPow1_sPow2_xPow1*cwuPow2*wvPow1 - 2*cu*su*cPow1_sPow2_xPow1*swuPow2*wvPow1 - 2*cu*su*cPow1_sPow2_yPow1*cwuPow1_swuPow1*wvPow1 - 2*cu*su*cPow1_sPow3*cwuPow1_swuPow1*wvPow2 + 2*cu*su*cPow2_sPow1_xPow1*cwuPow1_swuPow1*wvPow1 + 2*cu*su*cPow2_sPow1_yPow1*cwuPow2*wvPow1 - 2*cu*su*cPow2_sPow1_yPow1*swuPow2*wvPow1 + 2*cu*su*cPow2_sPow2*cwuPow2*wvPow2 - 2*cu*su*cPow2_sPow2*swuPow2*wvPow2 + 2*cu*su*cPow2_xPow1_yPow1*cwuPow1_swuPow1 + 2*cu*su*cPow3_sPow1*cwuPow1_swuPow1*wvPow2 + 2*cu*su*cPow3_yPow1*cwuPow1_swuPow1*wvPow1 - 2*cu*su*cwuPow1_swuPow1*sPow2_xPow1_yPow1 - 2*cu*su*cwuPow1_swuPow1*sPow3_xPow1*wvPow1 + pow(su, 2)*pow(v, 2)*cPow1_sPow3*swuPow2 + pow(su, 2)*pow(v, 2)*cPow3_sPow1*cwuPow2 + pow(su, 2)*v*cPow1_sPow2_yPow1*swuPow2 + 2*pow(su, 2)*v*cPow1_sPow3*swuPow2*wvPow1 + pow(su, 2)*v*cPow2_sPow1_xPow1*cwuPow2 + 2*pow(su, 2)*v*cPow3_sPow1*cwuPow2*wvPow1 + pow(su, 2)*v*cPow3_yPow1*cwuPow2 + pow(su, 2)*v*sPow3_xPow1*swuPow2 + pow(su, 2)*cPow1_sPow2_yPow1*swuPow2*wvPow1 + pow(su, 2)*cPow1_sPow3*swuPow2*wvPow2 + pow(su, 2)*cPow2_sPow1_xPow1*cwuPow2*wvPow1 + pow(su, 2)*cPow2_xPow1_yPow1*cwuPow2 + pow(su, 2)*cPow3_sPow1*cwuPow2*wvPow2 + pow(su, 2)*cPow3_yPow1*cwuPow2*wvPow1 + pow(su, 2)*sPow2_xPow1_yPow1*swuPow2 + pow(su, 2)*sPow3_xPow1*swuPow2*wvPow1 + cPow1_sPow1_xPow1_yPow1*cwuPow1_swuPow1*(2*pow(cu, 2) - 2*pow(su, 2)) + cPow1_sPow2_xPow1*cwuPow1_swuPow1*wvPow1*(2*pow(cu, 2) - 2*pow(su, 2)) + cPow1_sPow2_xPow1*cwuPow1_swuPow1*(2*pow(cu, 2)*v - 2*pow(su, 2)*v) + cPow2_sPow1_yPow1*cwuPow1_swuPow1*wvPow1*(2*pow(cu, 2) - 2*pow(su, 2)) + cPow2_sPow1_yPow1*cwuPow1_swuPow1*(2*pow(cu, 2)*v - 2*pow(su, 2)*v) + cPow2_sPow2*cwuPow1_swuPow1*wvPow1*(4*pow(cu, 2)*v - 4*pow(su, 2)*v) + cPow2_sPow2*cwuPow1_swuPow1*wvPow2*(2*pow(cu, 2) - 2*pow(su, 2)) + cPow2_sPow2*cwuPow1_swuPow1*(2*pow(cu, 2)*pow(v, 2) - 2*pow(su, 2)*pow(v, 2));
    result.xPow2_cPow1_sPow1 = -cu*su*pow(v, 2)*cPow2_sPow2*cwuPow2 + cu*su*pow(v, 2)*cPow2_sPow2*swuPow2 - 4*cu*su*pow(v, 2)*cPow3_sPow1*cwuPow1_swuPow1 + cu*su*pow(v, 2)*cPow4*cwuPow2 - cu*su*pow(v, 2)*cPow4*swuPow2 - 2*cu*su*v*cPow1_sPow2_xPow1*cwuPow2 + 2*cu*su*v*cPow1_sPow2_xPow1*swuPow2 - 8*cu*su*v*cPow2_sPow1_xPow1*cwuPow1_swuPow1 - 2*cu*su*v*cPow2_sPow2*cwuPow2*wvPow1 + 2*cu*su*v*cPow2_sPow2*swuPow2*wvPow1 - 8*cu*su*v*cPow3_sPow1*cwuPow1_swuPow1*wvPow1 + 2*cu*su*v*cPow3_xPow1*cwuPow2 - 2*cu*su*v*cPow3_xPow1*swuPow2 + 2*cu*su*v*cPow4*cwuPow2*wvPow1 - 2*cu*su*v*cPow4*swuPow2*wvPow1 - 4*cu*su*cPow1_sPow1_xPow2*cwuPow1_swuPow1 - 2*cu*su*cPow1_sPow2_xPow1*cwuPow2*wvPow1 + 2*cu*su*cPow1_sPow2_xPow1*swuPow2*wvPow1 - 8*cu*su*cPow2_sPow1_xPow1*cwuPow1_swuPow1*wvPow1 - cu*su*cPow2_sPow2*cwuPow2*wvPow2 + cu*su*cPow2_sPow2*swuPow2*wvPow2 + cu*su*cPow2_xPow2*cwuPow2 - cu*su*cPow2_xPow2*swuPow2 - 4*cu*su*cPow3_sPow1*cwuPow1_swuPow1*wvPow2 + 2*cu*su*cPow3_xPow1*cwuPow2*wvPow1 - 2*cu*su*cPow3_xPow1*swuPow2*wvPow1 + cu*su*cPow4*cwuPow2*wvPow2 - cu*su*cPow4*swuPow2*wvPow2 - cu*su*cwuPow2*sPow2_xPow2 + cu*su*sPow2_xPow2*swuPow2 + cPow1_sPow1_xPow2*cwuPow2*(pow(cu, 2) - pow(su, 2)) + cPow1_sPow1_xPow2*swuPow2*(-pow(cu, 2) + pow(su, 2)) + cPow1_sPow2_xPow1*cwuPow1_swuPow1*wvPow1*(-2*pow(cu, 2) + 2*pow(su, 2)) + cPow1_sPow2_xPow1*cwuPow1_swuPow1*(-2*pow(cu, 2)*v + 2*pow(su, 2)*v) + cPow2_sPow1_xPow1*cwuPow2*wvPow1*(2*pow(cu, 2) - 2*pow(su, 2)) + cPow2_sPow1_xPow1*cwuPow2*(2*pow(cu, 2)*v - 2*pow(su, 2)*v) + cPow2_sPow1_xPow1*swuPow2*wvPow1*(-2*pow(cu, 2) + 2*pow(su, 2)) + cPow2_sPow1_xPow1*swuPow2*(-2*pow(cu, 2)*v + 2*pow(su, 2)*v) + cPow2_sPow2*cwuPow1_swuPow1*wvPow1*(-2*pow(cu, 2)*v + 2*pow(su, 2)*v) + cPow2_sPow2*cwuPow1_swuPow1*wvPow2*(-pow(cu, 2) + pow(su, 2)) + cPow2_sPow2*cwuPow1_swuPow1*(-pow(cu, 2)*pow(v, 2) + pow(su, 2)*pow(v, 2)) + cPow2_xPow2*cwuPow1_swuPow1*(pow(cu, 2) - pow(su, 2)) + cPow3_sPow1*cwuPow2*wvPow1*(2*pow(cu, 2)*v - 2*pow(su, 2)*v) + cPow3_sPow1*cwuPow2*wvPow2*(pow(cu, 2) - pow(su, 2)) + cPow3_sPow1*cwuPow2*(pow(cu, 2)*pow(v, 2) - pow(su, 2)*pow(v, 2)) + cPow3_sPow1*swuPow2*wvPow1*(-2*pow(cu, 2)*v + 2*pow(su, 2)*v) + cPow3_sPow1*swuPow2*wvPow2*(-pow(cu, 2) + pow(su, 2)) + cPow3_sPow1*swuPow2*(-pow(cu, 2)*pow(v, 2) + pow(su, 2)*pow(v, 2)) + cPow3_xPow1*cwuPow1_swuPow1*wvPow1*(2*pow(cu, 2) - 2*pow(su, 2)) + cPow3_xPow1*cwuPow1_swuPow1*(2*pow(cu, 2)*v - 2*pow(su, 2)*v) + cPow4*cwuPow1_swuPow1*wvPow1*(2*pow(cu, 2)*v - 2*pow(su, 2)*v) + cPow4*cwuPow1_swuPow1*wvPow2*(pow(cu, 2) - pow(su, 2)) + cPow4*cwuPow1_swuPow1*(pow(cu, 2)*pow(v, 2) - pow(su, 2)*pow(v, 2)) + cwuPow1_swuPow1*sPow2_xPow2*(-pow(cu, 2) + pow(su, 2));
    result.yPow2_cPow1_sPow1 = -4*cu*su*pow(v, 2)*cPow1_sPow3*cwuPow1_swuPow1 + cu*su*pow(v, 2)*cPow2_sPow2*cwuPow2 - cu*su*pow(v, 2)*cPow2_sPow2*swuPow2 - cu*su*pow(v, 2)*cwuPow2*sPow4 + cu*su*pow(v, 2)*sPow4*swuPow2 - 8*cu*su*v*cPow1_sPow2_yPow1*cwuPow1_swuPow1 - 8*cu*su*v*cPow1_sPow3*cwuPow1_swuPow1*wvPow1 + 2*cu*su*v*cPow2_sPow1_yPow1*cwuPow2 - 2*cu*su*v*cPow2_sPow1_yPow1*swuPow2 + 2*cu*su*v*cPow2_sPow2*cwuPow2*wvPow1 - 2*cu*su*v*cPow2_sPow2*swuPow2*wvPow1 - 2*cu*su*v*cwuPow2*sPow3_yPow1 - 2*cu*su*v*cwuPow2*sPow4*wvPow1 + 2*cu*su*v*sPow3_yPow1*swuPow2 + 2*cu*su*v*sPow4*swuPow2*wvPow1 - 4*cu*su*cPow1_sPow1_yPow2*cwuPow1_swuPow1 - 8*cu*su*cPow1_sPow2_yPow1*cwuPow1_swuPow1*wvPow1 - 4*cu*su*cPow1_sPow3*cwuPow1_swuPow1*wvPow2 + 2*cu*su*cPow2_sPow1_yPow1*cwuPow2*wvPow1 - 2*cu*su*cPow2_sPow1_yPow1*swuPow2*wvPow1 + cu*su*cPow2_sPow2*cwuPow2*wvPow2 - cu*su*cPow2_sPow2*swuPow2*wvPow2 + cu*su*cPow2_yPow2*cwuPow2 - cu*su*cPow2_yPow2*swuPow2 - cu*su*cwuPow2*sPow2_yPow2 - 2*cu*su*cwuPow2*sPow3_yPow1*wvPow1 - cu*su*cwuPow2*sPow4*wvPow2 + cu*su*sPow2_yPow2*swuPow2 + 2*cu*su*sPow3_yPow1*swuPow2*wvPow1 + cu*su*sPow4*swuPow2*wvPow2 + cPow1_sPow1_yPow2*cwuPow2*(pow(cu, 2) - pow(su, 2)) + cPow1_sPow1_yPow2*swuPow2*(-pow(cu, 2) + pow(su, 2)) + cPow1_sPow2_yPow1*cwuPow2*wvPow1*(2*pow(cu, 2) - 2*pow(su, 2)) + cPow1_sPow2_yPow1*cwuPow2*(2*pow(cu, 2)*v - 2*pow(su, 2)*v) + cPow1_sPow2_yPow1*swuPow2*wvPow1*(-2*pow(cu, 2) + 2*pow(su, 2)) + cPow1_sPow2_yPow1*swuPow2*(-2*pow(cu, 2)*v + 2*pow(su, 2)*v) + cPow1_sPow3*cwuPow2*wvPow1*(2*pow(cu, 2)*v - 2*pow(su, 2)*v) + cPow1_sPow3*cwuPow2*wvPow2*(pow(cu, 2) - pow(su, 2)) + cPow1_sPow3*cwuPow2*(pow(cu, 2)*pow(v, 2) - pow(su, 2)*pow(v, 2)) + cPow1_sPow3*swuPow2*wvPow1*(-2*pow(cu, 2)*v + 2*pow(su, 2)*v) + cPow1_sPow3*swuPow2*wvPow2*(-pow(cu, 2) + pow(su, 2)) + cPow1_sPow3*swuPow2*(-pow(cu, 2)*pow(v, 2) + pow(su, 2)*pow(v, 2)) + cPow2_sPow1_yPow1*cwuPow1_swuPow1*wvPow1*(2*pow(cu, 2) - 2*pow(su, 2)) + cPow2_sPow1_yPow1*cwuPow1_swuPow1*(2*pow(cu, 2)*v - 2*pow(su, 2)*v) + cPow2_sPow2*cwuPow1_swuPow1*wvPow1*(2*pow(cu, 2)*v - 2*pow(su, 2)*v) + cPow2_sPow2*cwuPow1_swuPow1*wvPow2*(pow(cu, 2) - pow(su, 2)) + cPow2_sPow2*cwuPow1_swuPow1*(pow(cu, 2)*pow(v, 2) - pow(su, 2)*pow(v, 2)) + cPow2_yPow2*cwuPow1_swuPow1*(pow(cu, 2) - pow(su, 2)) + cwuPow1_swuPow1*sPow2_yPow2*(-pow(cu, 2) + pow(su, 2)) + cwuPow1_swuPow1*sPow3_yPow1*wvPow1*(-2*pow(cu, 2) + 2*pow(su, 2)) + cwuPow1_swuPow1*sPow3_yPow1*(-2*pow(cu, 2)*v + 2*pow(su, 2)*v) + cwuPow1_swuPow1*sPow4*wvPow1*(-2*pow(cu, 2)*v + 2*pow(su, 2)*v) + cwuPow1_swuPow1*sPow4*wvPow2*(-pow(cu, 2) + pow(su, 2)) + cwuPow1_swuPow1*sPow4*(-pow(cu, 2)*pow(v, 2) + pow(su, 2)*pow(v, 2));
    result.xPow1_yPow1_cPow1_sPow1 = -cu*su*pow(v, 2)*cPow1_sPow3*cwuPow2 + cu*su*pow(v, 2)*cPow1_sPow3*swuPow2 - 4*cu*su*pow(v, 2)*cPow2_sPow2*cwuPow1_swuPow1 + cu*su*pow(v, 2)*cPow3_sPow1*cwuPow2 - cu*su*pow(v, 2)*cPow3_sPow1*swuPow2 - 4*cu*su*v*cPow1_sPow2_xPow1*cwuPow1_swuPow1 - cu*su*v*cPow1_sPow2_yPow1*cwuPow2 + cu*su*v*cPow1_sPow2_yPow1*swuPow2 - 2*cu*su*v*cPow1_sPow3*cwuPow2*wvPow1 + 2*cu*su*v*cPow1_sPow3*swuPow2*wvPow1 + cu*su*v*cPow2_sPow1_xPow1*cwuPow2 - cu*su*v*cPow2_sPow1_xPow1*swuPow2 - 4*cu*su*v*cPow2_sPow1_yPow1*cwuPow1_swuPow1 - 8*cu*su*v*cPow2_sPow2*cwuPow1_swuPow1*wvPow1 + 2*cu*su*v*cPow3_sPow1*cwuPow2*wvPow1 - 2*cu*su*v*cPow3_sPow1*swuPow2*wvPow1 + cu*su*v*cPow3_yPow1*cwuPow2 - cu*su*v*cPow3_yPow1*swuPow2 - cu*su*v*cwuPow2*sPow3_xPow1 + cu*su*v*sPow3_xPow1*swuPow2 - 4*cu*su*cPow1_sPow1_xPow1_yPow1*cwuPow1_swuPow1 - 4*cu*su*cPow1_sPow2_xPow1*cwuPow1_swuPow1*wvPow1 - cu*su*cPow1_sPow2_yPow1*cwuPow2*wvPow1 + cu*su*cPow1_sPow2_yPow1*swuPow2*wvPow1 - cu*su*cPow1_sPow3*cwuPow2*wvPow2 + cu*su*cPow1_sPow3*swuPow2*wvPow2 + cu*su*cPow2_sPow1_xPow1*cwuPow2*wvPow1 - cu*su*cPow2_sPow1_xPow1*swuPow2*wvPow1 - 4*cu*su*cPow2_sPow1_yPow1*cwuPow1_swuPow1*wvPow1 - 4*cu*su*cPow2_sPow2*cwuPow1_swuPow1*wvPow2 + cu*su*cPow2_xPow1_yPow1*cwuPow2 - cu*su*cPow2_xPow1_yPow1*swuPow2 + cu*su*cPow3_sPow1*cwuPow2*wvPow2 - cu*su*cPow3_sPow1*swuPow2*wvPow2 + cu*su*cPow3_yPow1*cwuPow2*wvPow1 - cu*su*cPow3_yPow1*swuPow2*wvPow1 - cu*su*cwuPow2*sPow2_xPow1_yPow1 - cu*su*cwuPow2*sPow3_xPow1*wvPow1 + cu*su*sPow2_xPow1_yPow1*swuPow2 + cu*su*sPow3_xPow1*swuPow2*wvPow1 + cPow1_sPow1_xPow1_yPow1*cwuPow2*(pow(cu, 2) - pow(su, 2)) + cPow1_sPow1_xPow1_yPow1*swuPow2*(-pow(cu, 2) + pow(su, 2)) + cPow1_sPow2_xPow1*cwuPow2*wvPow1*(pow(cu, 2) - pow(su, 2)) + cPow1_sPow2_xPow1*cwuPow2*(pow(cu, 2)*v - pow(su, 2)*v) + cPow1_sPow2_xPow1*swuPow2*wvPow1*(-pow(cu, 2) + pow(su, 2)) + cPow1_sPow2_xPow1*swuPow2*(-pow(cu, 2)*v + pow(su, 2)*v) + cPow1_sPow2_yPow1*cwuPow1_swuPow1*wvPow1*(-pow(cu, 2) + pow(su, 2)) + cPow1_sPow2_yPow1*cwuPow1_swuPow1*(-pow(cu, 2)*v + pow(su, 2)*v) + cPow1_sPow3*cwuPow1_swuPow1*wvPow1*(-2*pow(cu, 2)*v + 2*pow(su, 2)*v) + cPow1_sPow3*cwuPow1_swuPow1*wvPow2*(-pow(cu, 2) + pow(su, 2)) + cPow1_sPow3*cwuPow1_swuPow1*(-pow(cu, 2)*pow(v, 2) + pow(su, 2)*pow(v, 2)) + cPow2_sPow1_xPow1*cwuPow1_swuPow1*wvPow1*(pow(cu, 2) - pow(su, 2)) + cPow2_sPow1_xPow1*cwuPow1_swuPow1*(pow(cu, 2)*v - pow(su, 2)*v) + cPow2_sPow1_yPow1*cwuPow2*wvPow1*(pow(cu, 2) - pow(su, 2)) + cPow2_sPow1_yPow1*cwuPow2*(pow(cu, 2)*v - pow(su, 2)*v) + cPow2_sPow1_yPow1*swuPow2*wvPow1*(-pow(cu, 2) + pow(su, 2)) + cPow2_sPow1_yPow1*swuPow2*(-pow(cu, 2)*v + pow(su, 2)*v) + cPow2_sPow2*cwuPow2*wvPow1*(2*pow(cu, 2)*v - 2*pow(su, 2)*v) + cPow2_sPow2*cwuPow2*wvPow2*(pow(cu, 2) - pow(su, 2)) + cPow2_sPow2*cwuPow2*(pow(cu, 2)*pow(v, 2) - pow(su, 2)*pow(v, 2)) + cPow2_sPow2*swuPow2*wvPow1*(-2*pow(cu, 2)*v + 2*pow(su, 2)*v) + cPow2_sPow2*swuPow2*wvPow2*(-pow(cu, 2) + pow(su, 2)) + cPow2_sPow2*swuPow2*(-pow(cu, 2)*pow(v, 2) + pow(su, 2)*pow(v, 2)) + cPow2_xPow1_yPow1*cwuPow1_swuPow1*(pow(cu, 2) - pow(su, 2)) + cPow3_sPow1*cwuPow1_swuPow1*wvPow1*(2*pow(cu, 2)*v - 2*pow(su, 2)*v) + cPow3_sPow1*cwuPow1_swuPow1*wvPow2*(pow(cu, 2) - pow(su, 2)) + cPow3_sPow1*cwuPow1_swuPow1*(pow(cu, 2)*pow(v, 2) - pow(su, 2)*pow(v, 2)) + cPow3_yPow1*cwuPow1_swuPow1*wvPow1*(pow(cu, 2) - pow(su, 2)) + cPow3_yPow1*cwuPow1_swuPow1*(pow(cu, 2)*v - pow(su, 2)*v) + cwuPow1_swuPow1*sPow2_xPow1_yPow1*(-pow(cu, 2) + pow(su, 2)) + cwuPow1_swuPow1*sPow3_xPow1*wvPow1*(-pow(cu, 2) + pow(su, 2)) + cwuPow1_swuPow1*sPow3_xPow1*(-pow(cu, 2)*v + pow(su, 2)*v);

    high_order_moments = std::make_shared<SimpleVehicleModel::HighOrderMoments>(result);

    return vehicle_model_->propagateStateMoments(state_info, control_inputs, dt, noise_map);
}

StateInfo SimpleVehicleHMKF::update(const SimpleVehicleModel::HighOrderMoments & predicted_moments,
                                    const Eigen::VectorXd & observed_values,
                                    const Eigen::Vector2d & landmark,
                                    const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    // Step1. Create Predicted Moments
    const double& x_land = landmark(0);
    const double& y_land = landmark(1);

    const double& cPow1 = predicted_moments.cPow1;
    const double& sPow1 = predicted_moments.sPow1;
    const double& cPow2 = predicted_moments.cPow2;
    const double& sPow2 = predicted_moments.sPow2;
    const double& xPow1_cPow2 = predicted_moments.xPow1_cPow2;
    const double& yPow1_cPow2 = predicted_moments.yPow1_cPow2;
    const double& xPow1_sPow2 = predicted_moments.xPow1_sPow2;
    const double& yPow1_sPow2 = predicted_moments.yPow1_sPow2;
    const double& xPow1_cPow1_sPow1 = predicted_moments.xPow1_cPow1_sPow1;
    const double& yPow1_cPow1_sPow1 = predicted_moments.yPow1_cPow1_sPow1;
    const double& xPow1_yPow1_cPow2 = predicted_moments.xPow1_yPow1_cPow2;
    const double& xPow1_yPow1_sPow2 = predicted_moments.xPow1_yPow1_sPow2;
    const double& xPow2_cPow1_sPow1 = predicted_moments.xPow2_cPow1_sPow1;
    const double& yPow2_cPow1_sPow1 = predicted_moments.yPow2_cPow1_sPow1;

    const double& xPow2_cPow2 = predicted_moments.xPow2_cPow2;
    const double& yPow2_cPow2 = predicted_moments.yPow2_cPow2;
    const double& xPow2_sPow2 = predicted_moments.xPow2_sPow2;
    const double& yPow2_sPow2 = predicted_moments.yPow2_sPow2;
    const double& xPow1_yPow1_cPow1_sPow1 = predicted_moments.xPow1_yPow1_cPow1_sPow1;

    const double& cPow1_sPow1 = predicted_moments.cPow1_sPow1;
    const double& xPow1_cPow1 = predicted_moments.xPow1_cPow1;
    const double& xPow1_sPow1 = predicted_moments.xPow1_sPow1;
    const double& yPow1_cPow1 = predicted_moments.yPow1_cPow1;
    const double& yPow1_sPow1 = predicted_moments.yPow1_sPow1;
    const double& yawPow1_cPow1 = predicted_moments.yawPow1_cPow1;
    const double& yawPow1_sPow1 = predicted_moments.yawPow1_sPow1;
    const double& xPow2_cPow1 = predicted_moments.xPow2_cPow1;
    const double& xPow2_sPow1 = predicted_moments.xPow2_sPow1;
    const double& yPow2_cPow1 = predicted_moments.yPow2_cPow1;
    const double& yPow2_sPow1 = predicted_moments.yPow2_sPow1;
    const double& xPow1_yPow1_cPow1 = predicted_moments.xPow1_yPow1_cPow1;
    const double& xPow1_yPow1_sPow1 = predicted_moments.xPow1_yPow1_sPow1;
    const double& xPow1_yawPow1_cPow1 = predicted_moments.xPow1_yawPow1_cPow1;
    const double& xPow1_yawPow1_sPow1 = predicted_moments.xPow1_yawPow1_sPow1;
    const double& yPow1_yawPow1_cPow1 = predicted_moments.yPow1_yawPow1_cPow1;
    const double& yPow1_yawPow1_sPow1 = predicted_moments.yPow1_yawPow1_sPow1;

    const double xPow1_caPow1 = x_land * xPow1_cPow1 - xPow2_cPow1 + y_land * xPow1_sPow1 - xPow1_yPow1_sPow1;
    const double xPow1_saPow1 = y_land * xPow1_cPow1 - xPow1_yPow1_cPow1 - x_land * xPow1_sPow1 + xPow2_sPow1;
    const double yPow1_caPow1 = x_land * yPow1_cPow1 - xPow1_yPow1_cPow1 + y_land * yPow1_sPow1 - yPow2_sPow1;
    const double yPow1_saPow1 = y_land * yPow1_cPow1 - yPow2_cPow1 - x_land * yPow1_sPow1 + xPow1_yPow1_sPow1;
    const double yawPow1_caPow1 = x_land * yawPow1_cPow1 - xPow1_yawPow1_cPow1 + y_land * yawPow1_sPow1 - yPow1_yawPow1_sPow1;
    const double yawPow1_saPow1 = y_land * yawPow1_cPow1 - yPow1_yawPow1_cPow1 - x_land * yawPow1_sPow1 + xPow1_yawPow1_sPow1;

    // Step2. Create Observation Noise
    const auto wr_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WR);
    const auto wa_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WA);
    const double wrPow1 = wr_dist_ptr->calc_moment(1);
    const double wrPow2 = wr_dist_ptr->calc_moment(2);
    const double cwaPow1 = wa_dist_ptr->calc_cos_moment(1);
    const double swaPow1 = wa_dist_ptr->calc_sin_moment(1);
    const double cwaPow2 = wa_dist_ptr->calc_cos_moment(2);
    const double swaPow2 = wa_dist_ptr->calc_sin_moment(2);
    const double cwaPow1_swaPow1 = wa_dist_ptr->calc_cos_sin_moment(1, 1);

    // Step3. Get Observation Moments
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

    StateInfo observed_info;
    observed_info.mean = Eigen::VectorXd::Zero(2);
    observed_info.covariance = Eigen::MatrixXd::Zero(2, 2);
    observed_info.mean(MEASUREMENT::IDX::RCOS) = rcosPow1;
    observed_info.mean(MEASUREMENT::IDX::RSIN) = rsinPow1;
    observed_info.covariance(MEASUREMENT::IDX::RCOS, MEASUREMENT::IDX::RCOS) = rcosPow2 - rcosPow1*rcosPow1;
    observed_info.covariance(MEASUREMENT::IDX::RSIN, MEASUREMENT::IDX::RSIN) = rsinPow2 - rsinPow1*rsinPow1;
    observed_info.covariance(MEASUREMENT::IDX::RCOS, MEASUREMENT::IDX::RSIN) = rcosPow1_rsinPow1 - rcosPow1*rsinPow1;
    observed_info.covariance(MEASUREMENT::IDX::RSIN, MEASUREMENT::IDX::RCOS) = observed_info.covariance(MEASUREMENT::IDX::RCOS, MEASUREMENT::IDX::RSIN);

    const auto observation_mean = observed_info.mean;
    const auto observation_cov = observed_info.covariance;

    // Predicted Values
    Eigen::VectorXd predicted_mean = Eigen::VectorXd::Zero(3);
    predicted_mean(STATE::IDX::X) = predicted_moments.xPow1;
    predicted_mean(STATE::IDX::Y) = predicted_moments.yPow1;
    predicted_mean(STATE::IDX::YAW) = predicted_moments.yawPow1;
    Eigen::MatrixXd predicted_cov = Eigen::MatrixXd::Zero(3, 3);
    predicted_cov(STATE::IDX::X, STATE::IDX::X) = predicted_moments.xPow2 - predicted_moments.xPow1 * predicted_moments.xPow1;
    predicted_cov(STATE::IDX::Y, STATE::IDX::Y) = predicted_moments.yPow2 - predicted_moments.yPow1 * predicted_moments.yPow1;
    predicted_cov(STATE::IDX::YAW, STATE::IDX::YAW) = predicted_moments.yawPow2 - predicted_moments.yawPow1 * predicted_moments.yawPow1;
    predicted_cov(STATE::IDX::X, STATE::IDX::Y) = predicted_moments.xPow1_yPow1 - predicted_moments.xPow1 * predicted_moments.yPow1;
    predicted_cov(STATE::IDX::X, STATE::IDX::YAW) = predicted_moments.xPow1_yawPow1 - predicted_moments.xPow1 * predicted_moments.yawPow1;
    predicted_cov(STATE::IDX::Y, STATE::IDX::YAW) = predicted_moments.yPow1_yawPow1 - predicted_moments.yPow1 * predicted_moments.yawPow1;
    predicted_cov(STATE::IDX::Y, STATE::IDX::X) = predicted_cov(STATE::IDX::X, STATE::IDX::Y);
    predicted_cov(STATE::IDX::YAW, STATE::IDX::X) = predicted_cov(STATE::IDX::X, STATE::IDX::YAW);
    predicted_cov(STATE::IDX::YAW, STATE::IDX::Y) = predicted_cov(STATE::IDX::Y, STATE::IDX::YAW);

    Eigen::MatrixXd state_observation_cov(3, 2); // sigma = E[XY^T] - E[X]E[Y]^T
    state_observation_cov(STATE::IDX::X, MEASUREMENT::IDX::RCOS)
            = wrPow1 * cwaPow1 * xPow1_caPow1 - wrPow1 * swaPow1 * xPow1_saPow1
              - predicted_mean(STATE::IDX::X) * observation_mean(MEASUREMENT::IDX::RCOS);
    state_observation_cov(STATE::IDX::X, MEASUREMENT::IDX::RSIN)
            = wrPow1 * cwaPow1 * xPow1_saPow1 + wrPow1 * swaPow1 * xPow1_caPow1
              - predicted_mean(STATE::IDX::X) * observation_mean(MEASUREMENT::IDX::RSIN); // x_p * yaw

    state_observation_cov(STATE::IDX::Y, MEASUREMENT::IDX::RCOS)
            =  wrPow1 * cwaPow1 * yPow1_caPow1 - wrPow1 * swaPow1 * yPow1_saPow1
               - predicted_mean(STATE::IDX::Y) * observation_mean(MEASUREMENT::IDX::RCOS); // yp * (xp^2 + yp^2)
    state_observation_cov(STATE::IDX::Y, MEASUREMENT::IDX::RSIN)
            =  wrPow1 * cwaPow1 * yPow1_saPow1 + wrPow1 * swaPow1 * yPow1_caPow1
               - predicted_mean(STATE::IDX::Y) * observation_mean(MEASUREMENT::IDX::RSIN); // y_p * yaw

    state_observation_cov(STATE::IDX::YAW, MEASUREMENT::IDX::RCOS)
            =   wrPow1 * cwaPow1 * yawPow1_caPow1 - wrPow1 * swaPow1 * yawPow1_saPow1
                - predicted_mean(STATE::IDX::YAW) * observation_mean(MEASUREMENT::IDX::RCOS);
    state_observation_cov(STATE::IDX::YAW, MEASUREMENT::IDX::RSIN)
            =   wrPow1 * cwaPow1 * yawPow1_saPow1 + wrPow1 * swaPow1 * yawPow1_caPow1
                 - predicted_mean(STATE::IDX::YAW) * observation_mean(MEASUREMENT::IDX::RSIN);

    // Kalman Gain
    const auto K = state_observation_cov * observation_cov.inverse();

    StateInfo updated_info;
    updated_info.mean = predicted_mean + K * (observed_values - observation_mean);
    updated_info.covariance = predicted_cov - K*observation_cov*K.transpose();

    return updated_info;
}

void SimpleVehicleHMKF::createHighOrderMoments(const StateInfo& state_info,
                                               std::shared_ptr<SimpleVehicleModel::HighOrderMoments>& high_order_moments)
{
    ThreeDimensionalNormalDistribution dist(state_info.mean, state_info.covariance);

    SimpleVehicleModel::HighOrderMoments result;
    result.xPow1 = dist.calc_moment(STATE::IDX::X, 1);
    result.yPow1 = dist.calc_moment(STATE::IDX::Y, 1);
    result.yawPow1 = dist.calc_moment(STATE::IDX::YAW, 1);
    result.cPow1 = dist.calc_cos_moment(STATE::IDX::YAW, 1);
    result.sPow1 = dist.calc_sin_moment(STATE::IDX::YAW, 1);

    result.xPow2 = dist.calc_moment(STATE::IDX::X, 2);
    result.yPow2 = dist.calc_moment(STATE::IDX::Y, 2);
    result.yawPow2 = dist.calc_moment(STATE::IDX::YAW, 2);
    result.cPow2 = dist.calc_cos_moment(STATE::IDX::YAW, 2);
    result.sPow2 = dist.calc_sin_moment(STATE::IDX::YAW, 2);
    result.xPow1_yPow1 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::Y, 1, 1, 0, 0);
    result.xPow1_yawPow1 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW, 1, 1, 0, 0);
    result.yPow1_yawPow1 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW, 1, 1, 0, 0);
    result.xPow1_cPow1 = dist.calc_x_cos_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    result.yPow1_cPow1 = dist.calc_x_cos_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    result.xPow1_sPow1 = dist.calc_x_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    result.yPow1_sPow1 = dist.calc_x_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    result.cPow1_sPow1 = dist.calc_cos_sin_moment(STATE::IDX::YAW, 1, 1);
    result.yawPow1_cPow1 = dist.calc_x_cos_x_moment(STATE::IDX::YAW, 1, 1);
    result.yawPow1_sPow1 = dist.calc_x_sin_x_moment(STATE::IDX::YAW, 1, 1);

    result.xPow1_cPow2 = dist.calc_x_cos_z_cos_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    result.xPow1_sPow2 = dist.calc_x_sin_z_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    result.yPow1_cPow2 = dist.calc_x_cos_z_cos_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    result.yPow1_sPow2 = dist.calc_x_sin_z_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    result.xPow2_cPow1 = dist.calc_xx_cos_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    result.yPow2_cPow1 = dist.calc_xx_cos_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    result.xPow2_sPow1 = dist.calc_xx_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    result.yPow2_sPow1 = dist.calc_xx_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    result.xPow1_yPow1_cPow1 = dist.calc_xy_cos_z_moment();
    result.xPow1_yPow1_sPow1 = dist.calc_xy_sin_z_moment();
    result.xPow1_cPow1_sPow1 = dist.calc_x_cos_z_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    result.yPow1_cPow1_sPow1 = dist.calc_x_cos_z_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    result.xPow1_yawPow1_cPow1 = dist.calc_xy_cos_y_moment(STATE::IDX::X, STATE::IDX::YAW);
    result.xPow1_yawPow1_sPow1 = dist.calc_xy_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW);
    result.yPow1_yawPow1_cPow1 = dist.calc_xy_cos_y_moment(STATE::IDX::Y, STATE::IDX::YAW);
    result.yPow1_yawPow1_sPow1 = dist.calc_xy_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW);

    result.xPow2_cPow2 = dist.calc_xx_cos_z_cos_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    result.yPow2_cPow2 = dist.calc_xx_cos_z_cos_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    result.xPow2_sPow2 = dist.calc_xx_sin_z_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    result.yPow2_sPow2 = dist.calc_xx_sin_z_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    result.xPow1_yPow1_cPow2 = dist.calc_xy_cos_z_cos_z_moment();
    result.xPow1_yPow1_sPow2 = dist.calc_xy_sin_z_sin_z_moment();
    result.xPow2_cPow1_sPow1 = dist.calc_xx_cos_z_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    result.yPow2_cPow1_sPow1 = dist.calc_xx_cos_z_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    result.xPow1_yPow1_cPow1_sPow1 = dist.calc_xy_cos_z_sin_z_moment();

    high_order_moments = std::make_shared<SimpleVehicleModel::HighOrderMoments>(result);
}
