#include "filter/simple_vehicle_hmkf.h"
#include "distribution/three_dimensional_normal_distribution.h"

using namespace SimpleVehicle;

SimpleVehicleHMKF::SimpleVehicleHMKF()
{
    vehicle_model_ = SimpleVehicleModel();
}

StateInfo SimpleVehicleHMKF::predict(const StateInfo& state_info,
                                     const Eigen::Vector2d & control_inputs,
                                     const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map,
                                     std::shared_ptr<SimpleVehicleModel::HighOrderMoments>& high_order_moments)
{
    // Step1. Approximate to Gaussian Distribution
    const auto state_mean = state_info.mean;
    const auto state_cov = state_info.covariance;
    ThreeDimensionalNormalDistribution dist(state_info.mean, state_info.covariance);

    // Step2. State Moment
    SimpleVehicleModel::StateMoments moment;
    moment.xPow1 = dist.calc_moment(STATE::IDX::X, 1); // x
    moment.yPow1 = dist.calc_moment(STATE::IDX::Y, 1); // y
    moment.cPow1 = dist.calc_cos_moment(STATE::IDX::YAW, 1); // cos(yaw)
    moment.sPow1 = dist.calc_sin_moment(STATE::IDX::YAW, 1); // sin(yaw)
    moment.yawPow1 = dist.calc_moment(STATE::IDX::YAW, 1); // yaw
    moment.xPow2 = dist.calc_moment(STATE::IDX::X, 2); // x^2
    moment.yPow2 = dist.calc_moment(STATE::IDX::Y, 2); // y^2
    moment.cPow2 = dist.calc_cos_moment(STATE::IDX::YAW, 2); // cos(yaw)^2
    moment.sPow2 = dist.calc_sin_moment(STATE::IDX::YAW, 2); // sin(yaw)^2
    moment.yawPow2 = dist.calc_moment(STATE::IDX::YAW, 2); // yaw^2
    moment.xPow1_yPow1 = dist.calc_cross_second_moment(STATE::IDX::X, STATE::IDX::Y); // xy
    moment.cPow1_xPow1 = dist.calc_x_cos_z_moment(STATE::IDX::X, STATE::IDX::YAW); // x*cos(yaw)
    moment.sPow1_xPow1 = dist.calc_x_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW); // x*sin(yaw)
    moment.cPow1_yPow1 = dist.calc_x_cos_z_moment(STATE::IDX::Y, STATE::IDX::YAW); // y*cos(yaw)
    moment.sPow1_yPow1 = dist.calc_x_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW); // y*sin(yaw)
    moment.cPow1_sPow1 = dist.calc_cos_sin_moment(STATE::IDX::YAW, 1, 1); // cos(yaw)*sin(yaw)
    moment.xPow1_yawPow1 = dist.calc_cross_second_moment(STATE::IDX::X, STATE::IDX::YAW); // x*yaw
    moment.yPow1_yawPow1 = dist.calc_cross_second_moment(STATE::IDX::Y, STATE::IDX::YAW); // y*yaw
    moment.cPow1_yawPow1 = dist.calc_x_cos_x_moment(STATE::IDX::YAW, 1, 1); // yaw*cos(yaw)
    moment.sPow1_yawPow1 = dist.calc_x_sin_x_moment(STATE::IDX::YAW, 1, 1); // yaw*sin(yaw)
    const double& cPow2 = moment.cPow2;
    const double& sPow2 = moment.sPow2;
    const double& cPow1_xPow1 = moment.cPow1_xPow1;
    const double& sPow1_xPow1 = moment.sPow1_xPow1;
    const double& cPow1_yPow1 = moment.cPow1_yPow1;
    const double& sPow1_yPow1 = moment.sPow1_yPow1;
    const double& cPow1_sPow1 = moment.cPow1_sPow1;
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
    SimpleVehicleModel::Controls controls;
    controls.v = control_inputs(INPUT::IDX::V);
    controls.u = control_inputs(INPUT::IDX::U);
    controls.cu = std::cos(controls.u);
    controls.su = std::sin(controls.u);
    const double& v = controls.v;
    const double& u = controls.u;
    const double& cu = controls.cu;
    const double& su = controls.su;

    // Step4. System Noise
    const auto wv_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WV);
    const auto wu_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WU);
    SimpleVehicleModel::SystemNoiseMoments system_noise_moments;
    system_noise_moments.wvPow1 = wv_dist_ptr->calc_moment(1);
    system_noise_moments.wvPow2 = wv_dist_ptr->calc_moment(2);
    system_noise_moments.wuPow1 = wu_dist_ptr->calc_moment(1);
    system_noise_moments.wuPow2 = wu_dist_ptr->calc_moment(2);
    system_noise_moments.cwuPow1 = wu_dist_ptr->calc_cos_moment(1);
    system_noise_moments.swuPow1 = wu_dist_ptr->calc_sin_moment(1);
    system_noise_moments.swuPow2 = wu_dist_ptr->calc_sin_moment(2);
    system_noise_moments.cwuPow2 = wu_dist_ptr->calc_cos_moment(2);
    system_noise_moments.cwuPow1_swuPow1 = wu_dist_ptr->calc_cos_sin_moment(1, 1);
    system_noise_moments.wuPow1_cwuPow1 = wu_dist_ptr->calc_x_cos_moment(1, 1);
    system_noise_moments.wuPow1_swuPow1 = wu_dist_ptr->calc_x_sin_moment(1, 1);
    const double& wvPow1 = system_noise_moments.wvPow1;
    const double& wvPow2 = system_noise_moments.wvPow2;
    const double& cwuPow1 = system_noise_moments.cwuPow1;
    const double& swuPow1 = system_noise_moments.swuPow1;
    const double& cwuPow2 = system_noise_moments.cwuPow2;
    const double& swuPow2 = system_noise_moments.swuPow2;
    const double& cwuPow1_swuPow1 = system_noise_moments.cwuPow1_swuPow1;
    const double cwuPow1_wuPow1 = wu_dist_ptr->calc_x_cos_moment(1, 1);
    const double swuPow1_wuPow1 = wu_dist_ptr->calc_x_sin_moment(1, 1);


    // propagate moments
    const auto predicted_moment = vehicle_model_.propagateStateMoments(moment, system_noise_moments, controls);

    SimpleVehicleModel::HighOrderMoments result;
    result.xPow1 = predicted_moment.xPow1;
    result.yPow1 = predicted_moment.yPow1;
    result.yawPow1 = predicted_moment.yawPow1;
    result.cPow1 = predicted_moment.cPow1;
    result.sPow1 = predicted_moment.sPow1;

    result.xPow2 = predicted_moment.xPow2;
    result.yPow2 = predicted_moment.yPow2;
    result.yawPow2 = predicted_moment.yawPow2;
    result.cPow2 = predicted_moment.cPow2;
    result.sPow2 = predicted_moment.sPow2;
    result.xPow1_yPow1 = predicted_moment.xPow1_yPow1;
    result.xPow1_yawPow1 = predicted_moment.xPow1_yawPow1;
    result.yPow1_yawPow1 = predicted_moment.yPow1_yawPow1;
    result.xPow1_cPow1 = predicted_moment.cPow1_xPow1;
    result.yPow1_cPow1 = predicted_moment.cPow1_yPow1;
    result.xPow1_sPow1 = predicted_moment.sPow1_xPow1;
    result.yPow1_sPow1 = predicted_moment.sPow1_yPow1;
    result.cPow1_sPow1 = predicted_moment.cPow1_sPow1;
    result.yawPow1_cPow1 = predicted_moment.cPow1_yawPow1;
    result.yawPow1_sPow1 = predicted_moment.sPow1_yawPow1;

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
    result.xPow1_yawPow1_cPow1 = -cu*u*v*cPow1_sPow1*swuPow1 + cu*u*v*cPow2*cwuPow1 - cu*u*cPow1_sPow1*swuPow1*wvPow1 + cu*u*cPow1_xPow1*cwuPow1 + cu*u*cPow2*cwuPow1*wvPow1 - cu*u*sPow1_xPow1*swuPow1 - cu*v*cPow1_sPow1*swuPow1_wuPow1 - cu*v*cPow1_sPow1_yawPow1*swuPow1 + cu*v*cPow2*cwuPow1_wuPow1 + cu*v*cPow2_yawPow1*cwuPow1 - cu*cPow1_sPow1*swuPow1_wuPow1*wvPow1 - cu*cPow1_sPow1_yawPow1*swuPow1*wvPow1 + cu*cPow1_xPow1*cwuPow1_wuPow1 + cu*xPow1_cPow1_yawPow1*cwuPow1 + cu*cPow2*cwuPow1_wuPow1*wvPow1 + cu*cPow2_yawPow1*cwuPow1*wvPow1 - cu*sPow1_xPow1*swuPow1_wuPow1 - cu*xPow1_sPow1_yawPow1*swuPow1 - su*u*v*cPow1_sPow1*cwuPow1 - su*u*v*cPow2*swuPow1 - su*u*cPow1_sPow1*cwuPow1*wvPow1 - su*u*cPow1_xPow1*swuPow1 - su*u*cPow2*swuPow1*wvPow1 - su*u*cwuPow1*sPow1_xPow1 - su*v*cPow1_sPow1*cwuPow1_wuPow1 - su*v*cPow1_sPow1_yawPow1*cwuPow1 - su*v*cPow2*swuPow1_wuPow1 - su*v*cPow2_yawPow1*swuPow1 - su*cPow1_sPow1*cwuPow1_wuPow1*wvPow1 - su*cPow1_sPow1_yawPow1*cwuPow1*wvPow1 - su*cPow1_xPow1*swuPow1_wuPow1 - su*xPow1_cPow1_yawPow1*swuPow1 - su*cPow2*swuPow1_wuPow1*wvPow1 - su*cPow2_yawPow1*swuPow1*wvPow1 - su*cwuPow1*xPow1_sPow1_yawPow1 - su*cwuPow1_wuPow1*sPow1_xPow1;
    result.xPow1_yawPow1_sPow1 = cu*u*v*cPow1_sPow1*cwuPow1 + cu*u*v*cPow2*swuPow1 + cu*u*cPow1_sPow1*cwuPow1*wvPow1 + cu*u*cPow1_xPow1*swuPow1 + cu*u*cPow2*swuPow1*wvPow1 + cu*u*cwuPow1*sPow1_xPow1 + cu*v*cPow1_sPow1*cwuPow1_wuPow1 + cu*v*cPow1_sPow1_yawPow1*cwuPow1 + cu*v*cPow2*swuPow1_wuPow1 + cu*v*cPow2_yawPow1*swuPow1 + cu*cPow1_sPow1*cwuPow1_wuPow1*wvPow1 + cu*cPow1_sPow1_yawPow1*cwuPow1*wvPow1 + cu*cPow1_xPow1*swuPow1_wuPow1 + cu*xPow1_cPow1_yawPow1*swuPow1 + cu*cPow2*swuPow1_wuPow1*wvPow1 + cu*cPow2_yawPow1*swuPow1*wvPow1 + cu*cwuPow1*xPow1_sPow1_yawPow1 + cu*cwuPow1_wuPow1*sPow1_xPow1 - su*u*v*cPow1_sPow1*swuPow1 + su*u*v*cPow2*cwuPow1 - su*u*cPow1_sPow1*swuPow1*wvPow1 + su*u*cPow1_xPow1*cwuPow1 + su*u*cPow2*cwuPow1*wvPow1 - su*u*sPow1_xPow1*swuPow1 - su*v*cPow1_sPow1*swuPow1_wuPow1 - su*v*cPow1_sPow1_yawPow1*swuPow1 + su*v*cPow2*cwuPow1_wuPow1 + su*v*cPow2_yawPow1*cwuPow1 - su*cPow1_sPow1*swuPow1_wuPow1*wvPow1 - su*cPow1_sPow1_yawPow1*swuPow1*wvPow1 + su*cPow1_xPow1*cwuPow1_wuPow1 + su*xPow1_cPow1_yawPow1*cwuPow1 + su*cPow2*cwuPow1_wuPow1*wvPow1 + su*cPow2_yawPow1*cwuPow1*wvPow1 - su*sPow1_xPow1*swuPow1_wuPow1 - su*xPow1_sPow1_yawPow1*swuPow1;
    result.yPow1_yawPow1_cPow1 = cu*u*v*cPow1_sPow1*cwuPow1 - cu*u*v*sPow2*swuPow1 + cu*u*cPow1_sPow1*cwuPow1*wvPow1 + cu*u*cPow1_yPow1*cwuPow1 - cu*u*sPow1_yPow1*swuPow1 - cu*u*sPow2*swuPow1*wvPow1 + cu*v*cPow1_sPow1*cwuPow1_wuPow1 + cu*v*cPow1_sPow1_yawPow1*cwuPow1 - cu*v*sPow2*swuPow1_wuPow1 - cu*v*sPow2_yawPow1*swuPow1 + cu*cPow1_sPow1*cwuPow1_wuPow1*wvPow1 + cu*cPow1_sPow1_yawPow1*cwuPow1*wvPow1 + cu*cPow1_yPow1*cwuPow1_wuPow1 + cu*yPow1_cPow1_yawPow1*cwuPow1 - cu*sPow1_yPow1*swuPow1_wuPow1 - cu*yPow1_sPow1_yawPow1*swuPow1 - cu*sPow2*swuPow1_wuPow1*wvPow1 - cu*sPow2_yawPow1*swuPow1*wvPow1 - su*u*v*cPow1_sPow1*swuPow1 - su*u*v*cwuPow1*sPow2 - su*u*cPow1_sPow1*swuPow1*wvPow1 - su*u*cPow1_yPow1*swuPow1 - su*u*cwuPow1*sPow1_yPow1 - su*u*cwuPow1*sPow2*wvPow1 - su*v*cPow1_sPow1*swuPow1_wuPow1 - su*v*cPow1_sPow1_yawPow1*swuPow1 - su*v*cwuPow1*sPow2_yawPow1 - su*v*cwuPow1_wuPow1*sPow2 - su*cPow1_sPow1*swuPow1_wuPow1*wvPow1 - su*cPow1_sPow1_yawPow1*swuPow1*wvPow1 - su*cPow1_yPow1*swuPow1_wuPow1 - su*yPow1_cPow1_yawPow1*swuPow1 - su*cwuPow1*yPow1_sPow1_yawPow1 - su*cwuPow1*sPow2_yawPow1*wvPow1 - su*cwuPow1_wuPow1*sPow1_yPow1 - su*cwuPow1_wuPow1*sPow2*wvPow1;
    result.yPow1_yawPow1_sPow1 = cu*u*v*cPow1_sPow1*swuPow1 + cu*u*v*cwuPow1*sPow2 + cu*u*cPow1_sPow1*swuPow1*wvPow1 + cu*u*cPow1_yPow1*swuPow1 + cu*u*cwuPow1*sPow1_yPow1 + cu*u*cwuPow1*sPow2*wvPow1 + cu*v*cPow1_sPow1*swuPow1_wuPow1 + cu*v*cPow1_sPow1_yawPow1*swuPow1 + cu*v*cwuPow1*sPow2_yawPow1 + cu*v*cwuPow1_wuPow1*sPow2 + cu*cPow1_sPow1*swuPow1_wuPow1*wvPow1 + cu*cPow1_sPow1_yawPow1*swuPow1*wvPow1 + cu*cPow1_yPow1*swuPow1_wuPow1 + cu*yPow1_cPow1_yawPow1*swuPow1 + cu*cwuPow1*yPow1_sPow1_yawPow1 + cu*cwuPow1*sPow2_yawPow1*wvPow1 + cu*cwuPow1_wuPow1*sPow1_yPow1 + cu*cwuPow1_wuPow1*sPow2*wvPow1 + su*u*v*cPow1_sPow1*cwuPow1 - su*u*v*sPow2*swuPow1 + su*u*cPow1_sPow1*cwuPow1*wvPow1 + su*u*cPow1_yPow1*cwuPow1 - su*u*sPow1_yPow1*swuPow1 - su*u*sPow2*swuPow1*wvPow1 + su*v*cPow1_sPow1*cwuPow1_wuPow1 + su*v*cPow1_sPow1_yawPow1*cwuPow1 - su*v*sPow2*swuPow1_wuPow1 - su*v*sPow2_yawPow1*swuPow1 + su*cPow1_sPow1*cwuPow1_wuPow1*wvPow1 + su*cPow1_sPow1_yawPow1*cwuPow1*wvPow1 + su*cPow1_yPow1*cwuPow1_wuPow1 + su*yPow1_cPow1_yawPow1*cwuPow1 - su*sPow1_yPow1*swuPow1_wuPow1 - su*yPow1_sPow1_yawPow1*swuPow1 - su*sPow2*swuPow1_wuPow1*wvPow1 - su*sPow2_yawPow1*swuPow1*wvPow1;

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

    StateInfo predicted_info;
    predicted_info.mean = Eigen::VectorXd::Zero(3);
    predicted_info.covariance = Eigen::MatrixXd::Zero(3, 3);
    predicted_info.mean(STATE::IDX::X) = predicted_moment.xPow1;
    predicted_info.mean(STATE::IDX::Y) = predicted_moment.yPow1;
    predicted_info.mean(STATE::IDX::YAW)= predicted_moment.yawPow1;
    predicted_info.covariance(STATE::IDX::X, STATE::IDX::X) = predicted_moment.xPow2 - predicted_moment.xPow1 * predicted_moment.xPow1;
    predicted_info.covariance(STATE::IDX::Y, STATE::IDX::Y) = predicted_moment.yPow2 - predicted_moment.yPow1 * predicted_moment.yPow1;
    predicted_info.covariance(STATE::IDX::YAW, STATE::IDX::YAW) = predicted_moment.yawPow2 - predicted_moment.yawPow1 * predicted_moment.yawPow1;
    predicted_info.covariance(STATE::IDX::X, STATE::IDX::Y) = predicted_moment.xPow1_yPow1 - predicted_moment.xPow1 * predicted_moment.yPow1;
    predicted_info.covariance(STATE::IDX::X, STATE::IDX::YAW) = predicted_moment.xPow1_yawPow1 - predicted_moment.xPow1 * predicted_moment.yawPow1;
    predicted_info.covariance(STATE::IDX::Y, STATE::IDX::YAW) = predicted_moment.yPow1_yawPow1 - predicted_moment.yPow1 * predicted_moment.yawPow1;
    predicted_info.covariance(STATE::IDX::Y, STATE::IDX::X) = predicted_info.covariance(STATE::IDX::X, STATE::IDX::Y);
    predicted_info.covariance(STATE::IDX::YAW, STATE::IDX::X) = predicted_info.covariance(STATE::IDX::X, STATE::IDX::YAW);
    predicted_info.covariance(STATE::IDX::YAW, STATE::IDX::Y) = predicted_info.covariance(STATE::IDX::Y, STATE::IDX::YAW);

    return predicted_info;
}

StateInfo SimpleVehicleHMKF::update(const SimpleVehicleModel::HighOrderMoments & predicted_moments,
                                    const Eigen::VectorXd & observed_values,
                                    const Eigen::Vector2d & landmark,
                                    const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    // Step1. Create Observation Noise
    const auto wr_dist_ptr = noise_map.at(OBSERVATION_NOISE::IDX::WR);
    const auto wa_dist_ptr = noise_map.at(OBSERVATION_NOISE::IDX::WA);
    SimpleVehicleModel::ObservationNoiseMoments observation_noise;
    observation_noise.wrPow1 = wr_dist_ptr->calc_moment(1);
    observation_noise.wrPow2 = wr_dist_ptr->calc_moment(2);
    observation_noise.cwaPow1 = wa_dist_ptr->calc_cos_moment(1);
    observation_noise.swaPow1 = wa_dist_ptr->calc_sin_moment(1);
    observation_noise.cwaPow2 = wa_dist_ptr->calc_cos_moment(2);
    observation_noise.swaPow2 = wa_dist_ptr->calc_sin_moment(2);
    observation_noise.cwaPow1_swaPow1 = wa_dist_ptr->calc_cos_sin_moment(1, 1);

    // Step2. Get Observation Moments
    const auto observation_moments = vehicle_model_.getObservationMoments(predicted_moments, observation_noise, landmark);

    StateInfo observed_info;
    observed_info.mean = Eigen::VectorXd::Zero(2);
    observed_info.covariance = Eigen::MatrixXd::Zero(2, 2);
    observed_info.mean(OBSERVATION::IDX::RCOS) = observation_moments.rcosPow1;
    observed_info.mean(OBSERVATION::IDX::RSIN) = observation_moments.rsinPow1;
    observed_info.covariance(OBSERVATION::IDX::RCOS, OBSERVATION::IDX::RCOS) = observation_moments.rcosPow2 - observation_moments.rcosPow1*observation_moments.rcosPow1;
    observed_info.covariance(OBSERVATION::IDX::RSIN, OBSERVATION::IDX::RSIN) = observation_moments.rsinPow2 - observation_moments.rsinPow1*observation_moments.rsinPow1;
    observed_info.covariance(OBSERVATION::IDX::RCOS, OBSERVATION::IDX::RSIN) = observation_moments.rcosPow1_rsinPow1 - observation_moments.rcosPow1*observation_moments.rsinPow1;
    observed_info.covariance(OBSERVATION::IDX::RSIN, OBSERVATION::IDX::RCOS) = observed_info.covariance(OBSERVATION::IDX::RCOS, OBSERVATION::IDX::RSIN);

    const auto observation_mean = observed_info.mean;
    const auto observation_cov = observed_info.covariance;

    const double& x_land = landmark(0);
    const double& y_land = landmark(1);

    const double& wrPow1 = observation_noise.wrPow1;
    const double& cwaPow1 = observation_noise.cwaPow1;
    const double& swaPow1 = observation_noise.swaPow1;

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
    state_observation_cov(STATE::IDX::X, OBSERVATION::IDX::RCOS)
            = wrPow1 * cwaPow1 * xPow1_caPow1 - wrPow1 * swaPow1 * xPow1_saPow1
              - predicted_mean(STATE::IDX::X) * observation_mean(OBSERVATION::IDX::RCOS);
    state_observation_cov(STATE::IDX::X, OBSERVATION::IDX::RSIN)
            = wrPow1 * cwaPow1 * xPow1_saPow1 + wrPow1 * swaPow1 * xPow1_caPow1
              - predicted_mean(STATE::IDX::X) * observation_mean(OBSERVATION::IDX::RSIN); // x_p * yaw

    state_observation_cov(STATE::IDX::Y, OBSERVATION::IDX::RCOS)
            =  wrPow1 * cwaPow1 * yPow1_caPow1 - wrPow1 * swaPow1 * yPow1_saPow1
               - predicted_mean(STATE::IDX::Y) * observation_mean(OBSERVATION::IDX::RCOS); // yp * (xp^2 + yp^2)
    state_observation_cov(STATE::IDX::Y, OBSERVATION::IDX::RSIN)
            =  wrPow1 * cwaPow1 * yPow1_saPow1 + wrPow1 * swaPow1 * yPow1_caPow1
               - predicted_mean(STATE::IDX::Y) * observation_mean(OBSERVATION::IDX::RSIN); // y_p * yaw

    state_observation_cov(STATE::IDX::YAW, OBSERVATION::IDX::RCOS)
            =   wrPow1 * cwaPow1 * yawPow1_caPow1 - wrPow1 * swaPow1 * yawPow1_saPow1
                - predicted_mean(STATE::IDX::YAW) * observation_mean(OBSERVATION::IDX::RCOS);
    state_observation_cov(STATE::IDX::YAW, OBSERVATION::IDX::RSIN)
            =   wrPow1 * cwaPow1 * yawPow1_saPow1 + wrPow1 * swaPow1 * yawPow1_caPow1
                 - predicted_mean(STATE::IDX::YAW) * observation_mean(OBSERVATION::IDX::RSIN);

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
