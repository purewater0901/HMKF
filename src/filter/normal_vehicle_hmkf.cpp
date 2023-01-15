#include "filter/normal_vehicle_hmkf.h"
#include "distribution/three_dimensional_normal_distribution.h"

using namespace NormalVehicle;

PredictedMoments NormalVehicleHMKF::predict(const StateInfo& state_info,
                                            const Eigen::Vector2d & control_inputs,
                                            const double dt,
                                            const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
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

    // third order
    const double xPow3 = dist.calc_moment(STATE::IDX::X, 3);
    const double yPow3 = dist.calc_moment(STATE::IDX::Y, 3);
    const double cPow3 = dist.calc_cos_moment(STATE::IDX::YAW, 3);
    const double sPow3 = dist.calc_sin_moment(STATE::IDX::YAW, 3);
    const double cPow2_sPow1 = dist.calc_cos_sin_moment(STATE::IDX::YAW, 2, 1);
    const double cPow1_sPow2 = dist.calc_cos_sin_moment(STATE::IDX::YAW, 1, 2);
    const double xPow2_yPow1 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::Y, 2, 1, 0, 0);
    const double xPow1_yPow2 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::Y, 1, 2, 0, 0);
    const double cPow1_xPow2 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW, 2, 0, 1, 0);
    const double cPow1_yPow2 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW, 2, 0, 1, 0);
    const double sPow1_xPow2 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW, 2, 0, 0, 1);
    const double sPow1_yPow2 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW, 2, 0, 0, 1);
    const double cPow2_xPow1 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW, 1, 0, 2, 0);
    const double cPow2_yPow1 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW, 1, 0, 2, 0);
    const double sPow2_xPow1 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW, 1, 0, 0, 2);
    const double sPow2_yPow1 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW, 1, 0, 0, 2);
    const double cPow1_sPow1_xPow1 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW, 1, 0, 1, 1);
    const double cPow1_sPow1_yPow1 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW, 1, 0, 1, 1);

    // fourth order
    const double xPow4 = dist.calc_moment(STATE::IDX::X, 4);
    const double yPow4 = dist.calc_moment(STATE::IDX::Y, 4);
    const double xPow3_yPow1 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::Y, 3, 1, 0, 0);
    const double xPow1_yPow3 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::Y, 1, 3, 0, 0);
    const double cPow4 = dist.calc_cos_moment(STATE::IDX::YAW, 4);
    const double sPow4 = dist.calc_sin_moment(STATE::IDX::YAW, 4);
    const double cPow3_xPow1 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW, 1, 0, 3, 0);
    const double sPow3_xPow1 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW, 1, 0, 0, 3);
    const double cPow3_yPow1 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW, 1, 0, 3, 0);
    const double sPow3_yPow1 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW, 1, 0, 0, 3);
    const double cPow2_xPow2 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW, 1, 0, 2, 0);
    const double cPow2_yPow2 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW, 1, 0, 2, 0);
    const double sPow2_xPow2 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW, 1, 0, 0, 2);
    const double sPow2_yPow2 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW, 1, 0, 0, 2);
    const double cPow1_xPow3 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW, 3, 0, 1, 0);
    const double cPow1_yPow3 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW, 3, 0, 1, 0);
    const double sPow1_xPow3 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW, 3, 0, 0, 1);
    const double sPow1_yPow3 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW, 3, 0, 0, 1);
    const double cPow3_sPow1 = dist.calc_cos_sin_moment(STATE::IDX::YAW, 3, 1);
    const double cPow1_sPow3 = dist.calc_cos_sin_moment(STATE::IDX::YAW, 1, 3);
    const double cPow2_sPow2 = dist.calc_cos_sin_moment(STATE::IDX::YAW, 2, 2);
    const double cPow2_sPow1_xPow1 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW, 1, 0, 2, 1);
    const double cPow2_sPow1_yPow1 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW, 1, 0, 2, 1);
    const double cPow1_sPow2_xPow1 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW, 1, 0, 1, 2);
    const double cPow1_sPow2_yPow1 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW, 1, 0, 1, 2);
    const double cPow1_sPow1_xPow2 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW, 2, 0, 1, 1);
    const double cPow1_sPow1_yPow2 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW, 2, 0, 1, 1);

    // fifth order
    const double xPow5 = dist.calc_moment(STATE::IDX::X, 5);
    const double yPow5 = dist.calc_moment(STATE::IDX::Y, 5);
    const double cPow5 = dist.calc_cos_moment(STATE::IDX::YAW, 5);
    const double sPow5 = dist.calc_sin_moment(STATE::IDX::YAW, 5);
    const double xPow1_yPow4 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::Y, 1, 4, 0, 0);
    const double xPow4_yPow1 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::Y, 4, 1, 0, 0);
    const double cPow1_xPow4 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW, 4, 0, 1, 0);
    const double sPow1_xPow4 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW, 4, 0, 0, 1);
    const double cPow1_yPow4 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW, 4, 0, 1, 0);
    const double sPow1_yPow4 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW, 4, 0, 0, 1);
    const double cPow4_xPow1 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW, 1, 0, 4, 0);
    const double sPow4_xPow1 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW, 1, 0, 0, 4);
    const double cPow4_yPow1 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW, 1, 0, 4, 0);
    const double sPow4_yPow1 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW, 1, 0, 0, 4);
    const double cPow3_xPow2 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW, 2, 0, 3, 0);
    const double cPow3_yPow2 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW, 2, 0, 3, 0);
    const double sPow3_xPow2 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW, 2, 0, 0, 3);
    const double sPow3_yPow2 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW, 2, 0, 0, 3);
    const double cPow2_xPow3 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW, 3, 0, 2, 0);
    const double cPow2_yPow3 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW, 3, 0, 2, 0);
    const double sPow2_xPow3 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW, 3, 0, 0, 2);
    const double sPow2_yPow3 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW, 3, 0, 0, 2);
    const double cPow1_sPow4 = dist.calc_cos_sin_moment(STATE::IDX::YAW, 1, 4);
    const double cPow4_sPow1 = dist.calc_cos_sin_moment(STATE::IDX::YAW, 4, 1);
    const double cPow3_sPow1_xPow1 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW, 1, 0, 3, 1);
    const double cPow3_sPow1_yPow1 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW, 1, 0, 3, 1);
    const double cPow1_sPow3_xPow1 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW, 1, 0, 1, 3);
    const double cPow1_sPow3_yPow1 = dist.calc_xy_cos_y_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW, 1, 0, 1, 3);

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
    const double wxPow3 = wx_dist_ptr->calc_moment(3);
    const double wyPow3 = wy_dist_ptr->calc_moment(3);
    const double wxPow4 = wx_dist_ptr->calc_moment(4);
    const double wyPow4 = wy_dist_ptr->calc_moment(4);
    const double wxPow5 = wx_dist_ptr->calc_moment(5);
    const double wyPow5 = wy_dist_ptr->calc_moment(5);

    // Dynamics updates.
    PredictedMoments next_moments;
    next_moments.xPow1 = xPow1 + v*cPow1 + wxPow1;
    next_moments.yPow1 = yPow1 + v*sPow1 + wyPow1;
    next_moments.yawPow1 = yawPow1 + u + wyawPow1;

    next_moments.xPow2 = xPow2 + v*v*cPow2 + wxPow2 + 2*v*cPow1*wxPow1 + 2*v*cPow1_xPow1 + 2*wxPow1*xPow1;
    next_moments.yPow2 = yPow2 + v*v*sPow2 + wyPow2 + 2*v*sPow1*wyPow1 + 2*v*sPow1_yPow1 + 2*wyPow1*yPow1;
    next_moments.yawPow2 = yawPow2 + u*u + wyawPow2 + 2*u*wyawPow1 + 2*u*yawPow1 + 2*wyawPow1*yawPow1;
    next_moments.xPow1_yPow1 = xPow1_yPow1 + v*sPow1_xPow1 + xPow1*wyPow1
                                    + v*cPow1_yPow1 + v*v*cPow1_sPow1 + v*cPow1*wyPow1
                                    + wxPow1*yPow1 + v*sPow1*wxPow1 + wxPow1*wyPow1;
    next_moments.xPow1_yawPow1 = u*v*cPow1 + u*wxPow1 + u*xPow1 + v*cPow1*wyawPow1 + v*cPow1_yawPow1
                                      + wxPow1*wyawPow1 + wxPow1*yawPow1 + wyawPow1*xPow1 + xPow1_yawPow1;
    next_moments.yPow1_yawPow1 = u*v*sPow1 + u*wyPow1 + u*yPow1 + v*sPow1*wyawPow1 + v*sPow1_yawPow1
                                      + wyPow1*wyawPow1 + wyPow1*yawPow1 + wyawPow1*yPow1 + yPow1_yawPow1;
    next_moments.xPow4 = pow(v, 4)*cPow4 + 4*pow(v, 3)*cPow3*wxPow1 + 4*pow(v, 3)*cPow3_xPow1 + 6*pow(v, 2)*cPow2*wxPow2 + 12*pow(v, 2)*cPow2_xPow1*wxPow1 + 6*pow(v, 2)*cPow2_xPow2 + 4*v*cPow1*wxPow3 + 12*v*cPow1_xPow1*wxPow2 + 12*v*cPow1_xPow2*wxPow1 + 4*v*cPow1_xPow3 + 4*wxPow1*xPow3 + 6*wxPow2*xPow2 + 4*wxPow3*xPow1 + wxPow4 + xPow4;
    next_moments.yPow4 = pow(v, 4)*sPow4 + 4*pow(v, 3)*sPow3*wyPow1 + 4*pow(v, 3)*sPow3_yPow1 + 6*pow(v, 2)*sPow2*wyPow2 + 12*pow(v, 2)*sPow2_yPow1*wyPow1 + 6*pow(v, 2)*sPow2_yPow2 + 4*v*sPow1*wyPow3 + 12*v*sPow1_yPow1*wyPow2 + 12*v*sPow1_yPow2*wyPow1 + 4*v*sPow1_yPow3 + 4*wyPow1*yPow3 + 6*wyPow2*yPow2 + 4*wyPow3*yPow1 + wyPow4 + yPow4;
    next_moments.xPow5 = pow(v, 5)*cPow5 + 5*pow(v, 4)*cPow4*wxPow1 + 5*pow(v, 4)*cPow4_xPow1 + 10*pow(v, 3)*cPow3*wxPow2 + 20*pow(v, 3)*cPow3_xPow1*wxPow1 + 10*pow(v, 3)*cPow3_xPow2 + 10*pow(v, 2)*cPow2*wxPow3 + 30*pow(v, 2)*cPow2_xPow1*wxPow2 + 30*pow(v, 2)*cPow2_xPow2*wxPow1 + 10*pow(v, 2)*cPow2_xPow3 + 5*v*cPow1*wxPow4 + 20*v*cPow1_xPow1*wxPow3 + 30*v*cPow1_xPow2*wxPow2 + 20*v*cPow1_xPow3*wxPow1 + 5*v*cPow1_xPow4 + 5*wxPow1*xPow4 + 10*wxPow2*xPow3 + 10*wxPow3*xPow2 + 5*wxPow4*xPow1 + wxPow5 + xPow5;
    next_moments.yPow5 = pow(v, 5)*sPow5 + 5*pow(v, 4)*sPow4*wyPow1 + 5*pow(v, 4)*sPow4_yPow1 + 10*pow(v, 3)*sPow3*wyPow2 + 20*pow(v, 3)*sPow3_yPow1*wyPow1 + 10*pow(v, 3)*sPow3_yPow2 + 10*pow(v, 2)*sPow2*wyPow3 + 30*pow(v, 2)*sPow2_yPow1*wyPow2 + 30*pow(v, 2)*sPow2_yPow2*wyPow1 + 10*pow(v, 2)*sPow2_yPow3 + 5*v*sPow1*wyPow4 + 20*v*sPow1_yPow1*wyPow3 + 30*v*sPow1_yPow2*wyPow2 + 20*v*sPow1_yPow3*wyPow1 + 5*v*sPow1_yPow4 + 5*wyPow1*yPow4 + 10*wyPow2*yPow3 + 10*wyPow3*yPow2 + 5*wyPow4*yPow1 + wyPow5 + yPow5;
    //next_moments.xPow1_yPow4 = pow(v, 5)*cPow1_sPow4 + 4*pow(v, 4)*cPow1_sPow3*wyPow1 + 4*pow(v, 4)*cPow1_sPow3_yPow1 + pow(v, 4)*sPow4*wxPow1 + pow(v, 4)*sPow4_xPow1 + 6*pow(v, 3)*cPow1_sPow2*wyPow2 + 12*pow(v, 3)*cPow1_sPow2_yPow1*wyPow1 + 6*pow(v, 3)*cPow1_sPow2_yPow2 + 4*pow(v, 3)*sPow3*wxPow1*wyPow1 + 4*pow(v, 3)*sPow3_xPow1*wyPow1 + 4*pow(v, 3)*sPow3_xPow1_yPow1 + 4*pow(v, 3)*sPow3_yPow1*wxPow1 + 4*pow(v, 2)*cPow1_sPow1*wyPow3 + 12*pow(v, 2)*cPow1_sPow1_yPow1*wyPow2 + 12*pow(v, 2)*cPow1_sPow1_yPow2*wyPow1 + 4*pow(v, 2)*cPow1_sPow1_yPow3 + 6*pow(v, 2)*sPow2*wxPow1*wyPow2 + 6*pow(v, 2)*sPow2_xPow1*wyPow2 + 12*pow(v, 2)*sPow2_xPow1_yPow1*wyPow1 + 6*pow(v, 2)*sPow2_xPow1_yPow2 + 12*pow(v, 2)*sPow2_yPow1*wxPow1*wyPow1 + 6*pow(v, 2)*sPow2_yPow2*wxPow1 + v*cPow1*wyPow4 + 4*v*cPow1_yPow1*wyPow3 + 6*v*cPow1_yPow2*wyPow2 + 4*v*cPow1_yPow3*wyPow1 + v*cPow1_yPow4 + 4*v*sPow1*wxPow1*wyPow3 + 4*v*sPow1_xPow1*wyPow3 + 12*v*sPow1_xPow1_yPow1*wyPow2 + 12*v*sPow1_xPow1_yPow2*wyPow1 + 4*v*sPow1_xPow1_yPow3 + 12*v*sPow1_yPow1*wxPow1*wyPow2 + 12*v*sPow1_yPow2*wxPow1*wyPow1 + 4*v*sPow1_yPow3*wxPow1 + 4*wxPow1*wyPow1*yPow3 + 6*wxPow1*wyPow2*yPow2 + 4*wxPow1*wyPow3*yPow1 + wxPow1*wyPow4 + wxPow1*yPow4 + 4*wyPow1*xPow1_yPow3 + 6*wyPow2*xPow1_yPow2 + 4*wyPow3*xPow1_yPow1 + wyPow4*xPow1 + xPow1_yPow4;
    //next_moments.xPow4_yPow1 = pow(v, 5)*cPow4_sPow1 + 4*pow(v, 4)*cPow3_sPow1*wxPow1 + 4*pow(v, 4)*cPow3_sPow1_xPow1 + pow(v, 4)*cPow4*wyPow1 + pow(v, 4)*cPow4_yPow1 + 6*pow(v, 3)*cPow2_sPow1*wxPow2 + 12*pow(v, 3)*cPow2_sPow1_xPow1*wxPow1 + 6*pow(v, 3)*cPow2_sPow1_xPow2 + 4*pow(v, 3)*cPow3*wxPow1*wyPow1 + 4*pow(v, 3)*cPow3_xPow1*wyPow1 + 4*pow(v, 3)*cPow3_xPow1_yPow1 + 4*pow(v, 3)*cPow3_yPow1*wxPow1 + 4*pow(v, 2)*cPow1_sPow1*wxPow3 + 12*pow(v, 2)*cPow1_sPow1_xPow1*wxPow2 + 12*pow(v, 2)*cPow1_sPow1_xPow2*wxPow1 + 4*pow(v, 2)*cPow1_sPow1_xPow3 + 6*pow(v, 2)*cPow2*wxPow2*wyPow1 + 12*pow(v, 2)*cPow2_xPow1*wxPow1*wyPow1 + 12*pow(v, 2)*cPow2_xPow1_yPow1*wxPow1 + 6*pow(v, 2)*cPow2_xPow2*wyPow1 + 6*pow(v, 2)*cPow2_xPow2_yPow1 + 6*pow(v, 2)*cPow2_yPow1*wxPow2 + 4*v*cPow1*wxPow3*wyPow1 + 12*v*cPow1_xPow1*wxPow2*wyPow1 + 12*v*cPow1_xPow1_yPow1*wxPow2 + 12*v*cPow1_xPow2*wxPow1*wyPow1 + 12*v*cPow1_xPow2_yPow1*wxPow1 + 4*v*cPow1_xPow3*wyPow1 + 4*v*cPow1_xPow3_yPow1 + 4*v*cPow1_yPow1*wxPow3 + v*sPow1*wxPow4 + 4*v*sPow1_xPow1*wxPow3 + 6*v*sPow1_xPow2*wxPow2 + 4*v*sPow1_xPow3*wxPow1 + v*sPow1_xPow4 + 4*wxPow1*wyPow1*xPow3 + 4*wxPow1*xPow3_yPow1 + 6*wxPow2*wyPow1*xPow2 + 6*wxPow2*xPow2_yPow1 + 4*wxPow3*wyPow1*xPow1 + 4*wxPow3*xPow1_yPow1 + wxPow4*wyPow1 + wxPow4*yPow1 + wyPow1*xPow4 + xPow4_yPow1;


    return next_moments;
}

StateInfo NormalVehicleHMKF::update(const PredictedMoments & predicted_moments,
                                    const Eigen::VectorXd & observed_values,
                                    const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    const auto measurement_info = getMeasurementMoments(predicted_moments, noise_map);
    const auto state_observation_cov = getStateMeasurementMatrix(predicted_moments, measurement_info, noise_map);

    Eigen::VectorXd predicted_mean = Eigen::VectorXd::Zero(3);
    predicted_mean(0) = predicted_moments.xPow1;
    predicted_mean(1) = predicted_moments.yPow1;
    predicted_mean(2) = predicted_moments.yawPow1;
    Eigen::MatrixXd predicted_cov = Eigen::MatrixXd::Zero(3, 3);
    predicted_cov(0,0) = predicted_moments.xPow2 - predicted_moments.xPow1*predicted_moments.xPow1;
    predicted_cov(1,1) = predicted_moments.yPow2 - predicted_moments.yPow1*predicted_moments.yPow1;
    predicted_cov(2,2) = predicted_moments.yawPow2 - predicted_moments.yawPow1*predicted_moments.yawPow1;
    predicted_cov(0,1) = predicted_moments.xPow1_yPow1 - predicted_moments.xPow1*predicted_moments.yPow1;
    predicted_cov(0,2) = predicted_moments.xPow1_yawPow1 - predicted_moments.xPow1*predicted_moments.yawPow1;
    predicted_cov(1,2) = predicted_moments.yPow1_yawPow1 - predicted_moments.yPow1*predicted_moments.yawPow1;
    predicted_cov(1,0) = predicted_cov(0,1);
    predicted_cov(2,0) = predicted_cov(0,2);
    predicted_cov(2,1) = predicted_cov(1,2);

    Eigen::VectorXd measurement_mean = Eigen::VectorXd::Zero(2);
    measurement_mean(0) = measurement_info.rPow1;
    measurement_mean(1) = measurement_info.yawPow1;
    Eigen::MatrixXd measurement_cov = Eigen::MatrixXd::Zero(2,2);
    measurement_cov(0,0) = measurement_info.rPow2 - measurement_info.rPow1*measurement_info.rPow1;
    measurement_cov(1,1) = measurement_info.yawPow2 - measurement_info.yawPow1*measurement_info.yawPow1;
    measurement_cov(0,1) = measurement_info.rPow1_yawPow1 - measurement_info.rPow1*measurement_info.yawPow1;
    measurement_cov(1,0) = measurement_cov(0,1);

    // Kalman Gain
    const auto K = state_observation_cov * measurement_cov.inverse();

    StateInfo updated_info;
    updated_info.mean = predicted_mean + K * (observed_values - measurement_mean);
    updated_info.covariance = predicted_cov - K * measurement_cov * K.transpose();

    return updated_info;
}

MeasurementMoments NormalVehicleHMKFgetMeasurementMoments(const PredictedMoments & predicted_moments,
                                                          const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    const double& yawPow1 = predicted_moments.yawPow1;
    const double& yawPow2 = predicted_moments.yawPow2;
    const double& xPow4 = predicted_moments.xPow4;
    const double& yPow4 = predicted_moments.yPow4;
    const double& xPow4_yawPow1 = predicted_moments.xPow4_yawPow1;
    const double& yPow4_yawPow1 = predicted_moments.yPow4_yawPow1;
    const double& xPow8 = predicted_moments.xPow8;
    const double& yPow8 = predicted_moments.yPow8;
    const double& xPow4_yPow4 = predicted_moments.xPow4_yPow4;

    // Observation Noise
    const auto wr_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WR);
    const auto wyaw_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WYAW);
    const double mrPow1 = wr_dist_ptr->calc_moment(1);
    const double mrPow2 = wr_dist_ptr->calc_moment(2);
    const double myawPow1 = wyaw_dist_ptr->calc_moment(1);
    const double myawPow2 = wyaw_dist_ptr->calc_moment(2);

    // measurement update
    MeasurementMoments meas_moments;
    meas_moments.rPow1 = xPow4 + yPow4 + mrPow1;
    meas_moments.rPow2 = xPow8 + yPow8 + mrPow2 + 2.0*xPow4_yPow4 + 2.0*xPow4*mrPow1 + 2.0*yPow4*mrPow1;
    meas_moments.yawPow1 = yawPow1 + myawPow1;
    meas_moments.yawPow2 = yawPow2 + myawPow2 + 2.0*yawPow1*myawPow1;
    meas_moments.rPow1_yawPow1 = xPow4_yawPow1 + yPow4_yawPow1 + yawPow1*mrPow1 + xPow4*myawPow1
                                 + yPow4*myawPow1 + mrPow1*myawPow1;

    return meas_moments;
}

Eigen::MatrixXd NormalVehicleHMKF::getStateMeasurementMatrix(const PredictedMoments& predicted_moments,
                                                             const MeasurementMoments & measurement_moments,
                                                             const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    // Observation Noise
    const auto wr_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WR);
    const auto wyaw_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WYAW);
    const double wrPow1 = wr_dist_ptr->calc_moment(1);
    const double wyawPow1 = wyaw_dist_ptr->calc_moment(1);

    // State Moment
    const double& xPow1 = predicted_moments.xPow1;
    const double& yPow1 = predicted_moments.yPow1;
    const double& yawPow1 = predicted_moments.yawPow1;
    const double& xPow1_yawPow1 = predicted_moments.xPow1_yawPow1;
    const double& yPow1_yawPow1 = predicted_moments.yPow1_yawPow1;
    const double& yawPow2 = predicted_moments.yawPow2;
    const double& xPow5 = predicted_moments.xPow5;
    const double& yPow5 = predicted_moments.yPow5;
    const double& xPow1_yPow4 = predicted_moments.xPow1_yPow4;
    const double& xPow4_yPow1 = predicted_moments.xPow4_yPow1;
    const double& xPow4_yawPow1 = predicted_moments.xPow4_yawPow1;
    const double& yPow4_yawPow1 = predicted_moments.yPow4_yawPow1;

    const double& mrPow1 = measurement_moments.rPow1;
    const double& myawPow1 = measurement_moments.yawPow1;

    Eigen::MatrixXd state_observation_cov(3, 2); // sigma = E[XY^T] - E[X]E[Y]^T
    state_observation_cov(STATE::IDX::X, MEASUREMENT::IDX::R)
            = xPow5 + xPow1_yPow4 + xPow1 * wrPow1 - xPow1 * mrPow1; // xp * (xp^4 + yp^4 + mr)
    state_observation_cov(STATE::IDX::X, MEASUREMENT::IDX::YAW)
            = xPow1_yawPow1 + xPow1 * wyawPow1 - xPow1 * myawPow1; // x_p * (yaw + myaw)
    state_observation_cov(STATE::IDX::Y, MEASUREMENT::IDX::R)
            = xPow4_yPow1 + yPow5  + yPow1 * wrPow1 - yPow1 * mrPow1; // yp * (xp^4 + yp^4 + mr)
    state_observation_cov(STATE::IDX::Y, MEASUREMENT::IDX::YAW)
            = yPow1_yawPow1 + yPow1 * wyawPow1 - yPow1 * myawPow1; // y_p * (yaw + myaw)
    state_observation_cov(STATE::IDX::YAW, MEASUREMENT::IDX::R)
            = xPow4_yawPow1 + yPow4_yawPow1 + yawPow1 * wrPow1 - yawPow1 * mrPow1; // yaw_p * (x_p^4 + y_p^4 + mr)
    state_observation_cov(STATE::IDX::YAW, MEASUREMENT::IDX::YAW)
            = yawPow2 + yawPow1 * wyawPow1 - yawPow1 * myawPow1; // yaw_p * (yaw_p + myaw)

    return state_observation_cov;

}