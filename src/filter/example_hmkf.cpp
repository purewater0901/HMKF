#include "filter/example_hmkf.h"
#include "distribution/two_dimensional_normal_distribution.h"

using namespace Example;

void ExampleHMKF::predict(const Example::StateInfo &state,
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
    const double xPow3 = dist.calc_moment(STATE::IDX::X, 3); // x^3
    const double cPow3 = dist.calc_xy_cos_y_sin_y_moment(0, 0, 3, 0); // cos(yaw)^3
    const double cPow1_xPow2 = dist.calc_xy_cos_y_sin_y_moment(2, 0, 1, 0); // xx*cos(yaw)
    const double cPow2_xPow1 = dist.calc_xy_cos_y_sin_y_moment(1, 0, 2, 0); // x*cos(yaw)*cos(yaw)
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
    const double next_xPow1 = xPow1 + v*cPow1 + wxPow1;
    const double next_yawPow1 = yawPow1 + u + wyawPow1;
    const double next_xPow2 = xPow2 + v*v*cPow2 + wxPow2 + 2*v*cPow1_xPow1 + 2*xPow1*wxPow1 + 2*v*wxPow1*cPow1;
    const double next_yawPow2 = yawPow2 + u*u + wyawPow2 + 2*u*yawPow1 + 2*yawPow1*wyawPow1 + 2*u*wyawPow1;
    const double next_xPow1_yawPow1 = xPow1_yawPow1 + u*xPow1 + wyawPow1*xPow1
                                    + v*cPow1_yawPow1 + v*u*cPow1 + v*wyawPow1*cPow1
                                    + wxPow1*yawPow1 + wxPow1*u + wxPow1*wyawPow1;
    const double next_xPow3 = v*v*v*cPow3 + 3*v*v*wxPow1*cPow2 + 3*v*v*cPow2_xPow1
                            + 3*v*wxPow2*cPow1 + 6*v*wxPow1*cPow1_xPow1 +3*v*cPow1_xPow2
                            + wxPow3 + 3*wxPow2*xPow1 + 3*wxPow1*xPow2 + xPow3;
    const double next_xPow4 = std::pow(v,4) * cPow4 + 4*std::pow(v,3)*wxPow1*cPow3
                            + 4*std::pow(v,3) * xPow1_cPow3 + 6*v*v*wxPow2*cPow2
                            + 12*v*v*wxPow1*cPow2_xPow1 + 6*v*v*xPow2_cPow2
                            + 4*v*wxPow3*cPow1 + 12*v*wxPow2*cPow1_xPow1
                            + 12*v*wxPow1*cPow1_xPow2 + 4*v*xPow3_cPow1
                            + wxPow4 + 4*wxPow3*xPow1 + 6*wxPow2*xPow2
                            + 4*wxPow1*xPow3+xPow4;
}