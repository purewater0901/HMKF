#include "filter/simple_vehicle_squared_hmkf.h"
#include "distribution/three_dimensional_normal_distribution.h"

using namespace SimpleVehicleSquared;

SimpleVehicleSquaredHMKF::SimpleVehicleSquaredHMKF(const std::shared_ptr<BaseModel>& vehicle_model)
: vehicle_model_(vehicle_model)
{
}

StateInfo SimpleVehicleSquaredHMKF::update(const StateInfo& state_info,
                                           const SimpleVehicleSquared::HighOrderMoments & predicted_moments,
                                           const Eigen::VectorXd & observed_values,
                                           const Eigen::Vector2d & landmark,
                                           const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    // Step1. Create Predicted Moments
    const double& x_land = landmark(0);
    const double& y_land = landmark(1);

    const double& cPow2 = predicted_moments.cPow2;
    const double& sPow2 = predicted_moments.sPow2;
    const double& cPow1_sPow1 = predicted_moments.cPow1_sPow1;

    const double& xPow1_cPow2 = predicted_moments.xPow1_cPow2;
    const double& xPow1_sPow2 = predicted_moments.xPow1_sPow2;
    const double& yPow1_cPow2 = predicted_moments.yPow1_cPow2;
    const double& yPow1_sPow2 = predicted_moments.yPow1_sPow2;
    const double& xPow1_cPow1_sPow1 = predicted_moments.xPow1_cPow1_sPow1;
    const double& yPow1_cPow1_sPow1 = predicted_moments.yPow1_cPow1_sPow1;
    const double& yawPow1_cPow2 = predicted_moments.yawPow1_cPow2;
    const double& yawPow1_sPow2 = predicted_moments.yawPow1_sPow2;
    const double& yawPow1_cPow1_sPow1 = predicted_moments.yawPow1_cPow1_sPow1;

    const double& cPow4 = predicted_moments.cPow4;
    const double& sPow4 = predicted_moments.sPow4;
    const double& cPow1_sPow3 = predicted_moments.cPow1_sPow3;
    const double& cPow3_sPow1 = predicted_moments.cPow3_sPow1;
    const double& cPow2_sPow2 = predicted_moments.cPow2_sPow2;
    const double& xPow2_cPow2 = predicted_moments.xPow2_cPow2;
    const double& xPow2_sPow2 = predicted_moments.xPow2_sPow2;
    const double& yPow2_cPow2 = predicted_moments.yPow2_cPow2;
    const double& yPow2_sPow2 = predicted_moments.yPow2_sPow2;
    const double& xPow2_cPow1_sPow1 = predicted_moments.xPow2_cPow1_sPow1;
    const double& yPow2_cPow1_sPow1 = predicted_moments.yPow2_cPow1_sPow1;
    const double& xPow1_yPow1_cPow2 = predicted_moments.xPow1_yPow1_cPow2;
    const double& xPow1_yPow1_sPow2 = predicted_moments.xPow1_yPow1_sPow2;
    const double& xPow1_yPow1_cPow1_sPow1 = predicted_moments.xPow1_yPow1_cPow1_sPow1;
    const double& xPow1_yawPow1_cPow1_sPow1 = predicted_moments.xPow1_yawPow1_cPow1_sPow1;
    const double& yPow1_yawPow1_cPow1_sPow1 = predicted_moments.yPow1_yawPow1_cPow1_sPow1;
    const double& xPow1_yawPow1_cPow2 = predicted_moments.xPow1_yawPow1_cPow2;
    const double& xPow1_yawPow1_sPow2 = predicted_moments.xPow1_yawPow1_sPow2;
    const double& yPow1_yawPow1_cPow2 = predicted_moments.yPow1_yawPow1_cPow2;
    const double& yPow1_yawPow1_sPow2 = predicted_moments.yPow1_yawPow1_sPow2;

    const double& xPow1_cPow4 = predicted_moments.xPow1_cPow4;
    const double& xPow1_sPow4 = predicted_moments.xPow1_sPow4;
    const double& yPow1_cPow4 = predicted_moments.yPow1_cPow4;
    const double& yPow1_sPow4 = predicted_moments.yPow1_sPow4;
    const double& xPow1_cPow3_sPow1 = predicted_moments.xPow1_cPow3_sPow1;
    const double& xPow1_cPow2_sPow2 = predicted_moments.xPow1_cPow2_sPow2;
    const double& xPow1_cPow1_sPow3 = predicted_moments.xPow1_cPow1_sPow3;
    const double& yPow1_cPow3_sPow1 = predicted_moments.yPow1_cPow3_sPow1;
    const double& yPow1_cPow2_sPow2 = predicted_moments.yPow1_cPow2_sPow2;
    const double& yPow1_cPow1_sPow3 = predicted_moments.yPow1_cPow1_sPow3;
    const double& xPow3_cPow2 = predicted_moments.xPow3_cPow2;
    const double& xPow3_sPow2 = predicted_moments.xPow3_sPow2;
    const double& yPow3_cPow2 = predicted_moments.yPow3_cPow2;
    const double& yPow3_sPow2 = predicted_moments.yPow3_sPow2;
    const double& xPow1_yPow2_cPow1_sPow1 = predicted_moments.xPow1_yPow2_cPow1_sPow1;
    const double& xPow2_yPow1_cPow1_sPow1 = predicted_moments.xPow2_yPow1_cPow1_sPow1;
    const double& xPow3_cPow1_sPow1 = predicted_moments.xPow3_cPow1_sPow1;
    const double& yPow3_cPow1_sPow1 = predicted_moments.yPow3_cPow1_sPow1;
    const double& xPow1_yPow2_cPow2 = predicted_moments.xPow1_yPow2_cPow2;
    const double& xPow1_yPow2_sPow2 = predicted_moments.xPow1_yPow2_sPow2;
    const double& xPow2_yPow1_cPow2 = predicted_moments.xPow2_yPow1_cPow2;
    const double& xPow2_yPow1_sPow2 = predicted_moments.xPow2_yPow1_sPow2;
    const double& xPow2_yawPow1_cPow2 = predicted_moments.xPow2_yawPow1_cPow2;
    const double& xPow2_yawPow1_sPow2 = predicted_moments.xPow2_yawPow1_sPow2;
    const double& yPow2_yawPow1_cPow2 = predicted_moments.yPow2_yawPow1_cPow2;
    const double& yPow2_yawPow1_sPow2 = predicted_moments.yPow2_yawPow1_sPow2;
    const double& xPow2_yawPow1_cPow1_sPow1 = predicted_moments.xPow2_yawPow1_cPow1_sPow1;
    const double& yPow2_yawPow1_cPow1_sPow1 = predicted_moments.yPow2_yawPow1_cPow1_sPow1;
    const double& xPow1_yPow1_yawPow1_cPow1_sPow1 = predicted_moments.xPow1_yPow1_yawPow1_cPow1_sPow1;
    const double& xPow1_yPow1_yawPow1_cPow2 = predicted_moments.xPow1_yPow1_yawPow1_cPow2;
    const double& xPow1_yPow1_yawPow1_sPow2 = predicted_moments.xPow1_yPow1_yawPow1_sPow2;

    const double& xPow2_cPow4 = predicted_moments.xPow2_cPow4;
    const double& xPow2_sPow4 = predicted_moments.xPow2_sPow4;
    const double& yPow2_cPow4 = predicted_moments.yPow2_cPow4;
    const double& yPow2_sPow4 = predicted_moments.yPow2_sPow4;
    const double& xPow1_yPow1_cPow4 = predicted_moments.xPow1_yPow1_cPow4;
    const double& xPow1_yPow1_sPow4 = predicted_moments.xPow1_yPow1_sPow4;
    const double& xPow2_cPow2_sPow2 = predicted_moments.xPow2_cPow2_sPow2;
    const double& xPow2_cPow3_sPow1 = predicted_moments.xPow2_cPow3_sPow1;
    const double& xPow2_cPow1_sPow3 = predicted_moments.xPow2_cPow1_sPow3;
    const double& yPow2_cPow2_sPow2 = predicted_moments.yPow2_cPow2_sPow2;
    const double& yPow2_cPow3_sPow1 = predicted_moments.yPow2_cPow3_sPow1;
    const double& yPow2_cPow1_sPow3 = predicted_moments.yPow2_cPow1_sPow3;
    const double& xPow1_yPow1_cPow3_sPow1 = predicted_moments.xPow1_yPow1_cPow3_sPow1;
    const double& xPow1_yPow1_cPow2_sPow2 = predicted_moments.xPow1_yPow1_cPow2_sPow2;
    const double& xPow1_yPow1_cPow1_sPow3 = predicted_moments.xPow1_yPow1_cPow1_sPow3;

    const double& xPow3_cPow4 = predicted_moments.xPow3_cPow4;
    const double& xPow3_sPow4 = predicted_moments.xPow3_sPow4;
    const double& yPow3_cPow4 = predicted_moments.yPow3_cPow4;
    const double& yPow3_sPow4 = predicted_moments.yPow3_sPow4;
    const double& xPow1_yPow2_cPow4 = predicted_moments.xPow1_yPow2_cPow4;
    const double& xPow1_yPow2_sPow4 = predicted_moments.xPow1_yPow2_sPow4;
    const double& xPow2_yPow1_cPow4 = predicted_moments.xPow2_yPow1_cPow4;
    const double& xPow2_yPow1_sPow4 = predicted_moments.xPow2_yPow1_sPow4;
    const double& xPow3_cPow3_sPow1 = predicted_moments.xPow3_cPow3_sPow1;
    const double& xPow3_cPow1_sPow3 = predicted_moments.xPow3_cPow1_sPow3;
    const double& xPow3_cPow2_sPow2 = predicted_moments.xPow3_cPow2_sPow2;
    const double& yPow3_cPow3_sPow1 = predicted_moments.yPow3_cPow3_sPow1;
    const double& yPow3_cPow1_sPow3 = predicted_moments.yPow3_cPow1_sPow3;
    const double& yPow3_cPow2_sPow2 = predicted_moments.yPow3_cPow2_sPow2;
    const double& xPow2_yPow1_cPow3_sPow1 = predicted_moments.xPow2_yPow1_cPow3_sPow1;
    const double& xPow2_yPow1_cPow1_sPow3 = predicted_moments.xPow2_yPow1_cPow1_sPow3;
    const double& xPow2_yPow1_cPow2_sPow2 = predicted_moments.xPow2_yPow1_cPow2_sPow2;
    const double& xPow1_yPow2_cPow2_sPow2 = predicted_moments.xPow1_yPow2_cPow2_sPow2;
    const double& xPow1_yPow2_cPow1_sPow3 = predicted_moments.xPow1_yPow2_cPow1_sPow3;
    const double& xPow1_yPow2_cPow3_sPow1 = predicted_moments.xPow1_yPow2_cPow3_sPow1;

    const double& xPow4_cPow4 = predicted_moments.xPow4_cPow4;
    const double& xPow4_sPow4 = predicted_moments.xPow4_sPow4;
    const double& yPow4_cPow4 = predicted_moments.yPow4_cPow4;
    const double& yPow4_sPow4 = predicted_moments.yPow4_sPow4;
    const double& xPow4_cPow2_sPow2 = predicted_moments.xPow4_cPow2_sPow2;
    const double& xPow4_cPow3_sPow1 = predicted_moments.xPow4_cPow3_sPow1;
    const double& xPow4_cPow1_sPow3 = predicted_moments.xPow4_cPow1_sPow3;
    const double& yPow4_cPow2_sPow2 = predicted_moments.yPow4_cPow2_sPow2;
    const double& yPow4_cPow3_sPow1 = predicted_moments.yPow4_cPow3_sPow1;
    const double& yPow4_cPow1_sPow3 = predicted_moments.yPow4_cPow1_sPow3;
    const double& xPow1_yPow3_cPow4 = predicted_moments.xPow1_yPow3_cPow4;
    const double& xPow1_yPow3_sPow4 = predicted_moments.xPow1_yPow3_sPow4;
    const double& xPow2_yPow2_cPow4 = predicted_moments.xPow2_yPow2_cPow4;
    const double& xPow2_yPow2_sPow4 = predicted_moments.xPow2_yPow2_sPow4;
    const double& xPow3_yPow1_cPow4 = predicted_moments.xPow3_yPow1_cPow4;
    const double& xPow3_yPow1_sPow4 = predicted_moments.xPow3_yPow1_sPow4;
    const double& xPow3_yPow1_cPow1_sPow3 = predicted_moments.xPow3_yPow1_cPow1_sPow3;
    const double& xPow3_yPow1_cPow3_sPow1 = predicted_moments.xPow3_yPow1_cPow3_sPow1;
    const double& xPow3_yPow1_cPow2_sPow2 = predicted_moments.xPow3_yPow1_cPow2_sPow2;
    const double& xPow2_yPow2_cPow1_sPow3 = predicted_moments.xPow2_yPow2_cPow1_sPow3;
    const double& xPow2_yPow2_cPow3_sPow1 = predicted_moments.xPow2_yPow2_cPow3_sPow1;
    const double& xPow2_yPow2_cPow2_sPow2 = predicted_moments.xPow2_yPow2_cPow2_sPow2;
    const double& xPow1_yPow3_cPow1_sPow3 = predicted_moments.xPow1_yPow3_cPow1_sPow3;
    const double& xPow1_yPow3_cPow3_sPow1 = predicted_moments.xPow1_yPow3_cPow3_sPow1;
    const double& xPow1_yPow3_cPow2_sPow2 = predicted_moments.xPow1_yPow3_cPow2_sPow2;

    // noise
    const auto wr_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WR);
    const auto wa_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WA);
    const double wrPow2 = wr_dist_ptr->calc_moment(2);
    const double wrPow4 = wr_dist_ptr->calc_moment(4);
    const double cwaPow2 = wa_dist_ptr->calc_cos_moment(2);
    const double swaPow2 = wa_dist_ptr->calc_sin_moment(2);
    const double cwaPow4 = wa_dist_ptr->calc_cos_moment(4);
    const double swaPow4 = wa_dist_ptr->calc_sin_moment(4);
    const double cwaPow1_swaPow1 = wa_dist_ptr->calc_cos_sin_moment(1, 1);
    const double cwaPow3_swaPow1 = wa_dist_ptr->calc_cos_sin_moment(3, 1);
    const double cwaPow1_swaPow3 = wa_dist_ptr->calc_cos_sin_moment(1, 3);
    const double cwaPow2_swaPow2 = wa_dist_ptr->calc_cos_sin_moment(2, 2);

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

    const double haPow4 = pow(x_land, 4)*cPow4 + 4*pow(x_land, 3)*y_land*cPow3_sPow1 - 4*pow(x_land, 3)*yPow1_cPow3_sPow1 - 4*pow(x_land, 3)*xPow1_cPow4 + 6*pow(x_land, 2)*pow(y_land, 2)*cPow2_sPow2 - 12*pow(x_land, 2)*y_land*yPow1_cPow2_sPow2 - 12*pow(x_land, 2)*y_land*xPow1_cPow3_sPow1 + 6*pow(x_land, 2)*yPow2_cPow2_sPow2 + 12*pow(x_land, 2)*xPow1_yPow1_cPow3_sPow1 + 6*pow(x_land, 2)*xPow2_cPow4 + 4*x_land*pow(y_land, 3)*cPow1_sPow3 - 12*x_land*pow(y_land, 2)*yPow1_cPow1_sPow3 - 12*x_land*pow(y_land, 2)*xPow1_cPow2_sPow2 + 12*x_land*y_land*yPow2_cPow1_sPow3 + 24*x_land*y_land*xPow1_yPow1_cPow2_sPow2 + 12*x_land*y_land*xPow2_cPow3_sPow1 - 4*x_land*yPow3_cPow1_sPow3 - 12*x_land*xPow1_yPow2_cPow2_sPow2 - 12*x_land*xPow2_yPow1_cPow3_sPow1 - 4*x_land*xPow3_cPow4 + pow(y_land, 4)*sPow4 - 4*pow(y_land, 3)*xPow1_cPow1_sPow3 - 4*pow(y_land, 3)*yPow1_sPow4 + 12*pow(y_land, 2)*xPow1_yPow1_cPow1_sPow3 + 6*pow(y_land, 2)*xPow2_cPow2_sPow2 + 6*pow(y_land, 2)*yPow2_sPow4 - 12*y_land*xPow1_yPow2_cPow1_sPow3 - 12*y_land*xPow2_yPow1_cPow2_sPow2 - 4*y_land*xPow3_cPow3_sPow1 - 4*y_land*yPow3_sPow4 + 4*xPow1_yPow3_cPow1_sPow3 + 6*xPow2_yPow2_cPow2_sPow2 + 4*xPow3_yPow1_cPow3_sPow1 + xPow4_cPow4 + yPow4_sPow4;
    const double hbPow4 = pow(x_land, 4)*sPow4 - 4*pow(x_land, 3)*y_land*cPow1_sPow3 + 4*pow(x_land, 3)*yPow1_cPow1_sPow3 - 4*pow(x_land, 3)*xPow1_sPow4 + 6*pow(x_land, 2)*pow(y_land, 2)*cPow2_sPow2 + 12*pow(x_land, 2)*y_land*xPow1_cPow1_sPow3 - 12*pow(x_land, 2)*y_land*yPow1_cPow2_sPow2 - 12*pow(x_land, 2)*xPow1_yPow1_cPow1_sPow3 + 6*pow(x_land, 2)*yPow2_cPow2_sPow2 + 6*pow(x_land, 2)*xPow2_sPow4 - 4*x_land*pow(y_land, 3)*cPow3_sPow1 - 12*x_land*pow(y_land, 2)*xPow1_cPow2_sPow2 + 12*x_land*pow(y_land, 2)*yPow1_cPow3_sPow1 - 12*x_land*y_land*xPow2_cPow1_sPow3 + 24*x_land*y_land*xPow1_yPow1_cPow2_sPow2 - 12*x_land*y_land*yPow2_cPow3_sPow1 + 12*x_land*xPow2_yPow1_cPow1_sPow3 - 12*x_land*xPow1_yPow2_cPow2_sPow2 + 4*x_land*yPow3_cPow3_sPow1 - 4*x_land*xPow3_sPow4 + pow(y_land, 4)*cPow4 + 4*pow(y_land, 3)*xPow1_cPow3_sPow1 - 4*pow(y_land, 3)*yPow1_cPow4 + 6*pow(y_land, 2)*xPow2_cPow2_sPow2 - 12*pow(y_land, 2)*xPow1_yPow1_cPow3_sPow1 + 6*pow(y_land, 2)*yPow2_cPow4 + 4*y_land*xPow3_cPow1_sPow3 - 12*y_land*xPow2_yPow1_cPow2_sPow2 + 12*y_land*xPow1_yPow2_cPow3_sPow1 - 4*y_land*yPow3_cPow4 - 4*xPow3_yPow1_cPow1_sPow3 + 6*xPow2_yPow2_cPow2_sPow2 - 4*xPow1_yPow3_cPow3_sPow1 + yPow4_cPow4 + xPow4_sPow4;
    const double haPow2_hbPow2 = pow(x_land, 2)*pow(y_land, 2)*cPow4 + pow(x_land, 2)*pow(y_land, 2)*sPow4 - 2*pow(x_land, 2)*y_land*yPow1_cPow4 - 2*pow(x_land, 2)*y_land*yPow1_sPow4 + pow(x_land, 2)*yPow2_cPow4 + pow(x_land, 2)*yPow2_sPow4 - 2*x_land*pow(y_land, 2)*xPow1_cPow4 - 2*x_land*pow(y_land, 2)*xPow1_sPow4 + 6*x_land*y_land*xPow2_cPow1_sPow3 - 6*x_land*y_land*yPow2_cPow1_sPow3 - 16*x_land*y_land*xPow1_yPow1_cPow2_sPow2 - 6*x_land*y_land*xPow2_cPow3_sPow1 + 6*x_land*y_land*yPow2_cPow3_sPow1 + 4*x_land*y_land*xPow1_yPow1_cPow4 + 4*x_land*y_land*xPow1_yPow1_sPow4 - 6*x_land*xPow2_yPow1_cPow1_sPow3 + 2*x_land*yPow3_cPow1_sPow3 + 8*x_land*xPow1_yPow2_cPow2_sPow2 - 4*x_land*xPow3_cPow2_sPow2 + 6*x_land*xPow2_yPow1_cPow3_sPow1 - 2*x_land*yPow3_cPow3_sPow1 - 2*x_land*xPow1_yPow2_cPow4 - 2*x_land*xPow1_yPow2_sPow4 + pow(y_land, 2)*xPow2_cPow4 + pow(y_land, 2)*xPow2_sPow4 + 6*y_land*xPow1_yPow2_cPow1_sPow3 - 2*y_land*xPow3_cPow1_sPow3 + 8*y_land*xPow2_yPow1_cPow2_sPow2 - 4*y_land*yPow3_cPow2_sPow2 - 6*y_land*xPow1_yPow2_cPow3_sPow1 + 2*y_land*xPow3_cPow3_sPow1 - 2*y_land*xPow2_yPow1_cPow4 - 2*y_land*xPow2_yPow1_sPow4 + cPow1_sPow3*(2*pow(x_land, 3)*y_land - 2*x_land*pow(y_land, 3)) + xPow1_cPow1_sPow3*(-6*pow(x_land, 2)*y_land + 2*pow(y_land, 3)) + xPow1_yPow1_cPow1_sPow3*(6*pow(x_land, 2) - 6*pow(y_land, 2)) - 2*xPow1_yPow3_cPow1_sPow3 + 2*xPow3_yPow1_cPow1_sPow3 + yPow1_cPow1_sPow3*(-2*pow(x_land, 3) + 6*x_land*pow(y_land, 2)) + cPow2_sPow2*(pow(x_land, 4) - 4*pow(x_land, 2)*pow(y_land, 2) + pow(y_land, 4)) + xPow1_cPow2_sPow2*(-4*pow(x_land, 3) + 8*x_land*pow(y_land, 2)) + xPow2_cPow2_sPow2*(6*pow(x_land, 2) - 4*pow(y_land, 2)) - 4*xPow2_yPow2_cPow2_sPow2 + xPow4_cPow2_sPow2 + yPow1_cPow2_sPow2*(8*pow(x_land, 2)*y_land - 4*pow(y_land, 3)) + yPow2_cPow2_sPow2*(-4*pow(x_land, 2) + 6*pow(y_land, 2)) + yPow4_cPow2_sPow2 + cPow3_sPow1*(-2*pow(x_land, 3)*y_land + 2*x_land*pow(y_land, 3)) + xPow1_cPow3_sPow1*(6*pow(x_land, 2)*y_land - 2*pow(y_land, 3)) + xPow1_yPow1_cPow3_sPow1*(-6*pow(x_land, 2) + 6*pow(y_land, 2)) + 2*xPow1_yPow3_cPow3_sPow1 - 2*xPow3_yPow1_cPow3_sPow1 + yPow1_cPow3_sPow1*(2*pow(x_land, 3) - 6*x_land*pow(y_land, 2)) + xPow2_yPow2_cPow4 + xPow2_yPow2_sPow4;
    const double haPow3_hbPow1 = pow(x_land, 3)*y_land*cPow4 - pow(x_land, 3)*yPow1_cPow4 - 6*pow(x_land, 2)*y_land*yPow1_cPow3_sPow1 - 3*pow(x_land, 2)*y_land*xPow1_cPow4 + 3*pow(x_land, 2)*yPow2_cPow3_sPow1 + 3*pow(x_land, 2)*xPow1_yPow1_cPow4 - x_land*pow(y_land, 3)*sPow4 + 6*x_land*pow(y_land, 2)*xPow1_cPow1_sPow3 + 3*x_land*pow(y_land, 2)*yPow1_sPow4 - 12*x_land*y_land*xPow1_yPow1_cPow1_sPow3 - 9*x_land*y_land*xPow2_cPow2_sPow2 + 9*x_land*y_land*yPow2_cPow2_sPow2 + 12*x_land*y_land*xPow1_yPow1_cPow3_sPow1 + 3*x_land*y_land*xPow2_cPow4 - 3*x_land*y_land*yPow2_sPow4 + 6*x_land*xPow1_yPow2_cPow1_sPow3 + 9*x_land*xPow2_yPow1_cPow2_sPow2 - 3*x_land*yPow3_cPow2_sPow2 - 6*x_land*xPow1_yPow2_cPow3_sPow1 + 4*x_land*xPow3_cPow3_sPow1 - 3*x_land*xPow2_yPow1_cPow4 + x_land*yPow3_sPow4 + pow(y_land, 3)*xPow1_sPow4 - 3*pow(y_land, 2)*xPow2_cPow1_sPow3 - 3*pow(y_land, 2)*xPow1_yPow1_sPow4 + 6*y_land*xPow2_yPow1_cPow1_sPow3 - 4*y_land*yPow3_cPow1_sPow3 - 9*y_land*xPow1_yPow2_cPow2_sPow2 + 3*y_land*xPow3_cPow2_sPow2 - 6*y_land*xPow2_yPow1_cPow3_sPow1 - y_land*xPow3_cPow4 + 3*y_land*xPow1_yPow2_sPow4 + cPow1_sPow3*(-3*pow(x_land, 2)*pow(y_land, 2) + pow(y_land, 4)) - 3*xPow2_yPow2_cPow1_sPow3 + yPow1_cPow1_sPow3*(6*pow(x_land, 2)*y_land - 4*pow(y_land, 3)) + yPow2_cPow1_sPow3*(-3*pow(x_land, 2) + 6*pow(y_land, 2)) + yPow4_cPow1_sPow3 + cPow2_sPow2*(-3*pow(x_land, 3)*y_land + 3*x_land*pow(y_land, 3)) + xPow1_cPow2_sPow2*(9*pow(x_land, 2)*y_land - 3*pow(y_land, 3)) + xPow1_yPow1_cPow2_sPow2*(-9*pow(x_land, 2) + 9*pow(y_land, 2)) + 3*xPow1_yPow3_cPow2_sPow2 - 3*xPow3_yPow1_cPow2_sPow2 + yPow1_cPow2_sPow2*(3*pow(x_land, 3) - 9*x_land*pow(y_land, 2)) + cPow3_sPow1*(-pow(x_land, 4) + 3*pow(x_land, 2)*pow(y_land, 2)) + xPow1_cPow3_sPow1*(4*pow(x_land, 3) - 6*x_land*pow(y_land, 2)) + xPow2_cPow3_sPow1*(-6*pow(x_land, 2) + 3*pow(y_land, 2)) + 3*xPow2_yPow2_cPow3_sPow1 - xPow4_cPow3_sPow1 + xPow3_yPow1_cPow4 - xPow1_yPow3_sPow4;
    const double haPow1_hbPow3 = -pow(x_land, 3)*y_land*sPow4 + pow(x_land, 3)*yPow1_sPow4 - 6*pow(x_land, 2)*y_land*yPow1_cPow1_sPow3 + 3*pow(x_land, 2)*y_land*xPow1_sPow4 + 3*pow(x_land, 2)*yPow2_cPow1_sPow3 - 3*pow(x_land, 2)*xPow1_yPow1_sPow4 + x_land*pow(y_land, 3)*cPow4 + 6*x_land*pow(y_land, 2)*xPow1_cPow3_sPow1 - 3*x_land*pow(y_land, 2)*yPow1_cPow4 + 12*x_land*y_land*xPow1_yPow1_cPow1_sPow3 + 9*x_land*y_land*xPow2_cPow2_sPow2 - 9*x_land*y_land*yPow2_cPow2_sPow2 - 12*x_land*y_land*xPow1_yPow1_cPow3_sPow1 + 3*x_land*y_land*yPow2_cPow4 - 3*x_land*y_land*xPow2_sPow4 - 6*x_land*xPow1_yPow2_cPow1_sPow3 + 4*x_land*xPow3_cPow1_sPow3 - 9*x_land*xPow2_yPow1_cPow2_sPow2 + 3*x_land*yPow3_cPow2_sPow2 + 6*x_land*xPow1_yPow2_cPow3_sPow1 - x_land*yPow3_cPow4 + 3*x_land*xPow2_yPow1_sPow4 - pow(y_land, 3)*xPow1_cPow4 - 3*pow(y_land, 2)*xPow2_cPow3_sPow1 + 3*pow(y_land, 2)*xPow1_yPow1_cPow4 - 6*y_land*xPow2_yPow1_cPow1_sPow3 + 9*y_land*xPow1_yPow2_cPow2_sPow2 - 3*y_land*xPow3_cPow2_sPow2 + 6*y_land*xPow2_yPow1_cPow3_sPow1 - 4*y_land*yPow3_cPow3_sPow1 - 3*y_land*xPow1_yPow2_cPow4 + y_land*xPow3_sPow4 + cPow1_sPow3*(-pow(x_land, 4) + 3*pow(x_land, 2)*pow(y_land, 2)) + xPow1_cPow1_sPow3*(4*pow(x_land, 3) - 6*x_land*pow(y_land, 2)) + xPow2_cPow1_sPow3*(-6*pow(x_land, 2) + 3*pow(y_land, 2)) + 3*xPow2_yPow2_cPow1_sPow3 - xPow4_cPow1_sPow3 + cPow2_sPow2*(3*pow(x_land, 3)*y_land - 3*x_land*pow(y_land, 3)) + xPow1_cPow2_sPow2*(-9*pow(x_land, 2)*y_land + 3*pow(y_land, 3)) + xPow1_yPow1_cPow2_sPow2*(9*pow(x_land, 2) - 9*pow(y_land, 2)) - 3*xPow1_yPow3_cPow2_sPow2 + 3*xPow3_yPow1_cPow2_sPow2 + yPow1_cPow2_sPow2*(-3*pow(x_land, 3) + 9*x_land*pow(y_land, 2)) + cPow3_sPow1*(-3*pow(x_land, 2)*pow(y_land, 2) + pow(y_land, 4)) - 3*xPow2_yPow2_cPow3_sPow1 + yPow1_cPow3_sPow1*(6*pow(x_land, 2)*y_land - 4*pow(y_land, 3)) + yPow2_cPow3_sPow1*(-3*pow(x_land, 2) + 6*pow(y_land, 2)) + yPow4_cPow3_sPow1 + xPow1_yPow3_cPow4 - xPow3_yPow1_sPow4;

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
    const Eigen::VectorXd& measurement_mean = meas_info.mean;
    const Eigen::VectorXd& predicted_mean = state_info.mean;

    const double xPow1_haPow2 =  std::pow(x_land, 2) * xPow1_cPow2 + xPow3_cPow2 - 2.0 * x_land * xPow2_cPow2
                                 + std::pow(y_land, 2) * xPow1_sPow2 + xPow1_yPow2_sPow2 - 2.0 * y_land * xPow1_yPow1_sPow2
                                 + 2.0 * x_land * y_land * xPow1_cPow1_sPow1 - 2.0 * x_land * xPow1_yPow1_cPow1_sPow1
                                 - 2.0 * y_land * xPow2_cPow1_sPow1 + 2.0 * xPow2_yPow1_cPow1_sPow1;
    const double xPow1_hbPow2 = std::pow(y_land, 2) * xPow1_cPow2 + xPow1_yPow2_cPow2 - 2.0 * y_land * xPow1_yPow1_cPow2
                                + std::pow(x_land, 2) * xPow1_sPow2 + xPow3_sPow2 - 2.0 * x_land * xPow2_sPow2
                                - 2.0 * x_land * y_land * xPow1_cPow1_sPow1 + 2.0 * x_land * xPow1_yPow1_cPow1_sPow1
                                + 2.0 * y_land * xPow2_cPow1_sPow1 - 2.0 * xPow2_yPow1_cPow1_sPow1;
    const double xPow1_haPow1_hbPow1 =  x_land * y_land * xPow1_cPow2 + xPow2_yPow1_cPow2 - x_land * xPow1_yPow1_cPow2
                                        - y_land * xPow2_cPow2 - std::pow(x_land, 2) * xPow1_cPow1_sPow1
                                        - xPow3_cPow1_sPow1 + 2.0 * x_land * xPow2_cPow1_sPow1
                                        + std::pow(y_land, 2) * xPow1_cPow1_sPow1 + xPow1_yPow2_cPow1_sPow1
                                        - 2.0 * y_land * xPow1_yPow1_cPow1_sPow1 - x_land * y_land * xPow1_sPow2
                                        - xPow2_yPow1_sPow2 + x_land * xPow1_yPow1_sPow2 + y_land * xPow2_sPow2;
    const double yPow1_haPow2 = std::pow(x_land, 2) * yPow1_cPow2 + xPow2_yPow1_cPow2 - 2.0 * x_land * xPow1_yPow1_cPow2
                                + std::pow(y_land, 2) * yPow1_sPow2 + yPow3_sPow2 - 2.0 * y_land * yPow2_sPow2
                                + 2.0 * x_land * y_land * yPow1_cPow1_sPow1 - 2.0 * x_land * yPow2_cPow1_sPow1
                                - 2.0 * y_land * xPow1_yPow1_cPow1_sPow1 + 2.0 * xPow1_yPow2_cPow1_sPow1;
    const double yPow1_hbPow2 = std::pow(y_land, 2) * yPow1_cPow2 + yPow3_cPow2 - 2.0 * y_land * yPow2_cPow2
                                + std::pow(x_land, 2) * yPow1_sPow2 + xPow2_yPow1_sPow2 - 2.0 * x_land * xPow1_yPow1_sPow2
                                - 2.0 * x_land * y_land * yPow1_cPow1_sPow1 + 2.0 * x_land * yPow2_cPow1_sPow1
                                + 2.0 * y_land * xPow1_yPow1_cPow1_sPow1 - 2.0 * xPow1_yPow2_cPow1_sPow1;
    const double yPow1_haPow1_hbPow1 =  x_land * y_land * yPow1_cPow2 + xPow1_yPow2_cPow2 - x_land * yPow2_cPow2
                                        - y_land * xPow1_yPow1_cPow2 - std::pow(x_land, 2) * yPow1_cPow1_sPow1
                                        - xPow2_yPow1_cPow1_sPow1 + 2.0 * x_land * xPow1_yPow1_cPow1_sPow1
                                        + std::pow(y_land, 2) * yPow1_cPow1_sPow1 + yPow3_cPow1_sPow1
                                        - 2.0 * y_land * yPow2_cPow1_sPow1 - x_land * y_land * yPow1_sPow2
                                        - xPow1_yPow2_sPow2 + x_land * yPow2_sPow2 + y_land * xPow1_yPow1_sPow2;
    const double yawPow1_haPow2 = std::pow(x_land, 2) * yawPow1_cPow2 + xPow2_yawPow1_cPow2 - 2.0 * x_land * xPow1_yawPow1_cPow2
                                  + std::pow(y_land, 2) * yawPow1_sPow2 + yPow2_yawPow1_sPow2 - 2.0 * y_land * yPow1_yawPow1_sPow2
                                  + 2.0 * x_land * y_land * yawPow1_cPow1_sPow1 - 2.0 * x_land * yPow1_yawPow1_cPow1_sPow1
                                  - 2.0 * y_land * xPow1_yawPow1_cPow1_sPow1 + 2.0 * xPow1_yPow1_yawPow1_cPow1_sPow1;
    const double yawPow1_hbPow2 = std::pow(y_land, 2) * yawPow1_cPow2 + yPow2_yawPow1_cPow2 - 2.0 * y_land * yPow1_yawPow1_cPow2
                                  + std::pow(x_land, 2) * yawPow1_sPow2 + xPow2_yawPow1_sPow2 - 2.0 * x_land * xPow1_yawPow1_sPow2
                                  - 2.0 * x_land * y_land * yawPow1_cPow1_sPow1 + 2.0 * x_land * yPow1_yawPow1_cPow1_sPow1
                                  + 2.0 * y_land * xPow1_yawPow1_cPow1_sPow1 - 2.0 * xPow1_yPow1_yawPow1_cPow1_sPow1;
    const double yawPow1_haPow1_hbPow1 =  x_land * y_land * yawPow1_cPow2 + xPow1_yPow1_yawPow1_cPow2
                                          - x_land * yPow1_yawPow1_cPow2 - y_land * xPow1_yawPow1_cPow2
                                          - std::pow(x_land, 2) * yawPow1_cPow1_sPow1 - xPow2_yawPow1_cPow1_sPow1
                                          + 2.0 * x_land * xPow1_yawPow1_cPow1_sPow1 + std::pow(y_land, 2) * yawPow1_cPow1_sPow1
                                          + yPow2_yawPow1_cPow1_sPow1 - 2.0 * y_land * yPow1_yawPow1_cPow1_sPow1
                                          - x_land * y_land * yawPow1_sPow2 - xPow1_yPow1_yawPow1_sPow2
                                          + x_land * yPow1_yawPow1_sPow2 + y_land * xPow1_yawPow1_sPow2;

    Eigen::MatrixXd state_observation_cov(3, 2); // sigma = E[XY^T] - E[X]E[Y]^T
    state_observation_cov(STATE::IDX::X, MEASUREMENT::IDX::RCOS)
            = wrPow2 * cwaPow2 * xPow1_haPow2 + wrPow2 * swaPow2 * xPow1_hbPow2
              - 2.0 * wrPow2 * cwaPow1_swaPow1 * xPow1_haPow1_hbPow1
              - predicted_mean(STATE::IDX::X) * measurement_mean(MEASUREMENT::IDX::RCOS);
    state_observation_cov(STATE::IDX::X, MEASUREMENT::IDX::RSIN)
            = wrPow2 * cwaPow2 * xPow1_hbPow2 + wrPow2 * swaPow2 * xPow1_haPow2
              + 2.0 * wrPow2 * cwaPow1_swaPow1 * xPow1_haPow1_hbPow1
              - predicted_mean(STATE::IDX::X) * measurement_mean(MEASUREMENT::IDX::RSIN);

    state_observation_cov(STATE::IDX::Y, MEASUREMENT::IDX::RCOS)
            = wrPow2 * cwaPow2 * yPow1_haPow2 + wrPow2 * swaPow2 * yPow1_hbPow2
              - 2.0 * wrPow2 * cwaPow1_swaPow1 * yPow1_haPow1_hbPow1
              - predicted_mean(STATE::IDX::Y) * measurement_mean(MEASUREMENT::IDX::RCOS);
    state_observation_cov(STATE::IDX::Y, MEASUREMENT::IDX::RSIN)
            = wrPow2 * cwaPow2 * yPow1_hbPow2 + wrPow2 * swaPow2 * yPow1_haPow2
              + 2.0 * wrPow2 * cwaPow1_swaPow1 * yPow1_haPow1_hbPow1
              - predicted_mean(STATE::IDX::Y) * measurement_mean(MEASUREMENT::IDX::RSIN);

    state_observation_cov(STATE::IDX::YAW, MEASUREMENT::IDX::RCOS)
            = wrPow2 * cwaPow2 * yawPow1_haPow2 + wrPow2 * swaPow2 * yawPow1_hbPow2
              - 2.0 * wrPow2 * cwaPow1_swaPow1 * yawPow1_haPow1_hbPow1
              - predicted_mean(STATE::IDX::YAW) * measurement_mean(MEASUREMENT::IDX::RCOS);
    state_observation_cov(STATE::IDX::YAW, MEASUREMENT::IDX::RSIN)
            = wrPow2 * cwaPow2 * yawPow1_hbPow2 + wrPow2 * swaPow2 * yawPow1_haPow2
              + 2.0 * wrPow2 * cwaPow1_swaPow1 * yawPow1_haPow1_hbPow1
              - predicted_mean(STATE::IDX::YAW) * measurement_mean(MEASUREMENT::IDX::RSIN);

    // Kalman Gain
    const Eigen::MatrixXd& predicted_cov = state_info.covariance;
    const Eigen::MatrixXd& measurement_cov = meas_info.covariance;
    const auto K = state_observation_cov * measurement_cov.inverse();

    StateInfo updated_info;
    updated_info.mean = predicted_mean + K * (observed_values - measurement_mean);
    updated_info.covariance = predicted_cov - K*measurement_cov*K.transpose();

    return updated_info;
}