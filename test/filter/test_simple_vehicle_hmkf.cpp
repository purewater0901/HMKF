#include <iostream>
#include <random>
#include <vector>
#include <Eigen/Eigen>
#include <map>
#include <memory>
#include <cmath>
#include <gtest/gtest.h>
#include <boost/function.hpp>
#include <boost/bind.hpp>
#include <boost/random/beta_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>

#include "filter/simple_vehicle_hmkf.h"
#include "filter/mkf.h"
#include "filter/ekf.h"
#include "distribution/uniform_distribution.h"
#include "distribution/exponential_distribution.h"
#include "distribution/beta_distribution.h"
#include "distribution/normal_distribution.h"
#include "model/example_model.h"

using namespace SimpleVehicle;

/*
TEST(SimpleVehicleHMKF, Predict)
{
    SimpleVehicleHMKF hmkf;

    const double dt = 0.1;

    // Initial State
    StateInfo ini_state;
    ini_state.mean = Eigen::VectorXd::Zero(3);
    ini_state.mean(0) = 2.0;
    ini_state.mean(1) = 1.0;
    ini_state.mean(2) = M_PI/3.0;
    ini_state.covariance = Eigen::MatrixXd::Zero(3, 3);
    ini_state.covariance(0, 0) = 1.0*1.0; // V[x]
    ini_state.covariance(1, 1) = 1.0*1.0; // V[y]
    ini_state.covariance(0, 1) = 0.1; // V[xy]
    ini_state.covariance(1, 0) = 0.1; // V[xy]
    ini_state.covariance(0, 2) = 0.01;
    ini_state.covariance(2, 0) = 0.01;
    ini_state.covariance(1, 2) = 0.2;
    ini_state.covariance(2, 1) = 0.2;
    ini_state.covariance(2, 2) = (M_PI/10)*(M_PI/10); // V[yaw]

    // Input
    Eigen::Vector2d control_inputs = Eigen::Vector2d::Zero();
    control_inputs(0) = 1.0 * dt;
    control_inputs(1) = 0.1 * dt;

    // System Noise
    const double wv_lambda = 1.0;
    const double upper_wu = (M_PI/10.0) * dt;
    const double lower_wu = -(M_PI/10.0) * dt;
    std::map<int, std::shared_ptr<BaseDistribution>> system_noise_map{
            {SYSTEM_NOISE::IDX::WV, std::make_shared<ExponentialDistribution>(wv_lambda)},
            {SYSTEM_NOISE::IDX::WU, std::make_shared<UniformDistribution>(lower_wu, upper_wu)}};

    std::shared_ptr<SimpleVehicleModel::HighOrderMoments> high_order_moments;
    const auto predicted_moments = hmkf.predict(ini_state, control_inputs, system_noise_map, high_order_moments);

    if(high_order_moments) {
        std::cout << "E[xcos^2]: " << high_order_moments->xPow1_cPow2 << std::endl;
        std::cout << "E[ycos^2]: " << high_order_moments->yPow1_cPow2 << std::endl;
        std::cout << "E[xsin^2]: " << high_order_moments->xPow1_sPow2 << std::endl;
        std::cout << "E[ysin^2]: " << high_order_moments->yPow1_sPow2 << std::endl;
        std::cout << "E[x^2cos]: " << high_order_moments->xPow2_cPow1 << std::endl;
        std::cout << "E[y^2cos]: " << high_order_moments->yPow2_cPow1 << std::endl;
        std::cout << "E[x^2sin]: " << high_order_moments->xPow2_sPow1 << std::endl;
        std::cout << "E[y^2sin]: " << high_order_moments->yPow2_sPow1 << std::endl;
        std::cout << "E[xycos]: " << high_order_moments->xPow1_yPow1_cPow1 << std::endl;
        std::cout << "E[xysin]: " << high_order_moments->xPow1_yPow1_sPow1 << std::endl;
        std::cout << "E[xcossin]: " << high_order_moments->xPow1_cPow1_sPow1 << std::endl;
        std::cout << "E[ycossin]: " << high_order_moments->yPow1_cPow1_sPow1 << std::endl;
        std::cout << "E[xyawcos]: " << high_order_moments->xPow1_yawPow1_cPow1 << std::endl;
        std::cout << "E[xyawsin]: " << high_order_moments->xPow1_yawPow1_sPow1 << std::endl;
        std::cout << "E[yyawcos]: " << high_order_moments->yPow1_yawPow1_cPow1 << std::endl;
        std::cout << "E[yyawsin]: " << high_order_moments->yPow1_yawPow1_sPow1 << std::endl;

        std::cout << "E[x^2cos^2]: " << high_order_moments->xPow2_cPow2 << std::endl;
        std::cout << "E[y^2cos^2]: " << high_order_moments->yPow2_cPow2 << std::endl;
        std::cout << "E[x^2sin^2]: " << high_order_moments->xPow2_sPow2 << std::endl;
        std::cout << "E[y^2sin^2]: " << high_order_moments->yPow2_sPow2 << std::endl;
        std::cout << "E[xycos^2]: " << high_order_moments->xPow1_yPow1_cPow2 << std::endl;
        std::cout << "E[xysin^2]: " << high_order_moments->xPow1_yPow1_sPow2 << std::endl;
        std::cout << "E[x^2cossin]: " << high_order_moments->xPow2_cPow1_sPow1 << std::endl;
        std::cout << "E[y^2cossin]: " << high_order_moments->yPow2_cPow1_sPow1 << std::endl;
        std::cout << "E[xycossin]: " << high_order_moments->xPow1_yPow1_cPow1_sPow1 << std::endl;
    }

    // Update
    // measurement noise
    const double mr_lambda = 1.0;
    const double upper_ma = (M_PI/20.0);
    const double lower_ma = -(M_PI/20.0);
    std::map<int, std::shared_ptr<BaseDistribution>> measurement_noise_map{
            {OBSERVATION_NOISE::IDX::WR, std::make_shared<ExponentialDistribution>(mr_lambda)},
            {OBSERVATION_NOISE::IDX::WA, std::make_shared<UniformDistribution>(lower_ma, upper_ma)}};
    Eigen::VectorXd measured_values = Eigen::Vector2d::Zero();
    const double mx = predicted_moments.mean(0);
    const double my = predicted_moments.mean(1);
    const double myaw = predicted_moments.mean(2);
    measured_values(OBSERVATION::IDX::RCOS) = std::sqrt(mx*mx+my*my)*std::cos(myaw);
    measured_values(OBSERVATION::IDX::RSIN) = std::sqrt(mx*mx+my*my)*std::sin(myaw);
    Eigen::Vector2d landmark = Eigen::Vector2d::Zero();
    landmark(0) = 1.0;
    landmark(1) = 2.0;
    hmkf.update(*high_order_moments, measured_values, landmark, measurement_noise_map);
}
*/

TEST(SimpleVehicleHMKF, EdgeCase)
{
    SimpleVehicleHMKF hmkf;

    const double dt = 0.208;

    // Initial State
    StateInfo ini_state;
    ini_state.mean = Eigen::VectorXd::Zero(3);
    ini_state.mean(0) = 2.97581;
    ini_state.mean(1) = 1.35239;
    ini_state.mean(2) = 5.26836;
    ini_state.covariance = Eigen::MatrixXd::Zero(3, 3);
    ini_state.covariance(0, 0) = 0.000761754; // V[x]
    ini_state.covariance(1, 1) = 0.000783393; // V[y]
    ini_state.covariance(0, 1) = -1.2313e-05; // V[xy]
    ini_state.covariance(1, 0) = -1.2313e-05; // V[xy]
    ini_state.covariance(0, 2) = -0.000206351;
    ini_state.covariance(2, 0) = -0.000206351;
    ini_state.covariance(1, 2) = -9.27536e-05;
    ini_state.covariance(2, 1) = -9.27536e-05;
    ini_state.covariance(2, 2) = 0.00141191; // V[yaw]

    // Input
    Eigen::Vector2d control_inputs = Eigen::Vector2d::Zero();
    control_inputs(0) = 0.017888;
    control_inputs(1) = -0.0827841;

    // System Noise
    const double mean_wv = 0.0;
    const double cov_wv = std::pow(0.1, 2);
    const double mean_wu = 0.0;
    const double cov_wu = std::pow(1.0, 2);
    const std::map<int, std::shared_ptr<BaseDistribution>> system_noise_map = {
            {SYSTEM_NOISE::IDX::WV, std::make_shared<NormalDistribution>(mean_wv*dt, cov_wv*dt*dt)},
            {SYSTEM_NOISE::IDX::WU, std::make_shared<NormalDistribution>(mean_wu*dt, cov_wu*dt*dt)}};

    std::shared_ptr<SimpleVehicleModel::HighOrderMoments> high_order_moments;
    const auto predicted_moments = hmkf.predict(ini_state, control_inputs, system_noise_map, high_order_moments);

    if(high_order_moments) {
        std::cout << "E[x]: " << high_order_moments->xPow1 << std::endl;
        std::cout << "E[y]: " << high_order_moments->yPow1 << std::endl;
        std::cout << "E[yaw]: " << high_order_moments->yawPow1 << std::endl;
        std::cout << "E[cos]: " << high_order_moments->cPow1 << std::endl;
        std::cout << "E[sin]: " << high_order_moments->sPow1 << std::endl;
        std::cout << "E[x^2]: " << high_order_moments->xPow2 << std::endl;
        std::cout << "E[y^2]: " << high_order_moments->yPow2 << std::endl;
        std::cout << "E[yaw^2]: " << high_order_moments->yawPow2 << std::endl;
        std::cout << "E[cos^2]: " << high_order_moments->cPow2 << std::endl;
        std::cout << "E[sin^2]: " << high_order_moments->sPow2 << std::endl;
        std::cout << "E[xy]: " << high_order_moments->xPow1_yPow1 << std::endl;
        std::cout << "E[xyaw]: " << high_order_moments->xPow1_yawPow1 << std::endl;
        std::cout << "E[yyaw]: " << high_order_moments->yPow1_yawPow1 << std::endl;
        std::cout << "E[cos*sin]: " << high_order_moments->cPow1_sPow1<< std::endl;
        std::cout << "E[x*cos]: " << high_order_moments->xPow1_cPow1<< std::endl;
        std::cout << "E[x*sin]: " << high_order_moments->xPow1_sPow1<< std::endl;
        std::cout << "E[y*cos]: " << high_order_moments->yPow1_cPow1<< std::endl;
        std::cout << "E[y*sin]: " << high_order_moments->yPow1_sPow1<< std::endl;
        std::cout << "E[yaw*cos]: " << high_order_moments->yawPow1_cPow1<< std::endl;
        std::cout << "E[yaw*sin]: " << high_order_moments->yawPow1_sPow1<< std::endl;

        std::cout << "E[xcos^2]: " << high_order_moments->xPow1_cPow2 << std::endl;
        std::cout << "E[ycos^2]: " << high_order_moments->yPow1_cPow2 << std::endl;
        std::cout << "E[xsin^2]: " << high_order_moments->xPow1_sPow2 << std::endl;
        std::cout << "E[ysin^2]: " << high_order_moments->yPow1_sPow2 << std::endl;
        std::cout << "E[x^2cos]: " << high_order_moments->xPow2_cPow1 << std::endl;
        std::cout << "E[y^2cos]: " << high_order_moments->yPow2_cPow1 << std::endl;
        std::cout << "E[x^2sin]: " << high_order_moments->xPow2_sPow1 << std::endl;
        std::cout << "E[y^2sin]: " << high_order_moments->yPow2_sPow1 << std::endl;
        std::cout << "E[xycos]: " << high_order_moments->xPow1_yPow1_cPow1 << std::endl;
        std::cout << "E[xysin]: " << high_order_moments->xPow1_yPow1_sPow1 << std::endl;
        std::cout << "E[xcossin]: " << high_order_moments->xPow1_cPow1_sPow1 << std::endl;
        std::cout << "E[ycossin]: " << high_order_moments->yPow1_cPow1_sPow1 << std::endl;
        std::cout << "E[xyawcos]: " << high_order_moments->xPow1_yawPow1_cPow1 << std::endl;
        std::cout << "E[xyawsin]: " << high_order_moments->xPow1_yawPow1_sPow1 << std::endl;
        std::cout << "E[yyawcos]: " << high_order_moments->yPow1_yawPow1_cPow1 << std::endl;
        std::cout << "E[yyawsin]: " << high_order_moments->yPow1_yawPow1_sPow1 << std::endl;

        std::cout << "E[x^2cos^2]: " << high_order_moments->xPow2_cPow2 << std::endl;
        std::cout << "E[y^2cos^2]: " << high_order_moments->yPow2_cPow2 << std::endl;
        std::cout << "E[x^2sin^2]: " << high_order_moments->xPow2_sPow2 << std::endl;
        std::cout << "E[y^2sin^2]: " << high_order_moments->yPow2_sPow2 << std::endl;
        std::cout << "E[xycos^2]: " << high_order_moments->xPow1_yPow1_cPow2 << std::endl;
        std::cout << "E[xysin^2]: " << high_order_moments->xPow1_yPow1_sPow2 << std::endl;
        std::cout << "E[x^2cossin]: " << high_order_moments->xPow2_cPow1_sPow1 << std::endl;
        std::cout << "E[y^2cossin]: " << high_order_moments->yPow2_cPow1_sPow1 << std::endl;
        std::cout << "E[xycossin]: " << high_order_moments->xPow1_yPow1_cPow1_sPow1 << std::endl;
    }

    // Update
    // measurement noise
    const double mean_wr = 1.0;
    const double cov_wr = std::pow(0.09, 2);
    const double mean_wa = 0.0;
    const double cov_wa = std::pow(M_PI/100.0, 2);
    const std::shared_ptr<BaseDistribution> wr_dist = std::make_shared<NormalDistribution>(mean_wr, cov_wr);
    std::map<int, std::shared_ptr<BaseDistribution>> measurement_noise_map = {
            {OBSERVATION_NOISE::IDX::WR, wr_dist},
            {OBSERVATION_NOISE::IDX::WA, std::make_shared<NormalDistribution>(mean_wa, cov_wa)}};
    Eigen::VectorXd measured_values = Eigen::Vector2d::Zero();
    measured_values(OBSERVATION::IDX::RCOS) =  3.41384;
    measured_values(OBSERVATION::IDX::RSIN) = -0.886256;
    Eigen::Vector2d landmark = Eigen::Vector2d::Zero();
    landmark(0) = 3.7629;
    landmark(1) = -2.03092;
    const auto updated_moments = hmkf.update(*high_order_moments, measured_values, landmark, measurement_noise_map);

    std::cout << "updated_cov" << std::endl;
    std::cout << updated_moments.mean << std::endl;
    std::cout << updated_moments.covariance << std::endl;
}
