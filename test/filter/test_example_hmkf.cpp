#include <iostream>
#include <random>
#include <vector>
#include <Eigen/Eigen>
#include <map>
#include <memory>
#include <cmath>
#include <gtest/gtest.h>

#include "filter/example_hmkf.h"
#include "distribution/uniform_distribution.h"
#include "distribution/exponential_distribution.h"

using namespace Example;

TEST(ExampleHMKF, Predict)
{
    ExampleHMKF hmkf;

    const double dt = 0.1;

    // Initial State
    StateInfo ini_state;
    ini_state.mean = Eigen::VectorXd::Zero(2);
    ini_state.mean(0) = 5.0;
    ini_state.mean(1) = M_PI/2.0;
    ini_state.covariance = Eigen::MatrixXd::Zero(2, 2);
    ini_state.covariance(0, 0) = 1.0*1.0; // V[x]
    ini_state.covariance(1, 1) = (M_PI/30)*(M_PI/30); // V[yaw]

    // Input
    Eigen::Vector2d control_inputs = Eigen::Vector2d::Zero();
    control_inputs(0) = 1.0 * dt;
    control_inputs(1) = 0.1 * dt;

    // System Noise
    const double wx_lambda = 1.0;
    const double upper_wtheta = (M_PI/10.0);
    const double lower_wtheta = -(M_PI/10.0);
    std::map<int, std::shared_ptr<BaseDistribution>> system_noise_map{
            {SYSTEM_NOISE::IDX::WX, std::make_shared<ExponentialDistribution>(wx_lambda)},
            {SYSTEM_NOISE::IDX::WYAW, std::make_shared<UniformDistribution>(lower_wtheta, upper_wtheta)}};

    const auto predicted_moments = hmkf.predict(ini_state, control_inputs, dt, system_noise_map);

    std::cout << predicted_moments.xPow1 << std::endl;
    std::cout << predicted_moments.yawPow1 << std::endl;
    std::cout << "E[X^2]: " << predicted_moments.xPow2 << std::endl;
    std::cout << predicted_moments.yawPow2 << std::endl;
    std::cout << "E[X^2Yaw]: " << predicted_moments.xPow2_yawPow1 << std::endl;

    // Update
    // measurement noise
    const double mr_lambda = 0.5;
    const double upper_mtheta = (M_PI/20.0);
    const double lower_mtheta = -(M_PI/20.0);
    std::map<int, std::shared_ptr<BaseDistribution>> measurement_noise_map{
            {MEASUREMENT_NOISE::IDX::WR, std::make_shared<ExponentialDistribution>(mr_lambda)},
            {MEASUREMENT_NOISE::IDX::WYAW, std::make_shared<UniformDistribution>(lower_mtheta, upper_mtheta)}};
    Eigen::VectorXd measured_values = Eigen::Vector2d::Zero();
    measured_values(MEASUREMENT::IDX::R) = predicted_moments.xPow1*predicted_moments.xPow1;
    measured_values(MEASUREMENT::IDX::YAW) = predicted_moments.yawPow1;

    const auto measurement_moments = hmkf.getMeasurementMoments(predicted_moments, measurement_noise_map);
    const auto state_measurement_matrix = hmkf.getStateMeasurementMatrix(predicted_moments, measurement_moments, measurement_noise_map);
    std::cout << "E[R]: " << measurement_moments.rPow1 << std::endl;
    std::cout << "E[R^2]: " << measurement_moments.rPow2 << std::endl;
    std::cout << "E[YAW]: " << measurement_moments.yawPow1 << std::endl;
    std::cout << "E[YAW^2]: " << measurement_moments.yawPow2 << std::endl;
    std::cout << "E[R*YAW]: " << measurement_moments.rPow1_yawPow1 << std::endl;
    std::cout << state_measurement_matrix << std::endl;
}
