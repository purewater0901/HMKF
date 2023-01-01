#include <iostream>
#include <random>
#include <vector>
#include <Eigen/Eigen>
#include <map>
#include <memory>
#include <cmath>
#include <gtest/gtest.h>

#include "filter/example_hmkf.h"
#include "filter/mkf.h"
#include "filter/ekf.h"
#include "distribution/uniform_distribution.h"
#include "distribution/exponential_distribution.h"
#include "model/example_model.h"

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

TEST(ExampleHMKF, Simulation)
{
    std::shared_ptr<BaseModel> example_model = std::make_shared<ExampleVehicleModel>();

    ExampleHMKF hmkf;
    EKF ekf(example_model);
    MKF mkf(example_model);

    const double dt = 0.1;
    size_t N = 3000;

    // Initial State
    StateInfo ini_state;
    ini_state.mean = Eigen::VectorXd::Zero(2);
    ini_state.mean(0) = 5.0;
    ini_state.mean(1) = M_PI/2.0;
    ini_state.covariance = Eigen::MatrixXd::Zero(2, 2);
    ini_state.covariance(0, 0) = 1.0*1.0; // V[x]
    ini_state.covariance(1, 1) = (M_PI/30)*(M_PI/30); // V[yaw]

    Eigen::VectorXd x_true = ini_state.mean;

    // Input
    Eigen::VectorXd control_inputs = Eigen::VectorXd::Zero(2);
    control_inputs(0) = 1.0 * dt;
    control_inputs(1) = 0.1 * dt;

    // System Noise
    const double wx_lambda = 0.1;
    const double upper_wtheta = (M_PI/10.0);
    const double lower_wtheta = -(M_PI/10.0);
    std::map<int, std::shared_ptr<BaseDistribution>> system_noise_map{
            {SYSTEM_NOISE::IDX::WX, std::make_shared<ExponentialDistribution>(wx_lambda)},
            {SYSTEM_NOISE::IDX::WYAW, std::make_shared<UniformDistribution>(lower_wtheta, upper_wtheta)}};

    // measurement noise
    const double mr_lambda = 0.2;
    const double upper_mtheta = (M_PI/8.0);
    const double lower_mtheta = -(M_PI/8.0);
    std::map<int, std::shared_ptr<BaseDistribution>> measurement_noise_map{
            {MEASUREMENT_NOISE::IDX::WR, std::make_shared<ExponentialDistribution>(mr_lambda)},
            {MEASUREMENT_NOISE::IDX::WYAW, std::make_shared<UniformDistribution>(lower_mtheta, upper_mtheta)}};

    // Random Variable Generator
    std::default_random_engine generator;
    std::exponential_distribution<double> wx_dist(wx_lambda);
    std::uniform_real_distribution<double> wyaw_dist(lower_wtheta, upper_wtheta);
    std::exponential_distribution<double> mr_dist(mr_lambda);
    std::uniform_real_distribution<double> myaw_dist(lower_mtheta, upper_mtheta);

    std::vector<double> hmkf_x_diff_vec;
    std::vector<double> hmkf_yaw_diff_vec;
    std::vector<double> mkf_x_diff_vec;
    std::vector<double> mkf_yaw_diff_vec;
    std::vector<double> ekf_x_diff_vec;
    std::vector<double> ekf_yaw_diff_vec;
    StateInfo hmkf_state_info = ini_state;
    StateInfo mkf_state_info = ini_state;
    StateInfo ekf_state_info = ini_state;
    for(size_t i=0; i<N; ++i) {
        // System propagation
        Eigen::VectorXd system_noise = Eigen::VectorXd::Zero(2);
        system_noise(0) = wx_dist(generator);
        system_noise(1) = wyaw_dist(generator);
        x_true = example_model->propagate(x_true, control_inputs, system_noise, dt);

        // Measurement
        const Eigen::Vector2d measurement_noise = {mr_dist(generator), myaw_dist(generator)};
        const Eigen::VectorXd y = example_model->measure(x_true, measurement_noise);

        // Prediction
        const auto hmkf_predicted = hmkf.predict(hmkf_state_info, control_inputs, dt, system_noise_map);
        const auto ekf_predicted = ekf.predict(ekf_state_info, control_inputs, dt, system_noise_map);
        const auto mkf_predicted = mkf.predict(mkf_state_info, control_inputs, dt, system_noise_map);

        // Update
        hmkf_state_info = hmkf.update(hmkf_predicted, y, measurement_noise_map);
        mkf_state_info = mkf.update(mkf_predicted, y, measurement_noise_map);
        ekf_state_info = ekf.update(ekf_predicted, y, measurement_noise_map);

        // Evaluation
        // HMKF
        {
            const double x_diff = std::fabs(hmkf_state_info.mean(0) - x_true(0));
            const double yaw_diff = std::fabs(hmkf_state_info.mean(1) - x_true(1));
            hmkf_x_diff_vec.push_back(x_diff);
            hmkf_yaw_diff_vec.push_back(yaw_diff);
            std::cout << "hmkf_x_diff: " << x_diff << " [m]" << std::endl;
            std::cout << "hmkf_yaw_diff: " << yaw_diff << " [rad]" << std::endl;
            std::cout << "-------------" << std::endl;
        }
        {
            const double x_diff = std::fabs(mkf_state_info.mean(0) - x_true(0));
            const double yaw_diff = std::fabs(mkf_state_info.mean(1) - x_true(1));
            mkf_x_diff_vec.push_back(x_diff);
            mkf_yaw_diff_vec.push_back(yaw_diff);
            std::cout << "mkf_x_diff: " << x_diff << " [m]" << std::endl;
            std::cout << "mkf_yaw_diff: " << yaw_diff << " [rad]" << std::endl;
            std::cout << "-------------" << std::endl;
        }
        {
            const double x_diff = std::fabs(ekf_state_info.mean(0) - x_true(0));
            const double yaw_diff = std::fabs(ekf_state_info.mean(1) - x_true(1));
            ekf_x_diff_vec.push_back(x_diff);
            ekf_yaw_diff_vec.push_back(yaw_diff);
            std::cout << "ekf_x_diff: " << x_diff << " [m]" << std::endl;
            std::cout << "ekf_yaw_diff: " << yaw_diff << " [rad]" << std::endl;
            std::cout << "-------------" << std::endl;
        }
    }

    const double sum_hmkf_x_diff = std::accumulate(hmkf_x_diff_vec.begin(), hmkf_x_diff_vec.end(), 0.0);
    const double sum_hmkf_yaw_diff = std::accumulate(hmkf_yaw_diff_vec.begin(), hmkf_yaw_diff_vec.end(), 0.0);
    const double sum_mkf_x_diff = std::accumulate(mkf_x_diff_vec.begin(), mkf_x_diff_vec.end(), 0.0);
    const double sum_mkf_yaw_diff = std::accumulate(mkf_yaw_diff_vec.begin(), mkf_yaw_diff_vec.end(), 0.0);
    const double sum_ekf_x_diff = std::accumulate(ekf_x_diff_vec.begin(), ekf_x_diff_vec.end(), 0.0);
    const double sum_ekf_yaw_diff = std::accumulate(ekf_yaw_diff_vec.begin(), ekf_yaw_diff_vec.end(), 0.0);

    std::cout << "HMKF mean x_diff: " << sum_hmkf_x_diff / hmkf_x_diff_vec.size() << std::endl;
    std::cout << "HMKF mean yaw_diff: " << sum_hmkf_yaw_diff / hmkf_yaw_diff_vec.size() << std::endl;
    std::cout << "MKF mean x_diff: " << sum_mkf_x_diff / mkf_x_diff_vec.size() << std::endl;
    std::cout << "MKF mean yaw_diff: " << sum_mkf_yaw_diff / mkf_yaw_diff_vec.size() << std::endl;
    std::cout << "EKF mean x_diff: " << sum_ekf_x_diff / ekf_x_diff_vec.size() << std::endl;
    std::cout << "EKF mean yaw_diff: " << sum_ekf_yaw_diff / ekf_yaw_diff_vec.size() << std::endl;
}