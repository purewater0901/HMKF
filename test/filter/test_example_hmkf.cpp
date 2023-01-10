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

#include "filter/example_hmkf.h"
#include "filter/squared_example_hmkf.h"
#include "filter/mkf.h"
#include "filter/ekf.h"
#include "filter/ukf.h"
#include "distribution/uniform_distribution.h"
#include "distribution/exponential_distribution.h"
#include "distribution/beta_distribution.h"
#include "distribution/normal_distribution.h"
#include "model/example_model.h"
#include "model/squared_example_model.h"

using namespace Example;

TEST(ExampleHMKF, Predict)
{
    const size_t state_dim = 2;
    const size_t system_noise_dim = 2;
    const size_t measurement_dim = 1;
    const size_t measurement_noise_dim = 1;
    std::shared_ptr<BaseModel> example_model = std::make_shared<ExampleVehicleModel>(state_dim, system_noise_dim,
                                                                                     measurement_dim,
                                                                                     measurement_noise_dim);

    ExampleHMKF hmkf;
    MKF mkf(example_model);
    EKF ekf(example_model);
    UKF ukf(example_model, 0.15);

    const double dt = 0.1;

    // Initial State
    StateInfo ini_state;
    ini_state.mean = Eigen::VectorXd::Zero(2);
    ini_state.mean(0) = 0.0;
    ini_state.mean(1) = 0.0;
    ini_state.covariance = Eigen::MatrixXd::Zero(2, 2);
    ini_state.covariance(0, 0) = 0.1*0.1; // V[x]
    ini_state.covariance(1, 1) = 0.1*0.1; // V[y]

    // Input
    Eigen::Vector2d control_inputs = Eigen::Vector2d::Zero();
    control_inputs(0) = 1.0;
    control_inputs(1) = 0.1;

    // System Noise
    const double wx_lambda = 1.0;
    const double upper_wtheta = (M_PI/10.0);
    const double lower_wtheta = -(M_PI/10.0);
    std::map<int, std::shared_ptr<BaseDistribution>> system_noise_map{
            {SYSTEM_NOISE::IDX::WV, std::make_shared<ExponentialDistribution>(wx_lambda)},
            {SYSTEM_NOISE::IDX::WYAW, std::make_shared<UniformDistribution>(lower_wtheta, upper_wtheta)}};

    // Measurement noise
    const double mr_lambda = 10.0;
    std::map<int, std::shared_ptr<BaseDistribution>> measurement_noise_map{
            {MEASUREMENT_NOISE::IDX::WR, std::make_shared<ExponentialDistribution>(mr_lambda)}};

    const auto predicted_moments = hmkf.predict(ini_state, control_inputs, dt, system_noise_map);
    const auto mkf_predicted_info = mkf.predict(ini_state, control_inputs, dt, system_noise_map);
    const auto ekf_predicted_info = ekf.predict(ini_state, control_inputs, dt, system_noise_map);
    const auto ukf_predicted_info = ukf.predict(ini_state, control_inputs, dt, system_noise_map, measurement_noise_map);

    std::cout << "E[X]: " << predicted_moments.xPow1 << std::endl;
    std::cout << "E[Y]: " << predicted_moments.yPow1 << std::endl;
    std::cout << "E[X^2]: " << predicted_moments.xPow2 << std::endl;
    std::cout << "E[Y^2]: " << predicted_moments.yPow2 << std::endl;
    std::cout << "E[XY]: " << predicted_moments.xPow1_yPow1 << std::endl;
    std::cout << "E[X^3]: " << predicted_moments.xPow3 << std::endl;
    std::cout << "E[Y^3]: " << predicted_moments.yPow3 << std::endl;
    std::cout << "E[X^2Y]: " << predicted_moments.xPow2_yPow1 << std::endl;
    std::cout << "E[XY^2]: " << predicted_moments.xPow1_yPow2 << std::endl;
    std::cout << "E[X^4]: " << predicted_moments.xPow4 << std::endl;
    std::cout << "E[Y^4]: " << predicted_moments.yPow4 << std::endl;
    std::cout << "E[X^2Y^2]: " << predicted_moments.xPow2_yPow2 << std::endl;
    std::cout << "EKF E[X]: " << ekf_predicted_info.mean(0) << std::endl;
    std::cout << "EKF E[Y]: " << ekf_predicted_info.mean(1) << std::endl;
    std::cout << "EKF E[X^2]: " << ekf_predicted_info.covariance(0,0) + ekf_predicted_info.mean(0)*ekf_predicted_info.mean(0) << std::endl;
    std::cout << "EKF E[Y^2]: " << ekf_predicted_info.covariance(1,1) + ekf_predicted_info.mean(1)*ekf_predicted_info.mean(1) << std::endl;
    std::cout << "UKF E[X]: " << ukf_predicted_info.mean(0) << std::endl;
    std::cout << "UKF E[Y]: " << ukf_predicted_info.mean(1) << std::endl;
    std::cout << "UKF E[X^2]: " << ukf_predicted_info.covariance(0,0) + ukf_predicted_info.mean(0)*ukf_predicted_info.mean(0) << std::endl;
    std::cout << "UKF E[Y^2]: " << ukf_predicted_info.covariance(1,1) + ukf_predicted_info.mean(1)*ukf_predicted_info.mean(1) << std::endl;
    std::cout << "MKF E[X]: " << mkf_predicted_info.mean(0) << std::endl;
    std::cout << "MKF E[Y]: " << mkf_predicted_info.mean(1) << std::endl;
    std::cout << "MKF E[X^2]: " << mkf_predicted_info.covariance(0,0) + mkf_predicted_info.mean(0)*mkf_predicted_info.mean(0) << std::endl;
    std::cout << "MKF E[Y^2]: " << mkf_predicted_info.covariance(1,1) + mkf_predicted_info.mean(1)*mkf_predicted_info.mean(1) << std::endl;
    std::cout << "-----------------------" << std::endl;
    std::cout << "-----------------------" << std::endl;

    // Update
    const auto measurement_moments = hmkf.getMeasurementMoments(predicted_moments, measurement_noise_map);
    const auto state_measurement_matrix = hmkf.getStateMeasurementMatrix(predicted_moments, measurement_moments, measurement_noise_map);

    const auto mkf_measurement_moments = example_model->getMeasurementMoments(mkf_predicted_info, measurement_noise_map);
    const auto ekf_measurement_moments = ekf.getMeasurementInfo(ekf_predicted_info, measurement_noise_map);
    const auto ukf_measurement_moments = ukf.getMeasurementInfo(ukf_predicted_info, system_noise_map, measurement_noise_map);

    std::cout << "EKF: E[R]: " << ekf_measurement_moments.mean(0) << std::endl;
    std::cout << "EKF: E[R^2]: " << ekf_measurement_moments.covariance(0,0) + ekf_measurement_moments.mean(0)*ekf_measurement_moments.mean(0) << std::endl;
    std::cout << "UKF: E[R]: " << ukf_measurement_moments.mean(0) << std::endl;
    std::cout << "UKF: E[R^2]: " << ukf_measurement_moments.covariance(0,0) + ukf_measurement_moments.mean(0)*ukf_measurement_moments.mean(0) << std::endl;
    std::cout << "MKF: E[R]: " << mkf_measurement_moments.mean(0) << std::endl;
    std::cout << "MKF: E[R^2]: " << mkf_measurement_moments.covariance(0,0) + mkf_measurement_moments.mean(0)*mkf_measurement_moments.mean(0) << std::endl;
    std::cout << "HMKF E[R]: " << measurement_moments.rPow1 << std::endl;
    std::cout << "HMKF E[R^2]: " << measurement_moments.rPow2 << std::endl;
}

TEST(SquaredExampleHMKF, Predict)
{
    const size_t state_dim = 2;
    const size_t system_noise_dim = 2;
    const size_t measurement_dim = 1;
    const size_t measurement_noise_dim = 1;
    std::shared_ptr<BaseModel> example_model = std::make_shared<ExampleSquaredVehicleModel>(state_dim, system_noise_dim,
                                                                                     measurement_dim,
                                                                                     measurement_noise_dim);

    //ExampleHMKF hmkf;
    MKF mkf(example_model);
    EKF ekf(example_model);
    UKF ukf(example_model, 0.15);
    SquaredExampleHMKF hmkf;

    const double dt = 0.1;

    // Initial State
    StateInfo ini_state;
    ini_state.mean = Eigen::VectorXd::Zero(2);
    ini_state.mean(0) = 0.0;
    ini_state.mean(1) = 0.0;
    ini_state.covariance = Eigen::MatrixXd::Zero(2, 2);
    ini_state.covariance(0, 0) = 0.1*0.1; // V[x]
    ini_state.covariance(1, 1) = 0.1*0.1; // V[y]

    // Input
    Eigen::Vector2d control_inputs = Eigen::Vector2d::Zero();
    control_inputs(0) = 1.0;
    control_inputs(1) = 0.1;

    // System Noise
    const double wx_lambda = 1.0;
    const double upper_wtheta = (M_PI/10.0);
    const double lower_wtheta = -(M_PI/10.0);
    std::map<int, std::shared_ptr<BaseDistribution>> system_noise_map{
            {SYSTEM_NOISE::IDX::WV, std::make_shared<ExponentialDistribution>(wx_lambda)},
            {SYSTEM_NOISE::IDX::WYAW, std::make_shared<UniformDistribution>(lower_wtheta, upper_wtheta)}};

    // Measurement noise
    const double mr_lambda = 10.0;
    std::map<int, std::shared_ptr<BaseDistribution>> measurement_noise_map{
            {MEASUREMENT_NOISE::IDX::WR, std::make_shared<ExponentialDistribution>(mr_lambda)}};

    const auto predicted_moments = hmkf.predict(ini_state, control_inputs, dt, system_noise_map);
    const auto mkf_predicted_info = mkf.predict(ini_state, control_inputs, dt, system_noise_map);
    const auto ekf_predicted_info = ekf.predict(ini_state, control_inputs, dt, system_noise_map);
    const auto ukf_predicted_info = ukf.predict(ini_state, control_inputs, dt, system_noise_map, measurement_noise_map);

    std::cout << "HMKF E[X]: " << predicted_moments.xPow1 << std::endl;
    std::cout << "HMKF E[Y]: " << predicted_moments.yPow1 << std::endl;
    std::cout << "HMKF E[X^2]: " << predicted_moments.xPow2 << std::endl;
    std::cout << "HMKF E[Y^2]: " << predicted_moments.yPow2 << std::endl;
    std::cout << "EKF E[X]: " << ekf_predicted_info.mean(0) << std::endl;
    std::cout << "EKF E[Y]: " << ekf_predicted_info.mean(1) << std::endl;
    std::cout << "EKF E[X^2]: " << ekf_predicted_info.covariance(0,0) + ekf_predicted_info.mean(0)*ekf_predicted_info.mean(0) << std::endl;
    std::cout << "EKF E[Y^2]: " << ekf_predicted_info.covariance(1,1) + ekf_predicted_info.mean(1)*ekf_predicted_info.mean(1) << std::endl;
    std::cout << "UKF E[X]: " << ukf_predicted_info.mean(0) << std::endl;
    std::cout << "UKF E[Y]: " << ukf_predicted_info.mean(1) << std::endl;
    std::cout << "UKF E[X^2]: " << ukf_predicted_info.covariance(0,0) + ukf_predicted_info.mean(0)*ukf_predicted_info.mean(0) << std::endl;
    std::cout << "UKF E[Y^2]: " << ukf_predicted_info.covariance(1,1) + ukf_predicted_info.mean(1)*ukf_predicted_info.mean(1) << std::endl;
    std::cout << "MKF E[X]: " << mkf_predicted_info.mean(0) << std::endl;
    std::cout << "MKF E[Y]: " << mkf_predicted_info.mean(1) << std::endl;
    std::cout << "MKF E[X^2]: " << mkf_predicted_info.covariance(0,0) + mkf_predicted_info.mean(0)*mkf_predicted_info.mean(0) << std::endl;
    std::cout << "MKF E[Y^2]: " << mkf_predicted_info.covariance(1,1) + mkf_predicted_info.mean(1)*mkf_predicted_info.mean(1) << std::endl;
    std::cout << "-----------------------" << std::endl;
    std::cout << "-----------------------" << std::endl;

    // Update
    const auto measurement_moments = hmkf.getMeasurementMoments(predicted_moments, measurement_noise_map);
    const auto state_measurement_matrix = hmkf.getStateMeasurementMatrix(predicted_moments, measurement_moments, measurement_noise_map);

    const auto mkf_measurement_moments = example_model->getMeasurementMoments(mkf_predicted_info, measurement_noise_map);
    const auto ekf_measurement_moments = ekf.getMeasurementInfo(ekf_predicted_info, measurement_noise_map);
    const auto ukf_measurement_moments = ukf.getMeasurementInfo(ukf_predicted_info, system_noise_map, measurement_noise_map);

    std::cout << "EKF: E[R]: " << ekf_measurement_moments.mean(0) << std::endl;
    std::cout << "EKF: E[R^2]: " << ekf_measurement_moments.covariance(0,0) + ekf_measurement_moments.mean(0)*ekf_measurement_moments.mean(0) << std::endl;
    std::cout << "UKF: E[R]: " << ukf_measurement_moments.mean(0) << std::endl;
    std::cout << "UKF: E[R^2]: " << ukf_measurement_moments.covariance(0,0) + ukf_measurement_moments.mean(0)*ukf_measurement_moments.mean(0) << std::endl;
    std::cout << "MKF: E[R]: " << mkf_measurement_moments.mean(0) << std::endl;
    std::cout << "MKF: E[R^2]: " << mkf_measurement_moments.covariance(0,0) + mkf_measurement_moments.mean(0)*mkf_measurement_moments.mean(0) << std::endl;
    std::cout << "HMKF E[R]: " << measurement_moments.rPow1 << std::endl;
    std::cout << "HMKF E[R^2]: " << measurement_moments.rPow2 << std::endl;
}

/*
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
    ini_state.covariance(1, 1) = (M_PI/10)*(M_PI/10); // V[yaw]
    ini_state.covariance(0, 1) = 0.1*0.1; // V[x*yaw]
    ini_state.covariance(1, 0) = ini_state.covariance(0, 1); // V[x*yaw]

    Eigen::VectorXd x_true = ini_state.mean;

    // Input
    Eigen::VectorXd control_inputs = Eigen::VectorXd::Zero(2);
    control_inputs(0) = 1.0 * dt;
    control_inputs(1) = 0.1 * dt;

    // System Noise
    const double wx_lambda = 0.2;
    const double wyaw_alpha = 5.0;
    const double wyaw_beta = 1.0;
    std::map<int, std::shared_ptr<BaseDistribution>> system_noise_map{
            {SYSTEM_NOISE::IDX::WX, std::make_shared<ExponentialDistribution>(wx_lambda)},
            {SYSTEM_NOISE::IDX::WYAW, std::make_shared<BetaDistribution>(wyaw_alpha, wyaw_beta)}};

    // measurement noise
    const double upper_mr_lambda = 300;
    const double lower_mr_lambda = 0.0;
    const double mean_mtheta = M_PI / 8.0;
    const double cov_mtheta =  std::pow(M_PI/6.0, 2);
    std::map<int, std::shared_ptr<BaseDistribution>> measurement_noise_map{
            {MEASUREMENT_NOISE::IDX::WR, std::make_shared<UniformDistribution>(lower_mr_lambda, upper_mr_lambda)},
            {MEASUREMENT_NOISE::IDX::WYAW, std::make_shared<NormalDistribution>(mean_mtheta, cov_mtheta)}};

    // Random Variable Generator
    std::default_random_engine generator;
    std::exponential_distribution<double> wx_dist(wx_lambda);
    boost::random::mt19937 engine(1234567890);
    boost::function<double()> wyaw_dist = boost::bind(boost::random::beta_distribution<>(wyaw_alpha, wyaw_beta), engine);
    std::uniform_real_distribution<double> mr_dist(lower_mr_lambda, upper_mr_lambda);
    std::normal_distribution<double> myaw_dist(mean_mtheta, std::sqrt(cov_mtheta));

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
        system_noise(1) = wyaw_dist();
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
 */
