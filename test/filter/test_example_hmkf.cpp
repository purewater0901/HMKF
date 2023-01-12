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

constexpr double epsilon = 1e-3;

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
    Eigen::VectorXd control_inputs = Eigen::VectorXd::Zero(2);
    control_inputs(0) = 1.0;

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

    EXPECT_NEAR(predicted_moments.xPow1, 0.19672, epsilon);
    EXPECT_NEAR(predicted_moments.yPow1, 0.0, epsilon);
    EXPECT_NEAR(predicted_moments.xPow2, 0.058389339283061455, epsilon);
    EXPECT_NEAR(predicted_moments.yPow2, 0.011614470839380995, epsilon);
    EXPECT_NEAR(predicted_moments.xPow1_yPow1, 0.0, epsilon);
    EXPECT_NEAR(predicted_moments.xPow3, 0.021143192063091603, epsilon);
    EXPECT_NEAR(predicted_moments.yPow3, 0.0, epsilon);
    EXPECT_NEAR(predicted_moments.xPow2_yPow1, 0.0, epsilon);
    EXPECT_NEAR(predicted_moments.xPow1_yPow2, 0.0024687403860974098, epsilon);
    EXPECT_NEAR(predicted_moments.xPow4, 0.009301399614907455, epsilon);
    EXPECT_NEAR(predicted_moments.yPow4, 0.00040906469343830905, epsilon);
    EXPECT_NEAR(measurement_moments.rPow1, 0.17, epsilon);
    EXPECT_NEAR(measurement_moments.rPow2, 0.0453, epsilon);
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
    Eigen::VectorXd control_inputs = Eigen::VectorXd::Zero(2);
    control_inputs(0) = 1.0;

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

TEST(ExampleHMKF, Simulation)
{
    std::shared_ptr<BaseModel> example_model = std::make_shared<ExampleSquaredVehicleModel>(2, 2, 1, 1);

    ExampleHMKF hmkf;
    EKF ekf(example_model);
    MKF mkf(example_model);
    UKF ukf(example_model);

    const double dt = 0.1;
    size_t N = 3000;

    // Initial State
    StateInfo ini_state;
    ini_state.mean = Eigen::VectorXd::Zero(2);
    ini_state.mean(0) = 0.0;
    ini_state.mean(1) = 0.0;
    ini_state.covariance = Eigen::MatrixXd::Zero(2, 2);
    ini_state.covariance(0, 0) = 0.1*0.1; // V[x]
    ini_state.covariance(1, 1) = 0.1*0.1; // V[y]

    Eigen::VectorXd x_true = ini_state.mean;

    // Input
    Eigen::VectorXd control_inputs = Eigen::VectorXd::Zero(1);
    control_inputs(0) = 1.0;

    // System Noise
    const double wv_lambda = 0.2;
    //const double wyaw_alpha = 5.0;
    //const double wyaw_beta = 1.0;
    const double wyaw_mean = 0.0;
    const double wyaw_cov = M_PI/30 * M_PI/30;
    std::map<int, std::shared_ptr<BaseDistribution>> system_noise_map{
            {SYSTEM_NOISE::IDX::WV, std::make_shared<ExponentialDistribution>(wv_lambda)},
            {SYSTEM_NOISE::IDX::WYAW, std::make_shared<NormalDistribution>(wyaw_mean, wyaw_cov)}};

    // measurement noise
    const double upper_mr_lambda = 30;
    const double lower_mr_lambda = 0.0;
    std::map<int, std::shared_ptr<BaseDistribution>> measurement_noise_map{
            {MEASUREMENT_NOISE::IDX::WR, std::make_shared<UniformDistribution>(lower_mr_lambda,upper_mr_lambda)}};

    // Random Variable Generator
    std::default_random_engine generator;
    std::exponential_distribution<double> wv_dist(wv_lambda);
    /*
    boost::random::mt19937 engine(1234567890);
    boost::function<double()> wyaw_dist = boost::bind(boost::random::beta_distribution<>(wyaw_alpha, wyaw_beta), engine);
    */
    std::normal_distribution<double> wyaw_dist(wyaw_mean, std::sqrt(wyaw_cov));
    std::uniform_real_distribution<double> mr_dist(lower_mr_lambda, upper_mr_lambda);

    std::vector<double> hmkf_xy_diff_vec;
    std::vector<double> mkf_xy_diff_vec;
    std::vector<double> ekf_xy_diff_vec;
    std::vector<double> ukf_xy_diff_vec;
    StateInfo hmkf_state_info = ini_state;
    StateInfo mkf_state_info = ini_state;
    StateInfo ekf_state_info = ini_state;
    StateInfo ukf_state_info = ini_state;
    for(size_t i=0; i<N; ++i) {
        // System propagation
        Eigen::VectorXd system_noise = Eigen::VectorXd::Zero(2);
        system_noise(0) = wv_dist(generator);
        system_noise(1) = wyaw_dist(generator);
        x_true = example_model->propagate(x_true, control_inputs, system_noise, dt);

        // Measurement
        Eigen::VectorXd measurement_noise = Eigen::VectorXd::Zero(1);
        measurement_noise(0) = mr_dist(generator);
        const Eigen::VectorXd y = example_model->measure(x_true, measurement_noise);

        // Prediction
        const auto hmkf_predicted = hmkf.predict(hmkf_state_info, control_inputs, dt, system_noise_map);
        const auto ekf_predicted = ekf.predict(ekf_state_info, control_inputs, dt, system_noise_map);
        const auto mkf_predicted = mkf.predict(mkf_state_info, control_inputs, dt, system_noise_map);
        const auto ukf_predicted = ukf.predict(ukf_state_info, control_inputs, dt, system_noise_map, measurement_noise_map);

        // Update
        hmkf_state_info = hmkf.update(hmkf_predicted, y, measurement_noise_map);
        mkf_state_info = mkf.update(mkf_predicted, y, measurement_noise_map);
        ekf_state_info = ekf.update(ekf_predicted, y, measurement_noise_map);
        ukf_state_info = ukf.update(ukf_predicted, y, system_noise_map, measurement_noise_map);

        // Evaluation
        // HMKF
        {
            const double x_diff = std::fabs(hmkf_state_info.mean(0) - x_true(0));
            const double y_diff = std::fabs(hmkf_state_info.mean(1) - x_true(1));
            const double dist = std::hypot(x_diff, y_diff);
            hmkf_xy_diff_vec.push_back(dist);
            std::cout << "hmkf_dist_diff: " << dist << " [m]" << std::endl;
        }
        // MKF
        {
            const double x_diff = std::fabs(mkf_state_info.mean(0) - x_true(0));
            const double y_diff = std::fabs(mkf_state_info.mean(1) - x_true(1));
            const double dist = std::hypot(x_diff, y_diff);
            mkf_xy_diff_vec.push_back(dist);
            std::cout << "mkf_dist_diff: " << dist << " [m]" << std::endl;
        }
        // EKF
        {
            const double x_diff = std::fabs(ekf_state_info.mean(0) - x_true(0));
            const double y_diff = std::fabs(ekf_state_info.mean(1) - x_true(1));
            const double dist = std::hypot(x_diff, y_diff);
            ekf_xy_diff_vec.push_back(dist);
            std::cout << "ekf_dist_diff: " << dist << " [m]" << std::endl;
        }
        // UKF
        {
            const double x_diff = std::fabs(ukf_state_info.mean(0) - x_true(0));
            const double y_diff = std::fabs(ukf_state_info.mean(1) - x_true(1));
            const double dist = std::hypot(x_diff, y_diff);
            ukf_xy_diff_vec.push_back(dist);
            std::cout << "ukf_dist_diff: " << dist << " [m]" << std::endl;
            std::cout << "-------------" << std::endl;
        }
    }

    const double sum_hmkf_xy_diff = std::accumulate(hmkf_xy_diff_vec.begin(), hmkf_xy_diff_vec.end(), 0.0);
    const double sum_mkf_xy_diff = std::accumulate(mkf_xy_diff_vec.begin(), mkf_xy_diff_vec.end(), 0.0);
    const double sum_ekf_xy_diff = std::accumulate(ekf_xy_diff_vec.begin(), ekf_xy_diff_vec.end(), 0.0);
    const double sum_ukf_xy_diff = std::accumulate(ukf_xy_diff_vec.begin(), ukf_xy_diff_vec.end(), 0.0);

    std::cout << "HMKF mean dist diff: " << sum_hmkf_xy_diff / hmkf_xy_diff_vec.size() << std::endl;
    std::cout << "MKF mean dist diff: " << sum_mkf_xy_diff / mkf_xy_diff_vec.size() << std::endl;
    std::cout << "EKF mean dist diff: " << sum_ekf_xy_diff / ekf_xy_diff_vec.size() << std::endl;
    std::cout << "UKF mean dist diff: " << sum_ukf_xy_diff / ukf_xy_diff_vec.size() << std::endl;
}