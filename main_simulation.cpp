#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <Eigen/Eigen>
#include <fstream>
#include <filesystem>
#include <algorithm>

#include "matplotlibcpp.h"
#include "distribution/uniform_distribution.h"
#include "distribution/normal_distribution.h"
#include "distribution/three_dimensional_normal_distribution.h"
#include "filter/simple_vehicle_squared_hmkf.h"
#include "filter/mkf.h"
#include "filter/ukf.h"
#include "filter/ekf.h"
#include "model/simple_vehicle_squared_model.h"

using namespace SimpleVehicleSquared;

int main()
{
    // simulation setting value
    const double land_x = 0.0;
    const double land_y = 0.0;
    const Eigen::Vector2d landmark = {land_x, land_y};
    const double N = 400;
    const double dt = 1.0;

    // vehicle model
    std::shared_ptr<BaseModel> vehicle_model = std::make_shared<SimpleVehicleSquaredModel>(3, 2, 2, 2);

    // initial state
    const double x_ini = 3.0;
    const double y_ini = 5.0;
    const double yaw_ini = M_PI/4.0;
    const double x_cov = std::pow(0.1, 2);
    const double y_cov = std::pow(0.1, 2);
    const double yaw_cov = std::pow(M_PI/100, 2);
    StateInfo ini_state;
    ini_state.mean = Eigen::VectorXd::Zero(3);
    ini_state.covariance = Eigen::MatrixXd::Zero(3, 3);
    ini_state.mean(STATE::IDX::X) = x_ini;
    ini_state.mean(STATE::IDX::Y) = y_ini;
    ini_state.mean(STATE::IDX::YAW) = yaw_ini;
    ini_state.covariance(STATE::IDX::X, STATE::IDX::X) = x_cov;
    ini_state.covariance(STATE::IDX::Y, STATE::IDX::Y) = y_cov;
    ini_state.covariance(STATE::IDX::YAW, STATE::IDX::YAW) = yaw_cov;

    // true value
    Eigen::VectorXd true_state = ini_state.mean;

    // input
    const double v = 1.0;
    const double u = 0.01;
    Eigen::VectorXd control_input = Eigen::VectorXd::Zero(2);
    control_input(0) = v * dt;
    control_input(1) = u * dt;

    // system noise map
    const double mean_wv = 10.0*dt;
    const double cov_wv = std::pow(3.0*dt, 2);
    const double mean_wu = 0.0*dt;
    const double cov_wu = std::pow(0.1*dt, 2);
    std::map<int, std::shared_ptr<BaseDistribution>> system_noise_map = {
            //{SYSTEM_NOISE::IDX::WV, std::make_shared<NormalDistribution>(mean_wv, cov_wv)},
            {SYSTEM_NOISE::IDX::WV, std::make_shared<NormalDistribution>(6.5, 11.0)},
            {SYSTEM_NOISE::IDX::WU, std::make_shared<NormalDistribution>(mean_wu, cov_wu)}};
    //std::normal_distribution<double> wv_dist(mean_wv, std::sqrt(cov_wv));
    std::extreme_value_distribution<double> wv_dist(4.0, 5.0);
    std::normal_distribution<double> wu_dist(mean_wu, std::sqrt(cov_wu));

    // measurement noise map
    const double mean_mr = 1.0;
    const double cov_mr = std::pow(0.1, 2);
    const double mean_ma = 0.0;
    const double cov_ma = std::pow(M_PI/50.0, 2);
    std::map<int, std::shared_ptr<BaseDistribution>> measurement_noise_map = {
        {MEASUREMENT_NOISE::IDX::WR, std::make_shared<NormalDistribution>(mean_mr, cov_mr)},
        {MEASUREMENT_NOISE::IDX::WA, std::make_shared<NormalDistribution>(mean_ma, cov_ma)}};
    std::normal_distribution<double> mr_dist(mean_mr, std::sqrt(cov_mr));
    std::normal_distribution<double> ma_dist(mean_ma, std::sqrt(cov_ma));

    // random value engine
    //std::random_device seed_gen(0);
    //std::default_random_engine engine(seed_gen());
    std::default_random_engine engine;

    // Filters
    EKF ekf(vehicle_model);
    UKF ukf(vehicle_model);
    MKF mkf(vehicle_model);
    SimpleVehicleSquaredHMKF hmkf(vehicle_model);
    auto ekf_state_info = ini_state;
    auto ukf_state_info = ini_state;
    auto mkf_state_info = ini_state;
    auto hmkf_state_info = ini_state;
    std::shared_ptr<HighOrderMoments> hmkf_predicted_moments = nullptr;

    std::vector<double> times = {0.0};
    std::vector<double> ekf_xy_errors = {0.0};
    std::vector<double> ekf_yaw_errors = {0.0};
    std::vector<double> ukf_xy_errors = {0.0};
    std::vector<double> ukf_yaw_errors = {0.0};
    std::vector<double> mkf_xy_errors = {0.0};
    std::vector<double> mkf_yaw_errors = {0.0};
    std::vector<double> hmkf_xy_errors = {0.0};
    std::vector<double> hmkf_yaw_errors = {0.0};

    std::vector<double> x_true_vec = {true_state(0)};
    std::vector<double> y_true_vec = {true_state(1)};
    std::vector<double> yaw_true_vec = {true_state(2)};
    std::vector<double> hmkf_x_estimate = {true_state(0)};
    std::vector<double> hmkf_y_estimate = {true_state(1)};
    std::vector<double> hmkf_yaw_estimate = {true_state(2)};
    std::vector<double> mkf_x_estimate = {true_state(0)};
    std::vector<double> mkf_y_estimate = {true_state(1)};
    std::vector<double> mkf_yaw_estimate = {true_state(2)};
    std::vector<double> ekf_x_estimate = {true_state(0)};
    std::vector<double> ekf_y_estimate = {true_state(1)};
    std::vector<double> ekf_yaw_estimate = {true_state(2)};
    std::vector<double> ukf_x_estimate = {true_state(0)};
    std::vector<double> ukf_y_estimate = {true_state(1)};
    std::vector<double> ukf_yaw_estimate = {true_state(2)};

    for(size_t iter=0; iter<N; ++iter) {
        std::cout << "Iteration: " << iter << std::endl;
        const double wv = wv_dist(engine);
        const double wu = wu_dist(engine);
        Eigen::VectorXd system_noise = Eigen::VectorXd::Zero(2);
        system_noise(0) = wv;
        system_noise(1) = wu;

        // system propagation
        true_state = vehicle_model->propagate(true_state, control_input, system_noise, dt);

        // measurement
        const double mr = mr_dist(engine);
        const double ma = ma_dist(engine);
        Eigen::VectorXd measurement_noise = Eigen::VectorXd::Zero(2);
        measurement_noise(0) = mr;
        measurement_noise(1) = ma;
        const Eigen::VectorXd y = vehicle_model->measureWithLandmark(true_state, measurement_noise, landmark);

        // Filtering(Predict)
        hmkf_state_info = hmkf.predict(hmkf_state_info, control_input, dt, system_noise_map, hmkf_predicted_moments);
        mkf_state_info = mkf.predict(mkf_state_info, control_input, dt, system_noise_map);
        ekf_state_info = ekf.predict(ekf_state_info, control_input, dt, system_noise_map);
        ukf_state_info = ukf.predict(ukf_state_info, control_input, dt, system_noise_map, measurement_noise_map);

        // Filtering(Update)
        ekf_state_info = ekf.update(ekf_state_info, y, measurement_noise_map, {landmark(0), landmark(1)});
        ukf_state_info = ukf.update(ukf_state_info, y, system_noise_map, measurement_noise_map, {landmark(0), landmark(1)});
        mkf_state_info = mkf.update(mkf_state_info, y, measurement_noise_map, {landmark(0), landmark(1)});
        hmkf_state_info = hmkf.update(hmkf_state_info, *hmkf_predicted_moments, y, {landmark(0), landmark(1)}, measurement_noise_map);

        // insert true values
        times.push_back(times.back()+dt);
        x_true_vec.push_back(true_state(0));
        y_true_vec.push_back(true_state(1));
        yaw_true_vec.push_back(true_state(2));
        ekf_x_estimate.push_back(ekf_state_info.mean(0));
        ekf_y_estimate.push_back(ekf_state_info.mean(1));
        ekf_yaw_estimate.push_back(ekf_state_info.mean(2));
        ukf_x_estimate.push_back(ukf_state_info.mean(0));
        ukf_y_estimate.push_back(ukf_state_info.mean(1));
        ukf_yaw_estimate.push_back(ukf_state_info.mean(2));
        mkf_x_estimate.push_back(mkf_state_info.mean(0));
        mkf_y_estimate.push_back(mkf_state_info.mean(1));
        mkf_yaw_estimate.push_back(mkf_state_info.mean(2));
        hmkf_x_estimate.push_back(hmkf_state_info.mean(0));
        hmkf_y_estimate.push_back(hmkf_state_info.mean(1));
        hmkf_yaw_estimate.push_back(hmkf_state_info.mean(2));

        // compute errors
        // EKF
        {
            const double dx = true_state(0) - ekf_state_info.mean(0);
            const double dy = true_state(1) - ekf_state_info.mean(1);
            const double xy_error = std::hypot(dx, dy);
            const double dyaw = normalizeRadian(true_state(2) - ekf_state_info.mean(2));

            ekf_xy_errors.push_back(xy_error);
            ekf_yaw_errors.push_back(std::fabs(dyaw));
            ekf_x_estimate.push_back(ekf_state_info.mean(0));
            ekf_y_estimate.push_back(ekf_state_info.mean(1));
            ekf_yaw_estimate.push_back(ekf_state_info.mean(2));
        }

        // UKF
        {
            const double dx = true_state(0) - ukf_state_info.mean(0);
            const double dy = true_state(1) - ukf_state_info.mean(1);
            const double xy_error = std::hypot(dx, dy);
            const double dyaw = normalizeRadian(true_state(2) - ukf_state_info.mean(2));

            ukf_xy_errors.push_back(xy_error);
            ukf_yaw_errors.push_back(std::fabs(dyaw));
            ukf_x_estimate.push_back(ukf_state_info.mean(0));
            ukf_y_estimate.push_back(ukf_state_info.mean(1));
            ukf_yaw_estimate.push_back(ukf_state_info.mean(2));
        }

        // MKF
        {
            const double dx = true_state(0) - mkf_state_info.mean(0);
            const double dy = true_state(1) - mkf_state_info.mean(1);
            const double xy_error = std::hypot(dx, dy);
            const double dyaw = normalizeRadian(true_state(2) - mkf_state_info.mean(2));

            mkf_xy_errors.push_back(xy_error);
            mkf_yaw_errors.push_back(std::fabs(dyaw));
            mkf_x_estimate.push_back(mkf_state_info.mean(0));
            mkf_y_estimate.push_back(mkf_state_info.mean(1));
            mkf_yaw_estimate.push_back(mkf_state_info.mean(2));
        }

        // HMKF
        {
            const double dx = true_state(0) - hmkf_state_info.mean(0);
            const double dy = true_state(1) - hmkf_state_info.mean(1);
            const double xy_error = std::hypot(dx, dy);
            const double dyaw = normalizeRadian(true_state(2) - hmkf_state_info.mean(2));

            hmkf_xy_errors.push_back(xy_error);
            hmkf_yaw_errors.push_back(std::fabs(dyaw));
            hmkf_x_estimate.push_back(hmkf_state_info.mean(0));
            hmkf_y_estimate.push_back(hmkf_state_info.mean(1));
            hmkf_yaw_estimate.push_back(hmkf_state_info.mean(2));
        }
    }

    double hmkf_xy_error_sum = 0.0;
    double mkf_xy_error_sum = 0.0;
    double ekf_xy_error_sum = 0.0;
    double ukf_xy_error_sum = 0.0;
    double hmkf_yaw_error_sum = 0.0;
    double mkf_yaw_error_sum = 0.0;
    double ekf_yaw_error_sum = 0.0;
    double ukf_yaw_error_sum = 0.0;
    for(size_t i=0; i<ukf_xy_errors.size(); ++i) {
        hmkf_xy_error_sum += hmkf_xy_errors.at(i);
        mkf_xy_error_sum += mkf_xy_errors.at(i);
        ekf_xy_error_sum += ekf_xy_errors.at(i);
        ukf_xy_error_sum += ukf_xy_errors.at(i);
        hmkf_yaw_error_sum += hmkf_yaw_errors.at(i);
        mkf_yaw_error_sum += mkf_yaw_errors.at(i);
        ekf_yaw_error_sum += ekf_yaw_errors.at(i);
        ukf_yaw_error_sum += ukf_yaw_errors.at(i);
    }
    std::cout << "hmkf_xy_error mean: " << hmkf_xy_error_sum / hmkf_xy_errors.size() << std::endl;
    std::cout << "nkf_xy_error mean: " << mkf_xy_error_sum / mkf_xy_errors.size() << std::endl;
    std::cout << "ekf_xy_error mean: " << ekf_xy_error_sum / ekf_xy_errors.size() << std::endl;
    std::cout << "ukf_xy_error mean: " << ukf_xy_error_sum / ukf_xy_errors.size() << std::endl;
    std::cout << "hmkf_yaw_error mean: " << hmkf_yaw_error_sum / hmkf_yaw_errors.size() << std::endl;
    std::cout << "nkf_yaw_error mean: " << mkf_yaw_error_sum / mkf_yaw_errors.size() << std::endl;
    std::cout << "ekf_yaw_error mean: " << ekf_yaw_error_sum / ekf_yaw_errors.size() << std::endl;
    std::cout << "ukf_yaw_error mean: " << ukf_yaw_error_sum / ukf_yaw_errors.size() << std::endl;

    matplotlibcpp::figure_size(1500, 900);
    std::map<std::string, std::string> hmkf_keywords;
    std::map<std::string, std::string> mkf_keywords;
    std::map<std::string, std::string> ekf_keywords;
    std::map<std::string, std::string> ukf_keywords;
    hmkf_keywords.insert(std::pair<std::string, std::string>("label", "hmkf error"));
    mkf_keywords.insert(std::pair<std::string, std::string>("label", "mkf error"));
    ekf_keywords.insert(std::pair<std::string, std::string>("label", "ekf error"));
    ukf_keywords.insert(std::pair<std::string, std::string>("label", "ukf error"));
    matplotlibcpp::plot(hmkf_x_estimate, hmkf_y_estimate, hmkf_keywords);
    matplotlibcpp::plot(mkf_x_estimate, mkf_y_estimate, mkf_keywords);
    matplotlibcpp::plot(ekf_x_estimate, ekf_y_estimate, ekf_keywords);
    matplotlibcpp::plot(ukf_x_estimate, ukf_y_estimate, ukf_keywords);
    matplotlibcpp::named_plot("true", x_true_vec, y_true_vec);
    matplotlibcpp::legend();
    matplotlibcpp::title("Result");
    matplotlibcpp::show();

    return 0;
}