#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <Eigen/Eigen>

#include "matplotlibcpp.h"
#include "distribution/normal_distribution.h"
#include "model/normal_vehicle_model.h"
#include "filter/ekf.h"
#include "filter/ukf.h"
#include "filter/mkf.h"
#include "scenario/normal_vehicle_scenario.h"

using namespace NormalVehicle;

int main()
{
    NormalVehicleNonGaussianScenario scenario;
    const size_t N = scenario.N;
    const double dt = scenario.dt;

    // Vehicle Model
    std::shared_ptr<BaseModel> vehicle_model = std::make_shared<NormalVehicleModel>(3, 3, 2, 2);

    // Normal Vehicle Nonlinear Kalman Filter
    MKF normal_vehicle_mkf(vehicle_model);

    // Normal Vehicle Extended Kalman Filter
    EKF normal_vehicle_ekf(vehicle_model);

    // Normal Vehicle Unscented Kalman Filter
    UKF normal_vehicle_ukf(vehicle_model);

    // Uniform Distribution
    StateInfo nkf_state_info;
    nkf_state_info.mean = scenario.ini_mean_;
    nkf_state_info.covariance = scenario.ini_cov_;
    auto ekf_state_info = nkf_state_info;
    auto ukf_state_info = nkf_state_info;

    // Initial State
    Eigen::Vector3d x_true = nkf_state_info.mean;

    // Input
    const Eigen::VectorXd v_input = scenario.v_input_;
    const Eigen::VectorXd u_input = scenario.u_input_;

    // System Noise
    auto system_noise_map = scenario.system_noise_map_;

    // Observation Noise
    auto observation_noise_map = scenario.observation_noise_map_;

    // Random Variable Generator
    std::default_random_engine generator;
    auto wx_dist = scenario.wx_dist_;
    auto wy_dist = scenario.wy_dist_;
    auto wyaw_dist = scenario.wyaw_dist_;
    auto mr_dist = scenario.mr_dist_;
    auto myaw_dist = scenario.myaw_dist_;

    std::vector<double> times(N);
    std::vector<double> ukf_xy_errors(N);
    std::vector<double> nkf_xy_errors(N);
    std::vector<double> ekf_xy_errors(N);
    std::vector<double> ukf_yaw_errors(N);
    std::vector<double> nkf_yaw_errors(N);
    std::vector<double> ekf_yaw_errors(N);
    std::vector<double> x_true_vec(N);
    std::vector<double> y_true_vec(N);
    std::vector<double> nkf_x_estimate(N);
    std::vector<double> nkf_y_estimate(N);
    std::vector<double> ukf_x_estimate(N);
    std::vector<double> ukf_y_estimate(N);
    std::vector<double> ekf_x_estimate(N);
    std::vector<double> ekf_y_estimate(N);
    for(size_t i=0; i < N; ++i) {
        std::cout << "iteration: " << i << std::endl;
        // Control Inputs
        Eigen::Vector2d controls(v_input(i), u_input(i));

        // Simulate
        Eigen::Vector3d system_noise{wx_dist(generator), wy_dist(generator), wyaw_dist(generator)};
        Eigen::Vector2d observation_noise{std::max(0.0, mr_dist(generator)), myaw_dist()};
        x_true = vehicle_model->propagate(x_true, controls, system_noise, dt);
        auto nkf_y = vehicle_model->measure(x_true, observation_noise);
        auto ukf_y = nkf_y;
        auto ekf_y = nkf_y;

        // Predict
        const auto nkf_predicted_info = normal_vehicle_mkf.predict(nkf_state_info, controls, dt, system_noise_map);
        const auto ekf_predicted_info = normal_vehicle_ekf.predict(ekf_state_info, controls, dt, system_noise_map);
        const auto ukf_predicted_info = normal_vehicle_ukf.predict(ukf_state_info, controls, dt, system_noise_map, observation_noise_map);

        // Recalculate Yaw Angle to avoid the angle over 2*pi
        const double nkf_yaw_error = normalizeRadian(nkf_y(MEASUREMENT::IDX::YAW) - nkf_predicted_info.mean(STATE::IDX::YAW));
        const double ekf_yaw_error = normalizeRadian(ekf_y(MEASUREMENT::IDX::YAW) - ekf_predicted_info.mean(STATE::IDX::YAW));
        const double ukf_yaw_error = normalizeRadian(ukf_y(MEASUREMENT::IDX::YAW) - ukf_predicted_info.mean(STATE::IDX::YAW));
        nkf_y(MEASUREMENT::IDX::YAW) = nkf_yaw_error + nkf_predicted_info.mean(STATE::IDX::YAW);
        ekf_y(MEASUREMENT::IDX::YAW) = ekf_yaw_error + ekf_predicted_info.mean(STATE::IDX::YAW);
        ukf_y(MEASUREMENT::IDX::YAW) = ukf_yaw_error + ukf_predicted_info.mean(STATE::IDX::YAW);

        // Update
        const auto nkf_updated_info = normal_vehicle_mkf.update(nkf_predicted_info, nkf_y, observation_noise_map);
        const auto ekf_updated_info = normal_vehicle_ekf.update(ekf_predicted_info, ekf_y, observation_noise_map);
        const auto ukf_updated_info = normal_vehicle_ukf.update(ukf_predicted_info, ukf_y, system_noise_map, observation_noise_map);
        nkf_state_info = nkf_updated_info;
        ekf_state_info = ekf_updated_info;
        ukf_state_info = ukf_updated_info;

        // NKF
        {
            const double dx = x_true(STATE::IDX::X) - nkf_updated_info.mean(STATE::IDX::X);
            const double dy = x_true(STATE::IDX::Y) - nkf_updated_info.mean(STATE::IDX::Y);
            const double xy_error = std::sqrt(dx*dx + dy*dy);
            const double yaw_error = normalizeRadian(x_true(STATE::IDX::YAW) - nkf_updated_info.mean(STATE::IDX::YAW));

            std::cout << "mkf_xy_error: " << xy_error << std::endl;
            std::cout << "mkf_yaw_error: " << nkf_yaw_error << std::endl;
            nkf_xy_errors.at(i) = xy_error;
            nkf_yaw_errors.at(i) = yaw_error;
            nkf_yaw_errors.push_back(yaw_error);
            nkf_x_estimate.at(i) = nkf_state_info.mean(STATE::IDX::X);
            nkf_y_estimate.at(i) = nkf_state_info.mean(STATE::IDX::Y);
        }

        // UKF
        {
            const double dx = x_true(STATE::IDX::X) - ukf_updated_info.mean(STATE::IDX::X);
            const double dy = x_true(STATE::IDX::Y) - ukf_updated_info.mean(STATE::IDX::Y);
            const double xy_error = std::sqrt(dx*dx + dy*dy);
            const double yaw_error = normalizeRadian(x_true(STATE::IDX::YAW) - ukf_updated_info.mean(STATE::IDX::YAW));

            std::cout << "ukf_xy_error: " << xy_error << std::endl;
            std::cout << "ukf_yaw_error: " << ukf_yaw_error << std::endl;
            ukf_xy_errors.at(i) = xy_error;
            ukf_yaw_errors.at(i) = yaw_error;
            ukf_x_estimate.at(i) = ukf_updated_info.mean(STATE::IDX::X);
            ukf_y_estimate.at(i) = ukf_updated_info.mean(STATE::IDX::Y);
        }

        // EKF
        {
            const double dx = x_true(STATE::IDX::X) - ekf_updated_info.mean(STATE::IDX::X);
            const double dy = x_true(STATE::IDX::Y) - ekf_updated_info.mean(STATE::IDX::Y);
            const double xy_error = std::sqrt(dx*dx + dy*dy);
            const double yaw_error = normalizeRadian(x_true(STATE::IDX::YAW) - ekf_updated_info.mean(STATE::IDX::YAW));

            std::cout << "ekf_xy_error: " << xy_error << std::endl;
            std::cout << "ekf_yaw_error: " << ekf_yaw_error << std::endl;
            ekf_xy_errors.at(i) = xy_error;
            ekf_yaw_errors.at(i) = yaw_error;
            ekf_x_estimate.at(i) = ekf_state_info.mean(STATE::IDX::X);
            ekf_y_estimate.at(i) = ekf_state_info.mean(STATE::IDX::Y);
        }

        times.at(i) = i*dt;
        x_true_vec.at(i) = x_true(0);
        y_true_vec.at(i) = x_true(1);
    }

    double nkf_xy_error_sum = 0.0;
    double ekf_xy_error_sum = 0.0;
    double ukf_xy_error_sum = 0.0;
    double nkf_yaw_error_sum = 0.0;
    double ekf_yaw_error_sum = 0.0;
    double ukf_yaw_error_sum = 0.0;
    for(size_t i=0; i<ukf_xy_errors.size(); ++i) {
        nkf_xy_error_sum += nkf_xy_errors.at(i);
        ekf_xy_error_sum += ekf_xy_errors.at(i);
        ukf_xy_error_sum += ukf_xy_errors.at(i);
        nkf_yaw_error_sum += nkf_yaw_errors.at(i);
        ekf_yaw_error_sum += ekf_yaw_errors.at(i);
        ukf_yaw_error_sum += ukf_yaw_errors.at(i);
    }

    std::cout << "nkf_xy_error mean: " << nkf_xy_error_sum / N << std::endl;
    std::cout << "ekf_xy_error mean: " << ekf_xy_error_sum / N << std::endl;
    std::cout << "ukf_xy_error mean: " << ukf_xy_error_sum / N << std::endl;
    std::cout << "nkf_yaw_error mean: " << nkf_yaw_error_sum / N << std::endl;
    std::cout << "ekf_yaw_error mean: " << ekf_yaw_error_sum / N << std::endl;
    std::cout << "ukf_yaw_error mean: " << ukf_yaw_error_sum / N << std::endl;

    matplotlibcpp::figure_size(1500, 900);
    std::map<std::string, std::string> nkf_keywords;
    std::map<std::string, std::string> ukf_keywords;
    std::map<std::string, std::string> ekf_keywords;
    nkf_keywords.insert(std::pair<std::string, std::string>("label", "nkf error"));
    ukf_keywords.insert(std::pair<std::string, std::string>("label", "ukf error"));
    ekf_keywords.insert(std::pair<std::string, std::string>("label", "ekf error"));
    //matplotlibcpp::plot(times, nkf_xy_errors, nkf_keywords);
    //matplotlibcpp::plot(times, ukf_xy_errors, ukf_keywords);
    matplotlibcpp::plot(nkf_x_estimate, nkf_y_estimate, nkf_keywords);
    matplotlibcpp::plot(ukf_x_estimate, ukf_y_estimate, ukf_keywords);
    matplotlibcpp::plot(ekf_x_estimate, ekf_y_estimate, ekf_keywords);
    matplotlibcpp::named_plot("true", x_true_vec, y_true_vec);
    matplotlibcpp::legend();
    matplotlibcpp::title("Result");
    matplotlibcpp::show();

    return 0;
}
