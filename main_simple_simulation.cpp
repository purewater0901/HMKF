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
#include "filter/simple_vehicle_hmkf.h"
#include "filter/mkf.h"
#include "filter/ukf.h"
#include "filter/ekf.h"
#include "model/simple_vehicle_model.h"

using namespace SimpleVehicle;

int main()
{
    // simulation setting value
    const size_t montecarlo_num = 30;
    const double land_x = 0.0;
    const double land_y = 0.0;
    const Eigen::Vector2d landmark = {land_x, land_y};
    const size_t N = 400;
    const double dt = 0.2;

    // vehicle model
    std::shared_ptr<BaseModel> vehicle_model = std::make_shared<SimpleVehicleModel>(3, 2, 2, 2);

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

    // input
    const double v = 1.0;
    const double u = 0.01;
    Eigen::VectorXd control_input = Eigen::VectorXd::Zero(2);
    control_input(0) = v * dt;
    control_input(1) = u * dt;

    // system noise map
    const double mean_wv = 5.0*dt;
    const double cov_wv = std::pow(5.0*dt, 2);
    const double mean_wu = 0.0*dt;
    const double cov_wu = std::pow(M_PI/10*dt, 2);
    std::map<int, std::shared_ptr<BaseDistribution>> system_noise_map = {
            //{SYSTEM_NOISE::IDX::WV, std::make_shared<NormalDistribution>(mean_wv, cov_wv)},
            {SYSTEM_NOISE::IDX::WV, std::make_shared<UniformDistribution>(0.0*dt, 0.5*dt)},
            //{SYSTEM_NOISE::IDX::WU, std::make_shared<NormalDistribution>(mean_wu, cov_wu)}};
            {SYSTEM_NOISE::IDX::WU, std::make_shared<UniformDistribution>(-M_PI/10.0*dt, M_PI/10.0*dt)}};
//std::normal_distribution<double> wv_dist(mean_wv, std::sqrt(cov_wv));
    //std::extreme_value_distribution<double> wv_dist(3.0*dt, 4.0*dt);
    std::uniform_real_distribution<double> wv_dist(0.0*dt, 0.9*dt);
    //std::normal_distribution<double> wu_dist(mean_wu, std::sqrt(cov_wu));
    std::uniform_real_distribution<double> wu_dist(-M_PI/8.0*dt, M_PI/8.0*dt);

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
    SimpleVehicleHMKF hmkf(vehicle_model);

    std::vector<double> average_ekf_xy_errors;
    std::vector<double> average_ekf_yaw_errors;
    std::vector<double> average_ukf_xy_errors;
    std::vector<double> average_ukf_yaw_errors;
    std::vector<double> average_mkf_xy_errors;
    std::vector<double> average_mkf_yaw_errors;
    std::vector<double> average_hmkf_xy_errors;
    std::vector<double> average_hmkf_yaw_errors;
    // data
    std::vector<std::vector<double>> times(montecarlo_num, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> ekf_xy_errors(montecarlo_num, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> ekf_yaw_errors(montecarlo_num, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> ukf_xy_errors(montecarlo_num, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> ukf_yaw_errors(montecarlo_num, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> mkf_xy_errors(montecarlo_num, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> mkf_yaw_errors(montecarlo_num, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> hmkf_xy_errors(montecarlo_num, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> hmkf_yaw_errors(montecarlo_num, std::vector<double>(N, 0.0));

    std::vector<std::vector<double>> x_true_vec(montecarlo_num, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> y_true_vec(montecarlo_num, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> yaw_true_vec(montecarlo_num, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> hmkf_x_estimate(montecarlo_num, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> hmkf_y_estimate(montecarlo_num, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> hmkf_yaw_estimate(montecarlo_num, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> mkf_x_estimate(montecarlo_num, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> mkf_y_estimate(montecarlo_num, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> mkf_yaw_estimate(montecarlo_num, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> ekf_x_estimate(montecarlo_num, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> ekf_y_estimate(montecarlo_num, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> ekf_yaw_estimate(montecarlo_num, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> ukf_x_estimate(montecarlo_num, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> ukf_y_estimate(montecarlo_num, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> ukf_yaw_estimate(montecarlo_num, std::vector<double>(N, 0.0));

    for(size_t i=0; i<montecarlo_num; ++i) {
        std::cout << "montecarlo: " << i << std::endl;
        // initial true value
        Eigen::VectorXd true_state = ini_state.mean;
        auto ekf_state_info = ini_state;
        auto ukf_state_info = ini_state;
        auto mkf_state_info = ini_state;
        auto hmkf_state_info = ini_state;
        std::shared_ptr<SimpleVehicleModel::HighOrderMoments> hmkf_predicted_moments = nullptr;

        for(size_t iter=1; iter<N; ++iter) {
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
            hmkf_state_info = hmkf.update(*hmkf_predicted_moments, y, {landmark(0), landmark(1)}, measurement_noise_map);

            // insert true values
            times.at(i).at(iter) = times.at(i).at(iter-1) + dt;
            x_true_vec.at(i).at(iter) = true_state(0);
            y_true_vec.at(i).at(iter) = true_state(1);
            yaw_true_vec.at(i).at(iter) = true_state(2);

            // compute errors
            // EKF
            {
                const double dx = true_state(0) - ekf_state_info.mean(0);
                const double dy = true_state(1) - ekf_state_info.mean(1);
                const double xy_error = std::hypot(dx, dy);
                const double dyaw = normalizeRadian(true_state(2) - ekf_state_info.mean(2));

                ekf_xy_errors.at(i).at(iter) = xy_error;
                ekf_yaw_errors.at(i).at(iter) = std::fabs(dyaw);
                ekf_x_estimate.at(i).at(iter) = ekf_state_info.mean(0);
                ekf_y_estimate.at(i).at(iter) = ekf_state_info.mean(1);
                ekf_yaw_estimate.at(i).at(iter) = ekf_state_info.mean(2);
            }

            // UKF
            {
                const double dx = true_state(0) - ukf_state_info.mean(0);
                const double dy = true_state(1) - ukf_state_info.mean(1);
                const double xy_error = std::hypot(dx, dy);
                const double dyaw = normalizeRadian(true_state(2) - ukf_state_info.mean(2));

                ukf_xy_errors.at(i).at(iter) = xy_error;
                ukf_yaw_errors.at(i).at(iter) = std::fabs(dyaw);
                ukf_x_estimate.at(i).at(iter) = ukf_state_info.mean(0);
                ukf_y_estimate.at(i).at(iter) = ukf_state_info.mean(1);
                ukf_yaw_estimate.at(i).at(iter) = ukf_state_info.mean(2);
            }

            // MKF
            {
                const double dx = true_state(0) - mkf_state_info.mean(0);
                const double dy = true_state(1) - mkf_state_info.mean(1);
                const double xy_error = std::hypot(dx, dy);
                const double dyaw = normalizeRadian(true_state(2) - mkf_state_info.mean(2));

                mkf_xy_errors.at(i).at(iter) = xy_error;
                mkf_yaw_errors.at(i).at(iter) = std::fabs(dyaw);
                mkf_x_estimate.at(i).at(iter) = mkf_state_info.mean(0);
                mkf_y_estimate.at(i).at(iter) = mkf_state_info.mean(1);
                mkf_yaw_estimate.at(i).at(iter) = mkf_state_info.mean(2);
            }

            // HMKF
            {
                const double dx = true_state(0) - hmkf_state_info.mean(0);
                const double dy = true_state(1) - hmkf_state_info.mean(1);
                const double xy_error = std::hypot(dx, dy);
                const double dyaw = normalizeRadian(true_state(2) - hmkf_state_info.mean(2));

                hmkf_xy_errors.at(i).at(iter) = xy_error;
                hmkf_yaw_errors.at(i).at(iter) = std::fabs(dyaw);
                hmkf_x_estimate.at(i).at(iter) = hmkf_state_info.mean(0);
                hmkf_y_estimate.at(i).at(iter) = hmkf_state_info.mean(1);
                hmkf_yaw_estimate.at(i).at(iter) = hmkf_state_info.mean(2);
            }
        }

        const double mean_hmkf_xy_error = std::accumulate(hmkf_xy_errors.at(i).begin(), hmkf_xy_errors.at(i).end(), 0.0) / N;
        const double mean_mkf_xy_error = std::accumulate(mkf_xy_errors.at(i).begin(), mkf_xy_errors.at(i).end(), 0.0) / N;
        const double mean_ekf_xy_error = std::accumulate(ekf_xy_errors.at(i).begin(), ekf_xy_errors.at(i).end(), 0.0) / N;
        const double mean_ukf_xy_error = std::accumulate(ukf_xy_errors.at(i).begin(), ukf_xy_errors.at(i).end(), 0.0) / N;
        const double mean_hmkf_yaw_error = std::accumulate(hmkf_yaw_errors.at(i).begin(), hmkf_yaw_errors.at(i).end(), 0.0) / N;
        const double mean_mkf_yaw_error = std::accumulate(mkf_yaw_errors.at(i).begin(), mkf_yaw_errors.at(i).end(), 0.0) / N;
        const double mean_ekf_yaw_error = std::accumulate(ekf_yaw_errors.at(i).begin(), ekf_yaw_errors.at(i).end(), 0.0) / N;
        const double mean_ukf_yaw_error = std::accumulate(ukf_yaw_errors.at(i).begin(), ukf_yaw_errors.at(i).end(), 0.0) / N;
        average_ekf_xy_errors.push_back(mean_ekf_xy_error);
        average_ekf_yaw_errors.push_back(mean_ekf_yaw_error);
        average_ukf_xy_errors.push_back(mean_ukf_xy_error);
        average_ukf_yaw_errors.push_back(mean_ukf_yaw_error);
        average_mkf_xy_errors.push_back(mean_mkf_xy_error);
        average_mkf_yaw_errors.push_back(mean_mkf_yaw_error);
        average_hmkf_xy_errors.push_back(mean_hmkf_xy_error);
        average_hmkf_yaw_errors.push_back(mean_hmkf_yaw_error);
        std::cout << "hmkf_xy_error mean: " << mean_hmkf_xy_error<< std::endl;
        std::cout << "nkf_xy_error mean: " << mean_mkf_xy_error<< std::endl;
        std::cout << "ekf_xy_error mean: " << mean_ekf_xy_error<< std::endl;
        std::cout << "ukf_xy_error mean: " << mean_ukf_xy_error<< std::endl;
        std::cout << "hmkf_yaw_error mean: " << mean_hmkf_yaw_error<< std::endl;
        std::cout << "nkf_yaw_error mean: " << mean_mkf_yaw_error<< std::endl;
        std::cout << "ekf_yaw_error mean: " << mean_ekf_yaw_error<< std::endl;
        std::cout << "ukf_yaw_error mean: " << mean_ukf_yaw_error<< std::endl;
    }

    std::cout << "Final hmkf_xy_error mean: "
              << std::accumulate(average_hmkf_xy_errors.begin(), average_hmkf_xy_errors.end(), 0.0) / montecarlo_num << std::endl;
    std::cout << "Final mkf_xy_error mean: "
              << std::accumulate(average_mkf_xy_errors.begin(), average_mkf_xy_errors.end(), 0.0) / montecarlo_num << std::endl;
    std::cout << "Final ekf_xy_error mean: "
              << std::accumulate(average_ekf_xy_errors.begin(), average_ekf_xy_errors.end(), 0.0) / montecarlo_num << std::endl;
    std::cout << "Final ukf_xy_error mean: "
              << std::accumulate(average_ukf_xy_errors.begin(), average_ukf_xy_errors.end(), 0.0) / montecarlo_num << std::endl;
    std::cout << "Final hmkf_yaw_error mean: "
              << std::accumulate(average_hmkf_yaw_errors.begin(), average_hmkf_yaw_errors.end(), 0.0) / montecarlo_num << std::endl;
    std::cout << "Final mkf_yaw_error mean: "
              << std::accumulate(average_mkf_yaw_errors.begin(), average_mkf_yaw_errors.end(), 0.0) / montecarlo_num << std::endl;
    std::cout << "Final ekf_yaw_error mean: "
              << std::accumulate(average_ekf_yaw_errors.begin(), average_ekf_yaw_errors.end(), 0.0) / montecarlo_num << std::endl;
    std::cout << "Final ukf_yaw_error mean: "
              << std::accumulate(average_ukf_yaw_errors.begin(), average_ukf_yaw_errors.end(), 0.0) / montecarlo_num << std::endl;

    matplotlibcpp::figure_size(1500, 900);
    std::map<std::string, std::string> hmkf_keywords;
    std::map<std::string, std::string> mkf_keywords;
    std::map<std::string, std::string> ekf_keywords;
    std::map<std::string, std::string> ukf_keywords;
    hmkf_keywords.insert(std::pair<std::string, std::string>("label", "hmkf error"));
    mkf_keywords.insert(std::pair<std::string, std::string>("label", "mkf error"));
    ekf_keywords.insert(std::pair<std::string, std::string>("label", "ekf error"));
    ukf_keywords.insert(std::pair<std::string, std::string>("label", "ukf error"));
    matplotlibcpp::plot(hmkf_x_estimate.at(0), hmkf_y_estimate.at(0), hmkf_keywords);
    matplotlibcpp::plot(mkf_x_estimate.at(0), mkf_y_estimate.at(0), mkf_keywords);
    matplotlibcpp::plot(ekf_x_estimate.at(0), ekf_y_estimate.at(0), ekf_keywords);
    matplotlibcpp::plot(ukf_x_estimate.at(0), ukf_y_estimate.at(0), ukf_keywords);
    matplotlibcpp::named_plot("true", x_true_vec.at(0), y_true_vec.at(0));
    matplotlibcpp::legend();
    matplotlibcpp::title("Result");
    matplotlibcpp::show();

    return 0;
}