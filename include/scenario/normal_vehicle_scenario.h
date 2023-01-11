#ifndef HMKF_NORMAL_VEHICLE_SCENARIO_H
#define HMKF_NORMAL_VEHICLE_SCENARIO_H

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <memory>
#include <Eigen/Eigen>

#include "model/normal_vehicle_model.h"
#include "distribution/normal_distribution.h"
#include "distribution/uniform_distribution.h"
#include "distribution/exponential_distribution.h"
#include "distribution/beta_distribution.h"
#include "utilities.h"

#include <boost/function.hpp>
#include <boost/bind.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/beta_distribution.hpp>

using namespace NormalVehicle;

struct NormalVehicleGaussianScenario
{
    NormalVehicleGaussianScenario() : N(10000), dt(0.1), filename_("/normal_vehicle_gaussian.csv")
    {
        // Position normal distribution
        const double x_mean = 0.0;
        const double x_cov = 0.1*0.1;
        const double y_mean = 0.0;
        const double y_cov = 0.1*0.1;
        const double yaw_mean = M_PI/4.0;
        const double yaw_cov = M_PI/20 * M_PI/20;

        ini_mean_ = Eigen::VectorXd::Zero(3);
        ini_cov_ = Eigen::MatrixXd::Zero(3, 3);
        ini_mean_(0) = x_mean;
        ini_mean_(1) = y_mean;
        ini_mean_(2) = yaw_mean;
        ini_cov_(0, 0) = x_cov;
        ini_cov_(1, 1) = y_cov;
        ini_cov_(2, 2) = yaw_cov;

        // Input
        v_input_ = Eigen::VectorXd::Constant(N, 0.2);
        u_input_ = Eigen::VectorXd::Constant(N, 0.002);

        // System Noise
        const double mean_wx = 0.0;
        const double cov_wx = std::pow(0.01, 2);
        const double mean_wy = 0.0;
        const double cov_wy = std::pow(0.01, 2);
        const double mean_wyaw = 0.0;
        const double cov_wyaw = std::pow(M_PI/100, 2);
        system_noise_map_ = {
                {SYSTEM_NOISE::IDX::WX, std::make_shared<NormalDistribution>(mean_wx, cov_wx)},
                {SYSTEM_NOISE::IDX::WY, std::make_shared<NormalDistribution>(mean_wy, cov_wy)},
                {SYSTEM_NOISE::IDX::WYAW, std::make_shared<NormalDistribution>(mean_wyaw, cov_wyaw)}};

        // Observation Noise
        // Observation Noise
        const double mean_meas_noise_r = 100.0;
        const double cov_meas_noise_r = std::pow(10.5, 2);
        const double mean_meas_noise_yaw = 0.0;
        const double cov_meas_noise_yaw = std::pow(M_PI/10.0, 2);

        observation_noise_map_ = {
                {MEASUREMENT_NOISE::IDX::WR, std::make_shared<NormalDistribution>(mean_meas_noise_r, cov_meas_noise_r)},
                {MEASUREMENT_NOISE::IDX::WYAW, std::make_shared<NormalDistribution>(mean_meas_noise_yaw, cov_meas_noise_yaw)}};

        // Random Variable Generator
        wx_dist_ = std::normal_distribution<double>(mean_wx, std::sqrt(cov_wx));
        wy_dist_ = std::normal_distribution<double>(mean_wy, std::sqrt(cov_wy));
        wyaw_dist_ = std::normal_distribution<double>(mean_wyaw, std::sqrt(cov_wyaw));
        mr_dist_ = std::normal_distribution<double>(mean_meas_noise_r, std::sqrt(cov_meas_noise_r));
        myaw_dist_ = std::normal_distribution<double>(mean_meas_noise_yaw, std::sqrt(cov_meas_noise_yaw));
    }

    // Initial Setting
    const size_t N{10000};
    const double dt{0.1};
    const std::string filename_{"/normal_vehicle_gaussian.csv"};

    // Initial Distribution
    Eigen::VectorXd ini_mean_;
    Eigen::MatrixXd ini_cov_;

    //Input
    Eigen::VectorXd v_input_;
    Eigen::VectorXd u_input_;

    // Noise
    std::map<int, std::shared_ptr<BaseDistribution>> system_noise_map_;
    std::map<int, std::shared_ptr<BaseDistribution>> observation_noise_map_;

    std::normal_distribution<double> wx_dist_;
    std::normal_distribution<double> wy_dist_;
    std::normal_distribution<double> wyaw_dist_;
    std::normal_distribution<double> mr_dist_;
    std::normal_distribution<double> myaw_dist_;
};

struct NormalVehicleNonGaussianScenario
{
    NormalVehicleNonGaussianScenario() : N(20000), dt(0.1), filename_("/normal_vehicle_gaussian.csv")
    {
        // Position normal distribution
        const double x_mean = 0.0;
        const double x_cov = 0.1*0.1;
        const double y_mean = 0.0;
        const double y_cov = 0.1*0.1;
        const double yaw_mean = M_PI/4.0;
        const double yaw_cov = M_PI/20 * M_PI/20;

        ini_mean_ = Eigen::VectorXd::Zero(3);
        ini_cov_ = Eigen::MatrixXd::Zero(3, 3);
        ini_mean_(0) = x_mean;
        ini_mean_(1) = y_mean;
        ini_mean_(2) = yaw_mean;
        ini_cov_(0, 0) = x_cov;
        ini_cov_(1, 1) = y_cov;
        ini_cov_(2, 2) = yaw_cov;

        // Input
        v_input_ = Eigen::VectorXd::Constant(N, 0.2);
        u_input_ = Eigen::VectorXd::Constant(N, 0.002);

        // System Noise
        const double mean_wx = 0.0;
        const double cov_wx = std::pow(0.05, 2);
        const double mean_wy = 0.0;
        const double cov_wy = std::pow(0.01, 2);
        const double mean_wyaw = 0.0;
        const double cov_wyaw = std::pow(M_PI/100, 2);
        system_noise_map_ = {
                {SYSTEM_NOISE::IDX::WX, std::make_shared<NormalDistribution>(mean_wx, cov_wx)},
                {SYSTEM_NOISE::IDX::WY, std::make_shared<NormalDistribution>(mean_wy, cov_wy)},
                {SYSTEM_NOISE::IDX::WYAW, std::make_shared<NormalDistribution>(mean_wyaw, cov_wyaw)}};

        // Observation Noise
        const double lower_mean_noise_r = 100.0;
        const double upper_mean_noise_r = 0.0;
        const double meas_noise_alpha = 5.0;
        const double meas_noise_beta = 1.0;

        observation_noise_map_ = {
                {MEASUREMENT_NOISE::IDX::WR, std::make_shared<UniformDistribution>(lower_mean_noise_r, upper_mean_noise_r)},
                {MEASUREMENT_NOISE::IDX::WYAW, std::make_shared<BetaDistribution>(meas_noise_alpha, meas_noise_beta)}};

        // Random Variable Generator
        wx_dist_ = std::normal_distribution<double>(mean_wx, std::sqrt(cov_wx));
        wy_dist_ = std::normal_distribution<double>(mean_wy, std::sqrt(cov_wy));
        wyaw_dist_ = std::normal_distribution<double>(mean_wyaw, std::sqrt(cov_wyaw));
        mr_dist_ = std::uniform_real_distribution<double>(lower_mean_noise_r, upper_mean_noise_r);
        size_t seed = 1234567890;
        boost::random::mt19937 engine(seed);
        myaw_dist_ = boost::bind(boost::random::beta_distribution<>(meas_noise_alpha, meas_noise_beta), engine);
    }

    // Initial Setting
    const size_t N{10000};
    const double dt{0.1};
    const std::string filename_{"/normal_vehicle_gaussian.csv"};

    // Initial Distribution
    Eigen::VectorXd ini_mean_;
    Eigen::MatrixXd ini_cov_;

    //Input
    Eigen::VectorXd v_input_;
    Eigen::VectorXd u_input_;

    // Noise
    std::map<int, std::shared_ptr<BaseDistribution>> system_noise_map_;
    std::map<int, std::shared_ptr<BaseDistribution>> observation_noise_map_;

    std::normal_distribution<double> wx_dist_;
    std::normal_distribution<double> wy_dist_;
    std::normal_distribution<double> wyaw_dist_;
    std::uniform_real_distribution<double> mr_dist_;
    boost::function<double()> myaw_dist_;
};

#endif //HMKF_NORMAL_VEHICLE_SCENARIO_H
