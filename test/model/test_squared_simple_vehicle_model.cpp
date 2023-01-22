#include <iostream>
#include <random>
#include <vector>
#include <Eigen/Eigen>
#include <map>
#include <memory>
#include <gtest/gtest.h>

#include "model/simple_vehicle_squared_model.h"
#include "distribution/uniform_distribution.h"
#include "distribution/normal_distribution.h"
#include "distribution/four_dimensional_normal_distribution.h"

using namespace SimpleVehicleSquared;

TEST(SimpleVehicleSquaredModel, getObservationMoments)
{
    const double x_land = 3.7629;
    const double y_land = -2.03092;
    const double epsilon = 0.001;
    SimpleVehicleSquaredModel model;

    const double x_mean = 3.0045114484524555;
    const double y_mean = 1.5082653269460484;
    const double yaw_mean = 0.964168029865345;
    const double x_cov = 0.10000238971645992;
    const double y_cov = 0.10015079232158008;
    const double yaw_cov = 0.10870760813388891;
    const double xy_cov = 0.0009786270528993057;
    const double xyaw_cov = -0.000979207100381796;
    const double yyaw_cov = 0.00346968320345753;
    const Eigen::Vector3d mean = {x_mean, y_mean, yaw_mean};
    Eigen::Matrix3d cov;
    cov <<x_cov, xy_cov, xyaw_cov,
          xy_cov, y_cov, yyaw_cov,
          xyaw_cov, yyaw_cov, yaw_cov;

    ThreeDimensionalNormalDistribution dist(mean, cov);

    StateInfo state_info;
    state_info.mean = Eigen::VectorXd::Zero(3);
    state_info.mean(0) = x_mean;
    state_info.mean(1) = y_mean;
    state_info.mean(2) = yaw_mean;
    state_info.covariance = cov;

    // Step2. Create Observation Noise
    const double wr_mean = 1.0;
    const double wr_cov = std::pow(0.09, 2);
    const double wa_mean = 0.0;
    const double wa_cov = std::pow(M_PI/100, 2);
    auto wr_dist = NormalDistribution(wr_mean, wr_cov);
    auto wa_dist = NormalDistribution(wa_mean, wa_cov);
    std::map<int, std::shared_ptr<BaseDistribution>> observation_noise_map = {
        {MEASUREMENT_NOISE::IDX::WR , std::make_shared<NormalDistribution>(wr_mean, wr_cov)},
        {MEASUREMENT_NOISE::IDX::WA , std::make_shared<NormalDistribution>(wa_mean, wa_cov)}};

    // Step3. Get Observation Moments
    const auto observation_moments = model.getMeasurementMoments(state_info, observation_noise_map, {x_land, y_land});

    std::cout << "E[rcos^2]: " << observation_moments.mean(0) << std::endl;
    std::cout << "E[rsin^2]: " << observation_moments.mean(1) << std::endl;
    std::cout << "V[rcos^2]: " << observation_moments.covariance(0, 0) << std::endl;
    std::cout << "V[rsin^2]: " << observation_moments.covariance(1, 1) << std::endl;
    std::cout << "V[rcos^2sin^2]: " << observation_moments.covariance(0, 1) << std::endl;

    // Step4. getStateMeasurementMatrix
    const auto state_mea_matrix = model.getStateMeasurementMatrix(state_info, observation_moments, observation_noise_map, {x_land, y_land});
    std::cout << state_mea_matrix << std::endl;
}