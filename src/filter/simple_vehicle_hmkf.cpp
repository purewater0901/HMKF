#include "filter/simple_vehicle_hmkf.h"
#include "distribution/three_dimensional_normal_distribution.h"

using namespace SimpleVehicle;

SimpleVehicleHMKF::SimpleVehicleHMKF()
{
    vehicle_model_ = SimpleVehicleModel();
}

SimpleVehicleModel::HighOrderMoments SimpleVehicleHMKF::predict(const StateInfo& state_info,
                                                                const Eigen::Vector2d & control_inputs,
                                                                const double dt,
                                                                const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    // Step1. Approximate to Gaussian Distribution
    const auto state_mean = state_info.mean;
    const auto state_cov = state_info.covariance;
    ThreeDimensionalNormalDistribution dist(state_info.mean, state_info.covariance);

    // Step2. State Moment
    SimpleVehicleModel::StateMoments moment;
    moment.xPow1 = dist.calc_moment(STATE::IDX::X, 1); // x
    moment.yPow1 = dist.calc_moment(STATE::IDX::Y, 1); // y
    moment.cPow1 = dist.calc_cos_moment(STATE::IDX::YAW, 1); // cos(yaw)
    moment.sPow1 = dist.calc_sin_moment(STATE::IDX::YAW, 1); // sin(yaw)
    moment.yawPow1 = dist.calc_moment(STATE::IDX::YAW, 1); // yaw
    moment.xPow2 = dist.calc_moment(STATE::IDX::X, 2); // x^2
    moment.yPow2 = dist.calc_moment(STATE::IDX::Y, 2); // y^2
    moment.cPow2 = dist.calc_cos_moment(STATE::IDX::YAW, 2); // cos(yaw)^2
    moment.sPow2 = dist.calc_sin_moment(STATE::IDX::YAW, 2); // sin(yaw)^2
    moment.yawPow2 = dist.calc_moment(STATE::IDX::YAW, 2); // yaw^2
    moment.xPow1_yPow1 = dist.calc_cross_second_moment(STATE::IDX::X, STATE::IDX::Y); // xy
    moment.cPow1_xPow1 = dist.calc_x_cos_z_moment(STATE::IDX::X, STATE::IDX::YAW); // x*cos(yaw)
    moment.sPow1_xPow1 = dist.calc_x_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW); // x*sin(yaw)
    moment.cPow1_yPow1 = dist.calc_x_cos_z_moment(STATE::IDX::Y, STATE::IDX::YAW); // y*cos(yaw)
    moment.sPow1_yPow1 = dist.calc_x_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW); // y*sin(yaw)
    moment.cPow1_sPow1 = dist.calc_cos_sin_moment(STATE::IDX::YAW, 1, 1); // cos(yaw)*sin(yaw)
    moment.xPow1_yawPow1 = dist.calc_cross_second_moment(STATE::IDX::X, STATE::IDX::YAW); // x*yaw
    moment.yPow1_yawPow1 = dist.calc_cross_second_moment(STATE::IDX::Y, STATE::IDX::YAW); // y*yaw
    moment.cPow1_yawPow1 = dist.calc_x_cos_x_moment(STATE::IDX::YAW, 1, 1); // yaw*cos(yaw)
    moment.sPow1_yawPow1 = dist.calc_x_sin_x_moment(STATE::IDX::YAW, 1, 1); // yaw*sin(yaw)

    // Step3. Control Input
    SimpleVehicleModel::Controls controls;
    controls.v = control_inputs(INPUT::IDX::V);
    controls.u = control_inputs(INPUT::IDX::U);
    controls.cu = std::cos(controls.u);
    controls.su = std::sin(controls.u);

    // Step4. System Noise
    const auto wv_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WV);
    const auto wu_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WU);
    SimpleVehicleModel::SystemNoiseMoments system_noise_moments;
    system_noise_moments.wvPow1 = wv_dist_ptr->calc_moment(1);
    system_noise_moments.wvPow2 = wv_dist_ptr->calc_moment(2);
    system_noise_moments.wuPow1 = wu_dist_ptr->calc_moment(1);
    system_noise_moments.wuPow2 = wu_dist_ptr->calc_moment(2);
    system_noise_moments.cwuPow1 = wu_dist_ptr->calc_cos_moment(1);
    system_noise_moments.swuPow1 = wu_dist_ptr->calc_sin_moment(1);
    system_noise_moments.swuPow2 = wu_dist_ptr->calc_sin_moment(2);
    system_noise_moments.cwuPow2 = wu_dist_ptr->calc_cos_moment(2);
    system_noise_moments.cwuPow1_swuPow1 = wu_dist_ptr->calc_cos_sin_moment(1, 1);

    // propagate moments
    const auto predicted_moment = vehicle_model_.propagateStateMoments(moment, system_noise_moments, controls);

    SimpleVehicleModel::HighOrderMoments result;
    result.xPow1 = predicted_moment.xPow1;
    result.yPow1 = predicted_moment.yPow1;
    result.yawPow1 = predicted_moment.yawPow1;
    result.cPow1 = predicted_moment.cPow1;
    result.sPow1 = predicted_moment.sPow1;

    result.xPow2 = predicted_moment.xPow2;
    result.yPow2 = predicted_moment.yPow2;
    result.yawPow2 = predicted_moment.yawPow2;
    result.cPow2 = predicted_moment.cPow2;
    result.sPow2 = predicted_moment.sPow2;
    result.xPow1_yPow1 = predicted_moment.xPow1_yPow1;
    result.xPow1_yawPow1 = predicted_moment.xPow1_yawPow1;
    result.yPow1_yawPow1 = predicted_moment.yPow1_yawPow1;
    result.xPow1_cPow1 = predicted_moment.cPow1_xPow1;
    result.yPow1_cPow1 = predicted_moment.cPow1_yPow1;
    result.xPow1_sPow1 = predicted_moment.sPow1_xPow1;
    result.yPow1_sPow1 = predicted_moment.sPow1_yPow1;
    result.cPow1_sPow1 = predicted_moment.cPow1_sPow1;
    result.yawPow1_cPow1 = predicted_moment.cPow1_yawPow1;
    result.yawPow1_sPow1 = predicted_moment.sPow1_yawPow1;

    result.xPow1_cPow2; // = predicted_moment;
    result.yPow1_cPow2;
    result.xPow1_sPow2;
    result.yPow1_sPow2;
    result.xPow2_cPow1;
    result.xPow2_sPow1;
    result.yPow2_cPow1;
    result.yPow2_sPow1;
    result.xPow1_yPow1_cPow1;
    result.xPow1_yPow1_sPow1;
    result.xPow1_cPow1_sPow1;
    result.yPow1_cPow1_sPow1;
    result.xPow1_yawPow1_cPow1;
    result.xPow1_yawPow1_sPow1;
    result.yPow1_yawPow1_cPow1;
    result.yPow1_yawPow1_sPow1;

    result.xPow2_cPow2;
    result.yPow2_cPow2;
    result.xPow2_sPow2;
    result.yPow2_sPow2;
    result.xPow1_yPow1_cPow2;
    result.xPow1_yPow1_sPow2;
    result.xPow2_cPow1_sPow1;
    result.yPow2_cPow1_sPow1;
    result.xPow1_yPow1_cPow1_sPow1;

    return result;
}

StateInfo SimpleVehicleHMKF::update(const SimpleVehicleModel::HighOrderMoments & predicted_moments,
                                    const Eigen::VectorXd & observed_values,
                                    const Eigen::Vector2d & landmark,
                                    const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    // Step1. Create Observation Noise
    const auto wr_dist_ptr = noise_map.at(OBSERVATION_NOISE::IDX::WR);
    const auto wa_dist_ptr = noise_map.at(OBSERVATION_NOISE::IDX::WA);
    SimpleVehicleModel::ObservationNoiseMoments observation_noise;
    observation_noise.wrPow1 = wr_dist_ptr->calc_moment(1);
    observation_noise.wrPow2 = wr_dist_ptr->calc_moment(2);
    observation_noise.cwaPow1 = wa_dist_ptr->calc_cos_moment(1);
    observation_noise.swaPow1 = wa_dist_ptr->calc_sin_moment(1);
    observation_noise.cwaPow2 = wa_dist_ptr->calc_cos_moment(2);
    observation_noise.swaPow2 = wa_dist_ptr->calc_sin_moment(2);
    observation_noise.cwaPow1_swaPow1 = wa_dist_ptr->calc_cos_sin_moment(1, 1);

    // Step2. Get Observation Moments
    const auto observation_moments = vehicle_model_.getObservationMoments(predicted_moments, observation_noise, landmark);

    StateInfo observed_info;
    observed_info.mean = Eigen::VectorXd::Zero(2);
    observed_info.covariance = Eigen::MatrixXd::Zero(2, 2);
    observed_info.mean(OBSERVATION::IDX::RCOS) = observation_moments.rcosPow1;
    observed_info.mean(OBSERVATION::IDX::RSIN) = observation_moments.rsinPow1;
    observed_info.covariance(OBSERVATION::IDX::RCOS, OBSERVATION::IDX::RCOS) = observation_moments.rcosPow2 - observation_moments.rcosPow1*observation_moments.rcosPow1;
    observed_info.covariance(OBSERVATION::IDX::RSIN, OBSERVATION::IDX::RSIN) = observation_moments.rsinPow2 - observation_moments.rsinPow1*observation_moments.rsinPow1;
    observed_info.covariance(OBSERVATION::IDX::RCOS, OBSERVATION::IDX::RSIN) = observation_moments.rcosPow1_rsinPow1 - observation_moments.rcosPow1*observation_moments.rsinPow1;
    observed_info.covariance(OBSERVATION::IDX::RSIN, OBSERVATION::IDX::RCOS) = observed_info.covariance(OBSERVATION::IDX::RCOS, OBSERVATION::IDX::RSIN);

    const auto observation_mean = observed_info.mean;
    const auto observation_cov = observed_info.covariance;

    const double& x_land = landmark(0);
    const double& y_land = landmark(1);

    const double& wrPow1 = observation_noise.wrPow1;
    const double& cwaPow1 = observation_noise.cwaPow1;
    const double& swaPow1 = observation_noise.swaPow1;

    const double& xPow1_cPow1 = predicted_moments.xPow1_cPow1;
    const double& xPow1_sPow1 = predicted_moments.xPow1_sPow1;
    const double& yPow1_cPow1 = predicted_moments.yPow1_cPow1;
    const double& yPow1_sPow1 = predicted_moments.yPow1_sPow1;
    const double& yawPow1_cPow1 = predicted_moments.yawPow1_cPow1;
    const double& yawPow1_sPow1 = predicted_moments.yawPow1_sPow1;
    const double& xPow2_cPow1 = predicted_moments.xPow2_cPow1;
    const double& xPow2_sPow1 = predicted_moments.xPow2_sPow1;
    const double& yPow2_cPow1 = predicted_moments.yPow2_cPow1;
    const double& yPow2_sPow1 = predicted_moments.yPow2_sPow1;
    const double& xPow1_yPow1_cPow1 = predicted_moments.xPow1_yPow1_cPow1;
    const double& xPow1_yPow1_sPow1 = predicted_moments.xPow1_yPow1_sPow1;
    const double& xPow1_yawPow1_cPow1 = predicted_moments.xPow1_yawPow1_cPow1;
    const double& xPow1_yawPow1_sPow1 = predicted_moments.xPow1_yawPow1_sPow1;
    const double& yPow1_yawPow1_cPow1 = predicted_moments.yPow1_yawPow1_cPow1;
    const double& yPow1_yawPow1_sPow1 = predicted_moments.yPow1_yawPow1_sPow1;

    const double xPow1_caPow1 = x_land * xPow1_cPow1 - xPow2_cPow1 + y_land * xPow1_sPow1 - xPow1_yPow1_sPow1;
    const double xPow1_saPow1 = y_land * xPow1_cPow1 - xPow1_yPow1_cPow1 - x_land * xPow1_sPow1 + xPow2_sPow1;
    const double yPow1_caPow1 =  x_land * yPow1_cPow1 - xPow1_yPow1_cPow1 + y_land * yPow1_sPow1 - yPow2_sPow1;
    const double yPow1_saPow1 =  y_land * yPow1_cPow1 - yPow2_cPow1 - x_land * yPow1_sPow1 + xPow1_yPow1_sPow1;
    const double yawPow1_caPow1 = x_land * yawPow1_cPow1 - xPow1_yawPow1_cPow1 + y_land * yawPow1_sPow1 - yPow1_yawPow1_sPow1;
    const double yawPow1_saPow1 = y_land * yawPow1_cPow1 - yPow1_yawPow1_cPow1 - x_land * yawPow1_sPow1 + xPow1_yawPow1_sPow1;

    // Predicted Values
    Eigen::VectorXd predicted_mean = Eigen::VectorXd::Zero(3);
    predicted_mean(STATE::IDX::X) = predicted_moments.xPow1;
    predicted_mean(STATE::IDX::Y) = predicted_moments.yPow1;
    predicted_mean(STATE::IDX::YAW) = predicted_moments.yawPow1;
    Eigen::MatrixXd predicted_cov = Eigen::MatrixXd::Zero(3, 3);
    predicted_cov(STATE::IDX::X, STATE::IDX::X) = predicted_moments.xPow2 - predicted_moments.xPow1 * predicted_moments.xPow1;
    predicted_cov(STATE::IDX::Y, STATE::IDX::Y) = predicted_moments.yPow2 - predicted_moments.yPow1 * predicted_moments.yPow1;
    predicted_cov(STATE::IDX::YAW, STATE::IDX::YAW) = predicted_moments.yawPow2 - predicted_moments.yawPow1 * predicted_moments.yawPow1;
    predicted_cov(STATE::IDX::X, STATE::IDX::Y) = predicted_moments.xPow1_yPow1 - predicted_moments.xPow1 * predicted_moments.yPow1;
    predicted_cov(STATE::IDX::X, STATE::IDX::YAW) = predicted_moments.xPow1_yawPow1 - predicted_moments.xPow1 * predicted_moments.yawPow1;
    predicted_cov(STATE::IDX::Y, STATE::IDX::YAW) = predicted_moments.yPow1_yawPow1 - predicted_moments.yPow1 * predicted_moments.yawPow1;
    predicted_cov(STATE::IDX::Y, STATE::IDX::X) = predicted_cov(STATE::IDX::X, STATE::IDX::Y);
    predicted_cov(STATE::IDX::YAW, STATE::IDX::X) = predicted_cov(STATE::IDX::X, STATE::IDX::YAW);
    predicted_cov(STATE::IDX::YAW, STATE::IDX::Y) = predicted_cov(STATE::IDX::Y, STATE::IDX::YAW);

    Eigen::MatrixXd state_observation_cov(3, 2); // sigma = E[XY^T] - E[X]E[Y]^T
    state_observation_cov(STATE::IDX::X, OBSERVATION::IDX::RCOS)
            = wrPow1 * cwaPow1 * xPow1_caPow1 - wrPow1 * swaPow1 * xPow1_saPow1
              - predicted_mean(STATE::IDX::X) * observation_mean(OBSERVATION::IDX::RCOS);
    state_observation_cov(STATE::IDX::X, OBSERVATION::IDX::RSIN)
            = wrPow1 * cwaPow1 * xPow1_saPow1 + wrPow1 * swaPow1 * xPow1_caPow1
              - predicted_mean(STATE::IDX::X) * observation_mean(OBSERVATION::IDX::RSIN); // x_p * yaw

    state_observation_cov(STATE::IDX::Y, OBSERVATION::IDX::RCOS)
            =  wrPow1 * cwaPow1 * yPow1_caPow1 - wrPow1 * swaPow1 * yPow1_saPow1
               - predicted_mean(STATE::IDX::Y) * observation_mean(OBSERVATION::IDX::RCOS); // yp * (xp^2 + yp^2)
    state_observation_cov(STATE::IDX::Y, OBSERVATION::IDX::RSIN)
            =  wrPow1 * cwaPow1 * yPow1_saPow1 + wrPow1 * swaPow1 * yPow1_caPow1
               - predicted_mean(STATE::IDX::Y) * observation_mean(OBSERVATION::IDX::RSIN); // y_p * yaw

    state_observation_cov(STATE::IDX::YAW, OBSERVATION::IDX::RCOS)
            =   wrPow1 * cwaPow1 * yawPow1_caPow1 - wrPow1 * swaPow1 * yawPow1_saPow1
                - predicted_mean(STATE::IDX::YAW) * observation_mean(OBSERVATION::IDX::RCOS);
    state_observation_cov(STATE::IDX::YAW, OBSERVATION::IDX::RSIN)
            =    wrPow1 * cwaPow1 * yawPow1_saPow1 + wrPow1 * swaPow1 * yawPow1_caPow1
                 - predicted_mean(STATE::IDX::YAW) * observation_mean(OBSERVATION::IDX::RSIN);

    // Kalman Gain
    const auto K = state_observation_cov * observation_cov.inverse();

    StateInfo updated_info;
    updated_info.mean = predicted_mean + K * (observed_values - observation_mean);
    updated_info.covariance = predicted_cov - K*observation_cov*K.transpose();
}
