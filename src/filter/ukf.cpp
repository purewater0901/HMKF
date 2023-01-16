#include "filter/ukf.h"

UKF::UKF(const std::shared_ptr<BaseModel>& vehicle_model,
         const double alpha_squared,
         const double beta,
         const double kappa)
: vehicle_model_(vehicle_model) ,
  augmented_size_(vehicle_model_->getAugmentedSize()),
  alpha_squared_(alpha_squared),
  beta_(beta),
  kappa_(kappa),
  lambda_(alpha_squared_*(augmented_size_ + kappa_) - augmented_size_)
{

    Sigma_WM0_ = lambda_/(augmented_size_ + lambda_);
    Sigma_WC0_ = Sigma_WM0_ + (1.0 - alpha_squared_ + beta_);
    Sigma_WMI_ = 1.0 / (2.0 * (augmented_size_ + lambda_));
    Sigma_WCI_ = Sigma_WMI_;
}

StateInfo UKF::predict(const StateInfo& state_info,
                       const Eigen::VectorXd& control_inputs,
                       const double dt,
                       const std::map<int, std::shared_ptr<BaseDistribution>>& system_noise_map,
                       const std::map<int, std::shared_ptr<BaseDistribution>>& measurement_noise_map)
{
    const size_t state_dim = vehicle_model_->state_dim_;
    const size_t system_noise_dim = vehicle_model_->system_noise_dim_;
    const size_t measurement_noise_dim = vehicle_model_->measurement_noise_dim_;
    Eigen::VectorXd augmented_mean = Eigen::VectorXd::Zero(augmented_size_);
    Eigen::MatrixXd augmented_cov = Eigen::MatrixXd::Zero(augmented_size_, augmented_size_);

    // state
    augmented_mean.head(state_dim) = state_info.mean;
    augmented_cov.block(0, 0, state_dim, state_dim) = state_info.covariance;

    // system noise
    for(size_t i=0; i<system_noise_dim; ++i) {
        const auto dist_noise = system_noise_map.at(i);
        const size_t target_idx = i + state_dim;
        augmented_mean(target_idx) = dist_noise->calc_mean();
        augmented_cov(target_idx, target_idx) = dist_noise->calc_variance();
    }

    // measurement noise
    for(size_t i=0; i<measurement_noise_dim; ++i) {
        const auto dist_noise = measurement_noise_map.at(i);
        const size_t target_idx = i + state_dim + system_noise_dim;
        augmented_mean(target_idx) = dist_noise->calc_mean();
        augmented_cov(target_idx, target_idx) = dist_noise->calc_variance();
    }

    assert((augmented_cov*(augmented_size_ + lambda_)).llt().info() == Eigen::Success);
    const Eigen::MatrixXd augmented_cov_squared = (augmented_cov * (augmented_size_ + lambda_)).llt().matrixL();

    Eigen::MatrixXd sigma_points = Eigen::MatrixXd::Zero(augmented_size_, 2*augmented_size_+1);
    Eigen::VectorXd processed_augmented_mean = Eigen::VectorXd::Zero(augmented_size_);
    for(size_t i=0; i<augmented_size_; ++i) {
        sigma_points.col(i) = augmented_mean + augmented_cov_squared.col(i);
        const Eigen::VectorXd& curr_x = sigma_points.col(i).head(state_dim);
        const Eigen::VectorXd& system_noise = sigma_points.col(i).segment(state_dim, system_noise_dim);
        const Eigen::VectorXd processed_state = vehicle_model_->propagate(curr_x, control_inputs, system_noise, dt);

        sigma_points.col(i).head(state_dim) = processed_state;
        processed_augmented_mean += Sigma_WMI_ * sigma_points.col(i);
    }
    for(size_t i=augmented_size_; i<2*augmented_size_; ++i) {
        sigma_points.col(i) = augmented_mean - augmented_cov_squared.col(i-augmented_size_);

        const Eigen::VectorXd& curr_x = sigma_points.col(i).head(state_dim);
        const Eigen::VectorXd& system_noise = sigma_points.col(i).segment(state_dim, system_noise_dim);
        const Eigen::VectorXd processed_state = vehicle_model_->propagate(curr_x, control_inputs, system_noise, dt);
        sigma_points.col(i).head(state_dim) = processed_state;
        processed_augmented_mean += Sigma_WMI_ * sigma_points.col(i);
    }
    {
        sigma_points.col(2*augmented_size_) = augmented_mean;

        const Eigen::VectorXd& curr_x = sigma_points.col(2*augmented_size_).head(state_dim);
        const Eigen::VectorXd& system_noise = sigma_points.col(2*augmented_size_).segment(state_dim, system_noise_dim);
        const Eigen::VectorXd processed_state = vehicle_model_->propagate(curr_x, control_inputs, system_noise, dt);

        sigma_points.col(2*augmented_size_).head(state_dim) = processed_state;
        processed_augmented_mean += Sigma_WM0_ * sigma_points.col(2*augmented_size_);
    }

    Eigen::MatrixXd processed_augmented_cov = Eigen::MatrixXd::Zero(augmented_size_, augmented_size_);
    for(size_t i=0; i<2*augmented_size_; ++i) {
        const Eigen::VectorXd delta_x = sigma_points.col(i) - processed_augmented_mean;
        processed_augmented_cov += Sigma_WCI_ * (delta_x * delta_x.transpose());
    }
    {
        const Eigen::VectorXd delta_x = sigma_points.col(2*augmented_size_) - processed_augmented_mean;
        processed_augmented_cov += Sigma_WC0_ * (delta_x * delta_x.transpose());
    }

    StateInfo result;
    result.mean = processed_augmented_mean.head(state_dim);
    result.covariance = processed_augmented_cov.block(0, 0, state_dim, state_dim);

    return result;
}

StateInfo UKF::update(const StateInfo &state_info,
                      const Eigen::VectorXd& measurement_values,
                      const std::map<int, std::shared_ptr<BaseDistribution>>& system_noise_map,
                      const std::map<int, std::shared_ptr<BaseDistribution>>& measurement_noise_map)
{
    const size_t state_dim = vehicle_model_->state_dim_;
    const size_t system_noise_dim = vehicle_model_->system_noise_dim_;
    const size_t measurement_dim = vehicle_model_->measurement_dim_;
    const size_t measurement_noise_dim = vehicle_model_->measurement_noise_dim_;
    Eigen::VectorXd augmented_mean = Eigen::VectorXd::Zero(augmented_size_);
    Eigen::MatrixXd augmented_cov = Eigen::MatrixXd::Zero(augmented_size_, augmented_size_);

    // state
    augmented_mean.head(state_dim) = state_info.mean;
    augmented_cov.block(0, 0, state_dim, state_dim) = state_info.covariance;

    // system noise
    for(size_t i=0; i<system_noise_dim; ++i) {
        const auto dist_noise = system_noise_map.at(i);
        const size_t target_idx = i + state_dim;
        augmented_mean(target_idx) = dist_noise->calc_mean();
        augmented_cov(target_idx, target_idx) = dist_noise->calc_variance();
    }

    // measurement noise
    for(size_t i=0; i<measurement_noise_dim; ++i) {
        const auto dist_noise = measurement_noise_map.at(i);
        const size_t target_idx = i + state_dim + system_noise_dim;
        augmented_mean(target_idx) = dist_noise->calc_mean();
        augmented_cov(target_idx, target_idx) = dist_noise->calc_variance();
    }

    assert((augmented_cov*(augmented_size_ + lambda_)).llt().info() == Eigen::Success);
    const Eigen::MatrixXd augmented_cov_squared = (augmented_cov * (augmented_size_ + lambda_)).llt().matrixL();

    Eigen::MatrixXd sigma_points = Eigen::MatrixXd::Zero(augmented_size_, 2*augmented_size_+1);
    // Resample Sigma Points
    for(size_t i=0; i<augmented_size_; ++i) {
        sigma_points.col(i) = augmented_mean + augmented_cov_squared.col(i);
    }
    for(size_t i=augmented_size_; i<2*augmented_size_; ++i) {
        sigma_points.col(i) = augmented_mean - augmented_cov_squared.col(i-augmented_size_);
    }
    sigma_points.col(2*augmented_size_) = augmented_mean;

    // Calculate mean y
    Eigen::MatrixXd observed_sigma_points = Eigen::MatrixXd::Zero(measurement_dim, 2*augmented_size_+1);
    Eigen::VectorXd y_mean = Eigen::VectorXd::Zero(measurement_dim);
    for(size_t i=0; i<2*augmented_size_+1; ++i) {
        const Eigen::VectorXd& curr_x = sigma_points.col(i).head(state_dim);
        const Eigen::VectorXd& meas_noise = sigma_points.col(i).segment(state_dim+system_noise_dim, measurement_noise_dim);

        const Eigen::VectorXd y = vehicle_model_->measure(curr_x, meas_noise);
        observed_sigma_points.col(i) = y;
        if(i==2*augmented_size_) {
            y_mean += Sigma_WM0_ * y;
        } else {
            y_mean += Sigma_WMI_ * y;
        }
    }

    Eigen::MatrixXd Pyy = Eigen::MatrixXd::Zero(measurement_dim, measurement_dim);
    for(size_t i=0; i<2*augmented_size_+1; ++i) {
        const Eigen::VectorXd delta_y = observed_sigma_points.col(i) - y_mean;
        if(i==2*augmented_size_) {
            Pyy += Sigma_WC0_ * (delta_y * delta_y.transpose());
        } else {
            Pyy += Sigma_WCI_ * (delta_y * delta_y.transpose());
        }
    }

    Eigen::MatrixXd Pxy = Eigen::MatrixXd::Zero(state_dim, measurement_dim);
    for(size_t i=0; i<2*augmented_size_+1; ++i) {
        const Eigen::VectorXd delta_x = sigma_points.col(i).head(state_dim) - state_info.mean;
        const Eigen::VectorXd delta_y = observed_sigma_points.col(i) - y_mean;
        if(i==2*augmented_size_) {
            Pxy += Sigma_WC0_ * (delta_x * delta_y.transpose());
        } else {
            Pxy += Sigma_WCI_ * (delta_x * delta_y.transpose());
        }
    }

    const Eigen::MatrixXd K = Pxy*Pyy.inverse();

    StateInfo result;
    result.mean = state_info.mean + K*(measurement_values - y_mean);
    result.covariance = state_info.covariance - K * Pyy * K.transpose();

    return result;
}

StateInfo UKF::getMeasurementInfo(const StateInfo& state_info,
                                  const std::map<int, std::shared_ptr<BaseDistribution>>& system_noise_map,
                                  const std::map<int, std::shared_ptr<BaseDistribution>>& measurement_noise_map)
{
    const size_t state_dim = vehicle_model_->state_dim_;
    const size_t system_noise_dim = vehicle_model_->system_noise_dim_;
    const size_t measurement_dim = vehicle_model_->measurement_dim_;
    const size_t measurement_noise_dim = vehicle_model_->measurement_noise_dim_;
    Eigen::VectorXd augmented_mean = Eigen::VectorXd::Zero(augmented_size_);
    Eigen::MatrixXd augmented_cov = Eigen::MatrixXd::Zero(augmented_size_, augmented_size_);

    // state
    augmented_mean.head(state_dim) = state_info.mean;
    augmented_cov.block(0, 0, state_dim, state_dim) = state_info.covariance;

    // system noise
    for(size_t i=0; i<system_noise_dim; ++i) {
        const auto dist_noise = system_noise_map.at(i);
        const size_t target_idx = i + state_dim;
        augmented_mean(target_idx) = dist_noise->calc_mean();
        augmented_cov(target_idx, target_idx) = dist_noise->calc_variance();
    }

    // measurement noise
    for(size_t i=0; i<measurement_noise_dim; ++i) {
        const auto dist_noise = measurement_noise_map.at(i);
        const size_t target_idx = i + state_dim + system_noise_dim;
        augmented_mean(target_idx) = dist_noise->calc_mean();
        augmented_cov(target_idx, target_idx) = dist_noise->calc_variance();
    }

    assert((augmented_cov*(augmented_size_ + lambda_)).llt().info() == Eigen::Success);
    const Eigen::MatrixXd augmented_cov_squared = (augmented_cov * (augmented_size_ + lambda_)).llt().matrixL();

   Eigen::MatrixXd sigma_points = Eigen::MatrixXd::Zero(augmented_size_, 2*augmented_size_+1);
    // Resample Sigma Points
    for(size_t i=0; i<augmented_size_; ++i) {
        sigma_points.col(i) = augmented_mean + augmented_cov_squared.col(i);
    }
    for(size_t i=augmented_size_; i<2*augmented_size_; ++i) {
        sigma_points.col(i) = augmented_mean - augmented_cov_squared.col(i-augmented_size_);
    }
    sigma_points.col(2*augmented_size_) = augmented_mean;

    // Calculate mean y
    Eigen::MatrixXd observed_sigma_points = Eigen::MatrixXd::Zero(measurement_dim, 2*augmented_size_+1);
    Eigen::VectorXd y_mean = Eigen::VectorXd::Zero(measurement_dim);
    for(size_t i=0; i<2*augmented_size_+1; ++i) {
        const Eigen::VectorXd& curr_x = sigma_points.col(i).head(state_dim);
        const Eigen::VectorXd& meas_noise = sigma_points.col(i).segment(state_dim+system_noise_dim, measurement_noise_dim);

        const Eigen::VectorXd y = vehicle_model_->measure(curr_x, meas_noise);
        observed_sigma_points.col(i) = y;
        if(i==2*augmented_size_) {
            y_mean += Sigma_WM0_ * y;
        } else {
            y_mean += Sigma_WMI_ * y;
        }
    }

    Eigen::MatrixXd Pyy = Eigen::MatrixXd::Zero(measurement_dim, measurement_dim);
    for(size_t i=0; i<2*augmented_size_+1; ++i) {
        const Eigen::VectorXd delta_y = observed_sigma_points.col(i) - y_mean;
        if(i==2*augmented_size_) {
            Pyy += Sigma_WC0_ * (delta_y * delta_y.transpose());
        } else {
            Pyy += Sigma_WCI_ * (delta_y * delta_y.transpose());
        }
    }

    StateInfo measurement_info;
    measurement_info.mean = y_mean;
    measurement_info.covariance = Pyy;
    return measurement_info;
}

StateInfo UKF::update(const StateInfo& state_info,
                      const Eigen::VectorXd& measurement_values,
                      const std::map<int, std::shared_ptr<BaseDistribution>>& system_noise_map,
                      const std::map<int, std::shared_ptr<BaseDistribution>>& measurement_noise_map,
                      const Eigen::Vector2d& landmark)
{
    const size_t state_dim = vehicle_model_->state_dim_;
    const size_t system_noise_dim = vehicle_model_->system_noise_dim_;
    const size_t measurement_dim = vehicle_model_->measurement_dim_;
    const size_t measurement_noise_dim = vehicle_model_->measurement_noise_dim_;
    Eigen::VectorXd augmented_mean = Eigen::VectorXd::Zero(augmented_size_);
    Eigen::MatrixXd augmented_cov = Eigen::MatrixXd::Zero(augmented_size_, augmented_size_);

    // state
    augmented_mean.head(state_dim) = state_info.mean;
    augmented_cov.block(0, 0, state_dim, state_dim) = state_info.covariance;

    // system noise
    for(size_t i=0; i<system_noise_dim; ++i) {
        const auto dist_noise = system_noise_map.at(i);
        const size_t target_idx = i + state_dim;
        augmented_mean(target_idx) = dist_noise->calc_mean();
        augmented_cov(target_idx, target_idx) = dist_noise->calc_variance();
    }

    // measurement noise
    for(size_t i=0; i<measurement_noise_dim; ++i) {
        const auto dist_noise = measurement_noise_map.at(i);
        const size_t target_idx = i + state_dim + system_noise_dim;
        augmented_mean(target_idx) = dist_noise->calc_mean();
        augmented_cov(target_idx, target_idx) = dist_noise->calc_variance();
    }

    assert((augmented_cov*(augmented_size_ + lambda_)).llt().info() == Eigen::Success);
    const Eigen::MatrixXd augmented_cov_squared = (augmented_cov * (augmented_size_ + lambda_)).llt().matrixL();

    Eigen::MatrixXd sigma_points = Eigen::MatrixXd::Zero(augmented_size_, 2*augmented_size_+1);
    // Resample Sigma Points
    for(size_t i=0; i<augmented_size_; ++i) {
        sigma_points.col(i) = augmented_mean + augmented_cov_squared.col(i);
    }
    for(size_t i=augmented_size_; i<2*augmented_size_; ++i) {
        sigma_points.col(i) = augmented_mean - augmented_cov_squared.col(i-augmented_size_);
    }
    sigma_points.col(2*augmented_size_) = augmented_mean;

    // Calculate mean y
    Eigen::MatrixXd observed_sigma_points = Eigen::MatrixXd::Zero(measurement_dim, 2*augmented_size_+1);
    Eigen::VectorXd y_mean = Eigen::VectorXd::Zero(measurement_dim);
    for(size_t i=0; i<2*augmented_size_+1; ++i) {
        const Eigen::VectorXd& curr_x = sigma_points.col(i).head(state_dim);
        const Eigen::VectorXd& meas_noise = sigma_points.col(i).segment(state_dim+system_noise_dim, measurement_noise_dim);

        const Eigen::VectorXd y = vehicle_model_->measureWithLandmark(curr_x, meas_noise, landmark);
        observed_sigma_points.col(i) = y;
        if(i==2*augmented_size_) {
            y_mean += Sigma_WM0_ * y;
        } else {
            y_mean += Sigma_WMI_ * y;
        }
    }

    Eigen::MatrixXd Pyy = Eigen::MatrixXd::Zero(measurement_dim, measurement_dim);
    for(size_t i=0; i<2*augmented_size_+1; ++i) {
        const Eigen::VectorXd delta_y = observed_sigma_points.col(i) - y_mean;
        if(i==2*augmented_size_) {
            Pyy += Sigma_WC0_ * (delta_y * delta_y.transpose());
        } else {
            Pyy += Sigma_WCI_ * (delta_y * delta_y.transpose());
        }
    }

    Eigen::MatrixXd Pxy = Eigen::MatrixXd::Zero(state_dim, measurement_dim);
    for(size_t i=0; i<2*augmented_size_+1; ++i) {
        const Eigen::VectorXd delta_x = sigma_points.col(i).head(state_dim) - state_info.mean;
        const Eigen::VectorXd delta_y = observed_sigma_points.col(i) - y_mean;
        if(i==2*augmented_size_) {
            Pxy += Sigma_WC0_ * (delta_x * delta_y.transpose());
        } else {
            Pxy += Sigma_WCI_ * (delta_x * delta_y.transpose());
        }
    }

    const Eigen::MatrixXd K = Pxy*Pyy.inverse();

    StateInfo result;
    result.mean = state_info.mean + K*(measurement_values - y_mean);
    result.covariance = state_info.covariance - K * Pyy * K.transpose();

    return result;
}

StateInfo UKF::getMeasurementInfo(const StateInfo& state_info,
                                  const std::map<int, std::shared_ptr<BaseDistribution>>& system_noise_map,
                                  const std::map<int, std::shared_ptr<BaseDistribution>>& measurement_noise_map,
                                  const Eigen::Vector2d& landmark)
{
    const size_t state_dim = vehicle_model_->state_dim_;
    const size_t system_noise_dim = vehicle_model_->system_noise_dim_;
    const size_t measurement_dim = vehicle_model_->measurement_dim_;
    const size_t measurement_noise_dim = vehicle_model_->measurement_noise_dim_;
    Eigen::VectorXd augmented_mean = Eigen::VectorXd::Zero(augmented_size_);
    Eigen::MatrixXd augmented_cov = Eigen::MatrixXd::Zero(augmented_size_, augmented_size_);

    // state
    augmented_mean.head(state_dim) = state_info.mean;
    augmented_cov.block(0, 0, state_dim, state_dim) = state_info.covariance;

    // system noise
    for(size_t i=0; i<system_noise_dim; ++i) {
        const auto dist_noise = system_noise_map.at(i);
        const size_t target_idx = i + state_dim;
        augmented_mean(target_idx) = dist_noise->calc_mean();
        augmented_cov(target_idx, target_idx) = dist_noise->calc_variance();
    }

    // measurement noise
    for(size_t i=0; i<measurement_noise_dim; ++i) {
        const auto dist_noise = measurement_noise_map.at(i);
        const size_t target_idx = i + state_dim + system_noise_dim;
        augmented_mean(target_idx) = dist_noise->calc_mean();
        augmented_cov(target_idx, target_idx) = dist_noise->calc_variance();
    }

    assert((augmented_cov*(augmented_size_ + lambda_)).llt().info() == Eigen::Success);
    const Eigen::MatrixXd augmented_cov_squared = (augmented_cov * (augmented_size_ + lambda_)).llt().matrixL();

    Eigen::MatrixXd sigma_points = Eigen::MatrixXd::Zero(augmented_size_, 2*augmented_size_+1);
    // Resample Sigma Points
    for(size_t i=0; i<augmented_size_; ++i) {
        sigma_points.col(i) = augmented_mean + augmented_cov_squared.col(i);
    }
    for(size_t i=augmented_size_; i<2*augmented_size_; ++i) {
        sigma_points.col(i) = augmented_mean - augmented_cov_squared.col(i-augmented_size_);
    }
    sigma_points.col(2*augmented_size_) = augmented_mean;

    // Calculate mean y
    Eigen::MatrixXd observed_sigma_points = Eigen::MatrixXd::Zero(measurement_dim, 2*augmented_size_+1);
    Eigen::VectorXd y_mean = Eigen::VectorXd::Zero(measurement_dim);
    for(size_t i=0; i<2*augmented_size_+1; ++i) {
        const Eigen::VectorXd& curr_x = sigma_points.col(i).head(state_dim);
        const Eigen::VectorXd& meas_noise = sigma_points.col(i).segment(state_dim+system_noise_dim, measurement_noise_dim);

        const Eigen::VectorXd y = vehicle_model_->measureWithLandmark(curr_x, meas_noise, landmark);
        observed_sigma_points.col(i) = y;
        if(i==2*augmented_size_) {
            y_mean += Sigma_WM0_ * y;
        } else {
            y_mean += Sigma_WMI_ * y;
        }
    }

    Eigen::MatrixXd Pyy = Eigen::MatrixXd::Zero(measurement_dim, measurement_dim);
    for(size_t i=0; i<2*augmented_size_+1; ++i) {
        const Eigen::VectorXd delta_y = observed_sigma_points.col(i) - y_mean;
        if(i==2*augmented_size_) {
            Pyy += Sigma_WC0_ * (delta_y * delta_y.transpose());
        } else {
            Pyy += Sigma_WCI_ * (delta_y * delta_y.transpose());
        }
    }

    StateInfo measurement_info;
    measurement_info.mean = y_mean;
    measurement_info.covariance = Pyy;
    return measurement_info;
}