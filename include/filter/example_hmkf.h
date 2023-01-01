#ifndef HMKF_EXAMPLE_HMKF_H
#define HMKF_EXAMPLE_HMKF_H

#include <iostream>
#include <memory>
#include <vector>
#include <Eigen/Eigen>

#include "distribution/base_distribution.h"
#include "model/example_model.h"

namespace Example
{
    struct PredictedMoments {
        double xPow1{0.0};
        double yawPow1{0.0};
        double xPow2{0.0};
        double yawPow2{0.0};
        double xPow1_yawPow1{0.0};
        double xPow3{0.0};
        double xPow2_yawPow1{0.0};
        double xPow4{0.0};
    };

    struct MeasurementMoments {
        double rPow1{0.0};
        double yawPow1{0.0};
        double rPow2{0.0};
        double yawPow2{0.0};
        double rPow1_yawPow1{0.0};
    };
}

class ExampleHMKF
{
public:
    Example::PredictedMoments predict(const StateInfo& state,
                                      const Eigen::Vector2d & control_inputs,
                                      const double dt,
                                      const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map);
    StateInfo update(const Example::PredictedMoments & predicted_moments,
                     const Eigen::VectorXd & observed_values,
                     const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map);

    Example::MeasurementMoments getMeasurementMoments(const Example::PredictedMoments & predicted_moments,
                                                      const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map);

    Eigen::MatrixXd getStateMeasurementMatrix(const Example::PredictedMoments& predicted_moments,
                                              const Example::MeasurementMoments & measurement_moments,
                                              const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map);
};

#endif //HMKF_EXAMPLE_HMKF_H
