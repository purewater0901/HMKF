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
        double yPow1{0.0};
        double xPow2{0.0};
        double yPow2{0.0};
        double xPow1_yPow1{0.0};
        double xPow3{0.0};
        double yPow3{0.0};
        double xPow1_yPow2{0.0};
        double xPow2_yPow1{0.0};
        double xPow4{0.0};
        double yPow4{0.0};
        double xPow2_yPow2{0.0};
    };

    struct MeasurementMoments {
        double rPow1{0.0};
        double rPow2{0.0};
    };
}

class ExampleHMKF
{
public:
    Example::PredictedMoments predict(const StateInfo& state,
                                      const Eigen::VectorXd & control_inputs,
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
