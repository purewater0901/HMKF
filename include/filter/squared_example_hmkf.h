#ifndef HMKF_SQUARED_EXAMPLE_HMKF_H
#define HMKF_SQUARED_EXAMPLE_HMKF_H

#include <iostream>
#include <memory>
#include <vector>
#include <Eigen/Eigen>

#include "distribution/base_distribution.h"
#include "model/squared_example_model.h"

namespace SquaredExample
{
    struct PredictedMoments {
        double xPow1{0.0};
        double yPow1{0.0};
        double xPow2{0.0};
        double yPow2{0.0};
        double xPow1_yPow1{0.0};
        double xPow4{0.0};
        double yPow4{0.0};
        double xPow5{0.0};
        double yPow5{0.0};
        double xPow1_yPow4{0.0};
        double xPow4_yPow1{0.0};
        double xPow8{0.0};
        double yPow8{0.0};
        double xPow4_yPow4{0.0};
    };

    struct MeasurementMoments {
        double rPow1{0.0};
        double rPow2{0.0};
    };
}

class SquaredExampleHMKF
{
public:
    SquaredExample::PredictedMoments predict(const StateInfo& state,
                                             const Eigen::VectorXd & control_inputs,
                                             const double dt,
                                             const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map);
    StateInfo update(const SquaredExample::PredictedMoments & predicted_moments,
                     const Eigen::VectorXd & observed_values,
                     const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map);

    SquaredExample::MeasurementMoments getMeasurementMoments(const SquaredExample::PredictedMoments & predicted_moments,
                                                      const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map);

    Eigen::MatrixXd getStateMeasurementMatrix(const SquaredExample::PredictedMoments& predicted_moments,
                                              const SquaredExample::MeasurementMoments & measurement_moments,
                                              const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map);
};

#endif //HMKF_SQUARED_EXAMPLE_HMKF_H
