#ifndef HMKF_EXAMPLE_HMKF_H
#define HMKF_EXAMPLE_HMKF_H

#include <iostream>
#include <memory>
#include <vector>
#include <Eigen/Eigen>

#include "distribution/base_distribution.h"

namespace Example
{
    struct StateInfo {
        Eigen::VectorXd mean;
        Eigen::MatrixXd covariance;
    };

    namespace STATE {
        enum IDX {
            X = 0,
            YAW = 1,
        };
    }

    namespace MEASUREMENT {
        enum IDX {
            R = 0,
            YAW = 1,
        };
    }

    namespace INPUT {
        enum IDX {
            V = 0,
            U = 1,
        };
    }

    namespace SYSTEM_NOISE {
        enum IDX {
            WX = 0,
            WYAW = 1,
        };
    }

    namespace MEASUREMENT_NOISE {
        enum IDX {
            WR = 0,
            WYAW= 1,
        };
    }

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
    Example::PredictedMoments predict(const Example::StateInfo& state,
                                      const Eigen::Vector2d & control_inputs,
                                      const double dt,
                                      const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map);
    Example::StateInfo update(const Example::PredictedMoments & predicted_moments,
                              const Eigen::VectorXd & observed_values,
                              const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map);

    Example::MeasurementMoments getMeasurementMoments(const Example::PredictedMoments & predicted_moments,
                                                      const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map);

    Eigen::MatrixXd getStateMeasurementMatrix(const Example::PredictedMoments& predicted_moments,
                                              const Example::MeasurementMoments & measurement_moments,
                                              const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map);
};

#endif //HMKF_EXAMPLE_HMKF_H
