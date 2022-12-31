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
}

class ExampleHMKF
{
public:
    void predict(const Example::StateInfo& state,
                 const Eigen::Vector2d & control_inputs,
                 const double dt,
                 const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map);
};

#endif //HMKF_EXAMPLE_HMKF_H
