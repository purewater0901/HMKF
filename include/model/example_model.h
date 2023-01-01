#ifndef HMKF_EXAMPLE_MODEL_H
#define HMKF_EXAMPLE_MODEL_H

#include <iostream>
#include <cmath>
#include <vector>
#include <Eigen/Eigen>

#include "model/base_model.h"

namespace Example
{
    namespace STATE {
        enum IDX {
            X = 0,
            YAW = 1,
        };
    }

    namespace MEASUREMENT {
        enum IDX {
            R = 0,
            YAW = 1
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
            WYAW = 1,
        };
    }
}

class ExampleVehicleModel : public BaseModel
{
public:
    ExampleVehicleModel() = default;

    // dynamics model
    Eigen::VectorXd propagate(const Eigen::VectorXd& x_curr,
                              const Eigen::VectorXd& u_curr,
                              const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map,
                              const double dt) override;

    Eigen::VectorXd propagate(const Eigen::VectorXd& x_curr,
                              const Eigen::VectorXd& u_curr,
                              const Eigen::VectorXd& system_noise,
                              const double dt) override;

    // measurement model
    Eigen::VectorXd measure(const Eigen::VectorXd& x_curr,
                            const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map) override;

    Eigen::VectorXd measure(const Eigen::VectorXd& x_curr, const Eigen::VectorXd& measurement_noise) override;

    // get df/dx
    Eigen::MatrixXd getStateMatrix(const Eigen::VectorXd& x_curr,
                                   const Eigen::VectorXd& u_curr,
                                   const double dt) override;

    Eigen::MatrixXd getProcessNoiseMatrix(const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map) override;

    // get dh/dx
    Eigen::MatrixXd getMeasurementMatrix(const Eigen::VectorXd& x_curr,
                                         const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map) override;

    Eigen::MatrixXd getMeasurementNoiseMatrix(const Eigen::VectorXd& x_curr,
                                              const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map) override;

    // propagate dynamics moment
    StateInfo propagateStateMoments(const StateInfo &state_info,
                                    const Eigen::VectorXd &control_inputs,
                                    const double dt,
                                    const std::map<int, std::shared_ptr<BaseDistribution>> &noise_map) override;

    StateInfo getMeasurementMoments(const StateInfo &state_info,
                                    const std::map<int, std::shared_ptr<BaseDistribution>> &noise_map) override;

    Eigen::MatrixXd getStateMeasurementMatrix(const StateInfo& state_info, const StateInfo& measurement_info,
                                              const std::map<int, std::shared_ptr<BaseDistribution>> &noise_map) override;
};

#endif //HMKF_EXAMPLE_MODEL_H
