#ifndef UNCERTAINTY_PROPAGATION_MOBILE_ROBOT_MODEL_H
#define UNCERTAINTY_PROPAGATION_MOBILE_ROBOT_MODEL_H

#include <iostream>
#include <cmath>
#include <vector>
#include <Eigen/Eigen>

#include "model/base_model.h"

namespace MobileRobot
{
    namespace STATE {
        enum IDX {
            X = 0,
            Y = 1,
            V = 2,
            YAW = 3,
        };
    }

    namespace OBSERVATION {
        enum IDX {
            X = 0,
            Y = 1,
            VC = 2
        };
    }

    namespace INPUT {
        enum IDX {
            A = 0,
            U = 1,
        };
    }

    namespace SYSTEM_NOISE {
        enum IDX {
            WV = 0,
            WYAW = 1,
        };
    }

    namespace MEASUREMENT_NOISE {
        enum IDX {
            WX = 0,
            WY = 1,
            WVC = 2,
            WYAW = 3,
        };
    }
}

class MobileRobotModel : public BaseModel
{
public:
    MobileRobotModel() = default;

    Eigen::VectorXd propagate(const Eigen::VectorXd& x_curr,
                              const Eigen::VectorXd& u_curr,
                              const Eigen::VectorXd& system_noise,
                              const double dt) override;

    Eigen::VectorXd propagate(const Eigen::VectorXd& x_curr,
                              const Eigen::VectorXd& u_curr,
                              const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map,
                              const double dt) override;


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

#endif //UNCERTAINTY_PROPAGATION_MOBILE_ROBOT_MODEL_H
