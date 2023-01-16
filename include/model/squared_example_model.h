#ifndef HMKF_SQUARED_EXAMPLE_MODEL_H
#define HMKF_SQUARED_EXAMPLE_MODEL_H

#include <iostream>
#include <cmath>
#include <vector>
#include <Eigen/Eigen>

#include "model/base_model.h"

namespace SquaredExample
{
    namespace STATE {
        enum IDX {
            X = 0,
            Y = 1,
        };
    }

    namespace MEASUREMENT {
        enum IDX {
            R = 0,
        };
    }

    namespace INPUT {
        enum IDX {
            V = 0,
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
            WR = 0,
        };
    }
}

class ExampleSquaredVehicleModel : public BaseModel
{
public:
    ExampleSquaredVehicleModel(const size_t state_dim,
                               const size_t system_noise_dim,
                               const size_t measurement_dim,
                               const size_t measurement_noise_dim)
                               : BaseModel(state_dim, system_noise_dim, measurement_dim, measurement_noise_dim)
    {
    }

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

    Eigen::VectorXd measureWithLandmark(const Eigen::VectorXd& x_curr,
                                        const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map,
                                        const Eigen::Vector2d& landmark) override;

    Eigen::VectorXd measureWithLandmark(const Eigen::VectorXd& x_curr,
                                        const Eigen::VectorXd& observation_noise,
                                        const Eigen::Vector2d& landmark) override;

    // get df/dx
    Eigen::MatrixXd getStateMatrix(const Eigen::VectorXd& x_curr,
                                   const Eigen::VectorXd& u_curr,
                                   const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map,
                                   const double dt) override;

    Eigen::MatrixXd getProcessNoiseMatrix(const Eigen::VectorXd& x_curr,
                                          const Eigen::VectorXd& u_curr,
                                          const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map,
                                          const double dt) override;

    // get dh/dx
    Eigen::MatrixXd getMeasurementMatrix(const Eigen::VectorXd& x_curr,
                                         const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map) override;

    Eigen::MatrixXd getMeasurementNoiseMatrix(const Eigen::VectorXd& x_curr,
                                              const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map) override;

    Eigen::MatrixXd getMeasurementMatrix(const Eigen::VectorXd& x_curr,
                                         const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map,
                                         const Eigen::Vector2d& landmark) override;

    Eigen::MatrixXd getMeasurementNoiseMatrix(const Eigen::VectorXd& x_curr,
                                              const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map,
                                              const Eigen::Vector2d& landmark) override;

    // propagate dynamics moment
    StateInfo propagateStateMoments(const StateInfo &state_info,
                                    const Eigen::VectorXd &control_inputs,
                                    const double dt,
                                    const std::map<int, std::shared_ptr<BaseDistribution>> &noise_map) override;

    StateInfo getMeasurementMoments(const StateInfo &state_info,
                                    const std::map<int, std::shared_ptr<BaseDistribution>> &noise_map) override;

    Eigen::MatrixXd getStateMeasurementMatrix(const StateInfo& state_info, const StateInfo& measurement_info,
                                              const std::map<int, std::shared_ptr<BaseDistribution>> &noise_map) override;

    StateInfo getMeasurementMoments(const StateInfo &state_info,
                                    const std::map<int, std::shared_ptr<BaseDistribution>> &noise_map,
                                    const Eigen::Vector2d& landmark) override;

    Eigen::MatrixXd getStateMeasurementMatrix(const StateInfo& state_info, const StateInfo& measurement_info,
                                              const std::map<int, std::shared_ptr<BaseDistribution>> &noise_map,
                                              const Eigen::Vector2d& landmark) override;
};

#endif //HMKF_SQUARED_EXAMPLE_MODEL_H
