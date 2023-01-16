#ifndef UNCERTAINTY_PROPAGATION_SIMPLE_VEHICLE_MODEL_H
#define UNCERTAINTY_PROPAGATION_SIMPLE_VEHICLE_MODEL_H

#include <iostream>
#include <cmath>
#include <vector>
#include <Eigen/Eigen>

#include "model/base_model.h"

namespace SimpleVehicle
{
    namespace STATE {
        enum IDX {
            X = 0,
            Y = 1,
            YAW = 2,
        };
    }

    namespace MEASUREMENT {
        enum IDX {
            RCOS = 0,
            RSIN = 1,
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
            WV = 0,
            WU = 1,
        };
    }

    namespace MEASUREMENT_NOISE {
        enum IDX {
            WR = 0,
            WA = 1,
        };
    }
}

class SimpleVehicleModel : public BaseModel
{
public:
    struct HighOrderMoments{
        double xPow1{0.0};
        double yPow1{0.0};
        double yawPow1{0.0};
        double cPow1{0.0};
        double sPow1{0.0};

        double xPow2{0.0};
        double yPow2{0.0};
        double yawPow2{0.0};
        double cPow2{0.0};
        double sPow2{0.0};
        double xPow1_yPow1{0.0};
        double xPow1_yawPow1{0.0};
        double yPow1_yawPow1{0.0};
        double xPow1_cPow1{0.0};
        double yPow1_cPow1{0.0};
        double xPow1_sPow1{0.0};
        double yPow1_sPow1{0.0};
        double cPow1_sPow1{0.0};
        double yawPow1_cPow1{0.0};
        double yawPow1_sPow1{0.0};

        double xPow1_cPow2{0.0};
        double yPow1_cPow2{0.0};
        double xPow1_sPow2{0.0};
        double yPow1_sPow2{0.0};
        double xPow2_cPow1{0.0};
        double xPow2_sPow1{0.0};
        double yPow2_cPow1{0.0};
        double yPow2_sPow1{0.0};
        double xPow1_yPow1_cPow1{0.0};
        double xPow1_yPow1_sPow1{0.0};
        double xPow1_cPow1_sPow1{0.0};
        double yPow1_cPow1_sPow1{0.0};
        double xPow1_yawPow1_cPow1{0.0};
        double xPow1_yawPow1_sPow1{0.0};
        double yPow1_yawPow1_cPow1{0.0};
        double yPow1_yawPow1_sPow1{0.0};

        double xPow2_cPow2{0.0};
        double yPow2_cPow2{0.0};
        double xPow2_sPow2{0.0};
        double yPow2_sPow2{0.0};
        double xPow1_yPow1_cPow2{0.0};
        double xPow1_yPow1_sPow2{0.0};
        double xPow2_cPow1_sPow1{0.0};
        double yPow2_cPow1_sPow1{0.0};
        double xPow1_yPow1_cPow1_sPow1{0.0};
    };

    SimpleVehicleModel(const size_t state_dim,
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

    Eigen::VectorXd measure(const Eigen::VectorXd& x_curr, const Eigen::VectorXd& observation_noise) override;

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

#endif //UNCERTAINTY_PROPAGATION_SIMPLE_VEHICLE_MODEL_H
