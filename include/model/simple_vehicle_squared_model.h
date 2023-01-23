#ifndef HMKF_SIMPLE_VEHICLE_SQUARED_MODEL_H
#define HMKF_SIMPLE_VEHICLE_SQUARED_MODEL_H

#include <iostream>
#include <cmath>
#include <vector>
#include <Eigen/Eigen>

#include "model/base_model.h"

namespace SimpleVehicleSquared
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

    struct HighOrderMoments{
        double cPow2{0.0};
        double sPow2{0.0};
        double cPow1_sPow1{0.0};

        double xPow1_cPow2{0.0};
        double xPow1_sPow2{0.0};
        double yPow1_cPow2{0.0};
        double yPow1_sPow2{0.0};
        double xPow1_cPow1_sPow1{0.0};
        double yPow1_cPow1_sPow1{0.0};
        double yawPow1_cPow2{0.0};
        double yawPow1_sPow2{0.0};
        double yawPow1_cPow1_sPow1{0.0};

        double cPow4{0.0};
        double sPow4{0.0};
        double cPow1_sPow3{0.0};
        double cPow3_sPow1{0.0};
        double cPow2_sPow2{0.0};
        double xPow2_cPow2{0.0};
        double xPow2_sPow2{0.0};
        double yPow2_cPow2{0.0};
        double yPow2_sPow2{0.0};
        double xPow2_cPow1_sPow1{0.0};
        double yPow2_cPow1_sPow1{0.0};
        double xPow1_yPow1_cPow2{0.0};
        double xPow1_yPow1_sPow2{0.0};
        double xPow1_yPow1_cPow1_sPow1{0.0};
        double xPow1_yawPow1_cPow1_sPow1{0.0};
        double yPow1_yawPow1_cPow1_sPow1{0.0};
        double xPow1_yawPow1_cPow2{0.0};
        double xPow1_yawPow1_sPow2{0.0};
        double yPow1_yawPow1_cPow2{0.0};
        double yPow1_yawPow1_sPow2{0.0};

        double xPow1_cPow4{0.0};
        double xPow1_sPow4{0.0};
        double yPow1_cPow4{0.0};
        double yPow1_sPow4{0.0};
        double xPow1_cPow3_sPow1{0.0};
        double xPow1_cPow2_sPow2{0.0};
        double xPow1_cPow1_sPow3{0.0};
        double yPow1_cPow3_sPow1{0.0};
        double yPow1_cPow2_sPow2{0.0};
        double yPow1_cPow1_sPow3{0.0};
        double xPow3_cPow2{0.0};
        double xPow3_sPow2{0.0};
        double yPow3_cPow2{0.0};
        double yPow3_sPow2{0.0};
        double xPow1_yPow2_cPow1_sPow1{0.0};
        double xPow2_yPow1_cPow1_sPow1{0.0};
        double xPow3_cPow1_sPow1{0.0};
        double yPow3_cPow1_sPow1{0.0};
        double xPow1_yPow2_cPow2{0.0};
        double xPow1_yPow2_sPow2{0.0};
        double xPow2_yPow1_cPow2{0.0};
        double xPow2_yPow1_sPow2{0.0};
        double xPow2_yawPow1_cPow2{0.0};
        double xPow2_yawPow1_sPow2{0.0};
        double yPow2_yawPow1_cPow2{0.0};
        double yPow2_yawPow1_sPow2{0.0};
        double xPow2_yawPow1_cPow1_sPow1{0.0};
        double yPow2_yawPow1_cPow1_sPow1{0.0};
        double xPow1_yPow1_yawPow1_cPow1_sPow1{0.0};
        double xPow1_yPow1_yawPow1_cPow2{0.0};
        double xPow1_yPow1_yawPow1_sPow2{0.0};

        double xPow2_cPow4{0.0};
        double xPow2_sPow4{0.0};
        double yPow2_cPow4{0.0};
        double yPow2_sPow4{0.0};
        double xPow1_yPow1_cPow4{0.0};
        double xPow1_yPow1_sPow4{0.0};
        double xPow2_cPow2_sPow2{0.0};
        double xPow2_cPow3_sPow1{0.0};
        double xPow2_cPow1_sPow3{0.0};
        double yPow2_cPow2_sPow2{0.0};
        double yPow2_cPow3_sPow1{0.0};
        double yPow2_cPow1_sPow3{0.0};
        double xPow1_yPow1_cPow3_sPow1{0.0};
        double xPow1_yPow1_cPow2_sPow2{0.0};
        double xPow1_yPow1_cPow1_sPow3{0.0};

        double xPow3_cPow4{0.0};
        double xPow3_sPow4{0.0};
        double yPow3_cPow4{0.0};
        double yPow3_sPow4{0.0};
        double xPow1_yPow2_cPow4{0.0};
        double xPow1_yPow2_sPow4{0.0};
        double xPow2_yPow1_cPow4{0.0};
        double xPow2_yPow1_sPow4{0.0};
        double xPow3_cPow3_sPow1{0.0};
        double xPow3_cPow1_sPow3{0.0};
        double xPow3_cPow2_sPow2{0.0};
        double yPow3_cPow3_sPow1{0.0};
        double yPow3_cPow1_sPow3{0.0};
        double yPow3_cPow2_sPow2{0.0};
        double xPow2_yPow1_cPow3_sPow1{0.0};
        double xPow2_yPow1_cPow1_sPow3{0.0};
        double xPow2_yPow1_cPow2_sPow2{0.0};
        double xPow1_yPow2_cPow2_sPow2{0.0};
        double xPow1_yPow2_cPow1_sPow3{0.0};
        double xPow1_yPow2_cPow3_sPow1{0.0};

        double xPow4_cPow4{0.0};
        double xPow4_sPow4{0.0};
        double yPow4_cPow4{0.0};
        double yPow4_sPow4{0.0};
        double xPow4_cPow2_sPow2{0.0};
        double xPow4_cPow3_sPow1{0.0};
        double xPow4_cPow1_sPow3{0.0};
        double yPow4_cPow2_sPow2{0.0};
        double yPow4_cPow3_sPow1{0.0};
        double yPow4_cPow1_sPow3{0.0};
        double xPow3_yPow1_cPow4{0.0};
        double xPow1_yPow3_cPow4{0.0};
        double xPow3_yPow1_sPow4{0.0};
        double xPow1_yPow3_sPow4{0.0};
        double xPow2_yPow2_cPow4{0.0};
        double xPow2_yPow2_sPow4{0.0};
        double xPow3_yPow1_cPow1_sPow3{0.0};
        double xPow3_yPow1_cPow3_sPow1{0.0};
        double xPow3_yPow1_cPow2_sPow2{0.0};
        double xPow2_yPow2_cPow1_sPow3{0.0};
        double xPow2_yPow2_cPow3_sPow1{0.0};
        double xPow2_yPow2_cPow2_sPow2{0.0};
        double xPow1_yPow3_cPow1_sPow3{0.0};
        double xPow1_yPow3_cPow3_sPow1{0.0};
        double xPow1_yPow3_cPow2_sPow2{0.0};
    };
}

class SimpleVehicleSquaredModel : public BaseModel
{
public:
    SimpleVehicleSquaredModel(const size_t state_dim = 3,
                              const size_t system_noise_dim = 2,
                              const size_t measurement_dim = 2,
                              const size_t measurement_noise_dim = 2)
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

#endif //HMKF_SIMPLE_VEHICLE_SQUARED_MODEL_H
