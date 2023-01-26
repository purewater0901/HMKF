#include "filter/squared_example_hmkf.h"

#include "distribution/two_dimensional_normal_distribution.h"

using namespace SquaredExample;

PredictedMoments SquaredExampleHMKF::predict(const StateInfo &state,
                                             const Eigen::VectorXd &control_inputs,
                                             const double dt,
                                             const std::map<int, std::shared_ptr<BaseDistribution>> &noise_map)
{
    TwoDimensionalNormalDistribution dist(state.mean, state.covariance);

    const double xPow1 = dist.calc_moment(STATE::IDX::X, 1);
    const double yPow1 = dist.calc_moment(STATE::IDX::Y, 1);
    const double xPow2 = dist.calc_moment(STATE::IDX::X, 2); // x^2
    const double yPow2 = dist.calc_moment(STATE::IDX::Y, 2); // y^2
    const double xPow1_yPow1 = dist.calc_xy_cos_y_sin_y_moment(1, 1, 0, 0);
    const double xPow3 = dist.calc_moment(STATE::IDX::X, 3); // x^3
    const double yPow3 = dist.calc_moment(STATE::IDX::Y, 3); // y^3
    const double xPow2_yPow1 = dist.calc_xy_cos_y_sin_y_moment(2, 1, 0, 0);
    const double xPow1_yPow2 = dist.calc_xy_cos_y_sin_y_moment(1, 2, 0, 0);
    const double xPow4 = dist.calc_moment(STATE::IDX::X, 4); // x^4
    const double yPow4 = dist.calc_moment(STATE::IDX::Y, 4); // y^4
    const double xPow2_yPow2 = dist.calc_xy_cos_y_sin_y_moment(2, 2, 0, 0);
    const double xPow1_yPow3 = dist.calc_xy_cos_y_sin_y_moment(1, 3, 0, 0);
    const double xPow3_yPow1 = dist.calc_xy_cos_y_sin_y_moment(3, 1, 0, 0);

    const double xPow5 = dist.calc_xy_cos_y_sin_y_moment(5, 0, 0, 0);
    const double yPow5 = dist.calc_xy_cos_y_sin_y_moment(0, 5, 0, 0);
    const double xPow1_yPow4 = dist.calc_xy_cos_y_sin_y_moment(1, 4, 0, 0);
    const double xPow4_yPow1 = dist.calc_xy_cos_y_sin_y_moment(4, 1, 0, 0);
    const double xPow3_yPow2 = dist.calc_xy_cos_y_sin_y_moment(3, 2, 0, 0);
    const double xPow2_yPow3 = dist.calc_xy_cos_y_sin_y_moment(2, 3, 0, 0);

    const double xPow6 = dist.calc_xy_cos_y_sin_y_moment(6, 0, 0, 0);
    const double yPow6 = dist.calc_xy_cos_y_sin_y_moment(0, 6, 0, 0);
    const double xPow3_yPow3 = dist.calc_xy_cos_y_sin_y_moment(3, 3, 0, 0);
    const double xPow2_yPow4 = dist.calc_xy_cos_y_sin_y_moment(2, 4, 0, 0);
    const double xPow4_yPow2 = dist.calc_xy_cos_y_sin_y_moment(4, 2, 0, 0);

    const double xPow7 = dist.calc_xy_cos_y_sin_y_moment(7, 0, 0, 0);
    const double yPow7 = dist.calc_xy_cos_y_sin_y_moment(0, 7, 0, 0);
    const double xPow3_yPow4 = dist.calc_xy_cos_y_sin_y_moment(3, 4, 0, 0);
    const double xPow4_yPow3 = dist.calc_xy_cos_y_sin_y_moment(4, 3, 0, 0);

    const double xPow8 = dist.calc_moment(STATE::IDX::X, 8); // x^8
    const double yPow8 = dist.calc_moment(STATE::IDX::Y, 8); // y^8
    const double xPow4_yPow4 = dist.calc_xy_cos_y_sin_y_moment(4, 4, 0, 0);

    // Input
    const double v = control_inputs(INPUT::V);

    // System noise
    const auto wv_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WV);
    const auto wyaw_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WYAW);
    const double wvPow1 = wv_dist_ptr->calc_moment(1);
    const double wvPow2 = wv_dist_ptr->calc_moment(2);
    const double wvPow3 = wv_dist_ptr->calc_moment(3);
    const double wvPow4 = wv_dist_ptr->calc_moment(4);
    const double wvPow5 = wv_dist_ptr->calc_moment(5);
    const double wvPow6 = wv_dist_ptr->calc_moment(6);
    const double wvPow7 = wv_dist_ptr->calc_moment(7);
    const double wvPow8 = wv_dist_ptr->calc_moment(8);

    const double cwyawPow1 = wyaw_dist_ptr->calc_cos_moment(1);
    const double swyawPow1 = wyaw_dist_ptr->calc_sin_moment(1);
    const double cwyawPow1_swyawPow1 = wyaw_dist_ptr->calc_cos_sin_moment(1, 1);

    const double cwyawPow2 = wyaw_dist_ptr->calc_cos_moment(2);
    const double swyawPow2 = wyaw_dist_ptr->calc_sin_moment(2);

    const double cwyawPow3 = wyaw_dist_ptr->calc_cos_moment(3);
    const double swyawPow3 = wyaw_dist_ptr->calc_sin_moment(3);
    const double cwyawPow2_swyawPow2 = wyaw_dist_ptr->calc_cos_sin_moment(2, 2);
    const double cwyawPow2_swyawPow1 = wyaw_dist_ptr->calc_cos_sin_moment(2, 1);
    const double cwyawPow1_swyawPow2 = wyaw_dist_ptr->calc_cos_sin_moment(1, 2);

    const double cwyawPow4 = wyaw_dist_ptr->calc_cos_moment(4);
    const double swyawPow4 = wyaw_dist_ptr->calc_sin_moment(4);
    const double cwyawPow3_swyawPow1 = wyaw_dist_ptr->calc_cos_sin_moment(3, 1);
    const double cwyawPow1_swyawPow3 = wyaw_dist_ptr->calc_cos_sin_moment(1, 3);

    const double cwyawPow5 = wyaw_dist_ptr->calc_cos_moment(5);
    const double swyawPow5 = wyaw_dist_ptr->calc_sin_moment(5);
    const double cwyawPow2_swyawPow3= wyaw_dist_ptr->calc_cos_sin_moment(2, 3);
    const double cwyawPow3_swyawPow2= wyaw_dist_ptr->calc_cos_sin_moment(3, 2);
    const double cwyawPow1_swyawPow4= wyaw_dist_ptr->calc_cos_sin_moment(1, 4);
    const double cwyawPow4_swyawPow1= wyaw_dist_ptr->calc_cos_sin_moment(4, 1);

    const double cwyawPow6 = wyaw_dist_ptr->calc_cos_moment(6);
    const double swyawPow6 = wyaw_dist_ptr->calc_sin_moment(6);
    const double cwyawPow3_swyawPow3 = wyaw_dist_ptr->calc_cos_sin_moment(3, 3);
    const double cwyawPow2_swyawPow4 = wyaw_dist_ptr->calc_cos_sin_moment(2, 4);
    const double cwyawPow4_swyawPow2 = wyaw_dist_ptr->calc_cos_sin_moment(4, 2);

    const double cwyawPow7 = wyaw_dist_ptr->calc_cos_moment(7);
    const double swyawPow7 = wyaw_dist_ptr->calc_sin_moment(7);
    const double cwyawPow3_swyawPow4 = wyaw_dist_ptr->calc_cos_sin_moment(3, 4);
    const double cwyawPow4_swyawPow3 = wyaw_dist_ptr->calc_cos_sin_moment(4, 3);

    const double cwyawPow8 = wyaw_dist_ptr->calc_cos_moment(8);
    const double swyawPow8 = wyaw_dist_ptr->calc_sin_moment(8);
    const double cwyawPow4_swyawPow4 = wyaw_dist_ptr->calc_cos_sin_moment(4, 4);

    // moment propagation
    PredictedMoments next_moments;
    next_moments.xPow1 = xPow1 + (v+wvPow1)*cwyawPow1*dt;
    next_moments.yPow1 = yPow1 + (v+wvPow1)*swyawPow1*dt;
    next_moments.xPow2 = xPow2 + (v*v + wvPow2 + 2*v*wvPow1) * cwyawPow2 * dt * dt
                         + 2.0 * xPow1 * (v+wvPow1) * cwyawPow1 * dt;
    next_moments.yPow2 = yPow2 + (v*v + wvPow2 + 2*v*wvPow1) * swyawPow2 * dt * dt
                         + 2.0 * yPow1 * (v+wvPow1) * swyawPow1 * dt;
    next_moments.xPow1_yPow1 = xPow1_yPow1 + xPow1*(v+wvPow1)*swyawPow1*dt + yPow1*(v+wvPow1)*cwyawPow1*dt
                               + (v*v+wvPow2+2*v*wvPow1)*cwyawPow1_swyawPow1*dt*dt;
    next_moments.xPow4 = pow(dt, 4)*pow(v, 4)*cwyawPow4 + 4*pow(dt, 4)*pow(v, 3)*cwyawPow4*wvPow1 + 6*pow(dt, 4)*pow(v, 2)*cwyawPow4*wvPow2 + 4*pow(dt, 4)*v*cwyawPow4*wvPow3 + pow(dt, 4)*cwyawPow4*wvPow4 + 4*pow(dt, 3)*pow(v, 3)*cwyawPow3*xPow1 + 12*pow(dt, 3)*pow(v, 2)*cwyawPow3*wvPow1*xPow1 + 12*pow(dt, 3)*v*cwyawPow3*wvPow2*xPow1 + 4*pow(dt, 3)*cwyawPow3*wvPow3*xPow1 + 6*pow(dt, 2)*pow(v, 2)*cwyawPow2*xPow2 + 12*pow(dt, 2)*v*cwyawPow2*wvPow1*xPow2 + 6*pow(dt, 2)*cwyawPow2*wvPow2*xPow2 + 4*dt*v*cwyawPow1*xPow3 + 4*dt*cwyawPow1*wvPow1*xPow3 + xPow4;
    next_moments.yPow4 = pow(dt, 4)*pow(v, 4)*swyawPow4 + 4*pow(dt, 4)*pow(v, 3)*swyawPow4*wvPow1 + 6*pow(dt, 4)*pow(v, 2)*swyawPow4*wvPow2 + 4*pow(dt, 4)*v*swyawPow4*wvPow3 + pow(dt, 4)*swyawPow4*wvPow4 + 4*pow(dt, 3)*pow(v, 3)*swyawPow3*yPow1 + 12*pow(dt, 3)*pow(v, 2)*swyawPow3*wvPow1*yPow1 + 12*pow(dt, 3)*v*swyawPow3*wvPow2*yPow1 + 4*pow(dt, 3)*swyawPow3*wvPow3*yPow1 + 6*pow(dt, 2)*pow(v, 2)*swyawPow2*yPow2 + 12*pow(dt, 2)*v*swyawPow2*wvPow1*yPow2 + 6*pow(dt, 2)*swyawPow2*wvPow2*yPow2 + 4*dt*v*swyawPow1*yPow3 + 4*dt*swyawPow1*wvPow1*yPow3 + yPow4;
    next_moments.xPow5 = pow(dt, 5)*pow(v, 5)*cwyawPow5 + 5*pow(dt, 5)*pow(v, 4)*cwyawPow5*wvPow1 + 10*pow(dt, 5)*pow(v, 3)*cwyawPow5*wvPow2 + 10*pow(dt, 5)*pow(v, 2)*cwyawPow5*wvPow3 + 5*pow(dt, 5)*v*cwyawPow5*wvPow4 + pow(dt, 5)*cwyawPow5*wvPow5 + 5*pow(dt, 4)*pow(v, 4)*cwyawPow4*xPow1 + 20*pow(dt, 4)*pow(v, 3)*cwyawPow4*wvPow1*xPow1 + 30*pow(dt, 4)*pow(v, 2)*cwyawPow4*wvPow2*xPow1 + 20*pow(dt, 4)*v*cwyawPow4*wvPow3*xPow1 + 5*pow(dt, 4)*cwyawPow4*wvPow4*xPow1 + 10*pow(dt, 3)*pow(v, 3)*cwyawPow3*xPow2 + 30*pow(dt, 3)*pow(v, 2)*cwyawPow3*wvPow1*xPow2 + 30*pow(dt, 3)*v*cwyawPow3*wvPow2*xPow2 + 10*pow(dt, 3)*cwyawPow3*wvPow3*xPow2 + 10*pow(dt, 2)*pow(v, 2)*cwyawPow2*xPow3 + 20*pow(dt, 2)*v*cwyawPow2*wvPow1*xPow3 + 10*pow(dt, 2)*cwyawPow2*wvPow2*xPow3 + 5*dt*v*cwyawPow1*xPow4 + 5*dt*cwyawPow1*wvPow1*xPow4 + xPow5;
    next_moments.yPow5 = pow(dt, 5)*pow(v, 5)*swyawPow5 + 5*pow(dt, 5)*pow(v, 4)*swyawPow5*wvPow1 + 10*pow(dt, 5)*pow(v, 3)*swyawPow5*wvPow2 + 10*pow(dt, 5)*pow(v, 2)*swyawPow5*wvPow3 + 5*pow(dt, 5)*v*swyawPow5*wvPow4 + pow(dt, 5)*swyawPow5*wvPow5 + 5*pow(dt, 4)*pow(v, 4)*swyawPow4*yPow1 + 20*pow(dt, 4)*pow(v, 3)*swyawPow4*wvPow1*yPow1 + 30*pow(dt, 4)*pow(v, 2)*swyawPow4*wvPow2*yPow1 + 20*pow(dt, 4)*v*swyawPow4*wvPow3*yPow1 + 5*pow(dt, 4)*swyawPow4*wvPow4*yPow1 + 10*pow(dt, 3)*pow(v, 3)*swyawPow3*yPow2 + 30*pow(dt, 3)*pow(v, 2)*swyawPow3*wvPow1*yPow2 + 30*pow(dt, 3)*v*swyawPow3*wvPow2*yPow2 + 10*pow(dt, 3)*swyawPow3*wvPow3*yPow2 + 10*pow(dt, 2)*pow(v, 2)*swyawPow2*yPow3 + 20*pow(dt, 2)*v*swyawPow2*wvPow1*yPow3 + 10*pow(dt, 2)*swyawPow2*wvPow2*yPow3 + 5*dt*v*swyawPow1*yPow4 + 5*dt*swyawPow1*wvPow1*yPow4 + yPow5;
    next_moments.xPow1_yPow4 = pow(dt, 5)*pow(v, 5)*cwyawPow1_swyawPow4 + 5*pow(dt, 5)*pow(v, 4)*cwyawPow1_swyawPow4*wvPow1 + 10*pow(dt, 5)*pow(v, 3)*cwyawPow1_swyawPow4*wvPow2 + 10*pow(dt, 5)*pow(v, 2)*cwyawPow1_swyawPow4*wvPow3 + 5*pow(dt, 5)*v*cwyawPow1_swyawPow4*wvPow4 + pow(dt, 5)*cwyawPow1_swyawPow4*wvPow5 + 4*pow(dt, 4)*pow(v, 4)*cwyawPow1_swyawPow3*yPow1 + pow(dt, 4)*pow(v, 4)*swyawPow4*xPow1 + 16*pow(dt, 4)*pow(v, 3)*cwyawPow1_swyawPow3*wvPow1*yPow1 + 4*pow(dt, 4)*pow(v, 3)*swyawPow4*wvPow1*xPow1 + 24*pow(dt, 4)*pow(v, 2)*cwyawPow1_swyawPow3*wvPow2*yPow1 + 6*pow(dt, 4)*pow(v, 2)*swyawPow4*wvPow2*xPow1 + 16*pow(dt, 4)*v*cwyawPow1_swyawPow3*wvPow3*yPow1 + 4*pow(dt, 4)*v*swyawPow4*wvPow3*xPow1 + 4*pow(dt, 4)*cwyawPow1_swyawPow3*wvPow4*yPow1 + pow(dt, 4)*swyawPow4*wvPow4*xPow1 + 6*pow(dt, 3)*pow(v, 3)*cwyawPow1_swyawPow2*yPow2 + 4*pow(dt, 3)*pow(v, 3)*swyawPow3*xPow1_yPow1 + 18*pow(dt, 3)*pow(v, 2)*cwyawPow1_swyawPow2*wvPow1*yPow2 + 12*pow(dt, 3)*pow(v, 2)*swyawPow3*wvPow1*xPow1_yPow1 + 18*pow(dt, 3)*v*cwyawPow1_swyawPow2*wvPow2*yPow2 + 12*pow(dt, 3)*v*swyawPow3*wvPow2*xPow1_yPow1 + 6*pow(dt, 3)*cwyawPow1_swyawPow2*wvPow3*yPow2 + 4*pow(dt, 3)*swyawPow3*wvPow3*xPow1_yPow1 + 4*pow(dt, 2)*pow(v, 2)*cwyawPow1_swyawPow1*yPow3 + 6*pow(dt, 2)*pow(v, 2)*swyawPow2*xPow1_yPow2 + 8*pow(dt, 2)*v*cwyawPow1_swyawPow1*wvPow1*yPow3 + 12*pow(dt, 2)*v*swyawPow2*wvPow1*xPow1_yPow2 + 4*pow(dt, 2)*cwyawPow1_swyawPow1*wvPow2*yPow3 + 6*pow(dt, 2)*swyawPow2*wvPow2*xPow1_yPow2 + dt*v*cwyawPow1*yPow4 + 4*dt*v*swyawPow1*xPow1_yPow3 + dt*cwyawPow1*wvPow1*yPow4 + 4*dt*swyawPow1*wvPow1*xPow1_yPow3 + xPow1_yPow4;
    next_moments.xPow4_yPow1 = pow(dt, 5)*pow(v, 5)*cwyawPow4_swyawPow1 + 5*pow(dt, 5)*pow(v, 4)*cwyawPow4_swyawPow1*wvPow1 + 10*pow(dt, 5)*pow(v, 3)*cwyawPow4_swyawPow1*wvPow2 + 10*pow(dt, 5)*pow(v, 2)*cwyawPow4_swyawPow1*wvPow3 + 5*pow(dt, 5)*v*cwyawPow4_swyawPow1*wvPow4 + pow(dt, 5)*cwyawPow4_swyawPow1*wvPow5 + 4*pow(dt, 4)*pow(v, 4)*cwyawPow3_swyawPow1*xPow1 + pow(dt, 4)*pow(v, 4)*cwyawPow4*yPow1 + 16*pow(dt, 4)*pow(v, 3)*cwyawPow3_swyawPow1*wvPow1*xPow1 + 4*pow(dt, 4)*pow(v, 3)*cwyawPow4*wvPow1*yPow1 + 24*pow(dt, 4)*pow(v, 2)*cwyawPow3_swyawPow1*wvPow2*xPow1 + 6*pow(dt, 4)*pow(v, 2)*cwyawPow4*wvPow2*yPow1 + 16*pow(dt, 4)*v*cwyawPow3_swyawPow1*wvPow3*xPow1 + 4*pow(dt, 4)*v*cwyawPow4*wvPow3*yPow1 + 4*pow(dt, 4)*cwyawPow3_swyawPow1*wvPow4*xPow1 + pow(dt, 4)*cwyawPow4*wvPow4*yPow1 + 6*pow(dt, 3)*pow(v, 3)*cwyawPow2_swyawPow1*xPow2 + 4*pow(dt, 3)*pow(v, 3)*cwyawPow3*xPow1_yPow1 + 18*pow(dt, 3)*pow(v, 2)*cwyawPow2_swyawPow1*wvPow1*xPow2 + 12*pow(dt, 3)*pow(v, 2)*cwyawPow3*wvPow1*xPow1_yPow1 + 18*pow(dt, 3)*v*cwyawPow2_swyawPow1*wvPow2*xPow2 + 12*pow(dt, 3)*v*cwyawPow3*wvPow2*xPow1_yPow1 + 6*pow(dt, 3)*cwyawPow2_swyawPow1*wvPow3*xPow2 + 4*pow(dt, 3)*cwyawPow3*wvPow3*xPow1_yPow1 + 4*pow(dt, 2)*pow(v, 2)*cwyawPow1_swyawPow1*xPow3 + 6*pow(dt, 2)*pow(v, 2)*cwyawPow2*xPow2_yPow1 + 8*pow(dt, 2)*v*cwyawPow1_swyawPow1*wvPow1*xPow3 + 12*pow(dt, 2)*v*cwyawPow2*wvPow1*xPow2_yPow1 + 4*pow(dt, 2)*cwyawPow1_swyawPow1*wvPow2*xPow3 + 6*pow(dt, 2)*cwyawPow2*wvPow2*xPow2_yPow1 + 4*dt*v*cwyawPow1*xPow3_yPow1 + dt*v*swyawPow1*xPow4 + 4*dt*cwyawPow1*wvPow1*xPow3_yPow1 + dt*swyawPow1*wvPow1*xPow4 + xPow4_yPow1;
    next_moments.xPow8 = pow(dt, 8)*pow(v, 8)*cwyawPow8 + 8*pow(dt, 8)*pow(v, 7)*cwyawPow8*wvPow1 + 28*pow(dt, 8)*pow(v, 6)*cwyawPow8*wvPow2 + 56*pow(dt, 8)*pow(v, 5)*cwyawPow8*wvPow3 + 70*pow(dt, 8)*pow(v, 4)*cwyawPow8*wvPow4 + 56*pow(dt, 8)*pow(v, 3)*cwyawPow8*wvPow5 + 28*pow(dt, 8)*pow(v, 2)*cwyawPow8*wvPow6 + 8*pow(dt, 8)*v*cwyawPow8*wvPow7 + pow(dt, 8)*cwyawPow8*wvPow8 + 8*pow(dt, 7)*pow(v, 7)*cwyawPow7*xPow1 + 56*pow(dt, 7)*pow(v, 6)*cwyawPow7*wvPow1*xPow1 + 168*pow(dt, 7)*pow(v, 5)*cwyawPow7*wvPow2*xPow1 + 280*pow(dt, 7)*pow(v, 4)*cwyawPow7*wvPow3*xPow1 + 280*pow(dt, 7)*pow(v, 3)*cwyawPow7*wvPow4*xPow1 + 168*pow(dt, 7)*pow(v, 2)*cwyawPow7*wvPow5*xPow1 + 56*pow(dt, 7)*v*cwyawPow7*wvPow6*xPow1 + 8*pow(dt, 7)*cwyawPow7*wvPow7*xPow1 + 28*pow(dt, 6)*pow(v, 6)*cwyawPow6*xPow2 + 168*pow(dt, 6)*pow(v, 5)*cwyawPow6*wvPow1*xPow2 + 420*pow(dt, 6)*pow(v, 4)*cwyawPow6*wvPow2*xPow2 + 560*pow(dt, 6)*pow(v, 3)*cwyawPow6*wvPow3*xPow2 + 420*pow(dt, 6)*pow(v, 2)*cwyawPow6*wvPow4*xPow2 + 168*pow(dt, 6)*v*cwyawPow6*wvPow5*xPow2 + 28*pow(dt, 6)*cwyawPow6*wvPow6*xPow2 + 56*pow(dt, 5)*pow(v, 5)*cwyawPow5*xPow3 + 280*pow(dt, 5)*pow(v, 4)*cwyawPow5*wvPow1*xPow3 + 560*pow(dt, 5)*pow(v, 3)*cwyawPow5*wvPow2*xPow3 + 560*pow(dt, 5)*pow(v, 2)*cwyawPow5*wvPow3*xPow3 + 280*pow(dt, 5)*v*cwyawPow5*wvPow4*xPow3 + 56*pow(dt, 5)*cwyawPow5*wvPow5*xPow3 + 70*pow(dt, 4)*pow(v, 4)*cwyawPow4*xPow4 + 280*pow(dt, 4)*pow(v, 3)*cwyawPow4*wvPow1*xPow4 + 420*pow(dt, 4)*pow(v, 2)*cwyawPow4*wvPow2*xPow4 + 280*pow(dt, 4)*v*cwyawPow4*wvPow3*xPow4 + 70*pow(dt, 4)*cwyawPow4*wvPow4*xPow4 + 56*pow(dt, 3)*pow(v, 3)*cwyawPow3*xPow5 + 168*pow(dt, 3)*pow(v, 2)*cwyawPow3*wvPow1*xPow5 + 168*pow(dt, 3)*v*cwyawPow3*wvPow2*xPow5 + 56*pow(dt, 3)*cwyawPow3*wvPow3*xPow5 + 28*pow(dt, 2)*pow(v, 2)*cwyawPow2*xPow6 + 56*pow(dt, 2)*v*cwyawPow2*wvPow1*xPow6 + 28*pow(dt, 2)*cwyawPow2*wvPow2*xPow6 + 8*dt*v*cwyawPow1*xPow7 + 8*dt*cwyawPow1*wvPow1*xPow7 + xPow8;
    next_moments.yPow8 = pow(dt, 8)*pow(v, 8)*swyawPow8 + 8*pow(dt, 8)*pow(v, 7)*swyawPow8*wvPow1 + 28*pow(dt, 8)*pow(v, 6)*swyawPow8*wvPow2 + 56*pow(dt, 8)*pow(v, 5)*swyawPow8*wvPow3 + 70*pow(dt, 8)*pow(v, 4)*swyawPow8*wvPow4 + 56*pow(dt, 8)*pow(v, 3)*swyawPow8*wvPow5 + 28*pow(dt, 8)*pow(v, 2)*swyawPow8*wvPow6 + 8*pow(dt, 8)*v*swyawPow8*wvPow7 + pow(dt, 8)*swyawPow8*wvPow8 + 8*pow(dt, 7)*pow(v, 7)*swyawPow7*yPow1 + 56*pow(dt, 7)*pow(v, 6)*swyawPow7*wvPow1*yPow1 + 168*pow(dt, 7)*pow(v, 5)*swyawPow7*wvPow2*yPow1 + 280*pow(dt, 7)*pow(v, 4)*swyawPow7*wvPow3*yPow1 + 280*pow(dt, 7)*pow(v, 3)*swyawPow7*wvPow4*yPow1 + 168*pow(dt, 7)*pow(v, 2)*swyawPow7*wvPow5*yPow1 + 56*pow(dt, 7)*v*swyawPow7*wvPow6*yPow1 + 8*pow(dt, 7)*swyawPow7*wvPow7*yPow1 + 28*pow(dt, 6)*pow(v, 6)*swyawPow6*yPow2 + 168*pow(dt, 6)*pow(v, 5)*swyawPow6*wvPow1*yPow2 + 420*pow(dt, 6)*pow(v, 4)*swyawPow6*wvPow2*yPow2 + 560*pow(dt, 6)*pow(v, 3)*swyawPow6*wvPow3*yPow2 + 420*pow(dt, 6)*pow(v, 2)*swyawPow6*wvPow4*yPow2 + 168*pow(dt, 6)*v*swyawPow6*wvPow5*yPow2 + 28*pow(dt, 6)*swyawPow6*wvPow6*yPow2 + 56*pow(dt, 5)*pow(v, 5)*swyawPow5*yPow3 + 280*pow(dt, 5)*pow(v, 4)*swyawPow5*wvPow1*yPow3 + 560*pow(dt, 5)*pow(v, 3)*swyawPow5*wvPow2*yPow3 + 560*pow(dt, 5)*pow(v, 2)*swyawPow5*wvPow3*yPow3 + 280*pow(dt, 5)*v*swyawPow5*wvPow4*yPow3 + 56*pow(dt, 5)*swyawPow5*wvPow5*yPow3 + 70*pow(dt, 4)*pow(v, 4)*swyawPow4*yPow4 + 280*pow(dt, 4)*pow(v, 3)*swyawPow4*wvPow1*yPow4 + 420*pow(dt, 4)*pow(v, 2)*swyawPow4*wvPow2*yPow4 + 280*pow(dt, 4)*v*swyawPow4*wvPow3*yPow4 + 70*pow(dt, 4)*swyawPow4*wvPow4*yPow4 + 56*pow(dt, 3)*pow(v, 3)*swyawPow3*yPow5 + 168*pow(dt, 3)*pow(v, 2)*swyawPow3*wvPow1*yPow5 + 168*pow(dt, 3)*v*swyawPow3*wvPow2*yPow5 + 56*pow(dt, 3)*swyawPow3*wvPow3*yPow5 + 28*pow(dt, 2)*pow(v, 2)*swyawPow2*yPow6 + 56*pow(dt, 2)*v*swyawPow2*wvPow1*yPow6 + 28*pow(dt, 2)*swyawPow2*wvPow2*yPow6 + 8*dt*v*swyawPow1*yPow7 + 8*dt*swyawPow1*wvPow1*yPow7 + yPow8;
    next_moments.xPow4_yPow4 = pow(dt, 8)*pow(v, 8)*cwyawPow4_swyawPow4 + 8*pow(dt, 8)*pow(v, 7)*cwyawPow4_swyawPow4*wvPow1 + 28*pow(dt, 8)*pow(v, 6)*cwyawPow4_swyawPow4*wvPow2 + 56*pow(dt, 8)*pow(v, 5)*cwyawPow4_swyawPow4*wvPow3 + 70*pow(dt, 8)*pow(v, 4)*cwyawPow4_swyawPow4*wvPow4 + 56*pow(dt, 8)*pow(v, 3)*cwyawPow4_swyawPow4*wvPow5 + 28*pow(dt, 8)*pow(v, 2)*cwyawPow4_swyawPow4*wvPow6 + 8*pow(dt, 8)*v*cwyawPow4_swyawPow4*wvPow7 + pow(dt, 8)*cwyawPow4_swyawPow4*wvPow8 + 4*pow(dt, 7)*pow(v, 7)*cwyawPow3_swyawPow4*xPow1 + 4*pow(dt, 7)*pow(v, 7)*cwyawPow4_swyawPow3*yPow1 + 28*pow(dt, 7)*pow(v, 6)*cwyawPow3_swyawPow4*wvPow1*xPow1 + 28*pow(dt, 7)*pow(v, 6)*cwyawPow4_swyawPow3*wvPow1*yPow1 + 84*pow(dt, 7)*pow(v, 5)*cwyawPow3_swyawPow4*wvPow2*xPow1 + 84*pow(dt, 7)*pow(v, 5)*cwyawPow4_swyawPow3*wvPow2*yPow1 + 140*pow(dt, 7)*pow(v, 4)*cwyawPow3_swyawPow4*wvPow3*xPow1 + 140*pow(dt, 7)*pow(v, 4)*cwyawPow4_swyawPow3*wvPow3*yPow1 + 140*pow(dt, 7)*pow(v, 3)*cwyawPow3_swyawPow4*wvPow4*xPow1 + 140*pow(dt, 7)*pow(v, 3)*cwyawPow4_swyawPow3*wvPow4*yPow1 + 84*pow(dt, 7)*pow(v, 2)*cwyawPow3_swyawPow4*wvPow5*xPow1 + 84*pow(dt, 7)*pow(v, 2)*cwyawPow4_swyawPow3*wvPow5*yPow1 + 28*pow(dt, 7)*v*cwyawPow3_swyawPow4*wvPow6*xPow1 + 28*pow(dt, 7)*v*cwyawPow4_swyawPow3*wvPow6*yPow1 + 4*pow(dt, 7)*cwyawPow3_swyawPow4*wvPow7*xPow1 + 4*pow(dt, 7)*cwyawPow4_swyawPow3*wvPow7*yPow1 + 6*pow(dt, 6)*pow(v, 6)*cwyawPow2_swyawPow4*xPow2 + 16*pow(dt, 6)*pow(v, 6)*cwyawPow3_swyawPow3*xPow1_yPow1 + 6*pow(dt, 6)*pow(v, 6)*cwyawPow4_swyawPow2*yPow2 + 36*pow(dt, 6)*pow(v, 5)*cwyawPow2_swyawPow4*wvPow1*xPow2 + 96*pow(dt, 6)*pow(v, 5)*cwyawPow3_swyawPow3*wvPow1*xPow1_yPow1 + 36*pow(dt, 6)*pow(v, 5)*cwyawPow4_swyawPow2*wvPow1*yPow2 + 90*pow(dt, 6)*pow(v, 4)*cwyawPow2_swyawPow4*wvPow2*xPow2 + 240*pow(dt, 6)*pow(v, 4)*cwyawPow3_swyawPow3*wvPow2*xPow1_yPow1 + 90*pow(dt, 6)*pow(v, 4)*cwyawPow4_swyawPow2*wvPow2*yPow2 + 120*pow(dt, 6)*pow(v, 3)*cwyawPow2_swyawPow4*wvPow3*xPow2 + 320*pow(dt, 6)*pow(v, 3)*cwyawPow3_swyawPow3*wvPow3*xPow1_yPow1 + 120*pow(dt, 6)*pow(v, 3)*cwyawPow4_swyawPow2*wvPow3*yPow2 + 90*pow(dt, 6)*pow(v, 2)*cwyawPow2_swyawPow4*wvPow4*xPow2 + 240*pow(dt, 6)*pow(v, 2)*cwyawPow3_swyawPow3*wvPow4*xPow1_yPow1 + 90*pow(dt, 6)*pow(v, 2)*cwyawPow4_swyawPow2*wvPow4*yPow2 + 36*pow(dt, 6)*v*cwyawPow2_swyawPow4*wvPow5*xPow2 + 96*pow(dt, 6)*v*cwyawPow3_swyawPow3*wvPow5*xPow1_yPow1 + 36*pow(dt, 6)*v*cwyawPow4_swyawPow2*wvPow5*yPow2 + 6*pow(dt, 6)*cwyawPow2_swyawPow4*wvPow6*xPow2 + 16*pow(dt, 6)*cwyawPow3_swyawPow3*wvPow6*xPow1_yPow1 + 6*pow(dt, 6)*cwyawPow4_swyawPow2*wvPow6*yPow2 + 4*pow(dt, 5)*pow(v, 5)*cwyawPow1_swyawPow4*xPow3 + 24*pow(dt, 5)*pow(v, 5)*cwyawPow2_swyawPow3*xPow2_yPow1 + 24*pow(dt, 5)*pow(v, 5)*cwyawPow3_swyawPow2*xPow1_yPow2 + 4*pow(dt, 5)*pow(v, 5)*cwyawPow4_swyawPow1*yPow3 + 20*pow(dt, 5)*pow(v, 4)*cwyawPow1_swyawPow4*wvPow1*xPow3 + 120*pow(dt, 5)*pow(v, 4)*cwyawPow2_swyawPow3*wvPow1*xPow2_yPow1 + 120*pow(dt, 5)*pow(v, 4)*cwyawPow3_swyawPow2*wvPow1*xPow1_yPow2 + 20*pow(dt, 5)*pow(v, 4)*cwyawPow4_swyawPow1*wvPow1*yPow3 + 40*pow(dt, 5)*pow(v, 3)*cwyawPow1_swyawPow4*wvPow2*xPow3 + 240*pow(dt, 5)*pow(v, 3)*cwyawPow2_swyawPow3*wvPow2*xPow2_yPow1 + 240*pow(dt, 5)*pow(v, 3)*cwyawPow3_swyawPow2*wvPow2*xPow1_yPow2 + 40*pow(dt, 5)*pow(v, 3)*cwyawPow4_swyawPow1*wvPow2*yPow3 + 40*pow(dt, 5)*pow(v, 2)*cwyawPow1_swyawPow4*wvPow3*xPow3 + 240*pow(dt, 5)*pow(v, 2)*cwyawPow2_swyawPow3*wvPow3*xPow2_yPow1 + 240*pow(dt, 5)*pow(v, 2)*cwyawPow3_swyawPow2*wvPow3*xPow1_yPow2 + 40*pow(dt, 5)*pow(v, 2)*cwyawPow4_swyawPow1*wvPow3*yPow3 + 20*pow(dt, 5)*v*cwyawPow1_swyawPow4*wvPow4*xPow3 + 120*pow(dt, 5)*v*cwyawPow2_swyawPow3*wvPow4*xPow2_yPow1 + 120*pow(dt, 5)*v*cwyawPow3_swyawPow2*wvPow4*xPow1_yPow2 + 20*pow(dt, 5)*v*cwyawPow4_swyawPow1*wvPow4*yPow3 + 4*pow(dt, 5)*cwyawPow1_swyawPow4*wvPow5*xPow3 + 24*pow(dt, 5)*cwyawPow2_swyawPow3*wvPow5*xPow2_yPow1 + 24*pow(dt, 5)*cwyawPow3_swyawPow2*wvPow5*xPow1_yPow2 + 4*pow(dt, 5)*cwyawPow4_swyawPow1*wvPow5*yPow3 + 16*pow(dt, 4)*pow(v, 4)*cwyawPow1_swyawPow3*xPow3_yPow1 + 36*pow(dt, 4)*pow(v, 4)*cwyawPow2_swyawPow2*xPow2_yPow2 + 16*pow(dt, 4)*pow(v, 4)*cwyawPow3_swyawPow1*xPow1_yPow3 + pow(dt, 4)*pow(v, 4)*cwyawPow4*yPow4 + pow(dt, 4)*pow(v, 4)*swyawPow4*xPow4 + 64*pow(dt, 4)*pow(v, 3)*cwyawPow1_swyawPow3*wvPow1*xPow3_yPow1 + 144*pow(dt, 4)*pow(v, 3)*cwyawPow2_swyawPow2*wvPow1*xPow2_yPow2 + 64*pow(dt, 4)*pow(v, 3)*cwyawPow3_swyawPow1*wvPow1*xPow1_yPow3 + 4*pow(dt, 4)*pow(v, 3)*cwyawPow4*wvPow1*yPow4 + 4*pow(dt, 4)*pow(v, 3)*swyawPow4*wvPow1*xPow4 + 96*pow(dt, 4)*pow(v, 2)*cwyawPow1_swyawPow3*wvPow2*xPow3_yPow1 + 216*pow(dt, 4)*pow(v, 2)*cwyawPow2_swyawPow2*wvPow2*xPow2_yPow2 + 96*pow(dt, 4)*pow(v, 2)*cwyawPow3_swyawPow1*wvPow2*xPow1_yPow3 + 6*pow(dt, 4)*pow(v, 2)*cwyawPow4*wvPow2*yPow4 + 6*pow(dt, 4)*pow(v, 2)*swyawPow4*wvPow2*xPow4 + 64*pow(dt, 4)*v*cwyawPow1_swyawPow3*wvPow3*xPow3_yPow1 + 144*pow(dt, 4)*v*cwyawPow2_swyawPow2*wvPow3*xPow2_yPow2 + 64*pow(dt, 4)*v*cwyawPow3_swyawPow1*wvPow3*xPow1_yPow3 + 4*pow(dt, 4)*v*cwyawPow4*wvPow3*yPow4 + 4*pow(dt, 4)*v*swyawPow4*wvPow3*xPow4 + 16*pow(dt, 4)*cwyawPow1_swyawPow3*wvPow4*xPow3_yPow1 + 36*pow(dt, 4)*cwyawPow2_swyawPow2*wvPow4*xPow2_yPow2 + 16*pow(dt, 4)*cwyawPow3_swyawPow1*wvPow4*xPow1_yPow3 + pow(dt, 4)*cwyawPow4*wvPow4*yPow4 + pow(dt, 4)*swyawPow4*wvPow4*xPow4 + 24*pow(dt, 3)*pow(v, 3)*cwyawPow1_swyawPow2*xPow3_yPow2 + 24*pow(dt, 3)*pow(v, 3)*cwyawPow2_swyawPow1*xPow2_yPow3 + 4*pow(dt, 3)*pow(v, 3)*cwyawPow3*xPow1_yPow4 + 4*pow(dt, 3)*pow(v, 3)*swyawPow3*xPow4_yPow1 + 72*pow(dt, 3)*pow(v, 2)*cwyawPow1_swyawPow2*wvPow1*xPow3_yPow2 + 72*pow(dt, 3)*pow(v, 2)*cwyawPow2_swyawPow1*wvPow1*xPow2_yPow3 + 12*pow(dt, 3)*pow(v, 2)*cwyawPow3*wvPow1*xPow1_yPow4 + 12*pow(dt, 3)*pow(v, 2)*swyawPow3*wvPow1*xPow4_yPow1 + 72*pow(dt, 3)*v*cwyawPow1_swyawPow2*wvPow2*xPow3_yPow2 + 72*pow(dt, 3)*v*cwyawPow2_swyawPow1*wvPow2*xPow2_yPow3 + 12*pow(dt, 3)*v*cwyawPow3*wvPow2*xPow1_yPow4 + 12*pow(dt, 3)*v*swyawPow3*wvPow2*xPow4_yPow1 + 24*pow(dt, 3)*cwyawPow1_swyawPow2*wvPow3*xPow3_yPow2 + 24*pow(dt, 3)*cwyawPow2_swyawPow1*wvPow3*xPow2_yPow3 + 4*pow(dt, 3)*cwyawPow3*wvPow3*xPow1_yPow4 + 4*pow(dt, 3)*swyawPow3*wvPow3*xPow4_yPow1 + 16*pow(dt, 2)*pow(v, 2)*cwyawPow1_swyawPow1*xPow3_yPow3 + 6*pow(dt, 2)*pow(v, 2)*cwyawPow2*xPow2_yPow4 + 6*pow(dt, 2)*pow(v, 2)*swyawPow2*xPow4_yPow2 + 32*pow(dt, 2)*v*cwyawPow1_swyawPow1*wvPow1*xPow3_yPow3 + 12*pow(dt, 2)*v*cwyawPow2*wvPow1*xPow2_yPow4 + 12*pow(dt, 2)*v*swyawPow2*wvPow1*xPow4_yPow2 + 16*pow(dt, 2)*cwyawPow1_swyawPow1*wvPow2*xPow3_yPow3 + 6*pow(dt, 2)*cwyawPow2*wvPow2*xPow2_yPow4 + 6*pow(dt, 2)*swyawPow2*wvPow2*xPow4_yPow2 + 4*dt*v*cwyawPow1*xPow3_yPow4 + 4*dt*v*swyawPow1*xPow4_yPow3 + 4*dt*cwyawPow1*wvPow1*xPow3_yPow4 + 4*dt*swyawPow1*wvPow1*xPow4_yPow3 + xPow4_yPow4;

    return next_moments;
}

StateInfo SquaredExampleHMKF::update(const PredictedMoments & predicted_moments,
                              const Eigen::VectorXd & observed_values,
                              const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    // predicted moments
    const double& xPow1 = predicted_moments.xPow1;
    const double& yPow1 = predicted_moments.yPow1;
    const double& xPow2 = predicted_moments.xPow2;
    const double& yPow2 = predicted_moments.yPow2;
    const double& xPow1_yPow1 = predicted_moments.xPow1_yPow1;

    // Predicted mean and covariance
    Eigen::Vector2d predicted_mean = Eigen::Vector2d::Zero();
    Eigen::Matrix2d predicted_cov = Eigen::Matrix2d::Zero();
    predicted_mean(STATE::IDX::X) = xPow1;
    predicted_mean(STATE::IDX::Y) = yPow1;
    predicted_cov(STATE::IDX::X, STATE::IDX::X) = xPow2 - xPow1*xPow1;
    predicted_cov(STATE::IDX::Y, STATE::IDX::Y) = yPow2 - yPow1*yPow1;
    predicted_cov(STATE::IDX::X, STATE::IDX::Y) = xPow1_yPow1 - xPow1*yPow1;
    predicted_cov(STATE::IDX::Y, STATE::IDX::X) = predicted_cov(STATE::IDX::X, STATE::IDX::Y);

    // Measurement mean and covariance
    Eigen::VectorXd measurement_mean = Eigen::VectorXd::Zero(1);
    Eigen::MatrixXd measurement_cov = Eigen::MatrixXd::Zero(1, 1);

    const auto meas_moments = getMeasurementMoments(predicted_moments, noise_map);

    measurement_mean(MEASUREMENT::IDX::R) = meas_moments.rPow1;
    measurement_cov(MEASUREMENT::IDX::R, MEASUREMENT::IDX::R) = meas_moments.rPow2 - meas_moments.rPow1*meas_moments.rPow1;

    const auto state_observation_cov = getStateMeasurementMatrix(predicted_moments, meas_moments, noise_map);

    // Kalman Gain
    const auto K = state_observation_cov * measurement_cov.inverse();

    StateInfo updated_info;
    updated_info.mean = predicted_mean + K * (observed_values - measurement_mean);
    updated_info.covariance = predicted_cov - K * measurement_cov * K.transpose();

    return updated_info;
}

MeasurementMoments SquaredExampleHMKF::getMeasurementMoments(const PredictedMoments & predicted_moments,
                                                             const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    const auto wr_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WR);

    // Measurement noise
    const double wrPow1 = wr_dist_ptr->calc_moment(1);
    const double wrPow2 = wr_dist_ptr->calc_moment(2);

    // predicted moments
    const double& xPow4 = predicted_moments.xPow4;
    const double& yPow4 = predicted_moments.yPow4;
    const double& xPow8 = predicted_moments.xPow8;
    const double& yPow8 = predicted_moments.yPow8;
    const double& xPow4_yPow4 = predicted_moments.xPow4_yPow4;

    MeasurementMoments meas_moments;
    meas_moments.rPow1 = xPow4 + yPow4 + wrPow1;
    meas_moments.rPow2 = xPow8 + yPow8 + wrPow2 + 2.0*xPow4_yPow4 + 2.0*xPow4*wrPow1 + 2.0*yPow4*wrPow1;

    return meas_moments;
}

Eigen::MatrixXd SquaredExampleHMKF::getStateMeasurementMatrix(const PredictedMoments& predicted_moments,
                                                              const MeasurementMoments & measurement_moments,
                                                              const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map)
{
    const auto wr_dist_ptr = noise_map.at(MEASUREMENT_NOISE::IDX::WR);

    // predicted moments
    const double& xPow1 = predicted_moments.xPow1;
    const double& yPow1 = predicted_moments.yPow1;
    const double& xPow5 = predicted_moments.xPow5;
    const double& yPow5 = predicted_moments.yPow5;
    const double& xPow1_yPow4 = predicted_moments.xPow1_yPow4;
    const double& xPow4_yPow1 = predicted_moments.xPow4_yPow1;

    // Measurement noise
    const double wrPow1 = wr_dist_ptr->calc_moment(1);

    // measurement moments
    const double& mrPow1 = measurement_moments.rPow1;

    // x*(x^4 + y^4 + w_r) = x^5 + xy^4 + x*w_r
    // y*(x^4 + y^4 + w_r) = x^4y + y^5 + y*w_r
    Eigen::MatrixXd state_observation_cov = Eigen::MatrixXd::Zero(2, 1); // sigma = E[XY^T] - E[X]E[Y]^T
    state_observation_cov(STATE::IDX::X, MEASUREMENT::IDX::R) = xPow5 + xPow1_yPow4 + xPow1*wrPow1 - xPow1*mrPow1;
    state_observation_cov(STATE::IDX::Y, MEASUREMENT::IDX::R) = xPow4_yPow1 + yPow5 + yPow1*wrPow1 - yPow1*mrPow1;

    return state_observation_cov;
}
