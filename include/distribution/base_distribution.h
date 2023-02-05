#ifndef UNCERTAINTY_PROPAGATION_BASE_DISTRIBUTION_H
#define UNCERTAINTY_PROPAGATION_BASE_DISTRIBUTION_H

#include <iostream>
#include <cmath>
#include <vector>
#include <Eigen/Eigen>

class BaseDistribution{
public:
    BaseDistribution() = default;

    virtual double calc_mean() = 0;
    virtual double calc_variance() = 0;

    virtual std::complex<double> calc_characteristic(const std::complex<double>& t) = 0;
    virtual std::complex<double> calc_diff_characteristic(const std::complex<double>& t, const int order) = 0;

    double calc_moment(const int order);
    double calc_exp_moment(const int order);
    double calc_cos_moment(const int order);
    double calc_sin_moment(const int order);
    double calc_cos_sin_moment(const int cos_order, const int sin_order);
    double calc_x_cos_moment(const int x_order, const int cos_order);
    double calc_x_sin_moment(const int x_order, const int sin_order);
    double calc_exp_cos_moment(const int exp_order, const int cos_order);
    double calc_exp_sin_moment(const int exp_order, const int sin_order);
    double calc_x_cos_sin_moment(const int x_order, const int cos_order, const int sin_order);
    double calc_exp_x_cos_sin_moment(const int exp_order, const int x_order, const int cos_order, const int sin_order);
};

#endif //UNCERTAINTY_PROPAGATION_BASE_DISTRIBUTION_H
