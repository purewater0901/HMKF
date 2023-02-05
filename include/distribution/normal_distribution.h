#ifndef UNCERTAINTY_PROPAGATION_NORMAL_DISTRIBUTION_H
#define UNCERTAINTY_PROPAGATION_NORMAL_DISTRIBUTION_H

#include <iostream>
#include <vector>
#include <Eigen/Eigen>
#include <complex>

#include "distribution/base_distribution.h"
#include "utilities.h"

class NormalDistribution : public BaseDistribution{
public:
    NormalDistribution() : mean_(0.0), variance_(1.0) {}
    NormalDistribution(const double mean, const double variance);

    double calc_mean();
    double calc_variance();

    std::complex<double> calc_characteristic(const std::complex<double>& t);
    std::complex<double> calc_first_diff_characteristic(const std::complex<double>& t);

    std::complex<double> calc_diff_characteristic(const std::complex<double>& t, const int order);

    double mean_;
    double variance_;

private:
};

#endif //UNCERTAINTY_PROPAGATION_NORMAL_DISTRIBUTION_H
