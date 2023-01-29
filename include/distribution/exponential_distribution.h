#ifndef UNCERTAINTY_PROPAGATION_EXPONENTIAL_DISTRIBUTION_H
#define UNCERTAINTY_PROPAGATION_EXPONENTIAL_DISTRIBUTION_H

#include <iostream>
#include <vector>
#include <Eigen/Eigen>
#include <complex>

#include "distribution/base_distribution.h"
#include "utilities.h"

class ExponentialDistribution : public BaseDistribution {
public:
    ExponentialDistribution(const double lambda);

    double calc_mean();
    double calc_variance();

    std::complex<double> calc_characteristic(const std::complex<double>& t);
    std::complex<double> calc_diff_characteristic(const std::complex<double>& t, const int order);

    double lambda_;

private:
};

#endif //UNCERTAINTY_PROPAGATION_EXPONENTIAL_DISTRIBUTION_H
