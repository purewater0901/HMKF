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

    std::complex<double> calc_characteristic(const int t);
    std::complex<double> calc_diff_characteristic(const int t, const int order);

    double lambda_;

private:
};

#endif //UNCERTAINTY_PROPAGATION_EXPONENTIAL_DISTRIBUTION_H
