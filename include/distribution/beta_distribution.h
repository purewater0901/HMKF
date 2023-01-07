#ifndef HMKF_BETA_DISTRIBUTION_H
#define HMKF_BETA_DISTRIBUTION_H

#include <iostream>
#include <vector>
#include <Eigen/Eigen>
#include <complex>

#include "distribution/base_distribution.h"
#include "utilities.h"

class BetaDistribution : public BaseDistribution {
public:
    BetaDistribution(const double alpha, const double beta);

    double calc_mean();
    double calc_variance();

    std::complex<double> calc_characteristic(const int t);
    std::complex<double> calc_diff_characteristic(const int t, const int order);

    double alpha_;
    double beta_;

private:
};

#endif //HMKF_BETA_DISTRIBUTION_H
