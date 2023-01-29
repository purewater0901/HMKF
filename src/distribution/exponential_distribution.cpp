#include "distribution/exponential_distribution.h"

ExponentialDistribution::ExponentialDistribution(const double lambda) : lambda_(lambda)
{
}

double ExponentialDistribution::calc_mean()
{
    return 1.0/lambda_;
}

double ExponentialDistribution::calc_variance()
{
    return  1.0/(lambda_*lambda_);
}

std::complex<double> ExponentialDistribution::calc_characteristic(const std::complex<double>& t)
{
    const std::complex<double> i(0.0, 1.0);
    return lambda_ / (lambda_ - i *t);
}

std::complex<double> ExponentialDistribution::calc_diff_characteristic(const std::complex<double>& t, const int order)
{
    if(order==0) {
        return calc_characteristic(t);
    }

    const std::complex<double> i(0.0, 1.0);
    const auto tmp = lambda_ - i * t;
    return calc_diff_characteristic(t, order-1) * (static_cast<double>(order) * i / tmp);
}
