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

std::complex<double> ExponentialDistribution::calc_characteristic(const int t)
{
    const std::complex<double> i(0.0, 1.0);
    const auto t_double = static_cast<double>(t);
    return lambda_ / (lambda_ - i *t_double);
}

std::complex<double> ExponentialDistribution::calc_diff_characteristic(const int t, const int order)
{
    if(order==0) {
        return calc_characteristic(t);
    }

    const std::complex<double> i(0.0, 1.0);
    const auto t_double = static_cast<double>(t);
    const auto order_double = static_cast<double>(order);
    const auto tmp = lambda_ - i * t_double;
    return calc_diff_characteristic(t, order-1) * (order_double * i / tmp);
}
