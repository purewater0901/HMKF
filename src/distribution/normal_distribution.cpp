#include "distribution/normal_distribution.h"

NormalDistribution::NormalDistribution(const double mean, const double variance) : mean_(mean), variance_(variance)
{
}

double NormalDistribution::calc_mean()
{
    return mean_;
}

double NormalDistribution::calc_variance()
{
    return  variance_;
}

std::complex<double> NormalDistribution::calc_characteristic(const std::complex<double>& t)
{
    const std::complex<double> i(0.0, 1.0);
    return std::exp(i*t*mean_ - variance_*std::pow(t, 2)*0.5);
}

std::complex<double> NormalDistribution::calc_first_diff_characteristic(const std::complex<double>& t)
{
    const std::complex<double> i(0.0, 1.0);
    const auto tmp = i*mean_ - variance_*t;
    return tmp * calc_characteristic(t);
}

std::complex<double> NormalDistribution::calc_diff_characteristic(const std::complex<double>& t, const int order)
{
    if(order==0) {
        return calc_characteristic(t);
    }

    if(order == 1) {
        return calc_first_diff_characteristic(t);
    }

    const std::complex<double> i(0.0, 1.0);
    const auto tmp = (i*mean_ - variance_*t);

    return -(order-1) * variance_ * calc_diff_characteristic(t, order-2) + tmp* calc_diff_characteristic(t, order-1);
}
